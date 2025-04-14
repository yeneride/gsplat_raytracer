#include "auxiliary.h"
#include "float3_utils.h"
#include "grid.h"
#include "raytrace.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>

__device__ float3 computeColorFromSH(int idx, int deg, int max_coeffs,
                                     const float3 *means, float3 campos,
                                     const float *shs, bool *clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  float3 pos_float3 = means[idx];
  float3 pos = make_float3(pos_float3.x, pos_float3.y, pos_float3.z);
  float3 dir = normalize(pos - campos);

  float3 *sh = ((float3 *)shs) + idx * max_coeffs;
  float3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                 SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                 SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result = result + make_float3(0.5f, 0.5f, 0.5f);

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return max(result, make_float3(0.0f, 0.0f, 0.0f));
}

__global__ void traceRays(GridCell<64> *grid, float3 const *ellipsoidCenters,
                          float3 const *ellipsoidRadii, float4 const *rotations,
                          float scale_modifier, float3 cam_pos, float3 gridMin,
                          float3 cellSize, int cellsPerAxis, int numEllipsoids,
                          float tan_fovx, float tan_fovy, int width, int height,
                          const float *shs, int sh_deg, int max_coeffs,
                          float *out_color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float u = (2.0f * (x + 0.5f) / width - 1.0f) * tan_fovx;
  float v = (1.0f - 2.0f * (y + 0.5f) / height) * tan_fovy;

  float3 rayDir = normalize(make_float3(u, v, 1.0f));

  float3 tMin = (gridMin - cam_pos) / rayDir;
  float3 tMax =
      (gridMin +
       make_float3(cellsPerAxis * cellSize.x, cellsPerAxis * cellSize.y,
                   cellsPerAxis * cellSize.z) -
       cam_pos) /
      rayDir;

  float3 tEnter = min(tMin, tMax);
  float3 tExit = max(tMin, tMax);

  float tStart = max(0.0f, max(tEnter.x, max(tEnter.y, tEnter.z)));
  float tEnd = min(tExit.x, min(tExit.y, tExit.z));

  if (tStart >= tEnd)
    return;

  float3 hitColor = make_float3(0, 0, 0);
  float tMinHit = FLT_MAX;
  int hitEllipsoidIdx = -1;

  int3 step = make_int3(signbit(rayDir.x) ? -1 : 1, signbit(rayDir.y) ? -1 : 1,
                        signbit(rayDir.z) ? -1 : 1);

  int3 gridIndex =
      make_int3((cam_pos.x + tStart * rayDir.x - gridMin.x) / cellSize.x,
                (cam_pos.y + tStart * rayDir.y - gridMin.y) / cellSize.y,
                (cam_pos.z + tStart * rayDir.z - gridMin.z) / cellSize.z);

  float3 nextBoundary =
      make_float3((gridIndex.x + (step.x > 0)) * cellSize.x + gridMin.x,
                  (gridIndex.y + (step.y > 0)) * cellSize.y + gridMin.y,
                  (gridIndex.z + (step.z > 0)) * cellSize.z + gridMin.z);

  float3 tNext = (nextBoundary - cam_pos) / rayDir;
  float3 deltaT =
      make_float3(fabsf(cellSize.x / rayDir.x), fabsf(cellSize.y / rayDir.y),
                  fabsf(cellSize.z / rayDir.z));

  while (gridIndex.x >= 0 && gridIndex.x < cellsPerAxis && gridIndex.y >= 0 &&
         gridIndex.y < cellsPerAxis && gridIndex.z >= 0 &&
         gridIndex.z < cellsPerAxis && tStart < tEnd) {
    int cellIdx =
        gridIndex.x + cellsPerAxis * (gridIndex.y + cellsPerAxis * gridIndex.z);
    GridCell<64> cell = grid[cellIdx];

    for (int j = 0; j < cell.pointCount; j++) {
      int ellipsoidIdx = cell.point_idx[j];
      if (ellipsoidIdx >= numEllipsoids)
        continue;

      float3 center = ellipsoidCenters[ellipsoidIdx];
      float3 radii = ellipsoidRadii[ellipsoidIdx] * scale_modifier;

      float tNear;
      if (rayIntersectsEllipsoid(cam_pos, rayDir, center, radii, tNear) &&
          tNear < tMinHit) {
        tMinHit = tNear;
        hitEllipsoidIdx = ellipsoidIdx;
      }
    }

    if (tNext.x < tNext.y && tNext.x < tNext.z) {
      tStart = tNext.x;
      gridIndex.x += step.x;
      tNext.x += deltaT.x;
    } else if (tNext.y < tNext.z) {
      tStart = tNext.y;
      gridIndex.y += step.y;
      tNext.y += deltaT.y;
    } else {
      tStart = tNext.z;
      gridIndex.z += step.z;
      tNext.z += deltaT.z;
    }
  }

  int pixelIdx = (y * width + x) * 3;
  if (hitEllipsoidIdx >= 0) {
    bool clamped[3];
    float3 campos_glm = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);

    float4 rotation = rotations[hitEllipsoidIdx];
    float4 inverseRotation =
        make_float4(-rotation.x, -rotation.y, -rotation.z, rotation.w);
    float3 rotated_dir = applyRotation(rayDir, inverseRotation);

    // Temporarily modify campos to create an unrotated view direction for
    // computeColorFromSH We'll construct a fake camera position that would
    // result in our rotated direction
    float3 modified_campos = ellipsoidCenters[hitEllipsoidIdx] - rotated_dir;

    float3 color =
        computeColorFromSH(hitEllipsoidIdx, sh_deg, max_coeffs,
                           ellipsoidCenters, modified_campos, shs, clamped);

    out_color[pixelIdx] = color.x;
    out_color[pixelIdx + 1] = color.y;
    out_color[pixelIdx + 2] = color.z;
  } else {
    out_color[pixelIdx] = 0.0f;
    out_color[pixelIdx + 1] = 0.0f;
    out_color[pixelIdx + 2] = 0.0f;
  }
}

int Raytracer::forward(std::function<char *(size_t)> geometryBuffer,
                       std::function<char *(size_t)> binningBuffer,
                       std::function<char *(size_t)> imageBuffer, const int P,
                       int D, int M, const float *background, const int width,
                       int height, const float3 *means3D, const float *shs,
                       const float3 *scales, const float scale_modifier,
                       const float4 *rotations, const float *cov3D_precomp,
                       const float *viewmatrix, const float *projmatrix,
                       const float *cam_pos, const float tan_fovx,
                       float tan_fovy, const bool prefiltered, float *out_color,
                       int *radii, bool debug) {
  accelGrid->build(means3D, scales, P);

  GridCell<64> *d_grid = accelGrid->getDeviceGrid();
  float3 gridMin = accelGrid->getGridMin();
  float3 cellSize = accelGrid->getCellSize();
  int cellsPerAxis = accelGrid->getCellsPerAxis();

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  float3 camPos = {cam_pos[0], cam_pos[1], cam_pos[2]};

  traceRays<<<gridSize, blockSize>>>(d_grid, means3D, scales, rotations,
                                     scale_modifier, camPos, gridMin, cellSize,
                                     cellsPerAxis, P, tan_fovx, tan_fovy, width,
                                     height, shs, D, M, out_color);

  return 0;
}
