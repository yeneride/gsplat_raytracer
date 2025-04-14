#include "auxiliary.h"
#include "float3_utils.h"
#include "grid.h"
#include "raytrace.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ void computeColorFromSHBackward(int idx, int deg, int max_coeffs,
                                           const float3 *means, float3 campos,
                                           const float *shs,
                                           const bool *clamped,
                                           const float3 *dL_dcolor,
                                           float3 *dL_dmeans, float3 *dL_dshs) {
  float3 pos = means[idx];
  float3 dir_orig = pos - campos;
  float3 dir = normalize(dir_orig);

  float3 *sh = ((float3 *)shs) + idx * max_coeffs;
  float3 *dL_dsh = dL_dshs + idx * max_coeffs;

  // Use clamping rule: if clamping was applied, gradient becomes 0
  float3 dL_dRGB = dL_dcolor[idx];
  dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
  dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
  dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

  float3 dRGBdx = make_float3(0, 0, 0);
  float3 dRGBdy = make_float3(0, 0, 0);
  float3 dRGBdz = make_float3(0, 0, 0);
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;

  // No tricks here, just high school-level calculus.
  float dRGBdsh0 = SH_C0;
  dL_dsh[0] = dRGBdsh0 * dL_dRGB;

  if (deg > 0) {
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    dL_dsh[1] = dRGBdsh1 * dL_dRGB;
    dL_dsh[2] = dRGBdsh2 * dL_dRGB;
    dL_dsh[3] = dRGBdsh3 * dL_dRGB;

    dRGBdx = -SH_C1 * sh[3];
    dRGBdy = -SH_C1 * sh[1];
    dRGBdz = SH_C1 * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float dRGBdsh4 = SH_C2[0] * xy;
      float dRGBdsh5 = SH_C2[1] * yz;
      float dRGBdsh6 = SH_C2[2] * (2.0f * zz - xx - yy);
      float dRGBdsh7 = SH_C2[3] * xz;
      float dRGBdsh8 = SH_C2[4] * (xx - yy);
      dL_dsh[4] = dRGBdsh4 * dL_dRGB;
      dL_dsh[5] = dRGBdsh5 * dL_dRGB;
      dL_dsh[6] = dRGBdsh6 * dL_dRGB;
      dL_dsh[7] = dRGBdsh7 * dL_dRGB;
      dL_dsh[8] = dRGBdsh8 * dL_dRGB;

      dRGBdx = dRGBdx + SH_C2[0] * y * sh[4] + SH_C2[2] * 2.0f * -x * sh[6] +
               SH_C2[3] * z * sh[7] + SH_C2[4] * 2.0f * x * sh[8];
      dRGBdy = dRGBdy + SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] +
               SH_C2[2] * 2.0f * -y * sh[6] + SH_C2[4] * 2.0f * -y * sh[8];
      dRGBdz = dRGBdz + SH_C2[1] * y * sh[5] +
               SH_C2[2] * 2.0f * 2.0f * z * sh[6] + SH_C2[3] * x * sh[7];

      if (deg > 2) {
        float dRGBdsh9 = SH_C3[0] * y * (3.0f * xx - yy);
        float dRGBdsh10 = SH_C3[1] * xy * z;
        float dRGBdsh11 = SH_C3[2] * y * (4.0f * zz - xx - yy);
        float dRGBdsh12 = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        float dRGBdsh13 = SH_C3[4] * x * (4.0f * zz - xx - yy);
        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.0f * yy);
        dL_dsh[9] = dRGBdsh9 * dL_dRGB;
        dL_dsh[10] = dRGBdsh10 * dL_dRGB;
        dL_dsh[11] = dRGBdsh11 * dL_dRGB;
        dL_dsh[12] = dRGBdsh12 * dL_dRGB;
        dL_dsh[13] = dRGBdsh13 * dL_dRGB;
        dL_dsh[14] = dRGBdsh14 * dL_dRGB;
        dL_dsh[15] = dRGBdsh15 * dL_dRGB;

        dRGBdx =
            dRGBdx + (SH_C3[0] * sh[9] * 3.0f * 2.0f * xy +
                      SH_C3[1] * sh[10] * yz + SH_C3[2] * sh[11] * -2.0f * xy +
                      SH_C3[3] * sh[12] * -3.0f * 2.0f * xz +
                      SH_C3[4] * sh[13] * (-3.0f * xx + 4.0f * zz - yy) +
                      SH_C3[5] * sh[14] * 2.0f * xz +
                      SH_C3[6] * sh[15] * 3.0f * (xx - yy));

        dRGBdy =
            dRGBdy +
            (SH_C3[0] * sh[9] * 3.0f * (xx - yy) + SH_C3[1] * sh[10] * xz +
             SH_C3[2] * sh[11] * (-3.0f * yy + 4.0f * zz - xx) +
             SH_C3[3] * sh[12] * -3.0f * 2.0f * yz +
             SH_C3[4] * sh[13] * -2.0f * xy + SH_C3[5] * sh[14] * -2.0f * yz +
             SH_C3[6] * sh[15] * -3.0f * 2.0f * xy);

        dRGBdz = dRGBdz + (SH_C3[1] * sh[10] * xy +
                           SH_C3[2] * sh[11] * 4.0f * 2.0f * yz +
                           SH_C3[3] * sh[12] * 3.0f * (2.0f * zz - xx - yy) +
                           SH_C3[4] * sh[13] * 4.0f * 2.0f * xz +
                           SH_C3[5] * sh[14] * (xx - yy));
      }
    }
  }

  // The view direction is an input to the computation. View direction
  // is influenced by the Gaussian's mean, so SHs gradients
  // must propagate back into 3D position.
  float3 dL_ddir = make_float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB),
                               dot(dRGBdz, dL_dRGB));

  // Account for normalization of direction
  float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the view-dependent color.
  dL_dmeans[idx] = dL_dmeans[idx] + dL_dmean;
}

__device__ void computeCov3DBackward(int idx, const float3 scale, float mod,
                                     const float4 rot, const float *dL_dcov3Ds,
                                     float3 *dL_dscales, float4 *dL_drots) {
  // Recompute (intermediate) results for the 3D covariance computation.
  float4 q = rot;
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix R from quaternion
  float m00 = 1.f - 2.f * (y * y + z * z);
  float m01 = 2.f * (x * y - r * z);
  float m02 = 2.f * (x * z + r * y);

  float m10 = 2.f * (x * y + r * z);
  float m11 = 1.f - 2.f * (x * x + z * z);
  float m12 = 2.f * (y * z - r * x);

  float m20 = 2.f * (x * z - r * y);
  float m21 = 2.f * (y * z + r * x);
  float m22 = 1.f - 2.f * (x * x + y * y);

  // Compute scale matrix S
  float3 s = make_float3(mod * scale.x, mod * scale.y, mod * scale.z);

  // Compute M = S * R
  float M00 = s.x * m00;
  float M01 = s.x * m01;
  float M02 = s.x * m02;

  float M10 = s.y * m10;
  float M11 = s.y * m11;
  float M12 = s.y * m12;

  float M20 = s.z * m20;
  float M21 = s.z * m21;
  float M22 = s.z * m22;

  const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

  // Convert per-element covariance loss gradients to matrix form
  float dL_dSigma00 = dL_dcov3D[0];
  float dL_dSigma01 = 0.5f * dL_dcov3D[1];
  float dL_dSigma02 = 0.5f * dL_dcov3D[2];
  float dL_dSigma11 = dL_dcov3D[3];
  float dL_dSigma12 = 0.5f * dL_dcov3D[4];
  float dL_dSigma22 = dL_dcov3D[5];

  // Compute loss gradient w.r.t. matrix M
  // dSigma_dM = 2 * M
  float dL_dM00 =
      2.0f * (M00 * dL_dSigma00 + M01 * dL_dSigma01 + M02 * dL_dSigma02);
  float dL_dM01 =
      2.0f * (M00 * dL_dSigma01 + M01 * dL_dSigma11 + M02 * dL_dSigma12);
  float dL_dM02 =
      2.0f * (M00 * dL_dSigma02 + M01 * dL_dSigma12 + M02 * dL_dSigma22);

  float dL_dM10 =
      2.0f * (M10 * dL_dSigma00 + M11 * dL_dSigma01 + M12 * dL_dSigma02);
  float dL_dM11 =
      2.0f * (M10 * dL_dSigma01 + M11 * dL_dSigma11 + M12 * dL_dSigma12);
  float dL_dM12 =
      2.0f * (M10 * dL_dSigma02 + M11 * dL_dSigma12 + M12 * dL_dSigma22);

  float dL_dM20 =
      2.0f * (M20 * dL_dSigma00 + M21 * dL_dSigma01 + M22 * dL_dSigma02);
  float dL_dM21 =
      2.0f * (M20 * dL_dSigma01 + M21 * dL_dSigma11 + M22 * dL_dSigma12);
  float dL_dM22 =
      2.0f * (M20 * dL_dSigma02 + M21 * dL_dSigma12 + M22 * dL_dSigma22);

  // Gradients of loss w.r.t. scale
  float3 *dL_dscale = dL_dscales + idx;
  dL_dscale->x = m00 * dL_dM00 + m10 * dL_dM10 + m20 * dL_dM20;
  dL_dscale->y = m01 * dL_dM01 + m11 * dL_dM11 + m21 * dL_dM21;
  dL_dscale->z = m02 * dL_dM02 + m12 * dL_dM12 + m22 * dL_dM22;

  // Scale gradients by the scale modifier
  dL_dscale->x *= mod;
  dL_dscale->y *= mod;
  dL_dscale->z *= mod;

  // Gradients of loss w.r.t. rotation (quaternion)
  float4 *dL_drot = dL_drots + idx;
  dL_drot->x = 2 * z * (dL_dM01 - dL_dM10) + 2 * y * (dL_dM20 - dL_dM02) +
               2 * x * (dL_dM12 - dL_dM21);
  dL_drot->y = 2 * y * (dL_dM10 + dL_dM01) + 2 * z * (dL_dM20 + dL_dM02) +
               2 * r * (dL_dM12 - dL_dM21) - 4 * x * (dL_dM22 + dL_dM11);
  dL_drot->z = 2 * x * (dL_dM10 + dL_dM01) + 2 * r * (dL_dM20 - dL_dM02) +
               2 * z * (dL_dM12 + dL_dM21) - 4 * y * (dL_dM22 + dL_dM00);
  dL_drot->w = 2 * r * (dL_dM01 - dL_dM10) + 2 * x * (dL_dM20 + dL_dM02) +
               2 * y * (dL_dM12 + dL_dM21) - 4 * z * (dL_dM11 + dL_dM00);
}

__global__ void traceRaysBackward(
    GridCell<64> *grid, float3 const *ellipsoidCenters,
    float3 const *ellipsoidRadii, float4 const *rotations, float scale_modifier,
    float3 cam_pos, float3 gridMin, float3 cellSize, int cellsPerAxis,
    int numEllipsoids, float tan_fovx, float tan_fovy, int width, int height,
    const float *shs, int sh_deg, int max_coeffs, const float *dL_dpixels,
    float3 *dL_dmeans, float3 *dL_dscales, float4 *dL_drots, float *dL_dshs) {
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

    // Get the gradient of the loss w.r.t. the pixel color
    float3 dL_dcolor =
        make_float3(dL_dpixels[pixelIdx], dL_dpixels[pixelIdx + 1],
                    dL_dpixels[pixelIdx + 2]);

    // Temporarily modify campos to create an unrotated view direction for
    // computeColorFromSH
    float3 modified_campos = ellipsoidCenters[hitEllipsoidIdx] - rotated_dir;

    // Compute gradients for SH coefficients and means
    computeColorFromSHBackward(hitEllipsoidIdx, sh_deg, max_coeffs,
                               ellipsoidCenters, modified_campos, shs, clamped,
                               &dL_dcolor, dL_dmeans, (float3 *)dL_dshs);

    // Compute gradients for scales and rotations
    // Note: For ray tracing, we need to compute gradients for the intersection
    // point which affects the scales and rotations differently than in
    // rasterization
    float3 hitPoint = cam_pos + rayDir * tMinHit;
    float3 localHit = hitPoint - ellipsoidCenters[hitEllipsoidIdx];

    // Compute gradients for scales and rotations based on the intersection
    // This is a simplified approach and may need adjustment based on your exact
    // requirements
    float3 scale = ellipsoidRadii[hitEllipsoidIdx];
    float3 invScale =
        make_float3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z);
    float3 localHitScaled =
        make_float3(localHit.x * invScale.x, localHit.y * invScale.y,
                    localHit.z * invScale.z);

    // Compute gradient for scales based on how they affect the intersection
    float3 dL_dscale =
        make_float3(-2.0f * localHitScaled.x * localHitScaled.x * scale.x,
                    -2.0f * localHitScaled.y * localHitScaled.y * scale.y,
                    -2.0f * localHitScaled.z * localHitScaled.z * scale.z);

    // Accumulate scale gradients
    atomicAdd(&dL_dscales[hitEllipsoidIdx].x, dL_dscale.x * scale_modifier);
    atomicAdd(&dL_dscales[hitEllipsoidIdx].y, dL_dscale.y * scale_modifier);
    atomicAdd(&dL_dscales[hitEllipsoidIdx].z, dL_dscale.z * scale_modifier);

    // Compute gradient for rotation (simplified)
    // This would need a more rigorous derivation based on your quaternion
    // rotation
    float4 dL_drot = make_float4(0, 0, 0, 0);
    // ... (rotation gradient computation would go here)

    // Accumulate rotation gradients
    atomicAdd(&dL_drots[hitEllipsoidIdx].x, dL_drot.x);
    atomicAdd(&dL_drots[hitEllipsoidIdx].y, dL_drot.y);
    atomicAdd(&dL_drots[hitEllipsoidIdx].z, dL_drot.z);
    atomicAdd(&dL_drots[hitEllipsoidIdx].w, dL_drot.w);
  }
}

int Raytracer::backward(std::function<char *(size_t)> geometryBuffer,
                        std::function<char *(size_t)> binningBuffer,
                        std::function<char *(size_t)> imageBuffer, const int P,
                        int D, int M, const float *background, const int width,
                        int height, const float3 *means3D, const float *shs,
                        const float3 *scales, const float scale_modifier,
                        const float4 *rotations, const float *cov3D_precomp,
                        const float *viewmatrix, const float *projmatrix,
                        const float *cam_pos, const float tan_fovx,
                        float tan_fovy, const bool prefiltered,
                        const float *dL_dpixels, float3 *dL_dmeans,
                        float3 *dL_dscales, float4 *dL_drots, float *dL_dshs) {
  GridCell<64> *d_grid = accelGrid->getDeviceGrid();
  float3 gridMin = accelGrid->getGridMin();
  float3 cellSize = accelGrid->getCellSize();
  int cellsPerAxis = accelGrid->getCellsPerAxis();

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  float3 camPos = {cam_pos[0], cam_pos[1], cam_pos[2]};

  // Initialize gradients to zero
  cudaMemset(dL_dmeans, 0, P * sizeof(float3));
  cudaMemset(dL_dscales, 0, P * sizeof(float3));
  cudaMemset(dL_drots, 0, P * sizeof(float4));
  cudaMemset(dL_dshs, 0, P * M * 3 * sizeof(float));

  traceRaysBackward<<<gridSize, blockSize>>>(
      d_grid, means3D, scales, rotations, scale_modifier, camPos, gridMin,
      cellSize, cellsPerAxis, P, tan_fovx, tan_fovy, width, height, shs, D, M,
      dL_dpixels, dL_dmeans, dL_dscales, dL_drots, dL_dshs);

  return 0;
}
