#pragma once

#include <__clang_cuda_runtime_wrapper.h>
#ifndef RAYTRACER_H
#define RAYTRACER_H

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#endif

#include "grid.h"
#include "float3_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <iostream>

class Raytracer {
private:
  AccelerationGrid<64> *accelGrid;

public:
  Raytracer(float gridSizeX, float gridSizeY, float gridSizeZ,
            int cellsPerAxis) {
    accelGrid = new AccelerationGrid<64>({gridSizeX, gridSizeY, gridSizeZ},
                                         cellsPerAxis);
  }

  ~Raytracer() { delete accelGrid; }

  int forward(std::function<char *(size_t)> geometryBuffer,
              std::function<char *(size_t)> binningBuffer,
              std::function<char *(size_t)> imageBuffer, const int P, int D,
              int M, const float *background, const int width, int height,
              const float3 *means3D, const float *shs, const float3 *scales,
              const float scale_modifier, const float4 *rotations,
              const float *cov3D_precomp, const float *viewmatrix,
              const float *projmatrix, const float *cam_pos,
              const float tan_fovx, float tan_fovy, const bool prefiltered,
              float *out_color, int *radii, bool debug);

  int backward(std::function<char *(size_t)> geometryBuffer,
               std::function<char *(size_t)> binningBuffer,
               std::function<char *(size_t)> imageBuffer, const int P, int D,
               int M, const float *background, const int width, int height,
               const float3 *means3D, const float *shs, const float3 *scales,
               const float scale_modifier, const float4 *rotations,
               const float *cov3D_precomp, const float *viewmatrix,
               const float *projmatrix, const float *cam_pos,
               const float tan_fovx, float tan_fovy, const bool prefiltered,
               const float *dL_dpixels, float3 *dL_dmeans, float3 *dL_dscales,
               float4 *dL_drots, float *dL_dshs);
};

__forceinline__ __host__ __device__ float3 applyRotation(float3 direction,
                                                         float4 rotation) {
  // Extract quaternion components
  float qx = rotation.x;
  float qy = rotation.y;
  float qz = rotation.z;
  float qw = rotation.w;

  // Apply quaternion rotation using the formula:
  // v' = q * v * q^-1 (where v is represented as a quaternion with w=0)

  // This can be optimized to the following:
  float3 rotated;
  float qw2 = qw * qw;
  float qx2 = qx * qx;
  float qy2 = qy * qy;
  float qz2 = qz * qz;

  // Calculate rotation matrix terms
  float m00 = qw2 + qx2 - qy2 - qz2;
  float m01 = 2.0f * (qx * qy - qw * qz);
  float m02 = 2.0f * (qx * qz + qw * qy);

  float m10 = 2.0f * (qx * qy + qw * qz);
  float m11 = qw2 - qx2 + qy2 - qz2;
  float m12 = 2.0f * (qy * qz - qw * qx);

  float m20 = 2.0f * (qx * qz - qw * qy);
  float m21 = 2.0f * (qy * qz + qw * qx);
  float m22 = qw2 - qx2 - qy2 + qz2;

  // Apply the rotation matrix
  rotated.x = m00 * direction.x + m01 * direction.y + m02 * direction.z;
  rotated.y = m10 * direction.x + m11 * direction.y + m12 * direction.z;
  rotated.z = m20 * direction.x + m21 * direction.y + m22 * direction.z;

  return rotated;
}

__forceinline__ __host__ __device__ bool
rayIntersectsEllipsoid(float3 rayOrigin, float3 rayDir, float3 center,
                       float3 radii, float &tNear) {
  float3 oc = {
      rayOrigin.x - center.x,
      rayOrigin.y - center.y,
      rayOrigin.z - center.z,
  };
  float3 invRadii = make_float3(1.0f / radii.x, 1.0f / radii.y, 1.0f / radii.z);

  float3 oc_scaled = {
      oc.x * invRadii.x,
      oc.y * invRadii.y,
      oc.z * invRadii.z,
  };
  float3 dir_scaled = {
      rayDir.x * invRadii.x,
      rayDir.y * invRadii.y,
      rayDir.z * invRadii.z,
  };

  float a = dot(dir_scaled, dir_scaled);
  float b = 2.0f * dot(oc_scaled, dir_scaled);
  float c = dot(oc_scaled, oc_scaled) - 1.0f;

  float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f)
    return false;

  float sqrtD = sqrtf(discriminant);
  float t1 = (-b - sqrtD) / (2.0f * a);
  float t2 = (-b + sqrtD) / (2.0f * a);

  if (t2 < 0.0f)
    return false;

  tNear = (t1 > 0.0f) ? t1 : t2;
  return true;
}

#endif
