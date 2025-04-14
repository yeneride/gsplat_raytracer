#pragma once

#ifndef FLOAT3_UTILS
#define FLOAT3_UTILS

__forceinline__ __host__ __device__ static float dot(float3 x, float3 y) {
  return x.x * y.x + x.y * y.y + x.z * y.z;
}

__forceinline__ __host__ __device__ static float3 operator+(float3 x,
                                                            float3 y) {
  return {x.x + y.x, x.y + y.y, x.z + y.z};
}

__forceinline__ __host__ __device__ static float3 operator-(float3 x,
                                                            float3 y) {
  return {x.x - y.x, x.y - y.y, x.z - y.z};
}

__forceinline__ __host__ __device__ static float3 operator*(float3 x,
                                                            float3 y) {
  return {x.x * y.x, x.y * y.y, x.z * y.z};
}

__forceinline__ __host__ __device__ static float3 operator*(float3 x, float y) {
  return {x.x * y, x.y * y, x.z * y};
}

__forceinline__ __host__ __device__ static float3 operator*(float y, float3 x) {
  return {x.x * y, x.y * y, x.z * y};
}

__forceinline__ __host__ __device__ static float3 operator/(float3 x,
                                                            float3 y) {
  return {x.x / y.x, x.y / y.y, x.z / y.z};
}

__forceinline__ __host__ __device__ static float3 operator/(float3 x, float y) {
  return {x.x / y, x.y / y, x.z / y};
}

__forceinline__ __host__ __device__ static float3 operator/(float y, float3 x) {
  return {x.x / y, x.y / y, x.z / y};
}

__forceinline__ __host__ __device__ static float3 min(float3 x, float3 y) {
  return {min(x.x, y.x), min(x.y, y.y), min(x.z, y.z)};
}

__forceinline__ __host__ __device__ static float3 max(float3 x, float3 y) {
  return {max(x.x, y.x), max(x.y, y.y), max(x.z, y.z)};
}

__forceinline__ __host__ __device__ static float3 normalize(float3 x) {
  float len = sqrt(dot(x, x));
  return {x.x / len, x.y / len, x.z / len};
}

__forceinline__ __host__ __device__ float3 dnormvdv(float3 v, float3 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x =
      ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) *
      invsum32;
  dnormvdv.y =
      (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) *
      invsum32;
  dnormvdv.z =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdv;
}

#endif
