#pragma once

#ifndef FLOAT3_UTILS
#define FLOAT3_UTILS

__host__ __device__ static inline float dot(float3 x, float3 y) {
    return x.x * y.x + x.y * y.y + x.z * y.z;
}

__host__ __device__ static inline float3 operator+(float3 x, float3 y) {
    return {x.x + y.x, x.y + y.y, x.z + y.z};
}

__host__ __device__ static inline float3 operator-(float3 x, float3 y) {
    return {x.x - y.x, x.y - y.y, x.z - y.z};
}

__host__ __device__ static inline float3 operator*(float3 x, float3 y) {
    return {x.x * y.x, x.y * y.y, x.z * y.z};
}

__host__ __device__ static inline float3 operator*(float3 x, float y) {
    return {x.x * y, x.y * y, x.z * y};
}

__host__ __device__ static inline float3 operator*(float y, float3 x) {
    return {x.x * y, x.y * y, x.z * y};
}

__host__ __device__ static inline float3 operator/(float3 x, float3 y) {
    return {x.x / y.x, x.y / y.y, x.z / y.z};
}


__host__ __device__ static inline float3 operator/(float3 x, float y) {
    return {x.x / y, x.y / y, x.z / y};
}

__host__ __device__ static inline float3 operator/(float y, float3 x) {
    return {x.x / y, x.y / y, x.z / y};
}

__host__ __device__ static inline float3 min(float3 x, float3 y) {
    return {min(x.x, y.x), min(x.y, y.y), min(x.z, y.z)};
}

__host__ __device__ static inline float3 max(float3 x, float3 y) {
    return {max(x.x, y.x), max(x.y, y.y), max(x.z, y.z)};
}

__host__ __device__ static inline float3 normalize(float3 x) {
    float len = sqrt(dot(x, x));
    return {x.x / len, x.y / len, x.z / len};
}

#endif
