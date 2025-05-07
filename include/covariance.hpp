#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Covariance {
private:
  glm::vec3 scale;
  glm::vec4 rotation; // quaternion
  __host__ __device__ glm::mat3 scale_matrix() const {
    glm::mat3 S;
    S[0][0] = scale.x;
    S[0][1] = 0.0f;
    S[0][2] = 0.0f;
    S[1][0] = 0.0f;
    S[1][1] = scale.y;
    S[1][2] = 0.0f;
    S[2][0] = 0.0f;
    S[2][1] = 0.0f;
    S[2][2] = scale.z;
    return S;
  }
  __host__ __device__ glm::mat3 rotation_matrix() const {
    // TODO: Implement quaternion to matrix conversion for CUDA
    glm::mat3 R;
    // Identity for now
    R[0][0] = 1.0f;
    R[0][1] = 0.0f;
    R[0][2] = 0.0f;
    R[1][0] = 0.0f;
    R[1][1] = 1.0f;
    R[1][2] = 0.0f;
    R[2][0] = 0.0f;
    R[2][1] = 0.0f;
    R[2][2] = 1.0f;
    return R;
  }

public:
  __host__ __device__ Covariance() : scale(1.0f, 1.0f, 1.0f), rotation(0.0f, 0.0f, 0.0f, 1.0f) {}
  __host__ __device__ Covariance(glm::vec3 s, glm::vec4 r) : scale(s), rotation(r) {}
  __host__ __device__ glm::vec3 to_world(glm::vec3 v) const {
    glm::mat3 R = rotation_matrix();
    glm::mat3 S = scale_matrix();
    return R * S * v;
  }
  __host__ __device__ glm::vec3 to_local(glm::vec3 v) const {
    glm::mat3 R = rotation_matrix();
    glm::mat3 S = scale_matrix();
    // Inverse of S is just reciprocal of diagonal
    glm::mat3 S_inv = glm::inverse(S);
    // Transpose of R is its inverse (orthogonal)
    glm::mat3 R_inv = glm::transpose(R);
    return S_inv * R_inv * v;
  }
};
