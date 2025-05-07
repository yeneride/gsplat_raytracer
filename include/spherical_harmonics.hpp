#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class SphericalHarmonics {
private:
  glm::vec3 coeffs[3];

public:
  __host__ __device__ SphericalHarmonics()
      : coeffs{glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f)} {}
  __host__ __device__ SphericalHarmonics(glm::vec3 c0, glm::vec3 c1,
                                         glm::vec3 c2)
      : coeffs{c0, c1, c2} {}
  __host__ __device__ glm::vec3 color_at(glm::vec3 dir) const {
    // Just return the first coefficient for now
    auto color = coeffs[0] * 0.28209479177387814f + 0.5f;
    color.x = std::min(std::max(color.x, 0.0f), 1.0f);
    color.y = std::min(std::max(color.y, 0.0f), 1.0f);
    color.z = std::min(std::max(color.z, 0.0f), 1.0f);
    return color;
  }
};
