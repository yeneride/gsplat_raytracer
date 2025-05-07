#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Ray {
private:
  glm::vec3 origin;
  glm::vec3 direction;

public:
  __host__ __device__ Ray(glm::vec3 o, glm::vec3 d) : origin(o), direction(d) {}
  __host__ __device__ glm::vec3 get_origin() const { return origin; }
  __host__ __device__ glm::vec3 get_direction() const { return direction; }
};
