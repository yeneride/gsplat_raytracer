#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "ray.hpp"

class Camera {
private:
  glm::mat4 view;
  float focal_length;
  float aspect_ratio;
  __host__ __device__ Ray to_world(Ray r) {
    glm::vec4 origin = glm::vec4(r.get_origin(), 1.0f);
    glm::vec4 direction = glm::vec4(r.get_direction(), 0.0f);
    glm::vec4 world_origin = view * origin;
    glm::vec4 world_direction = view * direction;
    glm::vec3 world_origin_3 = glm::vec3(world_origin.x, world_origin.y, world_origin.z);
    glm::vec3 world_direction_3 = glm::vec3(world_direction.x, world_direction.y, world_direction.z);
    return Ray(world_origin_3, world_direction_3);
  }
  __host__ __device__ Ray local_at(float u, float v) {
    glm::vec3 origin = glm::vec3(0, 0, 0);
    glm::vec3 direction = glm::normalize(
        glm::vec3((2 * u - 1) * aspect_ratio, (2 * v - 1), focal_length));
    return Ray(origin, direction);
  }

public:
  __host__ __device__ Camera(glm::mat4 v, float f, float a) : view(v), focal_length(f), aspect_ratio(a) {}
  __host__ __device__ Ray ray_at(float u, float v) {
    return to_world(local_at(u, v));
  }
};
