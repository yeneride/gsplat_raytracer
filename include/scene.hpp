#pragma once
#include "acceleration_structure.hpp"
#include "ray.hpp"
#include <glm/glm.hpp>

class Scene {
private:
  SortedGaussiansAccelerationStructure gaussians;

public:
  Scene(SortedGaussiansAccelerationStructure &&s) : gaussians(std::move(s)) {}

  __device__ glm::vec3 render(Ray ray) const {
    glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
    float acc_transparency = 1.0f;
    const auto &candidates = gaussians.top_candidates_for(ray);
    for (const Gaussian* gptr : candidates) {
      const Gaussian& g = *gptr;
      float density = g.cumulative_density(ray);
      glm::vec3 radiance = g.radiance_from(ray);
      color += radiance * acc_transparency * density;
      acc_transparency *= (1.0f - density);
      if (acc_transparency < 0.01f) {
        break;
      }
    }
    return color;
  }
};
