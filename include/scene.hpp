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
    for (int i = 0; i < candidates.count; ++i) {
      const Gaussian &g = candidates.gaussians[i];
      const float distance = g.avg_depth(ray);
      if (distance < 0)
        continue;
      float density = g.cumulative_density(ray);
      if (density < 0.01f)
        continue;
      glm::vec3 radiance = g.radiance_from(ray);
      // color += radiance * acc_transparency * density;
      color += glm::vec3(density);
      break;
      acc_transparency *= (1 - density);
      if (acc_transparency < 0.01f) {
        break;
      }
    }
    return color;
  }
};
