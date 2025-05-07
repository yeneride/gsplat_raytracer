#pragma once
#include "covariance.hpp"
#include "ray.hpp"
#include "spherical_harmonics.hpp"
#include <glm/glm.hpp>
#include <math.h>

class Gaussian {
private:
  glm::vec3 mean;
  Covariance cov;
  SphericalHarmonics radiance;
  float opacity;
  __host__ __device__ Ray to_world(Ray r) const {
    glm::vec3 origin = cov.to_world(r.get_origin()) + mean;
    glm::vec3 direction = glm::normalize(cov.to_world(r.get_direction()));
    return Ray(origin, direction);
  }
  __host__ __device__ Ray to_local(Ray r) const {
    glm::vec3 origin = cov.to_local(r.get_origin() - mean);
    glm::vec3 direction = glm::normalize(cov.to_local(r.get_direction()));
    return Ray(origin, direction);
  }
  __host__ __device__ float normalized_distance_squared(Ray r) const {
    Ray local_ray = to_local(r);
    glm::vec3 o = local_ray.get_origin();
    glm::vec3 d = local_ray.get_direction();
    float o_dot_o = glm::dot(o, o);
    float o_dot_d = glm::dot(o, d);
    return o_dot_o - o_dot_d * o_dot_d;
  }

public:
  __host__ __device__ Gaussian()
      : mean(0.0f, 0.0f, 0.0f), cov(), radiance(), opacity(0.0f) {}
  __host__ __device__ Gaussian(glm::vec3 m, Covariance c, SphericalHarmonics r,
                               float o)
      : mean(m), cov(c), radiance(r), opacity(o) {}
  __host__ __device__ float cumulative_density(Ray r) const {
    float d = normalized_distance_squared(r);
    float coeff = 1.0f / sqrtf(2.0f * 3.14159265358979323846f);
    float exponent = -d / 2.0f;
    return d > 0.0f ? 1.0f : 0.0f;
    return coeff * std::exp(exponent) * opacity;
  }
  __host__ __device__ glm::vec3 radiance_from(Ray ray) const {
    glm::vec3 local_dir = glm::normalize(cov.to_local(ray.get_direction()));
    return radiance.color_at(local_dir);
  }
  __host__ __device__ float avg_depth(Ray ray) const {
    glm::vec3 local_origin = cov.to_local(glm::vec3(ray.get_origin() - mean));
    glm::vec3 local_dir = glm::normalize(cov.to_local(ray.get_direction()));
    float o_dot_d = glm::dot(local_origin, local_dir);
    glm::vec3 local_hit = local_dir * o_dot_d;
    glm::vec3 world_hit = cov.to_world(local_hit);
    float dir_dot_world = glm::dot(ray.get_direction(), world_hit);
    return -dir_dot_world;
  }
};
