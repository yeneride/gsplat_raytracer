#pragma once
#include "gaussian.hpp"
#include "ray.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

struct Gaussians {
  const Gaussian *gaussians;
  size_t count;
};

class SortedGaussiansAccelerationStructure {
private:
  Gaussians gaussians;

public:
  SortedGaussiansAccelerationStructure(thrust::device_vector<Gaussian> &g, const Ray &ray) {
    thrust::sort(thrust::device, g.begin(), g.end(),
                 [ray] __device__(const Gaussian &a, const Gaussian &b) {
                   return a.avg_depth(ray) < b.avg_depth(ray);
                 });
    gaussians = Gaussians{
        thrust::raw_pointer_cast(g.data()),
        g.size(),
    };
  }
  __host__ __device__ const Gaussians top_candidates_for(Ray ray) const {
    return gaussians;
  }
};
