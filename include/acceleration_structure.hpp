#pragma once
#include "gaussian.hpp"
#include "ray.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

constexpr size_t MAX_CANDIDATES = 64; // Set this to your desired maximum

class FilteredGaussians {
public:
  struct Entry {
    const Gaussian* ptr;
    float distance;
    float density;
  };

private:
  Entry entries[MAX_CANDIDATES];
  size_t count_;

public:
  __host__ __device__
  FilteredGaussians() : entries{}, count_(0) {}

  // Returns the largest distance in the array, or -infinity if empty
  __host__ __device__
  float max_distance() const {
    return count_ > 0 ? entries[count_ - 1].distance : -INFINITY;
  }

  // Submit a candidate. If not full, add; if full and this has lower distance than the max, replace the max (last entry)
  __host__ __device__
  void submit(const Gaussian* g, float distance, float density) {
    if (is_full()) {
      if (distance < max_distance()) {
        entries[count_ - 1] = {g, distance, density};
      } else {
        return;
      }
    } else {
      entries[count_++] = {g, distance, density};
    }
    // Keep entries sorted by distance (ascending)
    for (size_t i = count_ > 0 ? count_ - 1 : 0; i > 0; --i) {
      if (entries[i].distance < entries[i - 1].distance) {
        Entry tmp = entries[i];
        entries[i] = entries[i - 1];
        entries[i - 1] = tmp;
      } else {
        break;
      }
    }
    
    // After adding, check if total density exceeds 1.0
    // If so, remove entries from the end until density is close to 1.0 but not below it
    float total = total_density();
    while (total > 1.0f && count_ > 1) {
      // Calculate what the total would be if we removed the last entry
      float new_total = total - entries[count_ - 1].density;
      // Don't remove if it would make the total go below 1.0
      if (new_total < 1.0f) {
        break;
      }
      // Remove the last entry
      count_--;
      total = new_total;
    }
  }

  __host__ __device__
  bool is_full() const { return count_ >= MAX_CANDIDATES; }

  // Iterator interface for device code (yields Gaussian*)
  class iterator {
    const Entry* ptr_;
  public:
    __host__ __device__ iterator(const Entry* p) : ptr_(p) {}
    __host__ __device__ const Gaussian* operator*() const { return ptr_->ptr; }
    __host__ __device__ iterator& operator++() { ++ptr_; return *this; }
    __host__ __device__ bool operator!=(const iterator& other) const { return ptr_ != other.ptr_; }
  };

  __host__ __device__ iterator begin() const { return iterator(entries); }
  __host__ __device__ iterator end() const { return iterator(entries + count_); }

  __host__ __device__
  size_t size() const { return count_; }

  // Access to the sorted entries (for advanced use)
  __host__ __device__
  const Entry* data() const { return entries; }

  // Calculate total density as sum of all stored densities
  __host__ __device__
  float total_density() const {
    float sum = 0.0f;
    for (size_t i = 0; i < count_; i++) {
      sum += entries[i].density;
    }
    return sum;
  }
};

class SortedGaussiansAccelerationStructure {
private:
  Gaussian* d_gaussians;
  size_t count;

public:
  SortedGaussiansAccelerationStructure(thrust::device_vector<Gaussian>& g, const Ray& ray) {
    thrust::sort(thrust::device, g.begin(), g.end(),
                 [ray] __device__(const Gaussian& a, const Gaussian& b) {
                   return a.avg_depth(ray) < b.avg_depth(ray);
                 });
    d_gaussians = thrust::raw_pointer_cast(g.data());
    count = g.size();
  }

  // Filter gaussians to find the top candidates for a given ray
  __host__ __device__
  FilteredGaussians top_candidates_for(const Ray& ray) const {
    FilteredGaussians filtered;
    
    for (size_t i = 0; i < count && !filtered.is_full(); i++) {
      const Gaussian& g = d_gaussians[i];
      float distance = g.avg_depth(ray);
      if (distance < 0.0f)
        continue;
      float density = g.cumulative_density(ray);
      if (density < 0.01f)
        continue;

      filtered.submit(&g, distance, density);
    }
    
    return filtered;
  }

  // No bulk filtering; user should call submit for each candidate
  __host__ __device__
  const Gaussian* data() const { return d_gaussians; }
  __host__ __device__
  size_t size() const { return count; }
};
