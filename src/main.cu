#include "acceleration_structure.hpp"
#include "camera.hpp"
#include "covariance.hpp"
#include "gaussian.hpp"
#include "happly.hpp"
#include "image_io.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "spherical_harmonics.hpp"
#include "splat_loader.hpp"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>

struct RenderFunctor {
  Scene scene;
  Camera camera;
  float *image;
  int resolution;
  __device__ void operator()(int i) {
    int x = i % resolution;
    int y = i / resolution;
    float u = (float)x / resolution;
    float v = (float)y / resolution;
    Ray ray = camera.ray_at(u, v);
    glm::vec3 color = scene.render(ray);
    image[3 * i + 0] = color.x;
    image[3 * i + 1] = color.y;
    image[3 * i + 2] = color.z;
  }
};

int main() {
  std::cout << "Loading..." << std::endl;
  std::vector<Gaussian> gaussians_cpu = SplatLoader::load_from_ply("splat.ply");
  std::cout << "Loaded" << std::endl;

  size_t resolution = 1024;
  std::vector<float> image_cpu(resolution * resolution * 3);
  thrust::device_vector<float> image_gpu(resolution * resolution * 3);
  Camera camera(glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), 1.0f,
                1.0f);

  std::cout << "Sorting..." << std::endl;
  thrust::device_vector<Gaussian> gaussians_gpu = gaussians_cpu;
  Scene scene(SortedGaussiansAccelerationStructure(gaussians_gpu,
                                                   camera.ray_at(0.5, 0.5)));
  std::cout << "Sorted" << std::endl;

  RenderFunctor functor{scene, camera,
                        thrust::raw_pointer_cast(image_gpu.data()),
                        (int)resolution};
  std::cout << "Rendering..." << std::endl;
  thrust::for_each(thrust::device, thrust::counting_iterator(0),
                   thrust::counting_iterator((int)(resolution * resolution)),
                   functor);
  std::cout << "Rendered" << std::endl;

  thrust::copy(image_gpu.begin(), image_gpu.end(), image_cpu.begin());
  std::cout << "Copied" << std::endl;
  write_ppm("output.ppm", image_cpu.data(), resolution, resolution);
  return 0;
}
