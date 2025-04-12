#pragma once

#ifndef RAYTRACER_H
#define RAYTRACER_H

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#endif

#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <iostream>

class Raytracer {
private:
  AccelerationGrid<64> *accelGrid;

public:
  Raytracer(float gridSizeX, float gridSizeY, float gridSizeZ,
            int cellsPerAxis) {
    accelGrid = new AccelerationGrid<64>({gridSizeX, gridSizeY, gridSizeZ},
                                         cellsPerAxis);
  }

  ~Raytracer() { delete accelGrid; }

  int forward(std::function<char *(size_t)> geometryBuffer,
              std::function<char *(size_t)> binningBuffer,
              std::function<char *(size_t)> imageBuffer, const int P, int D,
              int M, const float *background, const int width, int height,
              const float3 *means3D, const float *shs, const float3 *scales,
              const float scale_modifier, const float4 *rotations,
              const float *cov3D_precomp, const float *viewmatrix,
              const float *projmatrix, const float *cam_pos,
              const float tan_fovx, float tan_fovy, const bool prefiltered,
              float *out_color, int *radii, bool debug);
};

#endif
