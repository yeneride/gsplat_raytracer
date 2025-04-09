#pragma once

#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

template <int MAX_POINTS> struct GridCell {
  int point_idx[MAX_POINTS];
  int pointCount;
};

template <int MAX_POINTS = 64> struct AccelerationGrid {
  GridCell<MAX_POINTS> *d_grid; // Device pointer to grid
  float3 gridSize = {100.0F, 100.0F, 100.0F};
  float3 cellSize = {1.0F, 1.0F, 1.0F};
  int cellsPerAxis = 100;

  AccelerationGrid(float3 size = {100.0F, 100.0F, 100.0F},
                   int cellsPerAxis = 100)
      : gridSize(size), cellSize{size.x / cellsPerAxis, size.y / cellsPerAxis,
                                 size.x / cellsPerAxis} {
    int totalCells = cellsPerAxis * cellsPerAxis * cellsPerAxis;
    cudaMalloc(&d_grid, totalCells * sizeof(GridCell<MAX_POINTS>));
  }

  ~AccelerationGrid() { cudaFree(d_grid); }

  float3 getGridMin() {
    return {0.0F, 0.0F, 0.0F};
  }

  float3 getCellSize() {
    return cellSize;
  }

  int getCellsPerAxis() {
    return cellsPerAxis;
  }

  __host__ __device__ static bool intersectsEllipsoid(float3 center, float3 radii,
                                      float3 cubeMin, float3 cubeMax) {
    float3 closestPoint =
        make_float3(fmaxf(cubeMin.x, fminf(center.x, cubeMax.x)),
                    fmaxf(cubeMin.y, fminf(center.y, cubeMax.y)),
                    fmaxf(cubeMin.z, fminf(center.z, cubeMax.z)));

    float dx = (closestPoint.x - center.x) / radii.x;
    float dy = (closestPoint.y - center.y) / radii.y;
    float dz = (closestPoint.z - center.z) / radii.z;

    return (dx * dx + dy * dy + dz * dz) <= 1.0f;
  }

  __global__ static void constructGrid(GridCell<MAX_POINTS> *grid,
                                       float3 *ellipsoidCenters,
                                       float3 *ellipsoidRadii,
                                       int numEllipsoids, float3 gridSize,
                                       float3 cellSize, int cellsPerAxis) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int gridIdx = x + y * cellsPerAxis + z * cellsPerAxis * cellsPerAxis;

    if (gridIdx >= cellsPerAxis * cellsPerAxis * cellsPerAxis)
      return;

    float3 cubeMin =
        make_float3(x * cellSize.x, y * cellSize.y, z * cellSize.z);
    float3 cubeMax = make_float3(cubeMin.x + cellSize.x, cubeMin.y + cellSize.y,
                                 cubeMin.z + cellSize.z);

    __shared__ GridCell<MAX_POINTS> localCell;
    localCell.pointCount = 0;

    for (int i = 0; i < numEllipsoids; i++) {
      if (intersectsEllipsoid(ellipsoidCenters[i], ellipsoidRadii[i], cubeMin,
                              cubeMax)) {
        for (int j = 0; j < MAX_POINTS; j++) {
          if (localCell.pointCount < MAX_POINTS) {
            localCell.point_idx[localCell.pointCount++] = i;
          }
        }
      }
    }

    grid[gridIdx] = localCell;
  }

  void clear() {
    int totalCells = cellsPerAxis * cellsPerAxis * cellsPerAxis;
    cudaMemset(d_grid, 0, totalCells * sizeof(GridCell<MAX_POINTS>));
  }

  void build(float3 *ellipsoidCenters, float3 *ellipsoidRadii,
             int numEllipsoids) {
    float3 *d_ellipsoidCenters, *d_ellipsoidRadii;
    dim3 gridSize(cellsPerAxis, cellsPerAxis, cellsPerAxis);

    clear();
    constructGrid<<<gridSize, 1>>>(
        d_grid, ellipsoidCenters, ellipsoidRadii, numEllipsoids,
        gridSize, cellSize);

    cudaDeviceSynchronize();
  }

  GridCell<MAX_POINTS> *getDeviceGrid() { return d_grid; }
};

#endif
