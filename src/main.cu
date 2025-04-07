#include <cuda_runtime.h>
#include <stdio.h>
#include "raytrace.h"

// CUDA kernel to add two arrays element by element
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

// Main program
int main(void) {
  // Print CUDA device information
  int deviceCount = 0;
  float milliseconds = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    printf("Error: Failed to get CUDA device count: %s\n",
           cudaGetErrorString(error));
    return EXIT_FAILURE;
  }

  if (deviceCount == 0) {
    printf("Warning: No CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  printf("Detected %d CUDA device(s)\n", deviceCount);

  // Use first device
  cudaSetDevice(0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("Using device 0: %s\n", deviceProp.name);

  // Set vector size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("Vector size: %d\n", numElements);

  // Allocate host memory
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize host arrays
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate device memory
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_C = NULL;

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch CUDA kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error: Failed to launch CUDA kernel: %s\n",
           cudaGetErrorString(error));
    goto cleanup;
  }

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify result (just checking a few elements)
  for (int i = 0; i < 5; ++i) {
    printf("%.2f + %.2f = %.2f\n", h_A[i], h_B[i], h_C[i]);
  }

  // Measure performance
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  // Timing run
  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Performance: %f ms per kernel launch (average of 100 launches)\n",
         milliseconds / 100);

cleanup:
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  // Reset device
  cudaDeviceReset();

  printf("CUDA demo completed successfully\n");
  return 0;
}
