#ifndef EXAMPLES_COMMON_CUDA_HELPER_H_
#define EXAMPLES_COMMON_CUDA_HELPER_H_

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

void __checkCudaErrors(cudaError_t result, char const *const func,
                       const char *const file, int const line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%u(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result),
            func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)

inline void __checkLastCudaError(const char *file, const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error: (%u) %s.\n", file,
            line, static_cast<unsigned int>(err), cudaGetErrorString(err));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define checkLastCudaError() __checkLastCudaError(__FILE__, __LINE__)

void checkCudaDevice() {
  printf("============= CUDA Device Info =============\n");
  int device_count = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_SUCCESS);
  }
  int driver_version = 0;
  int runtime_version = 0;
  checkCudaErrors(cudaDriverGetVersion(&driver_version));
  checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));
  printf("CUDA Driver Version: %d.%d\n", driver_version / 1000,
         (driver_version % 100) / 10);
  printf("Runtime Version: %d.%d\n", runtime_version / 1000,
         (runtime_version % 100) / 10);
  printf("Detected %d CUDA Capable device(s)\n", device_count);
  for (int dev = 0; dev < device_count; ++dev) {
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceProp device_prop;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev));
    printf("  Device %d: %s\n", dev, device_prop.name);
  }
  printf("============================================\n");
}

#endif  // EXAMPLES_COMMON_CUDA_HELPER_H_
