#ifndef CUDA_EXAMPLES_EXAMPLES_COMMON_CUDA_HELPER_H_
#define CUDA_EXAMPLES_EXAMPLES_COMMON_CUDA_HELPER_H_

#include <stdlib.h>

#include <cuda_runtime_api.h>

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

#endif  // CUDA_EXAMPLES_EXAMPLES_COMMON_CUDA_HELPER_H_