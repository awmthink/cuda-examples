#ifndef EXAMPLES_COMMON_GPU_TIMER_H_
#define EXAMPLES_COMMON_GPU_TIMER_H_

#include <cuda_runtime_api.h>

class GpuTimer {
 public:
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Start() { cudaEventRecord(start_, 0); }

  void Stop() { cudaEventRecord(stop_, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

#endif  // EXAMPLES_COMMON_GPU_TIMER_H_
