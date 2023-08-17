#include <iostream>
#include "common/cuda_helper.h"
#include "common/stopwatch.h"

constexpr int kElemNum = 1'000'000;

__global__ void simple_square_kernel(float *data, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    float temp = data[gid];
    data[gid] = temp * temp;
  }
}

__global__ void simple_exp_kernel(float *data, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    float temp = data[gid];
    data[gid] = __expf(temp);
  }
}

void CUDASyncStream() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::size_t sz = kElemNum * sizeof(float);

  float *h_a, *h_b;
  checkCudaErrors(cudaHostAlloc(&h_a, sz, cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&h_b, sz, cudaHostAllocDefault));

  float *d_a, *d_b;
  checkCudaErrors(cudaMalloc(&d_a, sz));
  checkCudaErrors(cudaMalloc(&d_b, sz));

  int threads = 512;
  int blocks = (kElemNum + threads - 1) / threads;

  Stopwatch sync_timer;
  sync_timer.Start();

  checkCudaErrors(cudaMemcpyAsync(d_a, h_a, sz, cudaMemcpyHostToDevice, stream));
  simple_square_kernel<<<blocks, threads, 0, stream>>>(d_a, kElemNum);
  checkCudaErrors(cudaMemcpyAsync(h_a, d_a, sz, cudaMemcpyDeviceToHost, stream));

  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sz, cudaMemcpyHostToDevice, stream));
  simple_exp_kernel<<<blocks, threads, 0, stream>>>(d_b, kElemNum);
  checkCudaErrors(cudaMemcpyAsync(h_b, d_b, sz, cudaMemcpyDeviceToHost, stream));

  checkCudaErrors(cudaStreamSynchronize(stream));

  std::cout << "sync time: " << sync_timer.Elapsed<Stopwatch::MICROSECONDS>() << " us\n";

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaStreamDestroy(stream));
}

void CUDAAsyncStream() {
  cudaStream_t stream1, stream2;
  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamCreate(&stream2));

  std::size_t sz = kElemNum * sizeof(float);

  float *h_a, *h_b;
  checkCudaErrors(cudaHostAlloc(&h_a, sz, cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&h_b, sz, cudaHostAllocDefault));

  float *d_a, *d_b;
  checkCudaErrors(cudaMalloc(&d_a, sz));
  checkCudaErrors(cudaMalloc(&d_b, sz));

  int threads = 512;
  int blocks = (kElemNum + threads - 1) / threads;

  Stopwatch sync_timer;
  sync_timer.Start();

  checkCudaErrors(cudaMemcpyAsync(d_a, h_a, sz, cudaMemcpyHostToDevice, stream1));
  simple_square_kernel<<<blocks, threads, 0, stream1>>>(d_a, kElemNum);
  checkCudaErrors(cudaMemcpyAsync(h_a, d_a, sz, cudaMemcpyDeviceToHost, stream1));

  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sz, cudaMemcpyHostToDevice, stream2));
  simple_exp_kernel<<<blocks, threads, 0, stream2>>>(d_b, kElemNum);
  checkCudaErrors(cudaMemcpyAsync(h_b, d_b, sz, cudaMemcpyDeviceToHost, stream2));

  checkCudaErrors(cudaStreamSynchronize(stream1));
  checkCudaErrors(cudaStreamSynchronize(stream2));

  std::cout << "async time: " << sync_timer.Elapsed<Stopwatch::MICROSECONDS>() << " us\n";

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaStreamDestroy(stream1));
  checkCudaErrors(cudaStreamDestroy(stream2));
}

int main() {
  // copy A -> compute A -> copy back A -> copy B -> compute B -> copy back B
  CUDASyncStream();

  // copy A -> compute A -> copy back A
  //   copy B -> compute B -> copy back B
  CUDAAsyncStream();

  return 0;
}