#include <iostream>
#include <numeric>
#include <vector>
#include "common/cuda_helper.h"
#include "common/gpu_timer.h"

__global__ void window_sum(float *input, float *output, int n, int winsize) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    float sum = 0;
    for (int i = 0; i < winsize; ++i) {
      sum += input[id + i];
    }
    output[id] = sum;
  }
}

__global__ void window_sum_shared(float *input, float *output, int n,
                                  int winsize) {
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int input_size = n + winsize - 1;

  // 先将每个线程块对应的输入复制到共享内存
  if (gid >= input_size) {
    return;
  }
  extern __shared__ float sinput[];
  sinput[tid] = input[gid];

  // 复制剩下的windows size - 1个
  if (tid < winsize - 1 && blockDim.x * (blockIdx.x + 1) + tid < input_size) {
    sinput[blockDim.x + tid] = input[blockDim.x * (blockIdx.x + 1) + tid];
  }
  __syncthreads();

  // 计算每个输出位置上的window sum
  if (gid >= n) {
    return;
  }
  float sum = 0;
  for (int i = 0; i < winsize; ++i) {
    sum += sinput[tid + i];
  }
  output[gid] = sum;
}

int main() {
  constexpr int length = 100;
  constexpr int winsize = 5;
  std::vector<float> h_input(length);
  std::iota(h_input.begin(), h_input.end(), 0);
  constexpr int output_len = length - winsize + 1;

  int device_id = 6;
  checkCudaErrors(cudaSetDevice(device_id));

  float *d_input, *d_output;
  checkCudaErrors(cudaMalloc(&d_input, length * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_output, output_len * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_input, h_input.data(), sizeof(float) * length,
                             cudaMemcpyHostToDevice));

  int threads_per_block = 256;
  int blocks = (output_len + threads_per_block - 1) / threads_per_block;
  std::size_t shared_size = (threads_per_block + winsize - 1) * sizeof(float);

  GpuTimer timer;
  timer.Start();
  window_sum<<<blocks, threads_per_block>>>(d_input, d_output, output_len,
                                            winsize);
  timer.Stop();
  std::cout << "window sum time: " << timer.Elapsed() << std::endl;

  timer.Start();
  window_sum_shared<<<blocks, threads_per_block, shared_size>>>(
      d_input, d_output, output_len, winsize);
  timer.Stop();
  std::cout << "window sum (shared) time: " << timer.Elapsed() << std::endl;

  std::vector<float> h_output(output_len);
  checkCudaErrors(cudaMemcpy(h_output.data(), d_output,
                             sizeof(float) * output_len,
                             cudaMemcpyDeviceToHost));
  for (auto out : h_output) {
    std::cout << out << ", ";
  }
  std::cout << std::endl;
}