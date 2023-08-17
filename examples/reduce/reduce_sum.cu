#include <iostream>
#include <random>
#include <vector>

#include "common/cuda_helper.h"
#include "common/stopwatch.h"

constexpr unsigned int array_size = 1'234'567;
constexpr auto num_per_block = 1024;

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

// 使用shuffle指令，并没有变快，时间为174 us
__device__ void warp_reduce_shuffle(float *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];  // setp = 32
  sdata[0] = WarpReduceSum(0);    // 在一个wrap里进行reduce
}

// 一个warp中的32个线程，对于函数中每行代码的执行都是同步的
// 加入warp_reduce优化，性能从208 us 优化到 168 us
// 这里把sdata声明为volatile 非常关键，防止编译器缓存sdata中的结果
__device__ void warp_reduce(volatile float *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];  // setp = 32
  sdata[tid] += sdata[tid + 16];  // setp = 16
  sdata[tid] += sdata[tid + 8];   // setp = 8
  sdata[tid] += sdata[tid + 4];   // setp = 4
  sdata[tid] += sdata[tid + 2];   // setp = 2
  sdata[tid] += sdata[tid + 1];   // setp = 1
}

__global__ void reduce_sum(float *d_in, float *d_block_result) {
  unsigned int tidx = threadIdx.x;
  unsigned int idx = blockIdx.x * num_per_block + threadIdx.x;

  extern __shared__ float sdata[];
  sdata[tidx] = d_in[idx] + d_in[idx + blockDim.x];
  __syncthreads();

  for (unsigned int step = blockDim.x / 2; step > warpSize; step = step / 2) {
    if (tidx < step) {
      sdata[tidx] += sdata[tidx + step];
    }
    __syncthreads();
  }
  if (tidx < warpSize) {
    warp_reduce(sdata, tidx);
  }

  if (tidx == 0) {
    d_block_result[blockIdx.x] = sdata[0];
  }
}

float ReduceSumCPU(const std::vector<float> &in) {
  return std::accumulate(in.begin(), in.end(), .0F);
}

float ReduceSumGPU(const std::vector<float> &in) {
  auto threads = num_per_block / 2;
  auto blocks = (array_size + num_per_block - 1) / num_per_block;

  float *d_in = nullptr;
  float *d_block_result = nullptr;

  // 按线程块对齐来申请device存储，并置0，可以使得kernel函数中不需要判定tid是否越界
  checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * num_per_block * blocks));
  checkCudaErrors(cudaMemset(d_in, 0, sizeof(float) * num_per_block * blocks));
  checkCudaErrors(cudaMalloc(&d_block_result, sizeof(float) * blocks));

  checkCudaErrors(cudaMemcpy(d_in, in.data(), sizeof(float) * in.size(), cudaMemcpyHostToDevice));

  std::cout << "Exectue the first reudce kernel ..." << std::endl;
  Stopwatch kernel_watch;
  reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_block_result);

  if (blocks > 1024) {
    std::cout << "Exectue the second reudce kernel ..." << std::endl;
    // 对d_block_result中的数据进行第二次reduce操作，重新计算blocks数量
    blocks = (blocks + threads - 1) / threads;
    // 复用d_in的存储空间用于存放第二次reduce的结果
    std::swap(d_in, d_block_result);
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_block_result);
  }
  checkLastCudaError();
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Reduce Sum Kernel elapsed: " << kernel_watch.Elapsed<mus>() << " us" << std::endl;

  // 对于cuda运算后的结果，进行最后的reduce，一般是针对数据量超过1M的数据时
  std::vector<float> block_result(blocks);
  checkCudaErrors(cudaMemcpy(block_result.data(), d_block_result, sizeof(float) * blocks,
                             cudaMemcpyDeviceToHost));

  float sum = std::accumulate(block_result.begin(), block_result.end(), .0f);

  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_block_result));

  return sum;
}

int main(int argc, char *argv[]) {
  std::vector<float> array(array_size);
  std::fill_n(array.begin(), array_size, 1.0f);

  Stopwatch cpu_timer;
  float cpu_sum = ReduceSumCPU(array);
  std::cout << "Reduce Sum CPU elapsed: " << cpu_timer.Elapsed<mus>() << " us" << std::endl;

  std::cout << "CPU SUM: " << cpu_sum << std::endl;

  float sum = ReduceSumGPU(array);
  std::cout << "CUDA SUM: " << sum << std::endl;

  return 0;
}