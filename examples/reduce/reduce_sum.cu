#include <iostream>
#include <random>
#include <vector>

#include "common/cuda_helper.h"

__global__ void reduce_sum(float *d_in, float *d_block_result) {
  unsigned int tidx = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ float sdata[];
  sdata[tidx] = d_in[idx];
  __syncthreads();

  for (unsigned int step = blockDim.x / 2; step > 0; step = step / 2) {
    if (tidx < step) {
      sdata[tidx] += sdata[tidx + step];
    }
    __syncthreads();
  }
  if (tidx == 0) {
    d_block_result[blockIdx.x] = sdata[0];
  }
}

int main(int argc, char *argv[]) {
  constexpr unsigned int array_size = 1234567;
  std::vector<float> array(array_size);
  std::fill_n(array.begin(), array_size, 1.0f);

  const dim3 block = 1024;
  const dim3 grid = (array_size + block.x - 1) / block.x;

  float *d_in = nullptr;
  float *d_block_result = nullptr;

  checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * block.x * grid.x));
  checkCudaErrors(cudaMemset(d_in, 0, sizeof(float) * block.x * grid.x));
  checkCudaErrors(cudaMalloc(&d_block_result, sizeof(float) * grid.x));

  checkCudaErrors(cudaMemcpy(d_in, array.data(), sizeof(float) * array_size,
                             cudaMemcpyHostToDevice));

  reduce_sum<<<grid, block, block.x * sizeof(float)>>>(d_in, d_block_result);
  checkLastCudaError();

  std::vector<float> block_result(grid.x);
  checkCudaErrors(cudaMemcpy(block_result.data(), d_block_result,
                             sizeof(float) * grid.x, cudaMemcpyDeviceToHost));

  float sum = std::accumulate(block_result.begin(), block_result.end(), .0f);
  std::cout << "SUM: " << sum << std::endl;

  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_block_result));

  return 0;
}