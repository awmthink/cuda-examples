#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.h"
#include "common/random.h"

__global__ void vector_add(const float *va, const float *vb, float *vc, int n) {
  // 获取当前线程在整个线程网络中的序号
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // 实际启动的gpu线程数的最小粒度是一个block，所以实际启动的线程数可能大于元素个数
  if (i < n) {
    vc[i] = va[i] + vb[i];
  }
}

int main(int argc, char *argv[]) {
    std::size_t num_elements = 1000;
    std::vector<int> vector_a(num_elements);
    std::vector<int> vector_b(num_elements);
    std::vector<int> vector_c(num_elements);

    FillSequenceNumbers(vector_a);
    FillSequenceNumbers(vector_b, num_elements);

    PrintElements(vector_a, 10);
    PrintElements(vector_b, 10);

    float *d_vector_a = nullptr;
    float *d_vector_b = nullptr;
    float *d_vector_c = nullptr;

    std::size_t size = num_elements * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_vector_a, size));
    checkCudaErrors(cudaMalloc(&d_vector_b, size));
    checkCudaErrors(cudaMalloc(&d_vector_c, size));

    checkCudaErrors(cudaMemcpy(d_vector_a, vector_a.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vector_b, vector_b.data(), size, cudaMemcpyHostToDevice));

    std::size_t threads_per_block = 64;
    // 向上取整，所以实际分配民的线程数量可能会比元素个数要多
    std::size_t block_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    vector_add<<<block_per_grid, threads_per_block>>>(d_vector_a, d_vector_b, d_vector_c, vector_a.size());
    checkCudaErrors(cudaMemcpy(vector_c.data(), d_vector_c, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_vector_a));
    checkCudaErrors(cudaFree(d_vector_b));
    checkCudaErrors(cudaFree(d_vector_c));

    PrintElements(vector_c, 10);

    return 0;
}