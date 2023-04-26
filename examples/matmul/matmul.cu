#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "common/cuda_helper.h"
#include "common/gpu_timer.h"

constexpr int BLOCK_SIZE = 16;

// MatMulCPU 是CPU版本的参考实现，它并非vanilla版本实现，而是优化了b的访存
void MatMulCPU(float *a, float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int l = 0; l < k; ++l) {
      float ail = a[i * k + l];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += ail * b[l * n + j];
      }
    }
  }
}

// MatMulV0 是最基本的矩阵乘法，每个线程计算一个输出
__global__ void MatMulV0(float *a, float *b, float *c, int m, int k, int n) {
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (gid_x >= n || gid_y >= m) {
    return;
  }

  float sum = 0;
  for (int i = 0; i < k; ++i) {
    sum += a[gid_y * k + i] * b[i * n + gid_x];
  }
  c[gid_y * n + gid_x] = sum;
}

// matmul_v1 使用了共享内存来加速矩阵乘法，对于每个线程块(BLOCK_SIZExBLOCK_SIZE)
// 内的所有线程，它们计算对应位置的输出时，读取共享内存，而不是全局内存
__global__ void MatMulV1(float *a, float *b, float *c, int m, int k, int n) {
  __shared__ float sa[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sb[BLOCK_SIZE][BLOCK_SIZE];
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int gid_x = blockDim.x * blockIdx.x + tid_x;
  int gid_y = blockDim.y * blockIdx.y + tid_y;
  float sum = 0;
  for (int l = 0; l < k; l += BLOCK_SIZE) {
    // 先将a中的一个tile加载到共享内存
    if (gid_y < m && l + tid_x < k) {
      sa[tid_y][tid_x] = a[gid_y * k + l + tid_x];
    } else {
      sa[tid_y][tid_x] = 0;
    }
    // 再将b中的一个tile加载到共享内存
    if (gid_x < n && l + tid_y < k) {
      sb[tid_y][tid_x] = b[(l + tid_y) * n + gid_x];
    } else {
      sb[tid_y][tid_x] = 0;
    }

    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sum += sa[tid_y][i] * sb[i][tid_x];
    }
    __syncthreads();
  }
  if (gid_y < m && gid_x < n) {
    c[gid_y * n + gid_x] = sum;
  }
}

// FillRandomNum 用随机数填充向量
void FillRandomNum(std::vector<float> &v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);
  for (auto &e : v) {
    e = dis(gen);
  }
}

// AverageDiff 计算两个向量的平均差异
float AverageDiff(const std::vector<float> &lhs,
                  const std::vector<float> &rhs) {
  float sum = 0;
  for (int i = 0; i < lhs.size(); ++i) {
    sum += std::fabs(lhs[i] - rhs[i]);
  }
  return sum / lhs.size();
}

int main() {
  // 检查Cuda设备的可用情况
  checkCudaDevice();

  constexpr int m = 512;
  constexpr int k = 256;
  constexpr int n = 128;

  // 创建Host侧的矩阵，并填充随机数
  std::vector<float> a(m * k);
  std::vector<float> b(k * n);
  std::vector<float> c(m * n);
  std::vector<float> c_cpu(m * n, 0);
  FillRandomNum(a);
  FillRandomNum(b);

  // 使用cpu实现来计算标准的输出
  GpuTimer timer;
  timer.Start();
  MatMulCPU(a.data(), b.data(), c_cpu.data(), m, k, n);
  timer.Stop();
  printf("matmul cpu time: %f ms\n", timer.Elapsed());

  // 分配设备侧的存储，并拷贝数据到设备侧
  float *d_a, *d_b, *d_c;
  checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * a.size()));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * b.size()));
  checkCudaErrors(cudaMalloc(&d_c, sizeof(float) * c.size()));

  checkCudaErrors(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(),
                             cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (m + blockDim.y - 1) / blockDim.y);

  // 使用MatMulV0来进行矩阵乘法计算
  timer.Start();
  MatMulV0<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, k, n);
  cudaDeviceSynchronize();
  timer.Stop();
  printf("matmul gpu v0 time: %f ms\n", timer.Elapsed());
  checkLastCudaError();

  // 使用MatMulV1来进行矩阵乘法计算
  timer.Start();
  MatMulV1<<<gridDim, blockDim, BLOCK_SIZE * BLOCK_SIZE * 2 * sizeof(float)>>>(
      d_a, d_b, d_c, m, k, n);
  cudaDeviceSynchronize();
  timer.Stop();
  printf("matmul gpu v1 time: %f ms\n", timer.Elapsed());
  checkLastCudaError();

  // 往Host侧拷贝回矩阵计算的结果
  checkCudaErrors(cudaMemcpy(c.data(), d_c, sizeof(float) * c.size(),
                             cudaMemcpyDeviceToHost));

  // 检查计算结果的正确性
  float diff = AverageDiff(c, c_cpu);
  if (diff > 1e-4) {
    printf("diff between c and c_cpu: %f\n", diff);
    exit(EXIT_FAILURE);
  }
  printf("diff between c and c_cpu: %f\n", diff);

  // 释放设备侧的存储
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));

  return 0;
}