#include <iostream>
#include <random>
#include <vector>

#include "common/cuda_helper.h"
#include "common/random.h"
#include "common/stopwatch.h"

constexpr int kBlockSize = 1024;

// 1, 3, 6, 10, 15   6, 13, 21, 30
template <typename T>
void BlellochScanCPU(T *in, T *out, int n) {
  int s = 0;
  for (s = 2; s <= n; s = s * 2) {
    for (int i = s - 1; i < n; i += s) {
      in[i] += in[i - s / 2];
    }
  }
  in[s / 2 - 1] = 0;
  for (s = n; s > 1; s = s / 2) {
    for (int i = s - 1; i < n; i += s) {
      float temp = in[i] + in[i - s / 2];
      in[i - s / 2] = in[i];
      in[i] = temp;
    }
  }
  for (int i = 0; i < n; ++i) {
    out[i] = in[i];
  }
}

template <typename T>
__global__ void blelloch_scan_kernel(const T *data, T *out) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;

  extern __shared__ T sdata[];
  sdata[tid] = data[gid];

  for (int stride = 2; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int idx = (tid + 1) * stride - 1;
    if (idx < blockDim.x) {
      sdata[idx] += sdata[idx - stride / 2];
    }
  }
  sdata[blockDim.x - 1] = 0;

  for (int stride = blockDim.x; stride >= 2; stride /= 2) {
    __syncthreads();
    int idx = (tid + 1) * stride - 1;
    if (idx < blockDim.x) {
      T temp1 = sdata[idx] + sdata[idx - stride / 2];
      T temp2 = sdata[idx];
      __syncthreads();
      sdata[idx] = temp1;
      sdata[idx - stride / 2] = temp2;
    }
  }
  out[gid] = sdata[tid];
}

// 代码只显示了一个block中的scan
template <typename T>
void BlellochScanGPU(T *vec, T *out, int n) {
  T *d_vec = nullptr;
  T *d_out = nullptr;

  int blocks = (n + kBlockSize - 1) / kBlockSize;
  int n_bytes = sizeof(T) * n;
  // 按kBlockSize对齐，进行超分配，末部全部填充0，不影响scan结果
  checkCudaErrors(cudaMalloc(&d_vec, sizeof(T) * blocks * kBlockSize));
  checkCudaErrors(cudaMemset(d_vec, 0, sizeof(T) * blocks * kBlockSize));
  checkCudaErrors(cudaMemcpy(d_vec, vec, n_bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(T) * blocks * kBlockSize));

  blelloch_scan_kernel<<<blocks, kBlockSize, kBlockSize * sizeof(T)>>>(d_vec, d_out);

  checkCudaErrors(cudaMemcpy(out, d_out, n_bytes, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_vec));
  checkCudaErrors(cudaFree(d_out));
}

constexpr bool IsPowerofTwo(int n) {
  if (n < 0) {
    return false;
  }
  return (n & (n - 1)) == 0;
}

int main() {
  constexpr int kNumSize = 32;
  static_assert(IsPowerofTwo(kNumSize), "kNumSize must be power of 2");
  std::vector<int> vec(kNumSize);
  FillSequenceNumbers(vec, 1);
  std::cout << "Origin Vector: ";
  PrintElements(vec);

  std::vector<int> out(kNumSize);
  BlellochScanCPU(vec.data(), out.data(), kNumSize);
  std::cout << "Scan Vecotr: ";
  PrintElements(out);

  // reset data
  FillSequenceNumbers(vec, 1);
  std::fill(out.begin(), out.end(), 0);

  static_assert(kNumSize < kBlockSize * kBlockSize, "NumSize is too large");
  BlellochScanGPU(vec.data(), out.data(), kNumSize);
  std::cout << "GPU Scan Vecotr: ";
  PrintElements(out);

  return 0;
}
