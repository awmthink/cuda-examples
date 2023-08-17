#include <iostream>
#include <random>
#include <vector>

#include "common/cuda_helper.h"
#include "common/random.h"
#include "common/stopwatch.h"

constexpr int kBlockSize = 1024;

template <typename T>
void HillisSteeleScanCPU(T *in, T *out, int n) {
  T *output = out;
  for (int s = 1; s < n; s = s * 2) {
    for (int i = 0; i < n; i++) {
      if (i - s >= 0) {
        out[i] = in[i] + in[i - s];
      } else {
        out[i] = in[i];
      }
    }
    std::swap(in, out);
  }
  // 经过迭代后，最终scan的结果是存储在in数组中的，将其拷贝到原始的输入数组中
  for (int i = 0; i < n; ++i) {
    output[i] = in[i];
  }
}

template <typename T>
__global__ void hillis_steele_scan_kernel(T *in, T *out, T *seg_out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // 申请一片 2*blockDim.x 大小的共享内存，然后切成2份
  extern __shared__ T sdata[];
  T *sdata_in = &sdata[0];
  T *sdata_out = &sdata[blockDim.x];
  // sout在每一step后存储最终的输出
  T *sout = sdata_out;

  sdata_in[tid] = in[gid];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid - s >= 0) {
      sdata_out[tid] = sdata_in[tid] + sdata_in[tid - s];
    } else {
      sdata_out[tid] = sdata_in[tid];
    }
    __syncthreads();
    sout = sdata_out;
    sdata_out = sdata_in;
    sdata_in = sout;
  }
  out[gid] = sout[tid];

  // 将每个block中最后一个结果，写入到seg_out中
  if (seg_out != nullptr && tid == blockDim.x - 1) {
    seg_out[blockIdx.x] = sout[tid];
  }
}

template <typename T>
__global__ void scan_add_segment_kernel(T *data, T *seg_data) {
  if (blockIdx.x > 0) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    data[gid] += seg_data[blockIdx.x - 1];
  }
}

template <typename T>
void HillisSteeleScanGPU(T *vec, T *out, int n) {
  T *d_vec = nullptr;
  T *d_out = nullptr;
  T *d_seg_out = nullptr;  // 存储每一个block的scan结果的最后一个元素

  int blocks = (n + kBlockSize - 1) / kBlockSize;
  int sub_blocks = (blocks + kBlockSize - 1) / kBlockSize;

  int n_bytes = sizeof(T) * n;
  // 按kBlockSize对齐，进行超分配，末部全部填充0，不影响scan结果
  checkCudaErrors(cudaMalloc(&d_vec, sizeof(T) * blocks * kBlockSize));
  checkCudaErrors(cudaMemset(d_vec, 0, sizeof(T) * blocks * kBlockSize));
  checkCudaErrors(cudaMemcpy(d_vec, vec, n_bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_out, n_bytes));

  checkCudaErrors(cudaMalloc(&d_seg_out, sub_blocks * kBlockSize * sizeof(T)));

  hillis_steele_scan_kernel<<<blocks, kBlockSize, 2 * kBlockSize * sizeof(T)>>>(d_vec, d_out,
                                                                                d_seg_out);
  hillis_steele_scan_kernel<<<sub_blocks, kBlockSize, 2 * kBlockSize * sizeof(T)>>>(
      d_seg_out, d_vec, static_cast<T *>(nullptr));

  scan_add_segment_kernel<<<blocks, kBlockSize>>>(d_out, d_vec);

  checkCudaErrors(cudaMemcpy(out, d_out, n_bytes, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_vec));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_seg_out));
}

int main() {
  constexpr int kNumSize = 10;
  std::vector<int> vec(kNumSize);
  FillSequenceNumbers(vec, 1);
  std::cout << "Origin Vector: ";
  PrintElements(vec);

  std::vector<int> out(kNumSize);
  HillisSteeleScanCPU(vec.data(), out.data(), kNumSize);
  std::cout << "Scan Vecotr: ";
  PrintElements(out);

  // reset data
  FillSequenceNumbers(vec, 1);
  std::fill(out.begin(), out.end(), 0);

  static_assert(kNumSize < kBlockSize * kBlockSize, "NumSize is too large");
  HillisSteeleScanGPU(vec.data(), out.data(), kNumSize);
  std::cout << "GPU Scan Vecotr: ";
  PrintElements(out);

  return 0;
}