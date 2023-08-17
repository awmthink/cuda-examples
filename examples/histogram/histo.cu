#include "common/cuda_helper.h"
#include "common/gpu_timer.h"
#include "common/random.h"

#include <vector>

constexpr float kMaxValue = 5.0f;
constexpr float kMinValue = -5.0f;
constexpr int kNums = 1'000'000;
constexpr int kBins = 8;

__host__ __device__ int compute_bin_index(float value) {
  constexpr float bin = (kMaxValue - kMinValue) / kBins;
  return static_cast<int>((value - kMinValue) / bin);
}

// 该版本要求，kbins必须小于一个block中的线程数量
__global__ void simple_histo_kernel(const float* array, int* hist_bins, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  // 每个block 计算一个局部的直方图
  __shared__ int shist_bins[kBins];
  if (tid < kBins) {
    shist_bins[tid] = 0;
  }
  __syncthreads();
  // 这里如果将该条件判断放在外围，会导致共享内存中的部分bin无法写入全局内存
  if (gid < n) {
    int bin_index = compute_bin_index(array[gid]);
    atomicAdd(&shist_bins[bin_index], 1);
  }

  __syncthreads();

  // 将每个block计算的局部直方图汇总到最终的直方图中
  if (tid < kBins) {
    atomicAdd(&hist_bins[tid], shist_bins[tid]);
  }
}

void SimpleHistoGPU(const std::vector<float>& array_data,
                    std::vector<int>& bin_data) {
  float* d_array_data = nullptr;
  int* d_bin_data = nullptr;
  checkCudaErrors(cudaMalloc(&d_array_data, sizeof(float) * array_data.size()));
  checkCudaErrors(cudaMemcpy(d_array_data, array_data.data(),
                             sizeof(float) * array_data.size(),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&d_bin_data, sizeof(int) * bin_data.size()));

  int threads = 512;
  int blocks = (array_data.size() + threads - 1) / threads;
  GpuTimer kernel_timer;
  kernel_timer.Start();
  simple_histo_kernel<<<blocks, threads>>>(d_array_data, d_bin_data,
                                           array_data.size());
  cudaDeviceSynchronize();
  kernel_timer.Stop();
  std::cout << "simple_histo_kernel execute time: " << kernel_timer.Elapsed()
            << " s" << std::endl;
  checkCudaErrors(cudaMemcpy(bin_data.data(), d_bin_data,
                             sizeof(int) * bin_data.size(),
                             cudaMemcpyDeviceToHost));
}

void SimpleHistoCPU(const std::vector<float>& array_data,
                    std::vector<int>& bin_data) {
  for (auto& data : array_data) {
    ++bin_data[compute_bin_index(data)];
  }
}

int main() {
  std::vector<float> num_array(kNums);
  FillRandomNumbers(num_array, kMinValue, kMaxValue);
  std::vector<int> dist_bins(kBins, 0);

  GpuTimer cpu_timer;
  cpu_timer.Start();
  SimpleHistoCPU(num_array, dist_bins);
  cpu_timer.Stop();
  std::cout << "SimpleHistoCPU execute time: " << cpu_timer.Elapsed() << " s"
            << std::endl;
  PrintElements(dist_bins);
  dist_bins.assign(dist_bins.size(), 0);
  SimpleHistoGPU(num_array, dist_bins);
  PrintElements(dist_bins);

  return 0;
}