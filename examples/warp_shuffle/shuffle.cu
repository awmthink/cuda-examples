#include "common/cuda_helper.h"
#include "common/random.h"
#include "common/stopwatch.h"

__global__ void shuffle_data(int *data, int n) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= n) {
    return;
  }
  int tid = threadIdx.x;
  int val = data[tid];
  int lane_idx = threadIdx.x % warpSize;
  // 将 lane_idx+1中的val向其他所有warp线程进行广播
  int new_val = __shfl_sync(0xFFFFFFFF, val, (lane_idx + 1) % warpSize);
  __syncthreads();
  data[tid] = new_val;
}

int main() {
  constexpr int kWarpSize = 32;
  std::vector<int> src_array(kWarpSize);
  FillSequenceNumbers(src_array, 1);
  PrintElements(src_array);
  std::vector<int> dst_array(kWarpSize);

  int *d_array = nullptr;

  std::size_t data_size = kWarpSize * sizeof(int);
  cudaMalloc(&d_array, data_size);
  cudaMemcpy(d_array, src_array.data(), data_size, cudaMemcpyHostToDevice);

  shuffle_data<<<1, kWarpSize>>>(d_array, kWarpSize);

  cudaMemcpy(dst_array.data(), d_array, data_size, cudaMemcpyDeviceToHost);
  PrintElements(dst_array);

  return 0;
}