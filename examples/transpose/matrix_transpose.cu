#include <iostream>
#include <vector>

#include "common/cuda_helper.h"
#include "common/random.h"
#include "common/stopwatch.h"

constexpr int kTileSize = 8;

static void TransposeCPU(const float *matrix, float *transposed_matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      transposed_matrix[j * rows + i] = matrix[i * cols + j];
    }
  }
}

__global__ void transpose_kernel(const float *mat, float *transposed_mat, int rows, int cols) {
  int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (tid_x >= cols || tid_y >= rows) {
    return;
  }

  transposed_mat[tid_x * rows + tid_y] = mat[tid_y * cols + tid_x];
}

__global__ void transpose_kernel_shared(const float *mat, float *transposed_mat, int rows,
                                        int cols) {
  int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  __shared__ float tile[kTileSize * kTileSize];

  // 合并读取mat中的数据，非合并写入共享内存
  if (gid_x < cols && gid_y < rows) {
    tile[tid_x * kTileSize + tid_y] = mat[gid_y * cols + gid_x];
  }
  // 上述代码会产成shared memory bank conflicts
  __syncthreads();

  int out_gid_x = blockIdx.y * blockDim.y + tid_x;
  int out_gid_y = blockIdx.x * blockDim.x + tid_y;

  if (out_gid_x < rows && out_gid_y < cols) {
    // 共享内存中的数据，合并写入transposed_mat中
    transposed_mat[out_gid_y * rows + out_gid_x] = tile[tid_y * kTileSize + tid_x];
  }
}

static void TransposeGPU(const float *matrix, float *transposed_matrix, int rows, int cols) {
  float *d_mat = nullptr;
  float *d_transposed_mat = nullptr;
  std::size_t data_size = rows * cols * sizeof(float);
  checkCudaErrors(cudaMalloc(&d_mat, data_size));
  checkCudaErrors(cudaMalloc(&d_transposed_mat, data_size));
  cudaMemcpy(d_mat, matrix, data_size, cudaMemcpyHostToDevice);

  dim3 threads(kTileSize, kTileSize, 1);
  dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

  Stopwatch transpose_watch;
  transpose_kernel<<<blocks, threads>>>(d_mat, d_transposed_mat, rows, cols);
  cudaDeviceSynchronize();
  std::cout << "Transpose Kernel elapsed: " << transpose_watch.Elapsed<Stopwatch::MICROSECONDS>()
            << " us" << std::endl;

  checkLastCudaError();

  checkCudaErrors(
      cudaMemcpy(transposed_matrix, d_transposed_mat, data_size, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_mat));
  checkCudaErrors(cudaFree(d_transposed_mat));
}

int main(int argc, char *argv[]) {
  constexpr int rows = 5000;
  constexpr int cols = 4000;
  std::vector<float> matrix(rows * cols);
  std::vector<float> transposed_matrix(rows * cols);
  FillRandomNumbers(matrix, -1.0f, 1.0f);
  PrintElements2D(matrix, cols);

  Stopwatch transpose_watch;
  transpose_watch.Start();
  TransposeCPU(matrix.data(), transposed_matrix.data(), rows, cols);
  std::cout << "Transpose CPU elapsed: " << transpose_watch.Elapsed() << " ms" << std::endl;

  TransposeGPU(matrix.data(), transposed_matrix.data(), rows, cols);

  PrintElements2D(transposed_matrix, rows);

  return 0;
}