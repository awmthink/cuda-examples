#include <iostream>
#include <vector>

#include "common/cuda_helper.h"
#include "common/random.h"
#include "common/stopwatch.h"

static void PrintMatrix(const std::vector<int> &matrix, int cols) {
  for (decltype(matrix.size()) i = 0; i < matrix.size(); i++) {
    if (i > 0 && i % cols == 0) {
      std::cout << std::endl;
    }
    std::cout << matrix[i] << "\t";
  }
  std::cout << std::endl;
}

static void TransposeCPU(const int *matrix, int *transposed_matrix, int rows,
                         int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      transposed_matrix[j * rows + i] = matrix[i * cols + j];
    }
  }
}

static inline void SimpleMatrixTranspose() {
  constexpr int rows = 5;
  constexpr int cols = 7;
  std::vector<int> matrix(rows * cols);
  std::vector<int> transposed_matrix(rows * cols);
  FillSequenceNumbers(matrix);
  std::cout << "Matrix:" << std::endl;
  PrintMatrix(matrix, cols);
  TransposeCPU(matrix.data(), transposed_matrix.data(), rows, cols);
  std::cout << "Transposed Matrix:" << std::endl;
  PrintMatrix(transposed_matrix, rows);
}

int main(int argc, char *argv[]) {
  SimpleMatrixTranspose();

  constexpr int rows = 3000;
  constexpr int cols = 4000;
  std::vector<int> matrix(rows * cols);
  std::vector<int> transposed_matrix(rows * cols);
  FillSequenceNumbers(matrix);

  Stopwatch transpose_watch;
  transpose_watch.Start();
  TransposeCPU(matrix.data(), transposed_matrix.data(), rows, cols);
  std::cout << "Transpose CPU elapsed: " << transpose_watch.Elapsed() << " ms"
            << std::endl;

  return 0;
}