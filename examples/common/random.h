#ifndef EXAMPLES_COMMON_RANDOM_H_
#define EXAMPLES_COMMON_RANDOM_H_

#include <iostream>
#include <random>
#include <vector>

// FillRandomNumbers 向输入序列中填充(min,max)之间的随机数
void FillRandomNumbers(std::vector<float>& array, float min = -1.0f,
                       float max = 1.0f, int64_t seed = 0L) {
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(min, max);
  for (auto& num : array) {
    num = distribution(generator);
  }
}
// FillRandomNumbers 向输入序列中填充(min,max)之间的随机数
void FillRandomNumbers(std::vector<int>& array, int min = -1, int max = 1,
                       int64_t seed = 0L) {
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(min, max);
  for (auto& num : array) {
    num = distribution(generator);
  }
}

// FillSequenceNumbers向输入序列中填充从start开始的顺序递增的数字
void FillSequenceNumbers(std::vector<int>& array, int start = 0) {
  std::iota(array.begin(), array.end(), start);
}

template <typename T>
void PrintElements(const std::vector<T>& vec, size_t num = 0) {
  if (vec.empty()) {
    return;
  }
  if (num == 0) {
    num = vec.size();
  }
  std::cout << "[" << vec.front();
  for (size_t i = 1; i != vec.size() && i < num; ++i) {
    std::cout << ", " << vec[i];
  }
  std::cout << "]" << std::endl;
}

template <typename T>
static void PrintElements2D(const std::vector<T>& matrix, int cols,
                            int max_rows = 5, int max_cols = 5) {
  int num_rows = 0;
  for (decltype(matrix.size()) i = 0; i < matrix.size(); i++) {
    if (i > 0 && i % cols == 0) {
      std::cout << std::endl;
      ++num_rows;
    }
    if (i % cols >= max_cols) {
      continue;
    }
    if (num_rows >= max_rows) {
      break;
    }
    std::cout << matrix[i] << "\t";
  }
}

#endif  // EXAMPLES_COMMON_RANDOM_H_
