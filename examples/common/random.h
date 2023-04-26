#ifndef EXAMPLES_COMMON_RANDOM_H_
#define EXAMPLES_COMMON_RANDOM_H_

#include <random>
#include <vector>

// FillRandomNumbers 向输入序列中填充(min,max)之间的随机数
void FillRandomNumbers(std::vector<float>& array, float min = -1.0f,
                       float max = 1.0f, int64_t seed = 0L) {
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(min, max);
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

#endif  // EXAMPLES_COMMON_RANDOM_H_
