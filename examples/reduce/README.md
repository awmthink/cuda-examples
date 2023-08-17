# 并行归约

* [GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
* [CUDA高性能计算经典问题——归约Reduction](https://zhuanlan.zhihu.com/p/416959273)


Pytorch中的版本：

1. 首先让所有线程执行 WarpReduceSum
2. 然后将每个线程束的 reduce 结果存储到 shared memory 中，注意这里是 lane_id=0 的线程去存储，因为前面提到了只有线程0上有正确的reduce结果
3. 从 shared memory 把数据读取出来，最后再用一个 warp 对其做 reduce，即可获得整个 block 的 reduce 结果


```cpp
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}


template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceSum(val);
  __syncthreads();
  if (laneid == 0) {
    shared[warpid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(0);
  if (warpid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}
```