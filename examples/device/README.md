# CUDA Device Management


## Device Query

本示例演示了使用CUDA Runtime API中的[设备特性查询的接口](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0)来获取CUDA设备相关的参数。

CUDA设备的相关特性都保存在`cudaDeviceProp`结构中，其中包括了CUDA设备的一些[详细特征](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)。

本示例在GeForce GTX 1060（6GB）设备上的运行结果如下：

```
Detected 1 CUDA Capable device(s)
CUDA Driver Version: 11.0
Runtime Version: 10.0

Device 0: "GeForce GTX 1060 6GB"
  CUDA capability Major/Minor verion number: 6.1
  Total Global Memory: 6075 MBytes
  Multiprocessor (SM) Count: 10
  CUDA Cores / SM: 128
  Total CUDA Cores: 1280
  Constant memory : 64 KBytes
  Shared memory per block : 48 KBytes
  Shared memory per SM : 96 KBytes
  Registers available per block : 65536
  Registers available per SM : 65536
  Maximum number of threads per block : 1024
  Maximum number of threads per SM : 2048
  Warp size: 32
```

更多设备管理相关的接口，可以参考CUDA Runtime API 6 Modules：[Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)