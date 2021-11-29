#include <cstdio>
#include <map>

#include <cuda_runtime_api.h>

#include "common/cuda_helper.h"

// ConvertSMVer2Cores convert gpu architecture version to cuda cores (ALU lanes)
// if the architectur version was not defined, return -1
// refer: https://en.wikipedia.org/wiki/CUDA
static inline int ConvertSMVer2Cores(int major, int minor) {
  const std::map<int, int> gpu_arch_cores_per_SM = {
      {0x10, 8},   {0x11, 8},   {0x12, 8},   {0x13, 8},   {0x20, 32},
      {0x21, 48},  {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
      {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128},
      {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
      {0x86, 128}, {0x87, 128}};
  int arch_version = (major << 4) + minor;
  auto found = gpu_arch_cores_per_SM.find(arch_version);
  if (found == gpu_arch_cores_per_SM.end()) {
    return -1;
  }
  return found->second;
}

int main(int argc, char *argv[]) {
  int device_count = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", device_count);
  }

  int driver_version = 0;
  int runtime_version = 0;
  checkCudaErrors(cudaDriverGetVersion(&driver_version));
  checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));
  printf("CUDA Driver Version: %d.%d\n", driver_version / 1000,
         (driver_version % 100) / 10);
  printf("Runtime Version: %d.%d\n", runtime_version / 1000,
         (runtime_version % 100) / 10);

  for (int dev = 0; dev < device_count; dev++) {
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceProp device_prop;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev));

    printf("\nDevice %d: \"%s\"\n", dev, device_prop.name);
    printf("  CUDA capability Major/Minor verion number: %d.%d\n",
           device_prop.major, device_prop.minor);
    printf("  Total Global Memory: %.0f MBytes\n",
           static_cast<float>(device_prop.totalGlobalMem) / 1024.0f / 1024.0f);
    printf("  Multiprocessor (SM) Count: %d\n",
           device_prop.multiProcessorCount);
    printf("  CUDA Cores / SM: %d\n",
           ConvertSMVer2Cores(device_prop.major, device_prop.minor));
    printf("  Total CUDA Cores: %d\n",
           ConvertSMVer2Cores(device_prop.major, device_prop.minor) *
               device_prop.multiProcessorCount);

    printf("  Constant memory : %zu KBytes\n",
           device_prop.totalConstMem / 1024);
    printf("  Shared memory per block : %zu KBytes\n",
           device_prop.sharedMemPerBlock / 1024);
    printf("  Shared memory per SM : %zu KBytes\n",
           device_prop.sharedMemPerMultiprocessor / 1024);
    printf("  Registers available per block : %d\n", device_prop.regsPerBlock);
    printf("  Registers available per SM : %d\n",
           device_prop.regsPerMultiprocessor);
    printf("  Maximum number of threads per block : %d\n",
           device_prop.maxThreadsPerBlock);
    printf("  Maximum number of threads per SM : %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf("  Warp size: %d\n", device_prop.warpSize);
  }

  return 0;
}