#include <stdio.h>

#include <cuda_runtime_api.h>

__global__ void hello() { printf("hello, "); }

int main(int argc, char *argv[]) {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("world!\n");
}