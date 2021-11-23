#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void hello() { printf("hello, "); }

int main(int argc, char *argv[]) {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("world!\n");
}