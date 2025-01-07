#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

constexpr int WARP_SIZE = 32;

__device__ __always_inline half ldg_f16_prefetch_64B(const half *const ptr) {
    half ret;
    asm ("ld.global.L2::64B.b16 %0, [%1];"  : "=h"(*(reinterpret_cast<unsigned short *>(&(ret)))) : "l"(ptr));
    return ret;
}

__device__ __always_inline half ldg_f16_prefetch_128B(const half *const ptr) {
    half ret;
    asm ("ld.global.L2::128B.b16 %0, [%1];"  : "=h"(*(reinterpret_cast<unsigned short *>(&(ret)))) : "l"(ptr));
    return ret;
}

__device__ __always_inline half ldg_f16_prefetch_256B(const half *const ptr) {
    half ret;
    asm ("ld.global.L2::256B.b16 %0, [%1];"  : "=h"(*(reinterpret_cast<unsigned short *>(&(ret)))) : "l"(ptr));
    return ret;
}

__device__ __always_inline float ldg_f32_prefetch_64B(const float *ptr) {
  float ret;
  asm volatile ("ld.global.L2::64B.f32 %0, [%1];"  : "=f"(ret) : "l"(ptr));
  return ret;
}

__device__ __always_inline float ldg_f32_prefetch_128B(const float *ptr) {
  float ret;
  asm volatile ("ld.global.L2::128B.f32 %0, [%1];"  : "=f"(ret) : "l"(ptr));
  return ret;
}

__device__ __always_inline float ldg_f32_prefetch_256B(const float *ptr) {
  float ret;
  asm volatile ("ld.global.L2::256B.f32 %0, [%1];"  : "=f"(ret) : "l"(ptr));
  return ret;
}

__global__
void prefetch_kernel_load_64_prefetch_0(const half* in, half* out) {
  half value_0 = in[threadIdx.x]; // Load 64B Prefetch 0B
  __nanosleep(1000000);  // Sleep 1us
  half value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  // half result = value_0 + value_1;
  half value_2 = in[threadIdx.x + WARP_SIZE * 2];
  half value_3 = in[threadIdx.x + WARP_SIZE * 3];
  half value_4 = in[threadIdx.x + WARP_SIZE * 4];
  half value_5 = in[threadIdx.x + WARP_SIZE * 5];
  half result = value_0 + value_1 + value_2 + value_3 + value_4 + value_5;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_64_prefetch_64(const half* in, half* out) {
  half value_0 = ldg_f16_prefetch_64B(&in[threadIdx.x]); // Load 64B Prefetch 64B
  __nanosleep(1000000);  // Sleep 1us
  half value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  // half result = value_0 + value_1;
  half value_2 = in[threadIdx.x + WARP_SIZE * 2];
  half value_3 = in[threadIdx.x + WARP_SIZE * 3];
  half value_4 = in[threadIdx.x + WARP_SIZE * 4];
  half value_5 = in[threadIdx.x + WARP_SIZE * 5];
  half result = value_0 + value_1 + value_2 + value_3 + value_4 + value_5;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_64_prefetch_128(const half* in, half* out) {
  half value_0 = ldg_f16_prefetch_128B(&in[threadIdx.x]); // Load 64B Prefetch 64B
  __nanosleep(1000000);  // Sleep 1us
  half value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  // half result = value_0 + value_1;
  half value_2 = in[threadIdx.x + WARP_SIZE * 2];
  half value_3 = in[threadIdx.x + WARP_SIZE * 3];
  half value_4 = in[threadIdx.x + WARP_SIZE * 4];
  half value_5 = in[threadIdx.x + WARP_SIZE * 5];
  half result = value_0 + value_1 + value_2 + value_3 + value_4 + value_5;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_64_prefetch_256(const half* in, half* out) {
  half value_0 = ldg_f16_prefetch_256B(&in[threadIdx.x]); // Load 64B Prefetch 64B
  __nanosleep(1000000);  // Sleep 1us
  half value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  // half result = value_0 + value_1;
  half value_2 = in[threadIdx.x + WARP_SIZE * 2];
  half value_3 = in[threadIdx.x + WARP_SIZE * 3];
  half value_4 = in[threadIdx.x + WARP_SIZE * 4];
  half value_5 = in[threadIdx.x + WARP_SIZE * 5];
  half result = value_0 + value_1 + value_2 + value_3 + value_4 + value_5;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_128_prefetch_0(const float* in, float* out) {
  float value_0 = in[threadIdx.x];
  __nanosleep(1000000);  // Sleep 1us
  float value_1 = in[threadIdx.x + WARP_SIZE * 1];
  float value_2 = in[threadIdx.x + WARP_SIZE * 2];
  float result = value_0 + value_1 + value_2;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_128_prefetch_64(const float* in, float* out) {
  float value_0 = ldg_f32_prefetch_64B(&in[threadIdx.x]);
  __nanosleep(1000000);  // Sleep 1us
  float value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  float value_2 = in[threadIdx.x + WARP_SIZE * 2];
  float result = value_0 + value_1 + value_2;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_128_prefetch_128(const float* in, float* out) {
  float value_0 = ldg_f32_prefetch_128B(&in[threadIdx.x]);
  __nanosleep(1000000);  // Sleep 1us
  float value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  float value_2 = in[threadIdx.x + WARP_SIZE * 2];
  float result = value_0 + value_1 + value_2;
  out[threadIdx.x] = result;
}

__global__
void prefetch_kernel_load_128_prefetch_256(const float* in, float* out) {
  float value_0 = ldg_f32_prefetch_256B(&in[threadIdx.x]);
  __nanosleep(1000000);  // Sleep 1us
  float value_1 = in[threadIdx.x + WARP_SIZE * 1]; // Load Next 64B
  float value_2 = in[threadIdx.x + WARP_SIZE * 2];
  float result = value_0 + value_1 + value_2;
  out[threadIdx.x] = result;
}

int launchKernelExperiment0() {
  constexpr int page_size = 4096;
  constexpr int nr_element_f16 = 4096 / sizeof(half);
  constexpr int offset = nr_element_f16 / 2;

  void* input;
  CUDACHECK(cudaMalloc(&input, page_size));
  CUDACHECK(cudaMemset(input, 0, 4096));

  dim3 grid(1, 1, 1);
  dim3 block(32, 1, 1); // Just one warp
  
  const half* in = reinterpret_cast<const half*>(input);
  half* out = reinterpret_cast<half* >(input) + offset;
  std::cout << "In: " << in << "  Out: " << out << std::endl;
  
  prefetch_kernel_load_64_prefetch_0<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_64_prefetch_64<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_64_prefetch_128<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_64_prefetch_256<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaFree(input));
  return 0;
}

int launchKernelExperiment1() {
  constexpr int page_size = 4096;
  constexpr int nr_element_f32 = 4096 / sizeof(float);
  constexpr int offset = nr_element_f32 / 2;

  void* input;
  CUDACHECK(cudaMalloc(&input, page_size));
  CUDACHECK(cudaMemset(input, 0, 4096));

  dim3 grid(1, 1, 1);
  dim3 block(32, 1, 1); // Just one warp
  
  const float* in = reinterpret_cast<const float*>(input);
  float* out = reinterpret_cast<float* >(input) + offset;
  std::cout << "In: " << in << "  Out: " << out << std::endl;
  
  prefetch_kernel_load_128_prefetch_0<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_128_prefetch_64<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_128_prefetch_128<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());
  prefetch_kernel_load_128_prefetch_256<<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaFree(input));
  return 0;
}

int main(void) {
  launchKernelExperiment0();
  launchKernelExperiment1();
  return 0;
}
