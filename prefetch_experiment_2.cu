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

__device__ __always_inline float4 ldg_f32v4_prefetch_64B(const float4 *ptr) {
  float4 ret;
  asm volatile ("ld.global.L2::64B.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
  return ret;
}

__device__ __always_inline float4 ldg_f32v4_prefetch_128B(const float4 *ptr) {
  float4 ret;
  asm volatile ("ld.global.L2::128B.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
  return ret;
}

__device__ __always_inline float4 ldg_f32v4_prefetch_256B(const float4 *ptr) {
  float4 ret;
  asm volatile ("ld.global.L2::256B.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
  return ret;
}

__global__
void prefetch_kernel_load_128_prefetch_0(const float* in, float* out) {
  const float4* value_0_ptr = reinterpret_cast<const float4*>(in);
  float4 value_0 = *(value_0_ptr + threadIdx.x);
  __nanosleep(1000000);  // Sleep 1us
  float4 value_1 = value_0_ptr[threadIdx.x + blockDim.x * 1];
  float4 value_2 = value_0_ptr[threadIdx.x + blockDim.x * 2];
  float4 result = make_float4(value_0.x + value_1.x + value_2.x,
                              value_0.y + value_1.y + value_2.y,
                              value_0.z + value_1.z + value_2.z,
                              value_0.w + value_1.w + value_2.w);
  float4* out_ptr = reinterpret_cast<float4*>(out);
  *(out_ptr + threadIdx.x) = result;
}

__global__
void prefetch_kernel_load_128_prefetch_64(const float* in, float* out) {
  const float4* value_0_ptr = reinterpret_cast<const float4*>(in);
  float4 value_0 = ldg_f32v4_prefetch_64B(value_0_ptr + threadIdx.x);
  __nanosleep(1000000);  // Sleep 1us
  float4 value_1 = value_0_ptr[threadIdx.x + blockDim.x * 1];
  float4 value_2 = value_0_ptr[threadIdx.x + blockDim.x * 2];
  float4 result = make_float4(value_0.x + value_1.x + value_2.x,
                              value_0.y + value_1.y + value_2.y,
                              value_0.z + value_1.z + value_2.z,
                              value_0.w + value_1.w + value_2.w);
  float4* out_ptr = reinterpret_cast<float4*>(out);
  *(out_ptr + threadIdx.x) = result;
}

__global__
void prefetch_kernel_load_128_prefetch_128(const float* in, float* out) {
  const float4* value_0_ptr = reinterpret_cast<const float4*>(in);
  float4 value_0 = ldg_f32v4_prefetch_128B(value_0_ptr + threadIdx.x);
  __nanosleep(1000000);  // Sleep 1us
  float4 value_1 = value_0_ptr[threadIdx.x + blockDim.x * 1];
  float4 value_2 = value_0_ptr[threadIdx.x + blockDim.x * 2];
  float4 result = make_float4(value_0.x + value_1.x + value_2.x,
                              value_0.y + value_1.y + value_2.y,
                              value_0.z + value_1.z + value_2.z,
                              value_0.w + value_1.w + value_2.w);
  float4* out_ptr = reinterpret_cast<float4*>(out);
  *(out_ptr + threadIdx.x) = result;
}

__global__
void prefetch_kernel_load_128_prefetch_256(const float* in, float* out) {
  const float4* value_0_ptr = reinterpret_cast<const float4*>(in);
  float4 value_0 = ldg_f32v4_prefetch_256B(value_0_ptr + threadIdx.x);
  __nanosleep(1000000);  // Sleep 1us
  float4 value_1 = value_0_ptr[threadIdx.x + blockDim.x * 1];
  float4 value_2 = value_0_ptr[threadIdx.x + blockDim.x * 2];
  float4 result = make_float4(value_0.x + value_1.x + value_2.x,
                              value_0.y + value_1.y + value_2.y,
                              value_0.z + value_1.z + value_2.z,
                              value_0.w + value_1.w + value_2.w);
  float4* out_ptr = reinterpret_cast<float4*>(out);
  *(out_ptr + threadIdx.x) = result;
}

int launchKernelExperiment() {
  constexpr int page_size = 4096;
  constexpr int nr_element_f32 = 4096 / sizeof(float);
  constexpr int offset = nr_element_f32 / 2;

  void* input;
  CUDACHECK(cudaMalloc(&input, page_size));
  CUDACHECK(cudaMemset(input, 0, 4096));

  dim3 grid(1, 1, 1);
  dim3 block(8, 1, 1); // Just 8 threads
  
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
  launchKernelExperiment();
  return 0;
}
