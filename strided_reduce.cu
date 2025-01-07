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

// constexpr int WARP_SIZE = 32;

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

#define FINAL_MASK 0xffffffff

template <typename T>
__device__ __always_inline
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__device__ __always_inline
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum<T>(val);
  if(lane == 0)
    shared[wid] = val;
  __syncthreads();
  return wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)0.0f) : 0.0f;
}

// Two Stage Reduce
template<int CG_SIZE, int LOOP_COUNT>
__global__ void prefetch_kernel(const float* in, float* out) {
  int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int group_id = thread_offset / CG_SIZE;
  int thread_id = thread_offset % CG_SIZE;

  constexpr int elements_per_group = 4 * CG_SIZE * LOOP_COUNT;  // unit: float
  const float4* ptr = reinterpret_cast<const float4*>(in + elements_per_group * group_id) + thread_id;
  float sum = 0.0;
  for (int i = 0; i < LOOP_COUNT; i++) {
#ifdef PREFETCH_256
    float4 vals = ldg_f32v4_prefetch_256B(ptr);
#else
    float4 vals = *ptr;
#endif
    // tanh
    sum += tanh(vals.x);
    sum += tanh(vals.y);
    sum += tanh(vals.z);
    sum += tanh(vals.w);
    ptr += CG_SIZE;
  }

  sum = blockReduceSum(sum);
  if(threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

int launchKernelExperiment() {
  constexpr int page_size = 4096;
  constexpr int page_count = 256 * 1024;
  constexpr int element_count = page_size * page_count / 4;
  constexpr int cg_size = 8;
  constexpr int loop_count = 64;

  void* input;
  void* output;
  CUDACHECK(cudaMalloc(&input, page_size * page_count));
  CUDACHECK(cudaMalloc(&output, page_size * page_count));
  CUDACHECK(cudaMemset(input, 0, page_size * page_count));

  dim3 grid(element_count / (4 * cg_size * loop_count) / (256 / 8), 1, 1);
  dim3 block(256, 1, 1);

  const float* in = reinterpret_cast<const float*>(input);
  float* out = reinterpret_cast<float*>(output);
  prefetch_kernel<cg_size, loop_count><<<grid, block>>>(in, out);
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(output));
  return 0;
}

int main(void) {
  launchKernelExperiment();
  return 0;
}
