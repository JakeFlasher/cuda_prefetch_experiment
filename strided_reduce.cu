#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

// Error-checking macro for CUDA
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

//--------------------------------------
// Prefetch Primitives
//--------------------------------------
// For 256-byte L2 prefetch loads (when enabled)
__device__ __always_inline float4 ldg_f32v4_prefetch_256B(const float4 *ptr) {
  float4 ret;
  asm volatile ("ld.global.L2::256B.v4.f32 {%0,%1,%2,%3}, [%4];"
                : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
                : "l"(ptr));
  return ret;
}

//--------------------------------------
// Warp & Block Reduction Utilities
//--------------------------------------
#define FINAL_MASK 0xffffffff

template <typename T>
__device__ __always_inline
T warpReduceSum(T val)
{
  // This loop performs a warp-level reduction
  // (each warp has 32 threads if blockDim.x >= 32).
  for(int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

template <typename T>
__device__ __always_inline
T blockReduceSum(T val)
{
  // This array resides in shared memory; it stores
  // one value per warp (of which there are blockDim.x/32).
  static __shared__ T shared[32];

  int lane = threadIdx.x & 0x1f;  // 0..31
  int wid  = threadIdx.x >> 5;    // warp ID
  val      = warpReduceSum<T>(val);
  if(lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // Now, only the first warp handles the final sum...
  return (wid == 0)
         ? warpReduceSum( (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f )
         : (T)0.0f;
}

//-------------------------------------------------------------------------------------------------
// Kernel 1: Uses 256-byte prefetch (equivalent to a #define PREFETCH_256 scenario)
//-------------------------------------------------------------------------------------------------
template<int CG_SIZE, int LOOP_COUNT>
__global__ void prefetch_kernel_256(const float* in, float* out)
{
  // Each thread calculates a unique offset into the input array
  int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;

  // Group info: group_id is which "cooperative group" we're in,
  // thread_id is our position within that group
  int group_id   = thread_offset / CG_SIZE;
  int thread_id  = thread_offset % CG_SIZE;

  // We aim to load 4 * CG_SIZE * LOOP_COUNT floats per group
  constexpr int elements_per_group = 4 * CG_SIZE * LOOP_COUNT;  // unit: float

  // Cast pointer to float4 for vectorized loads
  const float4* ptr = reinterpret_cast<const float4*>(in + elements_per_group * group_id) + thread_id;

  float sum = 0.0f;

  // Main loop for summation
  for (int i = 0; i < LOOP_COUNT; i++) {
    // Use 256-byte prefetch instruction
    float4 vals = ldg_f32v4_prefetch_256B(ptr);

    // Accumulate tanh of each element
    sum += tanhf(vals.x);
    sum += tanhf(vals.y);
    sum += tanhf(vals.z);
    sum += tanhf(vals.w);

    // Move pointer by CG_SIZE float4's forward
    ptr += CG_SIZE;
  }

  // Perform block-wide reduction
  sum = blockReduceSum(sum);

  // Only one thread in the block writes out the reduced result
  if(threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

//-------------------------------------------------------------------------------------------------
// Kernel 2: Does not use 256-byte prefetch (equivalent to no #define PREFETCH_256 scenario)
//-------------------------------------------------------------------------------------------------
template<int CG_SIZE, int LOOP_COUNT>
__global__ void prefetch_kernel_0(const float* in, float* out)
{
  int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int group_id   = thread_offset / CG_SIZE;
  int thread_id  = thread_offset % CG_SIZE;

  constexpr int elements_per_group = 4 * CG_SIZE * LOOP_COUNT;  // unit: float
  const float4* ptr = reinterpret_cast<const float4*>(in + elements_per_group * group_id) + thread_id;

  float sum = 0.0f;

  for (int i = 0; i < LOOP_COUNT; i++) {
    // Regular global load
    float4 vals = *ptr;

    sum += tanhf(vals.x);
    sum += tanhf(vals.y);
    sum += tanhf(vals.z);
    sum += tanhf(vals.w);

    ptr += CG_SIZE;
  }

  sum = blockReduceSum(sum);

  if(threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

//---------------------------------------------------------------------------------
// Launch function that runs both kernels for profiling or experimentation
//---------------------------------------------------------------------------------
int launchKernelExperimentDual()
{
  // Here we define how many pages, how many floats, etc.
  constexpr int page_size    = 4096;
  constexpr int page_count   = 256 * 1024;
  constexpr int element_count= page_size * page_count / 4;

  // Each cooperative group has cg_size threads,
  // and each group processes 4*cg_size*loop_count floats
  constexpr int cg_size      = 8;
  constexpr int loop_count   = 64;

  // Allocate memory on device
  void* input;
  void* output_prefetch256;
  void* output_no_prefetch256;
  CUDACHECK(cudaMalloc(&input,                page_size * page_count));
  CUDACHECK(cudaMalloc(&output_prefetch256,   page_size * page_count));
  CUDACHECK(cudaMalloc(&output_no_prefetch256,page_size * page_count));

  // Clear out the memory
  CUDACHECK(cudaMemset(input, 0,                page_size * page_count));
  CUDACHECK(cudaMemset(output_prefetch256, 0,   page_size * page_count));
  CUDACHECK(cudaMemset(output_no_prefetch256, 0,page_size * page_count));

  // Define grid & block
  //  - We want total threads to be element_count/(4*cg_size*loop_count)
  //    times (256/8). This is borrowed from the original code.
  dim3 grid(element_count / (4 * cg_size * loop_count) / (256 / 8), 1, 1);
  dim3 block(256, 1, 1);

  // Cast pointers for kernel launching
  const float* in_prefetch256  = reinterpret_cast<const float*>(input);
  float*       out_prefetch256 = reinterpret_cast<float*>(output_prefetch256);

  const float* in_no_prefetch256  = reinterpret_cast<const float*>(input);
  float*       out_no_prefetch256 = reinterpret_cast<float*>(output_no_prefetch256);

  //-----------------------------------
  // 1) Launch kernel with 256B prefetch
  //-----------------------------------
  prefetch_kernel_256<cg_size, loop_count><<<grid, block>>>(in_prefetch256, out_prefetch256);
  CUDACHECK(cudaDeviceSynchronize());

  //-----------------------------------
  // 2) Launch kernel without 256B prefetch
  //-----------------------------------
  prefetch_kernel_0<cg_size, loop_count><<<grid, block>>>(in_no_prefetch256, out_no_prefetch256);
  CUDACHECK(cudaDeviceSynchronize());

  // Free all device allocations
  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(output_prefetch256));
  CUDACHECK(cudaFree(output_no_prefetch256));

  return 0;
}

int main(void)
{
  // We launch our dual-kernel experiment,
  // which runs both variants sequentially.
  launchKernelExperimentDual();
  return 0;
}
