#include <iostream>
#include <cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

template <typename TensorS, typename G2STiledCopy, unsigned int CG_SIZE,
  unsigned int BLOCK_DIM, unsigned int VEC_SIZE, unsigned int LOOP_COUNT>
__global__ void copy_kernel_vectorized(TensorS S) {
  using namespace cute;
  using Dtype = float;
  constexpr unsigned int CG_TILE_SIZE = CG_SIZE * VEC_SIZE * LOOP_COUNT;
  // auto cg = cooperative_groups::tiled_partition<8>(cooperative_groups::this_thread_block());
  auto cg = cooperative_groups::tiled_partition<CG_SIZE>(cooperative_groups::this_thread_block());
  
  auto cta_divide_tile = tiled_divide(S, Shape<Int<BLOCK_DIM * VEC_SIZE * LOOP_COUNT>>{});  // ((CTAShape_M,), m')
  auto cta_tile = cta_divide_tile(make_coord(_), blockIdx.x);

  auto cg_divide_tile = tiled_divide(cta_tile, Shape<Int<CG_TILE_SIZE>>{}); // ((CGShape_M,), m')
  auto cg_tile = cg_divide_tile(make_coord(_), cg.meta_group_rank());
  auto cg_loop_divide_tile_S = tiled_divide(cg_tile, Shape<Int<CG_SIZE * VEC_SIZE>>{});
  constexpr int loop_count = size<1>(cg_loop_divide_tile_S);
  static_assert(loop_count == LOOP_COUNT);

  __align__(128) __shared__ Dtype shared_storage[BLOCK_DIM * VEC_SIZE * LOOP_COUNT];

  Tensor D = make_tensor(make_smem_ptr(shared_storage + cg.meta_group_rank() * CG_TILE_SIZE), Layout<Shape<Int<CG_TILE_SIZE>>>{});
  auto cg_loop_divide_tile_D = tiled_divide(D, Shape<Int<CG_SIZE * VEC_SIZE>>{});

  auto g2s_tiled_copy = G2STiledCopy{};
  // Construct a Tensor corresponding to each thread's slice.
  ThrCopy g2s_thr_copy = g2s_tiled_copy.get_thread_slice(cg.thread_rank());

  for (int i = 0; i < loop_count; i++) {
    Tensor thr_tile_S = g2s_thr_copy.partition_S(cg_loop_divide_tile_S(make_coord(_), i));
    Tensor thr_tile_D = g2s_thr_copy.partition_D(cg_loop_divide_tile_D(make_coord(_), i));
    // Copy from GMEM to SMEM via cp.async
    copy(g2s_tiled_copy, thr_tile_S, thr_tile_D);
  }
  // cp_async_fence();
  cp_async_wait<0>();
#ifdef DEBUG
  for (int i = 0; i < loop_count; i++) {
    float4 vals = *reinterpret_cast<float4*>(
      shared_storage + cg.meta_group_rank() * CG_TILE_SIZE + i * CG_SIZE * VEC_SIZE + cg.thread_rank() * VEC_SIZE);
    printf("threadIdx: %u, loop %d, vals: %f, %f, %f, %f\n", threadIdx.x, i, vals.x, vals.y, vals.z, vals.w);
  }
#endif
}

int main(void) {
  using namespace cute;
  using Dtype = float;
  constexpr unsigned int loop = 4;
  constexpr unsigned int num_grids = 2;
  constexpr unsigned int num_threads = 64;
  constexpr unsigned int num_cg_threads = 8;
  constexpr unsigned int num_elements = num_grids * num_threads * 4 * loop; // Every thread copy a float4 for one loop. 
  thrust::host_vector<Dtype> h_S(num_elements);
  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Dtype>(i);
  }
  thrust::device_vector<Dtype> d_S = h_S;

  auto tensor_shape = make_shape(num_elements); // Dynamic shape
  Tensor tensor_S = make_tensor(
    make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));

  // Construct a TiledCopy with a specific access pattern.
  Layout thr_layout = make_layout(make_shape(Int<num_cg_threads>{})); // (8,) -> thr_idx
  Layout val_layout = make_layout(make_shape(Int<4>{}));              // (4,) -> val_idx
  using Atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, Dtype>;
  TiledCopy g2s_tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);
  print(g2s_tiled_copy);

  // Launch Kernel
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  cudaLaunchConfig_t config;
  config.gridDim = num_grids;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.numAttrs = 0;
  config.stream = stream;
  std::cout << "Data Pointer: " << thrust::raw_pointer_cast(d_S.data()) << std::endl;
  CUDACHECK(cudaLaunchKernelEx(&config, copy_kernel_vectorized<decltype(tensor_S), decltype(g2s_tiled_copy),
    num_cg_threads, num_threads, 4, loop>, tensor_S));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaStreamDestroy(stream));
  return 0;
}
