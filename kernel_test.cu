#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Error-checking helper (optional)
static inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - "
                  << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------
// Blog 1 Kernels: kernel_A, kernel_B, kernel_C
// These illustrate how to measure occupancy, memory-bound vs. compute-bound.

__global__ void kernel_A(double* A, int N, int M)
{
    double d = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
#pragma unroll 100
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }
        A[idx] = d;
    }
}

__global__ void kernel_B(double* A, int N, int M)
{
    extern __shared__ char dynamicSM[]; // Requesting dynamic shared memory
    double d = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
#pragma unroll 100
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }
        A[idx] = d;
    }
}

__global__ void kernel_C(double* A, const double* B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Strided memory access: each thread in a warp accesses memory with a stride
    const int stride = 16;
    int strided_idx = threadIdx.x * stride
                      + (blockIdx.x % stride)
                      + (blockIdx.x / stride) * stride * blockDim.x;

    if (strided_idx < N && idx < N) {
        A[idx] = B[strided_idx] + B[strided_idx];
    }
}

// -----------------------------------------------------------------------------
// Blog 2 Bank-conflict demonstration
// A kernel that intentionally causes some bank conflicts if not carefully padded.

__global__ void lltm_cuda_forward_kernel(float* res)
{
    __shared__ float sharedMem[1024];
    // Each thread tries to read from an index that can trigger bank conflicts
    // (e.g., threadIdx.x * 4).
    // This can cause multiple threads to hit the same bank simultaneously.
    // In some versions, if we read multiple times we could hide conflicts, but for
    // demonstration, let's do a single read.
    res[0] = sharedMem[threadIdx.x * 4];
}

// -----------------------------------------------------------------------------
// Blog 3: 128-bit loads (LDS.128) with different warp arrangements
// These kernels show how quarter-warps can merge or not merge memory transactions,
// and whether bank conflicts occur. The shared arrays are 128 elements in size.

__global__ void smem_1(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    // Fill shared memory
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();

    // Only threads with tid == 15 or 16 do a 128-bit load
    // reinterpret_cast<uint4 *> is a way to treat 4 * 32-bit as a 128-bit
    if (tid == 15 || tid == 16) {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}

__global__ void smem_2(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();

    // Only threads with tid == 0 or 15 do a 128-bit load
    if (tid == 0 || tid == 15) {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}

__global__ void smem_3(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();

    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_4(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();

    uint32_t addr;
    if (tid < 16) {
        addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
    } else {
        addr = (tid / 8) * 2 + ((tid % 8) % 2);
    }
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_5(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();

    // This pattern intentionally causes partial merges & potential bank conflicts
    uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];

    // Optional debug print (commented by default)
    // printf("tid: %d, addr: %d\n", tid, addr);
}

// -----------------------------------------------------------------------------
// Main function to demonstrate usage
// You can selectively comment/uncomment kernel launches to examine behaviors.
// For runtime info, you can run Nsight Compute or Nsight Systems.

int main() {
    std::cout << "Beginning comprehensive CUDA test...\n";

    // 1) Test for Blog 1 kernels
    {
        const int N = 80 * 2048 * 10; // smaller than blog example
        const int M = 5000;
        size_t sz = N * sizeof(double);

        double* dA = nullptr;
        double* dB = nullptr;
        checkCuda(cudaMalloc(&dA, sz), "Alloc dA");
        checkCuda(cudaMalloc(&dB, sz), "Alloc dB");
        checkCuda(cudaMemset(dA, 0, sz), "Memset dA");
        checkCuda(cudaMemset(dB, 0, sz), "Memset dB");

        int threadsPerBlock = 64;
        int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "Launching kernel_A...\n";
        kernel_A<<<numBlocks, threadsPerBlock>>>(dA, N, M);
        checkCuda(cudaGetLastError(), "Kernel_A launch");

        std::cout << "Launching kernel_B with dynamic shared mem (48KB)...\n";
        // Setting large dynamic shared memory usage
        checkCuda(cudaFuncSetAttribute(kernel_B,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024),
            "Set dynamic shared mem for kernel_B");

        kernel_B<<<numBlocks, threadsPerBlock, 48 * 1024>>>(dA, N, M);
        checkCuda(cudaGetLastError(), "Kernel_B launch");

        std::cout << "Launching kernel_C...\n";
        kernel_C<<<numBlocks, threadsPerBlock>>>(dA, dB, N);
        checkCuda(cudaGetLastError(), "Kernel_C launch");

        checkCuda(cudaDeviceSynchronize(), "Sync after Blog1 kernels");

        cudaFree(dA);
        cudaFree(dB);
    }

    // 2) Test for Blog 2 bank conflicts
    {
        float* dOut = nullptr;
        checkCuda(cudaMalloc(&dOut, sizeof(float) * 16), "Alloc dOut for bank conflict");
        std::cout << "\nLaunching lltm_cuda_forward_kernel (possible bank conflict example)...\n";
        lltm_cuda_forward_kernel<<<1, 32>>>(dOut);
        checkCuda(cudaGetLastError(), "lltm_cuda_forward_kernel launch");
        checkCuda(cudaDeviceSynchronize(), "Sync after Bank conflict kernel");
        cudaFree(dOut);
    }

    // 3) Test for Blog 3 (LDS.128 examples: smem_1 to smem_5)
    {
        uint32_t* dA = nullptr;
        size_t memSize = sizeof(uint32_t) * 128;
        checkCuda(cudaMalloc(&dA, memSize), "Alloc dA for smem tests");
        std::cout << "\nLaunching smem_1...\n";
        smem_1<<<1, 32>>>(dA);
        checkCuda(cudaGetLastError(), "smem_1 launch");
        checkCuda(cudaDeviceSynchronize(), "Sync smem_1");

        std::cout << "Launching smem_2...\n";
        checkCuda(cudaMemset(dA, 0, memSize), "Memset dA");
        smem_2<<<1, 32>>>(dA);
        checkCuda(cudaGetLastError(), "smem_2 launch");
        checkCuda(cudaDeviceSynchronize(), "Sync smem_2");

        std::cout << "Launching smem_3...\n";
        checkCuda(cudaMemset(dA, 0, memSize), "Memset dA");
        smem_3<<<1, 32>>>(dA);
        checkCuda(cudaGetLastError(), "smem_3 launch");
        checkCuda(cudaDeviceSynchronize(), "Sync smem_3");

        std::cout << "Launching smem_4...\n";
        checkCuda(cudaMemset(dA, 0, memSize), "Memset dA");
        smem_4<<<1, 32>>>(dA);
        checkCuda(cudaGetLastError(), "smem_4 launch");
        checkCuda(cudaDeviceSynchronize(), "Sync smem_4");

        std::cout << "Launching smem_5...\n";
        checkCuda(cudaMemset(dA, 0, memSize), "Memset dA");
        smem_5<<<1, 32>>>(dA);
        checkCuda(cudaGetLastError(), "smem_5 launch");
        checkCuda(cudaDeviceSynchronize(), "Sync smem_5");

        cudaFree(dA);
    }

    std::cout << "\nAll kernels executed.\n";
    return 0;
}
