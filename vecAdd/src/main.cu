#include <cassert>
#include <iostream>

__global__ void vecAddKern1D(const float *A, const float *B, float *C, size_t N);
__global__ void vecAddKern2D(const float *A, const float *B, float *C, size_t N);

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
    int device, SM_Count;

    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaDeviceGetAttribute(&SM_Count, cudaDevAttrMultiProcessorCount, device));

    constexpr size_t N = 1 << 28; // 268,435,456 elements
    size_t size = N * sizeof(float);

    // Allocate and initialize host arrays
    auto *h_A = static_cast<float *>(malloc(size));
    auto *h_B = static_cast<float *>(malloc(size));
    auto *h_C = static_cast<float *>(malloc(size));

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;

    checkCuda(cudaMalloc(&d_A, size));
    checkCuda(cudaMalloc(&d_B, size));
    checkCuda(cudaMalloc(&d_C, size));

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    // Launch kernel
    constexpr int threads = 256; // Threads per block
    const int blocks = SM_Count * 32; // Total blocks
    vecAddKern1D<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // Error checking
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize()); // Synchronize device with host
    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < N; i++) {
        assert(h_C[i] == h_A[i] + h_B[i]);
    }

    std::cout << "Success!" << std::endl;

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return 0;
}