//
// Created by jm on 1/10/26.
//

__global__ void vecAddKern1D(const float *A, const float *B, float *C, const size_t N) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    #pragma unroll
    for (unsigned i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecAddKern2D(const float *A, const float *B, float *C, const size_t N) {
    const unsigned idx = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x * gridDim.y;

    #pragma unroll
    for (unsigned i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
