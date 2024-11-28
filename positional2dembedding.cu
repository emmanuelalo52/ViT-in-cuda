#include <cuda_runtime.h>
#include <math.h>

// Define the CUDA kernel for positional embedding
__global__ void positional_embedding_kernel(float* x, float* x_emb, float* y_emb, float* cls_token, int B, int N, int C, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * N * C;
    if (idx < total_elements) {
        int b = idx / (N * C);
        int n = (idx % (N * C)) / C;
        int c = idx % C;

        if (n == 0) {
            x[idx] = cls_token[c];
        } else {
            int i = (n - 1) / w;
            int j = (n - 1) % w;
            x[idx] += x_emb[i * (C / 2) + c % (C / 2)] + y_emb[j * (C / 2) + c % (C / 2)];
        }
    }
}

// Positional2DEmbedding forward pass using CUDA
void positional_2d_embedding_forward(float* x, float* x_emb, float* y_emb, float* cls_token, int B, int N, int C, int h, int w) {
    // Allocate memory on the GPU
    float* pos_emb_output;
    cudaMalloc(&pos_emb_output, B * N * C * sizeof(float));

    // Apply positional embedding
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((B * N * C + blockDim.x - 1) / blockDim.x, 1, 1);
    positional_embedding_kernel<<<gridDim, blockDim>>>(x, x_emb, y_emb, cls_token, B, N, C, h, w);

    // Copy the result back to the input tensor
    cudaMemcpy(x, pos_emb_output, B * N * C * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free allocated memory
    cudaFree(pos_emb_output);
}