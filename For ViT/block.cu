#include <cuda_runtime.h>
#include <math.h>
#include "Self_attention.cu"
#include "mlp.cu"
// Define the CUDA kernel for layer normalization
__global__ void layer_norm_kernel(float* x, float* mean, float* var, float* gamma, float* beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float eps = 1e-5;
        float std_dev = sqrt(var[idx] + eps);
        x[idx] = gamma[idx] * (x[idx] - mean[idx]) / std_dev + beta[idx];
    }
}

// Block forward pass using CUDA
void block_forward(float* x, float* msa_weight, float* msa_bias, float* mlp_fc1_weight, float* mlp_fc1_bias, float* mlp_fc2_weight, float* mlp_fc2_bias, float dropout_prob, float* gamma1, float* beta1, float* gamma2, float* beta2, int B, int N, int C) {
    // Allocate memory on the GPU
    float* norm1_mean;
    float* norm1_var;
    cudaMalloc(&norm1_mean, B * N * sizeof(float));
    cudaMalloc(&norm1_var, B * N * sizeof(float));

    float* norm2_mean;
    float* norm2_var;
    cudaMalloc(&norm2_mean, B * N * sizeof(float));
    cudaMalloc(&norm2_var, B * N * sizeof(float));

    float* norm1_output;
    cudaMalloc(&norm1_output, B * N * C * sizeof(float));

    float* norm2_output;
    cudaMalloc(&norm2_output, B * N * C * sizeof(float));

    float* msa_output;
    cudaMalloc(&msa_output, B * N * C * sizeof(float));

    float* mlp_output;
    cudaMalloc(&mlp_output, B * N * C * sizeof(float));

    // Compute mean and variance for layer normalization 1
    // (This part is complex and may require custom reduction kernels)

    // Apply layer normalization 1
    layer_norm_kernel<<<gridDim, blockDim>>>(x, norm1_mean, norm1_var, gamma1, beta1, B * N * C);

    // Apply self-attention
    self_attention_forward(norm1_output, msa_weight, msa_bias, B, N, C, C / 12, 1.0f / sqrt(C));

    // Add residual connection
    for (int i = 0; i < B * N * C; i++) {
        x[i] += msa_output[i];
    }

    // Compute mean and variance for layer normalization 2
    // (This part is complex and may require custom reduction kernels)

    // Apply layer normalization 2
    layer_norm_kernel<<<gridDim, blockDim>>>(x, norm2_mean, norm2_var, gamma2, beta2, B * N * C);

    // Apply MLP
    mlp_forward(norm2_output, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias, dropout_prob, B, N, C);

    // Add residual connection
    for (int i = 0; i < B * N * C; i++) {
        x[i] += mlp_output[i];
    }

    // Free allocated memory
    cudaFree(norm1_mean);
    cudaFree(norm1_var);
    cudaFree(norm2_mean);
    cudaFree(norm2_var);
    cudaFree(norm1_output);
    cudaFree(norm2_output);
    cudaFree(msa_output);
    cudaFree(mlp_output);
}