#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "Self_attention.cu"
#include "GRKAN.cu"
// Layer Norm Kernel
__global__ void layer_norm_kernel(float* x, float* gamma, float* beta, float* normalized_x, int B, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        float mean = 0.0f;
        for (int c = 0; c < C; c++) {
            mean += x[idx * C + c];
        }
        mean /= C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            var += (x[idx * C + c] - mean) * (x[idx * C + c] - mean);
        }
        var /= C;
        float inv_std = 1.0f / sqrt(var + 1e-5f);
        for (int c = 0; c < C; c++) {
            normalized_x[idx * C + c] = gamma[c] * (x[idx * C + c] - mean) * inv_std + beta[c];
        }
    }
}

// Block Forward Function
void block_forward(float* x, float* norm1_gamma, float* norm1_beta, float* msa_qkv_weight, float* msa_proj_weight, float* norm2_gamma, float* norm2_beta, float** grkan_rational_a, float** grkan_rational_b, float* grkan_proj_weight, float* grkan_proj_bias, float* dropout_mask, int B, int N, int C, int H, int grkan_groups, int group_dim, int numerator_order, int denominator_order, float dropout_prob, float scale) {
    // Allocate memory for intermediate tensors
    float* norm1_output;
    cudaMalloc(&norm1_output, B * N * C * sizeof(float));
    float* msa_output;
    cudaMalloc(&msa_output, B * N * C * sizeof(float));
    float* norm2_output;
    cudaMalloc(&norm2_output, B * N * C * sizeof(float));
    float* grkan_output;
    cudaMalloc(&grkan_output, B * N * C * sizeof(float));

    // Step 1: x + msa(norm1(x))
    // Apply layer norm 1
    layer_norm_kernel<<<(B * N + 255) / 256, 256>>>(x, norm1_gamma, norm1_beta, norm1_output, B, N, C);
    cudaDeviceSynchronize();

    // Apply self-attention
    self_attention_forward(norm1_output, msa_qkv_weight, msa_proj_weight, msa_output, B, N, C, H, scale);
    cudaDeviceSynchronize();

    // Add residual
    for (int i = 0; i < B * N * C; i++) {
        x[i] += msa_output[i];
    }

    // Step 2: x + grkan(norm2(x))
    // Apply layer norm 2
    layer_norm_kernel<<<(B * N + 255) / 256, 256>>>(x, norm2_gamma, norm2_beta, norm2_output, B, N, C);
    cudaDeviceSynchronize();

    // Apply GRKAN
    grkan_forward(norm2_output, norm2_gamma, norm2_beta, grkan_rational_a, grkan_rational_b, grkan_proj_weight, grkan_proj_bias, dropout_mask, grkan_output, B, N, C, grkan_groups, group_dim, numerator_order, denominator_order, dropout_prob);
    cudaDeviceSynchronize();

    // Add residual
    for (int i = 0; i < B * N * C; i++) {
        x[i] += grkan_output[i];
    }

    // Free allocated memory
    cudaFree(norm1_output);
    cudaFree(msa_output);
    cudaFree(norm2_output);
    cudaFree(grkan_output);
}