#include "rational_function.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Rational Function Kernel
__global__ void rational_function_kernel(float* group_input, float* a, float* b, float* group_output, int B, int N, int group_dim, int numerator_order, int denominator_order) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * group_dim) {
        float x = group_input[idx];
        // Evaluate numerator using Horner's method
        float numerator = a[numerator_order];
        for (int i = numerator_order - 1; i >= 0; i--) {
            numerator = numerator * x + a[i];
        }
        // Evaluate denominator using Horner's method
        float denominator = b[denominator_order];
        for (int i = denominator_order - 1; i >= 0; i--) {
            denominator = denominator * x + b[i];
        }
        // Avoid division by zero
        const float epsilon = 1e-8f;
        if (denominator == 0.0f)
            denominator = epsilon;
        group_output[idx] = numerator / denominator;
    }
}

// Layer Normalization Kernel
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

// Concatenation Kernel
__global__ void concatenation_kernel(float* group_outputs, float* concat_output, int B, int N, int C, int groups, int group_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * C) {
        int group = idx / group_dim;
        int dim = idx % group_dim;
        concat_output[idx] = group_outputs[group * B * N * group_dim + dim];
    }
}

// Projection and Dropout Kernel
__global__ void projection_dropout_kernel(float* concat_output, float* proj_weight, float* proj_bias, float* dropout_mask, float* output, int B, int N, int C, float dropout_prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * C) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += concat_output[idx * C + c] * proj_weight[c];
        }
        sum += proj_bias[idx % C];
        if (dropout_mask[idx] < dropout_prob)
            sum = 0.0f;
        else
            sum /= (1.0f - dropout_prob);
        output[idx] = sum;
    }
}

// Host Function
void grkan_forward(float* x, float* gamma, float* beta, float** rational_a, float** rational_b, float* proj_weight, float* proj_bias, float* dropout_mask, float* output, int B, int N, int C, int groups, int group_dim, int numerator_order, int denominator_order, float dropout_prob) {
    // Allocate memory for normalized_x
    float* d_normalized_x;
    cudaMalloc(&d_normalized_x, B * N * C * sizeof(float));
    // Launch layer_norm_kernel
    int block_size = 256;
    int grid_size = (B * N + block_size - 1) / block_size;
    layer_norm_kernel<<<grid_size, block_size>>>(x, gamma, beta, d_normalized_x, B, N, C);
    cudaDeviceSynchronize();
    // Allocate memory for group_outputs
    float* d_group_outputs[groups];
    for (int i = 0; i < groups; i++) {
        cudaMalloc(&d_group_outputs[i], B * N * group_dim * sizeof(float));
        // Launch rational_function_kernel for each group
        rational_function_kernel<<<grid_size, block_size>>>(d_normalized_x + i * group_dim, rational_a[i], rational_b[i], d_group_outputs[i], B, N, group_dim, numerator_order, denominator_order);
        cudaDeviceSynchronize();
    }
    // Allocate memory for concat_output
    float* d_concat_output;
    cudaMalloc(&d_concat_output, B * N * C * sizeof(float));
    // Launch concatenation_kernel
    concatenation_kernel<<<grid_size, block_size>>>(group_outputs, d_concat_output, B, N, C, groups, group_dim);
    cudaDeviceSynchronize();
    // Launch projection_dropout_kernel
    projection_dropout_kernel<<<grid_size, block_size>>>(d_concat_output, proj_weight, proj_bias, dropout_mask, output, B, N, C, dropout_prob);
    cudaDeviceSynchronize();
    // Free allocated memory
    for (int i = 0; i < groups; i++)
        cudaFree(d_group_outputs[i]);
    cudaFree(d_normalized_x);
    cudaFree(d_concat_output);
}