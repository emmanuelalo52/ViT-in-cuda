#include <cuda_runtime.h>
#include <math.h>

// Define the CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

// Define the CUDA kernel for GELU activation
__global__ void gelu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = 0.5 * x[idx] * (1.0 + tanh(sqrt(2.0 / M_PI) * (x[idx] + 0.044715 * pow(x[idx], 3))));
    }
}

// Define the CUDA kernel for dropout
__global__ void dropout_kernel(float* x, float* mask, float p, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (mask[idx] < p) {
            x[idx] = 0.0f;
        } else {
            x[idx] /= (1.0f - p);
        }
    }
}

// MLP forward pass using CUDA
void mlp_forward(float* x, float* fc1_weight, float* fc1_bias, float* fc2_weight, float* fc2_bias, float dropout_prob, int B, int N, int C) {
    int hidden_dim = 4 * C;

    // Allocate memory on the GPU
    float* fc1_output;
    cudaMalloc(&fc1_output, B * N * hidden_dim * sizeof(float));

    float* gelu_output;
    cudaMalloc(&gelu_output, B * N * hidden_dim * sizeof(float));

    float* dropout_mask;
    cudaMalloc(&dropout_mask, B * N * hidden_dim * sizeof(float));

    float* fc2_output;
    cudaMalloc(&fc2_output, B * N * C * sizeof(float));

    // Perform first linear transformation
    dim3 blockDim(32, 32);
    dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x, (B * N + blockDim.y - 1) / blockDim.y);
    matmul_kernel<<<gridDim, blockDim>>>(x, fc1_weight, fc1_output, B * N, C, hidden_dim);

    // Add bias
    for (int i = 0; i < B * N; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            fc1_output[i * hidden_dim + j] += fc1_bias[j];
        }
    }

    // Apply GELU activation
    dim3 gelu_grid((B * N * hidden_dim + blockDim.x - 1) / blockDim.x, 1, 1);
    gelu_kernel<<<gelu_grid, blockDim>>>(fc1_output, B * N * hidden_dim);

    // Apply dropout
    // Generate dropout mask
    curandGenerateUniform(curandGenerator, dropout_mask, B * N * hidden_dim);
    dropout_kernel<<<gelu_grid, blockDim>>>(fc1_output, dropout_mask, dropout_prob, B * N * hidden_dim);

    // Perform second linear transformation
    matmul_kernel<<<gridDim, blockDim>>>(fc1_output, fc2_weight, fc2_output, B * N, hidden_dim, C);

    // Add bias
    for (int i = 0; i < B * N; i++) {
        for (int j = 0; j < C; j++) {
            fc2_output[i * C + j] += fc2_bias[j];
        }
    }

    // Apply dropout
    // Generate dropout mask
    curandGenerateUniform(curandGenerator, dropout_mask, B * N * C);
    dropout_kernel<<<gelu_grid, blockDim>>>(fc2_output, dropout_mask, dropout_prob, B * N * C);

    // Copy the result back to the input tensor
    cudaMemcpy(x, fc2_output, B * N * C * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free allocated memory
    cudaFree(fc1_output);
    cudaFree(gelu_output);
    cudaFree(dropout_mask);
    cudaFree(fc2_output);
}