// rational_function.cu

#include <cuda_runtime.h>
#include <stdio.h>

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

void rational_function_forward(float* x, float* a, float* b, float* y, int B, int N, int D, int numerator_order, int denominator_order) {
    int total_elements = B * N * D;

    // Allocate device memory
    float* d_x, *d_a, *d_b, *d_y;
    cudaError_t err;
    
    err = cudaMalloc(&d_x, total_elements * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMalloc(&d_a, (numerator_order + 1) * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMalloc(&d_b, (denominator_order + 1) * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMalloc(&d_y, total_elements * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Copy inputs to device
    err = cudaMemcpy(d_x, x, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMemcpy(d_a, a, (numerator_order + 1) * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMemcpy(d_b, b, (denominator_order + 1) * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    rational_function_kernel<<<gridSize, blockSize>>>(d_x, d_a, d_b, d_y, B, N, D, numerator_order, denominator_order);
    
    // Check for kernel errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Copy output back to host
    err = cudaMemcpy(y, d_y, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
}