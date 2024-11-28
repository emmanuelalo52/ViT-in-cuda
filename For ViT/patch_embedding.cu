#include <cuda_runtime.h>
#include <math.h>

// Define the CUDA kernel for 2D convolution
__global__ void conv2d_kernel(float* input, float* output, float* kernel, int B, int C, int H, int W, int K, int stride) {
    int out_h = (H - K) / stride + 1;
    int out_w = (W - K) / stride + 1;
    int out_c = C;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < out_h && col < out_w && ch < out_c) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                sum += input[(row * stride + i) * W * C + (col * stride + j) * C + ch] * kernel[i * K + j];
            }
        }
        output[row * out_w * out_c + col * out_c + ch] = sum;
    }
}

// Define the CUDA kernel for flattening
__global__ void flatten_kernel(float* input, float* output, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H * W;
    if (idx < total_elements) {
        int b = idx / (C * H * W);
        int c = (idx % (C * H * W)) / (H * W);
        int h = (idx % (H * W)) / W;
        int w = idx % W;
        output[b * (C * H * W) + h * (C * W) + w * C + c] = input[idx];
    }
}

// PatchEmbedding forward pass using CUDA
void patch_embedding_forward(float* x, float* conv_weight, int B, int C, int H, int W, int patch_size) {
    int out_h = (H - patch_size) / patch_size + 1;
    int out_w = (W - patch_size) / patch_size + 1;
    int out_c = C;

    // Allocate memory on the GPU
    float* conv_output;
    cudaMalloc(&conv_output, B * out_c * out_h * out_w * sizeof(float));

    float* flatten_output;
    cudaMalloc(&flatten_output, B * out_c * out_h * out_w * sizeof(float));

    // Perform 2D convolution
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((out_w + blockDim.x - 1) / blockDim.x, (out_h + blockDim.y - 1) / blockDim.y, (out_c + blockDim.z - 1) / blockDim.z);
    conv2d_kernel<<<gridDim, blockDim>>>(x, conv_output, conv_weight, B, C, H, W, patch_size, patch_size);

    // Flatten the output
    dim3 flatten_grid((B * out_c * out_h * out_w + blockDim.x - 1) / blockDim.x, 1, 1);
    flatten_kernel<<<flatten_grid, blockDim>>>(conv_output, flatten_output, B, out_c, out_h, out_w);

    // Transpose the output
    // (This part is complex and may require custom indexing)

    // Copy the result back to the input tensor
    cudaMemcpy(x, flatten_output, B * out_c * out_h * out_w * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free allocated memory
    cudaFree(conv_output);
    cudaFree(flatten_output);
}