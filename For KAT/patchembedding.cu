#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <math.h>

// Layer normalization kernel from previous code
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

// Kernel to transpose last two dimensions
__global__ void transpose_last_two_dims_kernel(float* input, float* output, int B, int num_patches, int n_emb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * num_patches * n_emb) {
        int b = idx / (num_patches * n_emb);
        int patch = (idx % (num_patches * n_emb)) / n_emb;
        int emb = idx % n_emb;
        output[idx] = input[b * num_patches * n_emb + patch * n_emb + emb];
    }
}

// Patch embedding forward function
void patch_embedding_forward(float* input_images, float* conv_weights, float* conv_bias, float* ln_gamma, float* ln_beta, float* output_embeddings, int B, int C, int H, int W, int n_emb, int patch_size, int num_patches) {
    // Allocate device memory for intermediate tensors
    float* d_input;
    float* d_output_conv;
    float* d_output_flatten;
    float* d_output_transpose;
    float* d_output_ln;

    cudaMalloc(&d_input, B * C * H * W * sizeof(float));
    cudaMalloc(&d_output_conv, B * n_emb * (H / patch_size) * (W / patch_size) * sizeof(float));
    cudaMalloc(&d_output_flatten, B * n_emb * num_patches * sizeof(float));
    cudaMalloc(&d_output_transpose, B * num_patches * n_emb * sizeof(float));
    cudaMalloc(&d_output_ln, B * num_patches * n_emb * sizeof(float));

    cudaMemcpy(d_input, input_images, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // cuDNN setup
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    cudnnTensorDescriptor_t input_desc, output_desc, filter_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n_emb, C, patch_size, patch_size);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, patch_size, patch_size, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    int H_out = H / patch_size;
    int W_out = W / patch_size;
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, n_emb, H_out, W_out);
    cudnnSetTensor1dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_emb);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, d_input, filter_desc, conv_weights, conv_desc, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT, d_output_conv, &beta, output_desc, d_output_conv);
    cudnnAddTensor(cudnn_handle, &alpha, bias_desc, conv_bias, &alpha, output_desc, d_output_conv);

    cudaMemcpy(d_output_flatten, d_output_conv, B * n_emb * num_patches * sizeof(float), cudaMemcpyDeviceToDevice);

    int grid_size = (B * num_patches * n_emb + 255) / 256;
    transpose_last_two_dims_kernel<<<grid_size, 256>>>(d_output_flatten, d_output_transpose, B, num_patches, n_emb);

    layer_norm_kernel<<<(B * num_patches + 255) / 256, 256>>>(d_output_transpose, ln_gamma, ln_beta, d_output_ln, B, num_patches, n_emb);

    cudaMemcpy(output_embeddings, d_output_ln, B * num_patches * n_emb * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output_conv);
    cudaFree(d_output_flatten);
    cudaFree(d_output_transpose);
    cudaFree(d_output_ln);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroy(cudnn_handle);
}