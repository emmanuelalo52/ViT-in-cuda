#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to broadcast cls_token
__global__ void broadcast_cls_token_kernel(float* cls_token, float* cls_token_expanded, int B, int n_emb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * n_emb) {
        int batch = idx / n_emb;
        int emb = idx % n_emb;
        cls_token_expanded[batch * n_emb + emb] = cls_token[emb];
    }
}

// CUDA kernel to concatenate cls_token_expanded and x
__global__ void concatenate_kernel(float* cls_token_expanded, float* x, float* concat, int B, int num_patches, int n_emb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * (num_patches + 1) * n_emb) {
        int b = idx / ((num_patches + 1) * n_emb);
        int pos = (idx % ((num_patches + 1) * n_emb)) / n_emb;
        int e = idx % n_emb;
        if (pos == 0) {
            concat[idx] = cls_token_expanded[b * n_emb + e];
        } else {
            concat[idx] = x[b * num_patches * n_emb + (pos - 1) * n_emb + e];
        }
    }
}

// CUDA kernel to add positional embeddings
__global__ void add_pos_embed_kernel(float* concat, float* pos_embed, float* output, int B, int seq_len, int n_emb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * seq_len * n_emb) {
        int b = idx / (seq_len * n_emb);
        int pos = (idx % (seq_len * n_emb)) / n_emb;
        int e = idx % n_emb;
        output[idx] = concat[idx] + pos_embed[pos * n_emb + e];
    }
}

// Positional2DEmbedding forward function
void positional2d_embedding_forward(float* x_device, float* cls_token_device, float* pos_embed_device, float* output_device, int B, int num_patches, int n_emb) {
    int seq_len = num_patches + 1;
    int concat_size = B * seq_len * n_emb;

    // Allocate memory for expanded cls_token
    float* cls_token_expanded_device;
    cudaMalloc(&cls_token_expanded_device, B * n_emb * sizeof(float));

    // Broadcast cls_token to all batches
    broadcast_cls_token_kernel<<<(B * n_emb + 255)/256, 256>>>(cls_token_device, cls_token_expanded_device, B, n_emb);

    // Allocate memory for concatenated tensor
    float* concat_device;
    cudaMalloc(&concat_device, concat_size * sizeof(float));

    // Concatenate cls_token_expanded and x
    concatenate_kernel<<<(concat_size + 255)/256, 256>>>(cls_token_expanded_device, x_device, concat_device, B, num_patches, n_emb);

    // Add positional embeddings
    add_pos_embed_kernel<<<(B * seq_len * n_emb + 255)/256, 256>>>(concat_device, pos_embed_device, output_device, B, seq_len, n_emb);

    // Free temporary memory
    cudaFree(cls_token_expanded_device);
    cudaFree(concat_device);
}