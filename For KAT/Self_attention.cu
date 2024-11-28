#include <cuda_runtime.h>
#include <math.h>
// QKV Projection Kernel
__global__ void qkv_projection_kernel(float* x, float* qkv_weight, float* qkv_output, int B, int N, int C, int proj_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * proj_size) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += x[idx * C + c] * qkv_weight[c * proj_size + idx % proj_size];
        }
        qkv_output[idx] = sum;
    }
}


// Reshape and Transpose Kernel
__global__ void reshape_transpose_kernel(float* qkv_output, float* q, float* k, float* v, int B, int N, int C, int H, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * N * T) {
        int batch = idx / (H * N * T);
        int head = (idx % (H * N * T)) / (N * T);
        int seq = (idx % (N * T)) / T;
        int dim = idx % T;
        q[idx] = qkv_output[batch * N * 3 * H * T + seq * 3 * H * T + head * 3 * T + dim];
        k[idx] = qkv_output[batch * N * 3 * H * T + seq * 3 * H * T + head * 3 * T + T + dim];
        v[idx] = qkv_output[batch * N * 3 * H * T + seq * 3 * H * T + head * 3 * T + 2 * T + dim];
    }
}
// Attention Score Kernel
__global__ void attention_score_kernel(float* q, float* k, float* attn_scores, int B, int H, int N, int T, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * N * N) {
        int batch = idx / (H * N * N);
        int head = (idx % (H * N * N)) / (N * N);
        int row = (idx % (N * N)) / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            sum += q[batch * H * N * T + head * N * T + row * T + t] * k[batch * H * N * T + head * N * T + col * T + t];
        }
        attn_scores[idx] = sum * scale;
    }
}

// Softmax Kernel
__global__ void softmax_kernel(float* attn_scores, int B, int H, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * N) {
        float max_val = -FLT_MAX;
        for (int n = 0; n < N; n++) {
            if (attn_scores[idx * N + n] > max_val)
                max_val = attn_scores[idx * N + n];
        }
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            attn_scores[idx * N + n] = exp(attn_scores[idx * N + n] - max_val);
            sum += attn_scores[idx * N + n];
        }
        for (int n = 0; n < N; n++) {
            attn_scores[idx * N + n] /= sum;
        }
    }
}


// Output Projection Kernel
__global__ void output_projection_kernel(float* attn_probs, float* v, float* proj_weight, float* output, int B, int N, int C, int H, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * C) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int n = 0; n < N; n++) {
                for (int t = 0; t < T; t++) {
                    sum += attn_probs[idx * H * N + h * N + n] * v[h * N * T + n * T + t] * proj_weight[h * T * C + t * C + idx % C];
                }
            }
        }
        output[idx] = sum;
    }
}

// Self-Attention Forward Function
void self_attention_forward(float* x, float* qkv_weight, float* proj_weight, float* output, int B, int N, int C, int H, float scale) {
    int T = C / H;
    int proj_size = 3 * C;

    float* d_qkv_output;
    cudaMalloc(&d_qkv_output, B * N * proj_size * sizeof(float));
    qkv_projection_kernel<<<(B * N * proj_size + 255) / 256, 256>>>(x, qkv_weight, d_qkv_output, B, N, C, proj_size);
    cudaDeviceSynchronize();

    float* d_q;
    cudaMalloc(&d_q, B * H * N * T * sizeof(float));
    float* d_k;
    cudaMalloc(&d_k, B * H * N * T * sizeof(float));
    float* d_v;
    cudaMalloc(&d_v, B * H * N * T * sizeof(float));
    reshape_transpose_kernel<<<(B * H * N * T + 255) / 256, 256>>>(d_qkv_output, d_q, d_k, d_v, B, N, C, H, T);
    cudaDeviceSynchronize();

    float* d_attn_scores;
    cudaMalloc(&d_attn_scores, B * H * N * N * sizeof(float));
    attention_score_kernel<<<(B * H * N * N + 255) / 256, 256>>>(d_q, d_k, d_attn_scores, B, H, N, T, scale);
    cudaDeviceSynchronize();

    softmax_kernel<<<(B * H * N + 255) / 256, 256>>>(d_attn_scores, B, H, N);
    cudaDeviceSynchronize();

    output_projection_kernel<<<(B * N * C + 255) / 256, 256>>>(d_attn_scores, d_v, proj_weight, output, B, N, C, H, T);
    cudaDeviceSynchronize();

    cudaFree(d_qkv_output);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_attn_scores);
}