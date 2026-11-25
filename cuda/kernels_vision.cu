// Native Inkling-2-Mini Vision Tower (model.visual.*): 23-layer BF16 ViT + 2x2 Merger.
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0) to squid5090's
// TigerStyle single-arena weight layout and cuBLAS BF16 Tensor Core GEMMs.

#include "kernels.hpp"
#include "kernels_common.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace squid {

namespace {

constexpr int V_PATCH = 20;
constexpr int V_TILE = 40;
constexpr int V_PATCH_DIM = V_PATCH * V_PATCH * 3; // 1200
constexpr int V_HIDDEN = 1024;
constexpr int V_HEADS = 16;
constexpr int V_HEAD_DIM = 64;
constexpr int V_MLP_DIM = 4096;
constexpr int V_MERGED = 4 * V_HIDDEN; // 4096
constexpr int V_POS_GRID = 32;
constexpr int V_ROPE_FREQS = 16;
constexpr float V_LN_EPS = 1e-5f;
constexpr int V_ROW_T = 256;
constexpr int V_ELEM_T = 256;
constexpr size_t V_SCORE_BYTES = 128ull * 1024 * 1024; // 128 MiB attention score slab

__device__ __forceinline__ uint16_t v_bf16_bits(float v) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float v_block_sum(float v, float* smem) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int w = 0; w < V_ROW_T / 32; ++w) tot += smem[w];
    __syncthreads();
    return tot;
}

__device__ __forceinline__ void v_patch_position(int p, int cols, int& row, int& col) {
    const int tile = p >> 2;
    const int in = p & 3;
    row = (tile / cols) * 2 + (in >> 1);
    col = (tile % cols) * 2 + (in & 1);
}

__global__ void v_patchify_kernel(
    const uint8_t* __restrict__ rgb, int height, int width, int cols, int patches,
    uint16_t* __restrict__ out, int64_t total) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int p = int(i / V_PATCH_DIM);
    const int e = int(i % V_PATCH_DIM);
    if (p >= patches) {
        out[i] = 0;
        return;
    }
    int row, col;
    v_patch_position(p, cols, row, col);
    const int y = row * V_PATCH + e / (V_PATCH * 3);
    const int x = col * V_PATCH + (e / 3) % V_PATCH;
    const int c = e % 3;
    const float v = (y < height && x < width) ? float(rgb[(size_t(y) * width + x) * 3 + c]) : -1.0f;
    out[i] = v_bf16_bits(__fdiv_rn(__fsub_rn(v, 127.5f), 127.5f));
}

__device__ __forceinline__ void v_axis_taps(int index, int size, int& t0, int& t1, float& w1) {
    const float src = __fsub_rn(__fdiv_rn(__fmul_rn(__fadd_rn(float(index), 0.5f), float(V_POS_GRID)), float(size)), 0.5f);
    const float low = floorf(src);
    w1 = __fsub_rn(src, low);
    const int l = int(low);
    t0 = min(max(l, 0), V_POS_GRID - 1);
    t1 = min(max(l + 1, 0), V_POS_GRID - 1);
}

__global__ void __launch_bounds__(V_ROW_T)
v_embed_assemble_kernel(
    const float* __restrict__ patches,
    const float* __restrict__ pos_table,
    const float* __restrict__ cls,
    int rows, int cols,
    float* __restrict__ h) {
    const int token = blockIdx.x;
    const int c = threadIdx.x * 4;
    const int S = 1 + 4 * rows * cols;
    float4* out = reinterpret_cast<float4*>(h + size_t(token) * V_HIDDEN + c);
    if (token >= S) {
        *out = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    if (token == 0) {
        *out = *reinterpret_cast<const float4*>(cls + c);
        return;
    }
    const int p = token - 1;
    int row, col;
    v_patch_position(p, cols, row, col);
    int r0, r1, c0, c1;
    float wr, wc;
    v_axis_taps(row, rows * 2, r0, r1, wr);
    v_axis_taps(col, cols * 2, c0, c1, wc);
    const float w00 = __fmul_rn(1.f - wr, 1.f - wc), w01 = __fmul_rn(1.f - wr, wc);
    const float w10 = __fmul_rn(wr, 1.f - wc), w11 = __fmul_rn(wr, wc);
    auto entry = [&](int r, int cc) {
        return *reinterpret_cast<const float4*>(pos_table + size_t(r * V_POS_GRID + cc) * V_HIDDEN + c);
    };
    const float4 a = entry(r0, c0), b = entry(r0, c1), d = entry(r1, c0), e = entry(r1, c1);
    const float4 x = *reinterpret_cast<const float4*>(patches + size_t(p) * V_HIDDEN + c);
    auto blend = [&](float xa, float pa, float pb, float pd, float pe) {
        return xa + (((pa * w00 + pb * w01) + pd * w10) + pe * w11);
    };
    *out = make_float4(
        blend(x.x, a.x, b.x, d.x, e.x),
        blend(x.y, a.y, b.y, d.y, e.y),
        blend(x.z, a.z, b.z, d.z, e.z),
        blend(x.w, a.w, b.w, d.w, e.w));
}

__global__ void __launch_bounds__(V_ROW_T)
v_layernorm_kernel(
    const float* x, const float* __restrict__ w, const float* __restrict__ b, void* out, bool out_fp32) {
    __shared__ float smem[V_ROW_T / 32];
    const int c = threadIdx.x * 4;
    const size_t off = size_t(blockIdx.x) * V_HIDDEN + c;
    const float4 v = *reinterpret_cast<const float4*>(x + off);
    const float mean = v_block_sum((v.x + v.y) + (v.z + v.w), smem) * (1.f / float(V_HIDDEN));
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
    const float var = v_block_sum((d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3), smem) * (1.f / float(V_HIDDEN));
    const float r = 1.f / sqrtf(var + V_LN_EPS);
    const float4 w4 = *reinterpret_cast<const float4*>(w + c);
    const float4 b4 = *reinterpret_cast<const float4*>(b + c);
    const float y0 = d0 * r * w4.x + b4.x, y1 = d1 * r * w4.y + b4.y;
    const float y2 = d2 * r * w4.z + b4.z, y3 = d3 * r * w4.w + b4.w;
    if (out_fp32) {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + off) = make_float4(y0, y1, y2, y3);
    } else {
        uint16_t* o = static_cast<uint16_t*>(out) + off;
        o[0] = v_bf16_bits(y0); o[1] = v_bf16_bits(y1);
        o[2] = v_bf16_bits(y2); o[3] = v_bf16_bits(y3);
    }
}

__global__ void __launch_bounds__(V_ROW_T)
v_qkv_post_kernel(
    const float* __restrict__ qkv, const float* __restrict__ bias,
    const float* __restrict__ inv_freq, int cols, int S, int Sp,
    uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ vt) {
    const int token = blockIdx.x;
    const bool valid = (token < S);
    float pos_row = 0.f, pos_col = 0.f;
    if (valid && token > 0) {
        int row, col;
        v_patch_position(token - 1, cols, row, col);
        pos_row = float(row + 1);
        pos_col = float(col + 1);
    }
    const float* x = qkv + size_t(token) * (3 * V_HIDDEN);
    for (int pair = threadIdx.x; pair < V_HIDDEN / 2; pair += V_ROW_T) {
        const int head = pair / (V_HEAD_DIM / 2);
        const int j = pair % (V_HEAD_DIM / 2);
        const int c = head * V_HEAD_DIM + 2 * j;
        const size_t qk = (size_t(head) * Sp + token) * V_HEAD_DIM + 2 * j;
        const size_t v0 = (size_t(head) * V_HEAD_DIM + 2 * j) * Sp + token;
        if (!valid) {
            q[qk] = q[qk + 1] = k[qk] = k[qk + 1] = 0;
            vt[v0] = vt[v0 + Sp] = 0;
            continue;
        }
        const float angle = __fmul_rn((j < V_ROPE_FREQS) ? pos_col : pos_row, inv_freq[j % V_ROPE_FREQS]);
        float sn, cs;
        sincosf(angle, &sn, &cs);
        const float qa = x[c] + bias[c], qb = x[c + 1] + bias[c + 1];
        const float ka = x[V_HIDDEN + c] + bias[V_HIDDEN + c], kb = x[V_HIDDEN + c + 1] + bias[V_HIDDEN + c + 1];
        q[qk]     = v_bf16_bits(__fadd_rn(__fmul_rn(qa, cs), __fmul_rn(-qb, sn)));
        q[qk + 1] = v_bf16_bits(__fadd_rn(__fmul_rn(qb, cs), __fmul_rn(qa, sn)));
        k[qk]     = v_bf16_bits(__fadd_rn(__fmul_rn(ka, cs), __fmul_rn(-kb, sn)));
        k[qk + 1] = v_bf16_bits(__fadd_rn(__fmul_rn(kb, cs), __fmul_rn(ka, sn)));
        vt[v0]      = v_bf16_bits(x[2 * V_HIDDEN + c] + bias[2 * V_HIDDEN + c]);
        vt[v0 + Sp] = v_bf16_bits(x[2 * V_HIDDEN + c + 1] + bias[2 * V_HIDDEN + c + 1]);
    }
}

__global__ void __launch_bounds__(V_ROW_T)
v_softmax_kernel(const float* __restrict__ scores, int ld, int S, float scale, uint16_t* __restrict__ p) {
    __shared__ float smem[V_ROW_T / 32];
    __shared__ float s_max[V_ROW_T / 32];
    const float* row = scores + size_t(blockIdx.x) * ld;
    uint16_t* out = p + size_t(blockIdx.x) * ld;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < S; c += V_ROW_T) m = fmaxf(m, row[c] * scale);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = m;
    __syncthreads();
    m = s_max[0];
    #pragma unroll
    for (int w = 1; w < V_ROW_T / 32; ++w) m = fmaxf(m, s_max[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < S; c += V_ROW_T) sum += expf(row[c] * scale - m);
    const float inv = 1.0f / v_block_sum(sum, smem);
    for (int c = threadIdx.x; c < ld; c += V_ROW_T) out[c] = (c < S) ? v_bf16_bits(expf(row[c] * scale - m) * inv) : 0;
}

__global__ void v_bias_gelu_kernel(
    const float* __restrict__ x, const float* __restrict__ bias, int64_t total, int N, uint16_t* __restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i];
    if (bias) v += bias[i % N];
    v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[i] = v_bf16_bits(v);
}

__global__ void __launch_bounds__(V_ROW_T)
v_residual_kernel(
    const float* __restrict__ y, const float* __restrict__ bias, const float* __restrict__ gamma, float* __restrict__ h) {
    const int c = threadIdx.x * 4;
    const size_t off = size_t(blockIdx.x) * V_HIDDEN + c;
    const float4 v = *reinterpret_cast<const float4*>(y + off);
    const float4 b = *reinterpret_cast<const float4*>(bias + c);
    const float4 g = *reinterpret_cast<const float4*>(gamma + c);
    float4* out = reinterpret_cast<float4*>(h + off);
    const float4 r = *out;
    *out = make_float4(
        r.x + g.x * (v.x + b.x),
        r.y + g.y * (v.y + b.y),
        r.z + g.z * (v.z + b.z),
        r.w + g.w * (v.w + b.w));
}

__global__ void v_merge_rows_kernel(const float* __restrict__ h, int64_t valid, int64_t total, uint16_t* __restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = (i < valid) ? v_bf16_bits(h[V_HIDDEN + i]) : 0;
}

// Scatter BF16 image embeddings [count, HIDDEN] directly into FP32 hidden[dst_offset..dst_offset+count-1]
// (overwriting embed_norm output; image embeddings intentionally skip embed_norm!).
__global__ void v_scatter_embeds_kernel(
    const __nv_bfloat16* __restrict__ src_bf16,
    float* __restrict__ dst_hidden,
    int dst_offset,
    int count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = int64_t(count) * HIDDEN;
    if (i >= total) return;
    dst_hidden[int64_t(dst_offset) * HIDDEN + i] = __bfloat162float(src_bf16[i]);
}

static cublasHandle_t get_vision_cublas() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        cublasStatus_t st = cublasCreate(&h);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCreate (vision) failed: %d\n", int(st));
            std::abort();
        }
    }
    return h;
}

// Row-major C[M, N] = A[M, K] @ B[N, K]^T (all inputs BF16; C is FP32 when out_f32 else BF16).
static void v_gemm(
    cublasHandle_t h, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    void* C, int ldc, bool out_f32, int M, int N, int K, cudaStream_t stream) {
    cublasSetStream(h, stream);
    const float alpha = 1.0f, beta = 0.0f;
    const cudaDataType c_type = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, c_type, ldc,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasGemmEx (vision) failed: %d (M=%d N=%d K=%d)\n", int(st), M, N, K);
        std::abort();
    }
}

static inline int v_round_up(int v, int to) { return ((v + to - 1) / to) * to; }
static inline unsigned v_blocks(int64_t n) { return static_cast<unsigned>((n + V_ELEM_T - 1) / V_ELEM_T); }

} // namespace

void launch_vision_encode(
    const VisionWeights& w,
    const uint8_t* rgb_host,
    int height,
    int width,
    __nv_bfloat16* out_bf16_dev,
    cudaStream_t stream) {
    const int rows = (height + V_TILE - 1) / V_TILE;
    const int cols = (width + V_TILE - 1) / V_TILE;
    const int N = rows * cols;
    const int S = 4 * N + 1;
    const int Sp = v_round_up(S, 64);
    const int Np = v_round_up(N, 64);
    const int Qc = int(std::clamp<size_t>((V_SCORE_BYTES / (size_t(Sp) * 6)) / 64 * 64, 64, size_t(Sp)));

    size_t at = 0;
    auto take = [&](size_t bytes) -> size_t {
        size_t off = at;
        at += ((bytes + 255) / 256) * 256;
        return off;
    };
    const size_t off_pix    = take(size_t(height) * width * 3);
    const size_t off_h      = take(size_t(Sp) * V_HIDDEN * 4);
    const size_t off_xb     = take(size_t(Sp) * V_HIDDEN * 2);
    const size_t off_big32  = take(size_t(Sp) * V_MLP_DIM * 4);
    const size_t off_big16  = take(size_t(Sp) * V_MLP_DIM * 2);
    const size_t off_attn   = take(size_t(Sp) * V_HIDDEN * 2);
    const size_t off_y      = take(size_t(Sp) * V_HIDDEN * 4);
    const size_t off_scores = take(size_t(Qc) * Sp * 4);
    const size_t off_probs  = take(size_t(Qc) * Sp * 2);

    uint8_t* work = nullptr;
    CUDA_CHECK(cudaMalloc(&work, at));

    uint8_t* pixels       = work + off_pix;
    float* h              = reinterpret_cast<float*>(work + off_h);
    __nv_bfloat16* xb     = reinterpret_cast<__nv_bfloat16*>(work + off_xb);
    float* big32          = reinterpret_cast<float*>(work + off_big32);
    __nv_bfloat16* big16  = reinterpret_cast<__nv_bfloat16*>(work + off_big16);
    __nv_bfloat16* attn   = reinterpret_cast<__nv_bfloat16*>(work + off_attn);
    float* y              = reinterpret_cast<float*>(work + off_y);
    float* scores         = reinterpret_cast<float*>(work + off_scores);
    __nv_bfloat16* probs  = reinterpret_cast<__nv_bfloat16*>(work + off_probs);

    cublasHandle_t ch = get_vision_cublas();

    CUDA_CHECK(cudaMemcpyAsync(pixels, rgb_host, size_t(height) * width * 3, cudaMemcpyHostToDevice, stream));
    const int64_t patch_total = int64_t(Sp) * V_PATCH_DIM;
    v_patchify_kernel<<<v_blocks(patch_total), V_ELEM_T, 0, stream>>>(
        pixels, height, width, cols, 4 * N, reinterpret_cast<uint16_t*>(big16), patch_total);
    v_gemm(ch, big16, V_PATCH_DIM, w.patch_w, V_PATCH_DIM, y, V_HIDDEN, true, Sp, V_HIDDEN, V_PATCH_DIM, stream);
    v_embed_assemble_kernel<<<Sp, V_ROW_T, 0, stream>>>(y, w.pos, w.cls, rows, cols, h);
    v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(h, w.ln_pre_w, w.ln_pre_b, h, true);

    __nv_bfloat16* q  = big16;
    __nv_bfloat16* k  = q + size_t(V_HEADS) * Sp * V_HEAD_DIM;
    __nv_bfloat16* vt = k + size_t(V_HEADS) * Sp * V_HEAD_DIM;
    const float scale = 1.0f / std::sqrt(float(V_HEAD_DIM));

    for (int i = 0; i < VISION_LAYERS; ++i) {
        const VisionLayerWeights& l = w.layers[i];
        v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(h, l.ln1_w, l.ln1_b, xb, false);
        v_gemm(ch, xb, V_HIDDEN, l.wqkv, V_HIDDEN, big32, 3 * V_HIDDEN, true, Sp, 3 * V_HIDDEN, V_HIDDEN, stream);
        v_qkv_post_kernel<<<Sp, V_ROW_T, 0, stream>>>(
            big32, l.bqkv, w.inv_freq, cols, S, Sp,
            reinterpret_cast<uint16_t*>(q), reinterpret_cast<uint16_t*>(k), reinterpret_cast<uint16_t*>(vt));
        for (int head = 0; head < V_HEADS; ++head) {
            const __nv_bfloat16* qh  = q  + size_t(head) * Sp * V_HEAD_DIM;
            const __nv_bfloat16* kh  = k  + size_t(head) * Sp * V_HEAD_DIM;
            const __nv_bfloat16* vth = vt + size_t(head) * V_HEAD_DIM * Sp;
            for (int q0 = 0; q0 < Sp; q0 += Qc) {
                const int Q = std::min(Qc, Sp - q0);
                v_gemm(ch, qh + size_t(q0) * V_HEAD_DIM, V_HEAD_DIM, kh, V_HEAD_DIM, scores, Sp, true, Q, Sp, V_HEAD_DIM, stream);
                v_softmax_kernel<<<Q, V_ROW_T, 0, stream>>>(scores, Sp, S, scale, reinterpret_cast<uint16_t*>(probs));
                v_gemm(ch, probs, Sp, vth, Sp, attn + size_t(q0) * V_HIDDEN + head * V_HEAD_DIM, V_HIDDEN, false, Q, V_HEAD_DIM, Sp, stream);
            }
        }
        v_gemm(ch, attn, V_HIDDEN, l.wo, V_HIDDEN, y, V_HIDDEN, true, Sp, V_HIDDEN, V_HIDDEN, stream);
        v_residual_kernel<<<Sp, V_ROW_T, 0, stream>>>(y, l.bo, l.ls1, h);

        v_layernorm_kernel<<<Sp, V_ROW_T, 0, stream>>>(h, l.ln2_w, l.ln2_b, xb, false);
        v_gemm(ch, xb, V_HIDDEN, l.fc, V_HIDDEN, big32, V_MLP_DIM, true, Sp, V_MLP_DIM, V_HIDDEN, stream);
        const int64_t mlp_tot = int64_t(Sp) * V_MLP_DIM;
        v_bias_gelu_kernel<<<v_blocks(mlp_tot), V_ELEM_T, 0, stream>>>(big32, l.bfc, mlp_tot, V_MLP_DIM, reinterpret_cast<uint16_t*>(big16));
        v_gemm(ch, big16, V_MLP_DIM, l.proj, V_MLP_DIM, y, V_HIDDEN, true, Sp, V_HIDDEN, V_MLP_DIM, stream);
        v_residual_kernel<<<Sp, V_ROW_T, 0, stream>>>(y, l.bproj, l.ls2, h);
    }

    // 2x2 patch merger -> proj_in [4096, 4096] -> exact GELU -> proj_out [2304, 4096] -> BF16 [Np, 2304]
    const int64_t merge_tot = int64_t(Np) * V_MERGED;
    v_merge_rows_kernel<<<v_blocks(merge_tot), V_ELEM_T, 0, stream>>>(
        h, int64_t(N) * V_MERGED, merge_tot, reinterpret_cast<uint16_t*>(big16));
    v_gemm(ch, big16, V_MERGED, w.proj_in, V_MERGED, big32, V_MERGED, true, Np, V_MERGED, V_MERGED, stream);
    v_bias_gelu_kernel<<<v_blocks(merge_tot), V_ELEM_T, 0, stream>>>(big32, nullptr, merge_tot, V_MERGED, reinterpret_cast<uint16_t*>(big16));
    v_gemm(ch, big16, V_MERGED, w.proj_out, V_MERGED, out_bf16_dev, HIDDEN, false, Np, HIDDEN, V_MERGED, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(work));
}

void launch_scatter_image_embeds(
    const __nv_bfloat16* src_bf16,
    float* dst_hidden,
    int dst_offset,
    int count,
    cudaStream_t stream) {
    if (count <= 0) return;
    const int64_t total = int64_t(count) * HIDDEN;
    v_scatter_embeds_kernel<<<v_blocks(total), V_ELEM_T, 0, stream>>>(src_bf16, dst_hidden, dst_offset, count);
}

} // namespace squid
