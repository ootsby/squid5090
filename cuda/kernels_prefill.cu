// Fast Batched Prefill Kernels (B <= 256):
// 1. Coalesced NVFP4/FP8/BF16 -> FP16 dequant + cuBLAS FP16 Tensor Core GEMM for dense, router & controllers.
// 2. Blackwell mma.sync.aligned.m16n8k16 FP16 Tensor Core Multi-Token MoE (w13 softcapped SwiGLU + w2 zero-atomic).
// 3. 1-launch batched B<=256 causal sconv + MoE slot reduction + residual + depth-attn controller.

#include "config.hpp"
#include "kernels.hpp"
#include "kernels_common.cuh"
#include "weights.hpp"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace squid {

namespace {

static cublasHandle_t g_cublas = nullptr;
static cublasLtHandle_t g_cublaslt = nullptr;
static void* g_cublaslt_ws = nullptr;
static constexpr size_t CUBLASLT_WS_BYTES = 32u << 20; // 32 MiB
static constexpr int FP8_ALGO_CACHE_CAP = 64;
struct Fp8AlgoEntry {
    int m, n, k;
    bool valid;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t lA, lB, lC;
    cublasLtMatmulAlgo_t algo;
};
static Fp8AlgoEntry g_fp8_algo_cache[FP8_ALGO_CACHE_CAP];
static int g_fp8_algo_count = 0;

static cublasHandle_t get_cublas(cudaStream_t stream) {
    if (!g_cublas) {
        cublasStatus_t st = cublasCreate(&g_cublas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", (int)st);
            std::exit(1);
        }
        cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(g_cublas, stream);
    return g_cublas;
}

static cublasLtHandle_t get_cublaslt() {
    if (!g_cublaslt) {
        cublasStatus_t st = cublasLtCreate(&g_cublaslt);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasLtCreate failed: %d\n", (int)st);
            std::exit(1);
        }
        if (cudaMalloc(&g_cublaslt_ws, CUBLASLT_WS_BYTES) != cudaSuccess) {
            fprintf(stderr, "cublasLt workspace alloc failed\n");
            std::exit(1);
        }
    }
    return g_cublaslt;
}

__global__ void p_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Per-token dynamic FP16 -> E4M3 quantization for native Blackwell FP8 cublasLtMatmul!
// Single-pass register-cached for K <= 4096 (covers 2304, 3072, etc. with 1 DRAM read!)
__global__ void __launch_bounds__(256)
p_quant_f16_to_e4m3_rowwise_kernel(
    const __half* __restrict__ x_h,
    uint8_t* __restrict__ out_e4m3,
    float* __restrict__ out_x_scale,
    int K) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n_vec8 = K >> 3;
    const uint4* src = reinterpret_cast<const uint4*>(x_h + (size_t)b * K);
    uint2* dst = reinterpret_cast<uint2*>(out_e4m3 + (size_t)b * K);

    uint4 reg_q[2];
    bool cached = (n_vec8 <= 512);
    float amax = 0.0f;
    if (cached) {
        if (tid < n_vec8) {
            reg_q[0] = __ldg(src + tid);
            const __half2* h = reinterpret_cast<const __half2*>(&reg_q[0]);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __half22float2(h[i]);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        if (tid + 256 < n_vec8) {
            reg_q[1] = __ldg(src + tid + 256);
            const __half2* h = reinterpret_cast<const __half2*>(&reg_q[1]);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __half22float2(h[i]);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    } else {
        for (int v = tid; v < n_vec8; v += 256) {
            uint4 q = __ldg(src + v);
            const __half2* h = reinterpret_cast<const __half2*>(&q);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __half22float2(h[i]);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }
    amax = warp_max(amax);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = amax;
    __syncthreads();
    __shared__ float s_inv_sc;
    if (tid == 0) {
        float m = sm[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) m = fmaxf(m, sm[w]);
        float sc = fmaxf(m * (1.0f / 448.0f), 1e-7f);
        out_x_scale[b] = sc;
        s_inv_sc = 1.0f / sc;
    }
    __syncthreads();
    float inv_sc = s_inv_sc;
    auto pack8 = [&](const uint4& q) -> uint2 {
        const __half2* h = reinterpret_cast<const __half2*>(&q);
        uint32_t out_w[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 f0 = __half22float2(h[2 * j + 0]);
            float2 f1 = __half22float2(h[2 * j + 1]);
            uint16_t p0 = __nv_cvt_float2_to_fp8x2(make_float2(f0.x * inv_sc, f0.y * inv_sc), __NV_SATFINITE, __NV_E4M3);
            uint16_t p1 = __nv_cvt_float2_to_fp8x2(make_float2(f1.x * inv_sc, f1.y * inv_sc), __NV_SATFINITE, __NV_E4M3);
            out_w[j] = uint32_t(p0) | (uint32_t(p1) << 16);
        }
        return make_uint2(out_w[0], out_w[1]);
    };
    if (cached) {
        if (tid < n_vec8) dst[tid] = pack8(reg_q[0]);
        if (tid + 256 < n_vec8) dst[tid + 256] = pack8(reg_q[1]);
    } else {
        for (int v = tid; v < n_vec8; v += 256) {
            dst[v] = pack8(__ldg(src + v));
        }
    }
}

// Vectorized float4 post-scale for FP8 cublasLtMatmul: y[b, n] *= row_scales[n] * x_scale[b]
__global__ void __launch_bounds__(256)
p_scale_y_fp8_kernel(
    float* __restrict__ y,
    const float* __restrict__ row_scales,
    const float* __restrict__ x_scale,
    int B, int N) {
    int n4 = N >> 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * n4;
    if (idx >= total) return;
    int b = (int)(idx / n4);
    int c4 = (int)(idx % n4);
    float xs = __ldg(x_scale + b);
    float4 rs = __ldg(reinterpret_cast<const float4*>(row_scales) + c4);
    float4* yp = reinterpret_cast<float4*>(y) + idx;
    float4 v = *yp;
    v.x *= rs.x * xs;
    v.y *= rs.y * xs;
    v.z *= rs.z * xs;
    v.w *= rs.w * xs;
    *yp = v;
}

__global__ void p_dequant_nvfp4_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    __half* __restrict__ out_f16,
    size_t total_blocks) {
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    uint2 w8 = __ldg(reinterpret_cast<const uint2*>(w_data) + bidx);
    // Keep e2m1 * e4m3 exact in FP16 and apply global_scale via cuBLAS alpha in FP32!
    __half2 hsc = __float2half2_rn(cvt_e4m3_to_float(__ldg(w_scales + bidx)));
    __half2 h[8];
    cvt_e2m1x8_to_half2x4(w8.x, h[0], h[1], h[2], h[3]);
    cvt_e2m1x8_to_half2x4(w8.y, h[4], h[5], h[6], h[7]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __hmul2(h[i], hsc);
    uint4* dst4 = reinterpret_cast<uint4*>(out_f16 + bidx * 16);
    dst4[0] = *reinterpret_cast<const uint4*>(&h[0]);
    dst4[1] = *reinterpret_cast<const uint4*>(&h[4]);
}

__global__ void p_dequant_fp8_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    __half* __restrict__ out_f16,
    int N, int K) {
    // Vectorized 16 FP8 -> 16 FP16 (uint4 load -> 2x uint4 stores) per thread!
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t chunks_per_row = (size_t)(K >> 4);
    size_t total = (size_t)N * chunks_per_row;
    if (idx >= total) return;
    int row = (int)(idx / chunks_per_row);
    __half2 hsc = __float2half2_rn(__ldg(row_scales + row));
    uint4 q = __ldg(reinterpret_cast<const uint4*>(w_data) + idx);
    const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
    __half2 h[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[2 * i + 0] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(ws[i] & 0xFFFFu)), hsc);
        h[2 * i + 1] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(ws[i] >> 16)), hsc);
    }
    uint4* dst = reinterpret_cast<uint4*>(out_f16) + idx * 2;
    dst[0] = *reinterpret_cast<const uint4*>(&h[0]);
    dst[1] = *reinterpret_cast<const uint4*>(&h[4]);
}

__global__ void p_dequant_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ w_data,
    __half* __restrict__ out_f16,
    size_t total_vec8) {
    // Vectorized 8 BF16 -> 8 FP16 (uint4 load -> uint4 store) per thread!
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec8) return;
    uint4 q = __ldg(reinterpret_cast<const uint4*>(w_data) + idx);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&q);
    __half2 h[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(b[i]);
        h[i] = __floats2half2_rn(f.x, f.y);
    }
    reinterpret_cast<uint4*>(out_f16)[idx] = *reinterpret_cast<const uint4*>(&h[0]);
}

__device__ __forceinline__ float4 load_sconv_src_f4(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    int b,
    int c4) {
    constexpr int HIDDEN4 = HIDDEN >> 2; // 576
    const float4* base4 = reinterpret_cast<const float4*>(src_or_slots);
    if (is_slotted) {
        const float4* sb = base4 + (size_t)b * TOP_K * HIDDEN4 + c4;
        float4 acc = sb[0 * HIDDEN4];
        #pragma unroll
        for (int k = 1; k < TOP_K; ++k) {
            float4 v = sb[k * HIDDEN4];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        return acc;
    }
    return base4[(size_t)b * HIDDEN4 + c4];
}

// 2D-parallel vectorized float4 batched causal hidden-dim 4-tap FIR sconv + optional 8-slot MoE reduction + residual add!
// Grid: dim3(9, (B+T_BLK-1)/T_BLK), Block: 64 (9 * 64 = 576 = HIDDEN/4 exact -> 576 CTAs for B=2048!).
constexpr int SCONV_T_BLK = 32;

__global__ void __launch_bounds__(64)
p_sconv_post_res_kernel(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    const __nv_bfloat16* __restrict__ sconv_w,
    float* __restrict__ ring,
    const float* __restrict__ residual,
    float* __restrict__ out_res,
    float* __restrict__ pre_tap_out, // optional (e.g. prelude)
    float* __restrict__ blocks,      // optional: commit partial into blocks[b, block_idx]
    const float* __restrict__ partial,
    int block_idx,
    int is_boundary,
    int write_conv_only,
    int update_ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    constexpr int HIDDEN4 = HIDDEN >> 2; // 576
    int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= HIDDEN4) return;

    int b0 = blockIdx.y * SCONV_T_BLK;
    int b1 = min(b0 + SCONV_T_BLK, B);
    if (b0 >= B) return;

    int pos0 = *d_pos;
    // Load 4 channels x 4 taps = 16 BF16 weights contiguously (2 uint4 loads!)
    const uint4* sw4 = reinterpret_cast<const uint4*>(sconv_w + (size_t)c4 * 16);
    uint4 wq0 = __ldg(sw4);
    uint4 wq1 = __ldg(sw4 + 1);
    const __nv_bfloat162* wb0 = reinterpret_cast<const __nv_bfloat162*>(&wq0);
    const __nv_bfloat162* wb1 = reinterpret_cast<const __nv_bfloat162*>(&wq1);
    float2 t0a = __bfloat1622float2(wb0[0]), t0b = __bfloat1622float2(wb0[1]); // ch0: tap 0,1,2,3
    float2 t1a = __bfloat1622float2(wb0[2]), t1b = __bfloat1622float2(wb0[3]); // ch1: tap 0,1,2,3
    float2 t2a = __bfloat1622float2(wb1[0]), t2b = __bfloat1622float2(wb1[1]); // ch2: tap 0,1,2,3
    float2 t3a = __bfloat1622float2(wb1[2]), t3b = __bfloat1622float2(wb1[3]); // ch3: tap 0,1,2,3
    float4 wc0 = make_float4(t0a.x, t1a.x, t2a.x, t3a.x);
    float4 wc1 = make_float4(t0a.y, t1a.y, t2a.y, t3a.y);
    float4 wc2 = make_float4(t0b.x, t1b.x, t2b.x, t3b.x);
    float4 wc3 = make_float4(t0b.y + 1.0f, t1b.y + 1.0f, t2b.y + 1.0f, t3b.y + 1.0f);

    float4* ring4 = reinterpret_cast<float4*>(ring);
    const float4* res4 = reinterpret_cast<const float4*>(residual);
    float4* out4 = reinterpret_cast<float4*>(out_res);
    float4* tap4 = reinterpret_cast<float4*>(pre_tap_out);
    float4* blk4 = reinterpret_cast<float4*>(blocks);
    const float4* part4 = reinterpret_cast<const float4*>(partial);

    auto get_hist = [&](int b_idx) -> float4 {
        if (b_idx >= 0) return load_sconv_src_f4(src_or_slots, is_slotted, b_idx, c4);
        int p_hist = pos0 + b_idx;
        return (p_hist >= 0) ? ring4[(p_hist & 15) * HIDDEN4 + c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    float4 h0 = get_hist(b0 - 3);
    float4 h1 = get_hist(b0 - 2);
    float4 h2 = get_hist(b0 - 1);

    for (int b = b0; b < b1; ++b) {
        float4 cur = load_sconv_src_f4(src_or_slots, is_slotted, b, c4);
        if (update_ring && b >= B - 16) {
            ring4[((pos0 + b) & 15) * HIDDEN4 + c4] = cur;
        }
        float4 conv = make_float4(
            (wc0.x * h0.x + wc1.x * h1.x + wc2.x * h2.x + wc3.x * cur.x) * INV_SQRT_5,
            (wc0.y * h0.y + wc1.y * h1.y + wc2.y * h2.y + wc3.y * cur.y) * INV_SQRT_5,
            (wc0.z * h0.z + wc1.z * h1.z + wc2.z * h2.z + wc3.z * cur.z) * INV_SQRT_5,
            (wc0.w * h0.w + wc1.w * h1.w + wc2.w * h2.w + wc3.w * cur.w) * INV_SQRT_5);
        h0 = h1; h1 = h2; h2 = cur;

        size_t off = (size_t)b * HIDDEN4 + c4;
        if (write_conv_only) {
            out4[off] = conv;
        } else {
            float4 res_v;
            if (is_boundary) {
                if (blk4 && part4) {
                    blk4[((size_t)b * 4 + block_idx) * HIDDEN4 + c4] = part4[off];
                }
                res_v = conv;
            } else {
                float4 rv = res4[off];
                res_v = make_float4(rv.x + conv.x, rv.y + conv.y, rv.z + conv.z, rv.w + conv.w);
            }
            out4[off] = res_v;
            if (tap4) tap4[off] = res_v;
        }
    }
}

__global__ void p_sconv_ring_commit_kernel(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    float* __restrict__ ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    constexpr int HIDDEN4 = HIDDEN >> 2;
    int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= HIDDEN4) return;
    int pos0 = *d_pos;
    int b0 = max(0, B - 16);
    float4* ring4 = reinterpret_cast<float4*>(ring);
    for (int b = b0; b < B; ++b) {
        ring4[((pos0 + b) & 15) * HIDDEN4 + c4] = load_sconv_src_f4(src_or_slots, is_slotted, b, c4);
    }
}

// 1-CTA-per-token (576 threads) runtime-gated directional ablation + residual/boundary commit for prefill.
__global__ void __launch_bounds__(576)
p_ablate_post_res_kernel(
    float* __restrict__ out_res,         // in: raw post-sconv y; out: ablated residual
    const float* __restrict__ residual,
    float* __restrict__ pre_tap_out,
    float* __restrict__ blocks,
    const float* __restrict__ partial,
    int block_idx,
    int is_boundary,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    __shared__ float sm_r[18];
    __shared__ float sm_d[18];
    __shared__ float s_coeff;

    float4 y = *reinterpret_cast<const float4*>(out_res + (size_t)b * HIDDEN + c4);
    float4 rv = *reinterpret_cast<const float4*>(abl_r + c4);
    float dr = warp_sum(y.x * rv.x + y.y * rv.y + y.z * rv.z + y.w * rv.w);
    if (lane == 0) sm_r[warp] = dr;
    if (abl_d) {
        float4 dv = *reinterpret_cast<const float4*>(abl_d + c4);
        float dd = warp_sum(y.x * dv.x + y.y * dv.y + y.z * dv.z + y.w * dv.w);
        if (lane == 0) sm_d[warp] = dd;
    }
    __syncthreads();
    if (warp == 0) {
        float tot_r = warp_sum((lane < 18) ? sm_r[lane] : 0.0f);
        float tot_d = abl_d ? warp_sum((lane < 18) ? sm_d[lane] : 0.0f) : 0.0f;
        if (lane == 0) {
            float4 ap = __ldg(reinterpret_cast<const float4*>(abl_p));
            s_coeff = (!abl_d || tot_d > ap.w) ? (ap.x * fmaxf(tot_r - ap.y, ap.z)) : 0.0f;
        }
    }
    __syncthreads();
    float k = s_coeff;
    y = make_float4(y.x - k * rv.x, y.y - k * rv.y, y.z - k * rv.z, y.w - k * rv.w);

    float4 res_v;
    if (is_boundary) {
        if (blocks && partial) {
            *reinterpret_cast<float4*>(blocks + ((size_t)b * 4 + block_idx) * HIDDEN + c4) =
                *reinterpret_cast<const float4*>(partial + (size_t)b * HIDDEN + c4);
        }
        res_v = y;
    } else {
        float4 base = *reinterpret_cast<const float4*>(residual + (size_t)b * HIDDEN + c4);
        res_v = make_float4(base.x + y.x, base.y + y.y, base.z + y.z, base.w + y.w);
    }
    *reinterpret_cast<float4*>(out_res + (size_t)b * HIDDEN + c4) = res_v;
    if (pre_tap_out) *reinterpret_cast<float4*>(pre_tap_out + (size_t)b * HIDDEN + c4) = res_v;
}

// Copy tap slot into flat [4, B, HIDDEN] layout (arbitrary B, coalesced!) for 1-shot batched DSpark commit!
__global__ void p_copy_taps_chunked_kernel(
    const float* __restrict__ src, // [B, HIDDEN]
    float* __restrict__ dst_taps,  // [4, B, HIDDEN]
    int tap_slot,
    int B) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_f4 = ((size_t)B * HIDDEN) >> 2;
    if (idx >= total_f4) return;
    float* dst = dst_taps + (size_t)tap_slot * B * HIDDEN;
    reinterpret_cast<float4*>(dst)[idx] = reinterpret_cast<const float4*>(src)[idx];
}

// Batched Controller Prep (grid = B, block = 576 = HIDDEN/4 exact, 1-pass float4 zero loops!):
// Computes msg = depth_attn(saved, prior_last, cp_q), cat_h = [rmsnorm(x), rmsnorm(msg)] in FP16!
__global__ void __launch_bounds__(576)
p_ctrl_prep_h_kernel(
    const float* __restrict__ x,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    const float* __restrict__ cp_q,
    float* __restrict__ out_msg,   // [B, HIDDEN]
    __half* __restrict__ out_cat_h // [B, 2*HIDDEN]
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    float4 sv = *reinterpret_cast<const float4*>(saved + (size_t)b * HIDDEN + c4);
    float4 pl = *reinterpret_cast<const float4*>(prior_last + (size_t)b * HIDDEN + c4);
    float4 xv = *reinterpret_cast<const float4*>(x + (size_t)b * HIDDEN + c4);
    float4 qv = *reinterpret_cast<const float4*>(cp_q + c4);

    float d0 = warp_sum(qv.x * sv.x + qv.y * sv.y + qv.z * sv.z + qv.w * sv.w);
    float d1 = warp_sum(qv.x * pl.x + qv.y * pl.y + qv.z * pl.z + qv.w * pl.w);
    float xsq = warp_sum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w);

    __shared__ float sm0[18], sm1[18], smx[18];
    if (lane == 0) { sm0[warp] = d0; sm1[warp] = d1; smx[warp] = xsq; }
    __syncthreads();

    __shared__ float w0, w1, inv_x;
    if (warp == 0) {
        float t0 = warp_sum((lane < 18) ? sm0[lane] : 0.0f);
        float t1 = warp_sum((lane < 18) ? sm1[lane] : 0.0f);
        float tx = warp_sum((lane < 18) ? smx[lane] : 0.0f);
        if (lane == 0) {
            float l0 = t0 * INV_SQRT_HIDDEN;
            float l1 = t1 * INV_SQRT_HIDDEN;
            float mx = fmaxf(l0, l1);
            float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
            float inv = 1.0f / (e0 + e1);
            w0 = e0 * inv;
            w1 = e1 * inv;
            inv_x = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
        }
    }
    __syncthreads();

    float ww0 = w0, ww1 = w1, ix = inv_x;
    float4 mv = make_float4(
        ww0 * sv.x + ww1 * pl.x,
        ww0 * sv.y + ww1 * pl.y,
        ww0 * sv.z + ww1 * pl.z,
        ww0 * sv.w + ww1 * pl.w);
    *reinterpret_cast<float4*>(out_msg + (size_t)b * HIDDEN + c4) = mv;

    __half* cat_b = out_cat_h + (size_t)b * (2 * HIDDEN);
    {
        __half2 h01 = __floats2half2_rn(xv.x * ix, xv.y * ix);
        __half2 h23 = __floats2half2_rn(xv.z * ix, xv.w * ix);
        *reinterpret_cast<uint2*>(cat_b + c4) = make_uint2(*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23));
    }

    float msq = warp_sum(mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w);
    if (lane == 0) sm0[warp] = msq;
    __syncthreads();
    __shared__ float inv_m;
    if (warp == 0) {
        float tm = warp_sum((lane < 18) ? sm0[lane] : 0.0f);
        if (lane == 0) inv_m = rsqrtf(tm / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float im = inv_m;
    {
        __half2 h01 = __floats2half2_rn(mv.x * im, mv.y * im);
        __half2 h23 = __floats2half2_rn(mv.z * im, mv.w * im);
        *reinterpret_cast<uint2*>(cat_b + HIDDEN + c4) = make_uint2(*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23));
    }
}

// Batched Controller Finish (grid = B, block = 576 = HIDDEN/4 exact, 1-pass float4 zero loops!):
// merged = (1 - tanh(forget)) * x + tanh(gain) * msg -> RMSNorm * post_norm -> out (f32) & out_h (f16)
__global__ void __launch_bounds__(576)
p_ctrl_finish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ msg,
    const float* __restrict__ forget,
    const float* __restrict__ gain,
    const __nv_bfloat16* __restrict__ post_norm,
    float* __restrict__ out,
    __half* __restrict__ out_h) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    float4 xv = *reinterpret_cast<const float4*>(x + (size_t)b * HIDDEN + c4);
    float4 mv = *reinterpret_cast<const float4*>(msg + (size_t)b * HIDDEN + c4);
    float4 fv = *reinterpret_cast<const float4*>(forget + (size_t)b * HIDDEN + c4);
    float4 gv = *reinterpret_cast<const float4*>(gain + (size_t)b * HIDDEN + c4);

    float4 vv = make_float4(
        (1.0f - fast_tanh(fv.x)) * xv.x + fast_tanh(gv.x) * mv.x,
        (1.0f - fast_tanh(fv.y)) * xv.y + fast_tanh(gv.y) * mv.y,
        (1.0f - fast_tanh(fv.z)) * xv.z + fast_tanh(gv.z) * mv.z,
        (1.0f - fast_tanh(fv.w)) * xv.w + fast_tanh(gv.w) * mv.w);

    float sq = warp_sum(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z + vv.w * vv.w);
    __shared__ float sm[18];
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ir = inv_rms;
    uint2 nw4 = *reinterpret_cast<const uint2*>(post_norm + c4);
    float2 n01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nw4.x));
    float2 n23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nw4.y));
    vv = make_float4(vv.x * ir * n01.x, vv.y * ir * n01.y, vv.z * ir * n23.x, vv.w * ir * n23.y);

    *reinterpret_cast<float4*>(out + (size_t)b * HIDDEN + c4) = vv;
    if (out_h) {
        __half2 h01 = __floats2half2_rn(vv.x, vv.y);
        __half2 h23 = __floats2half2_rn(vv.z, vv.w);
        *reinterpret_cast<uint2*>(out_h + (size_t)b * HIDDEN + c4) = make_uint2(*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23));
    }
}

// Per-token top-8 selection (1 warp per token, grid = B) — identical tie-breaking to kernels_inkling.cu!
__global__ void p_route_topk_only_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ router_bias,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights) {
    int b = blockIdx.x;
    int lane = threadIdx.x;

    float sc[4], bi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = lane + 32 * i;
        float raw = fast_sigmoid(logits[(size_t)b * NUM_EXPERTS + e]);
        sc[i] = raw;
        bi[i] = raw + router_bias[e];
    }

    int winners[TOP_K];
    float chosen[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        float best = -1e30f;
        int best_e = NUM_EXPERTS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane + 32 * i;
            if (bi[i] > best || (bi[i] == best && e < best_e)) {
                best = bi[i];
                best_e = e;
            }
        }
        float top = warp_max(best);
        int cand = (best == top) ? best_e : NUM_EXPERTS;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
        }
        cand = min(cand, NUM_EXPERTS - 1);
        float sel = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + 32 * i == cand) {
                sel = sc[i];
                bi[i] = -1e30f;
            }
        }
        chosen[k] = warp_sum(sel);
        winners[k] = cand;
    }

    if (lane == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) total += chosen[k];
        float inv = ROUTE_SCALE / (total + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            out_ids[b * TOP_K + k] = winners[k];
            out_weights[b * TOP_K + k] = chosen[k] * inv;
        }
    }
}

__global__ void p_sort_experts_kernel(
    const int32_t* __restrict__ route_ids,
    const float* __restrict__ route_w,
    int B,
    int TR,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ sorted_tok,
    int32_t* __restrict__ sorted_slot,
    float* __restrict__ sorted_w) {
    __shared__ int32_t counts[NUM_EXPERTS];
    __shared__ int32_t cursors[NUM_EXPERTS];
    int tid = threadIdx.x;
    counts[tid] = 0;
    __syncthreads();

    int total = B * TOP_K;
    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        atomicAdd(&counts[route_ids[idx]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        int nt = 0;
        int32_t* tile_exp = expert_offsets + 130;
        int32_t* tile_st  = expert_offsets + 386;
        int32_t* tile_en  = expert_offsets + 642;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            expert_offsets[e] = sum;
            cursors[e] = sum;
            int cnt = counts[e];
            for (int off = 0; off < cnt; off += TR) {
                tile_exp[nt] = e;
                tile_st[nt]  = sum + off;
                tile_en[nt]  = sum + min(off + TR, cnt);
                nt++;
            }
            sum += cnt;
        }
        expert_offsets[NUM_EXPERTS] = sum;
        expert_offsets[129] = nt;
    }
    __syncthreads();

    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        int e = route_ids[idx];
        int dst = atomicAdd(&cursors[e], 1);
        sorted_tok[dst] = idx >> 3;
        sorted_slot[dst] = idx & 7;
        sorted_w[dst] = route_w[idx];
    }
}

// ============================================================================
// Multi-Stage cp.async 128x128x64 Blackwell FP16 TC Grouped MoE GEMM (Prefill)
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine wgemm.cu (Apache-2.0)
// and specialized for Blackwell sm_120a hardware cvt.rn.f16x2.e2m1x2 + FP16 MMA!
// ============================================================================

enum class WgemmMode { Up, Down };
constexpr int WG_TN = 128;
constexpr int WG_BK = 64;
constexpr int WG_AS = 3;
constexpr int WG_NT = 256;
constexpr int WG_KCH = WG_BK / 8; // 8 16-byte chunks per activation row

__device__ __forceinline__ uint32_t wg_smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void wg_cp_async16(void* smem_dst, const void* gmem_src, int src_bytes = 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(wg_smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void wg_cp_async4(void* smem_dst, const void* gmem_src, int src_bytes = 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(wg_smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void wg_cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void wg_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Groups));
}

__device__ __forceinline__ void wg_ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void wg_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int Chunks>
__device__ __forceinline__ int wg_swz(int row, int chunk) {
    return row * (Chunks * 8) + ((chunk ^ (row & 7)) << 3);
}

template <int FMT, int K, int TM>
struct WgCfg {
    static_assert(FMT != 2 || K == HIDDEN);
    static constexpr int CODE_ROW  = (FMT == 0) ? (K / 2) : (FMT == 2) ? IQ3_ROW_BYTES : K;
    static constexpr int SCALE_ROW = (FMT == 0) ? (K / 16) : (FMT == 2) ? IQ3_D_ROW_BYTES : (K / 32);
    static constexpr int WK        = (FMT != 1) ? 128 : 64;
    static constexpr int R         = WK / WG_BK;
    static constexpr int KC        = K / WK;
    static constexpr int WS        = (TM <= 32) ? ((FMT != 1) ? 6 : 7) : ((FMT != 1) ? 3 : 4);
    static_assert(R * (WS - 1) >= WG_AS);
    static constexpr int WCODE     = (FMT == 0) ? (WK / 2) : (FMT == 2) ? 48 : WK;
    static constexpr int WSTRIDE   = (FMT == 2) ? WCODE : WCODE + 16; // 48 and 80 both spread 8 rows across the smem banks
    static constexpr int W_BYTES   = WG_TN * WSTRIDE;
    static constexpr int A_BYTES   = TM * WG_BK * 2;
    static constexpr int SC_BYTES  = WG_TN * SCALE_ROW;
    static constexpr int OFF_W     = WG_AS * A_BYTES;
    static constexpr int OFF_SC    = OFF_W + WS * W_BYTES;
    static constexpr int OFF_GRID  = (OFF_SC + SC_BYTES + 15) / 16 * 16;
    static constexpr int SMEM      = OFF_GRID + ((FMT == 2) ? 2048 : 0);
    static_assert(K % WK == 0 && SMEM <= 101376);
};

template <int MT, int TM, class After>
__device__ __forceinline__ void wg_mma_step(
    float (&acc)[TM / 16][2][4], const uint16_t* A,
    const uint32_t (&b)[WG_BK / 16][2][2], int lane, After&& after) {
    #pragma unroll
    for (int kk = 0; kk < WG_BK / 16; ++kk) {
        uint32_t a[MT][4];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            wg_ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                           wg_smem_addr(A + wg_swz<WG_KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
        }
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            wg_mma_f16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
            wg_mma_f16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
        }
        after(kk);
    }
}

template <int FMT, WgemmMode MODE, int K, int TM>
__global__ void __launch_bounds__(WG_NT, 1)
p_wgemm_kernel(
    const uint8_t* __restrict__ w_codes,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ w_factors,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    const float* __restrict__ sorted_w,
    __half* __restrict__ out_act_h,
    float* __restrict__ out_slots) {
    constexpr int KS = K / WG_BK;
    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    using C = WgCfg<FMT, K, TM>;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* a_smem = smem;
    unsigned char* w_smem = smem + C::OFF_W;
    uint8_t* sc_smem = smem + C::OFF_SC;
    uint2* iq3_grid = reinterpret_cast<uint2*>(smem + C::OFF_GRID);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_idx = blockIdx.x / NTILES_N;
    if (tile_idx >= expert_offsets[129]) return;

    const int e  = expert_offsets[130 + tile_idx];
    const int pb = expert_offsets[386 + tile_idx];
    const int en = expert_offsets[642 + tile_idx];
    const int n  = en - pb;
    const int n0 = (blockIdx.x % NTILES_N) * WG_TN;

    const uint8_t* code_tile  = w_codes  + ((size_t)e * NE + n0) * C::CODE_ROW;
    const uint8_t* scale_tile = w_scales + ((size_t)e * NE + n0) * C::SCALE_ROW;
    const uint16_t* a_base    = reinterpret_cast<const uint16_t*>(x_h);

    const uint16_t* arow[TM / 32];
    #pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
        const int r = (tid >> 3) + 32 * i;
        const int rr = (r < n) ? r : 0;
        if constexpr (MODE == WgemmMode::Up) {
            arow[i] = a_base + (size_t)sorted_tok[pb + rr] * K;
        } else {
            arow[i] = a_base + (size_t)(pb + rr) * K;
        }
    }

    auto load_a = [&](int ks) {
        uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % WG_AS) * C::A_BYTES);
        #pragma unroll
        for (int i = 0; i < TM / 32; ++i) {
            const int r = (tid >> 3) + 32 * i;
            const int c = tid & 7;
            wg_cp_async16(A + wg_swz<WG_KCH>(r, c), arow[i] + ks * WG_BK + c * 8, (r < n) ? 16 : 0);
        }
    };
    auto load_w = [&](int chunk) {
        unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
        constexpr int CPR = C::WCODE / 16;
        for (int id = tid; id < WG_TN * CPR; id += WG_NT) { // CPR = 3 (IQ3) does not divide evenly into WG_NT
            const int row = id / CPR;
            const int ch = id % CPR;
            wg_cp_async16(ws + row * C::WSTRIDE + ch * 16,
                          code_tile + (size_t)row * C::CODE_ROW + chunk * C::WCODE + ch * 16);
        }
    };
    auto issue = [&](int j) {
        if (j < KS) load_a(j);
        wg_cp_commit();
    };

    static_assert(WG_NT == 256);
    if constexpr (FMT == 2) iq3_grid[tid] = iq3_grid_entry(tid);
    for (int id = tid; id < C::SC_BYTES / 4; id += WG_NT) {
        int byte_off = id * 4;
        wg_cp_async4(sc_smem + byte_off, scale_tile + (size_t)(byte_off / C::SCALE_ROW) * C::SCALE_ROW + (byte_off % C::SCALE_ROW));
    }
    #pragma unroll
    for (int j = 0; j < C::WS - 1; ++j) {
        if (j < C::KC) load_w(j);
    }
    #pragma unroll
    for (int j = 0; j < WG_AS - 1; ++j) issue(j);

    const int c = lane & 3;
    const float factor = (FMT != 1) ? __ldg(w_factors + e) : 1.0f;
    const int mact = min((n + 15) / 16, TM / 16);
    float acc[TM / 16][2][4];
    #pragma unroll
    for (int a = 0; a < TM / 16; ++a) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.0f;
    }

    auto scales_of = [&](int ks, __half2 (&sc)[2][4]) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int row = warp * 16 + nt * 8 + (lane >> 2);
            if constexpr (FMT == 0) {
                const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
                const __half2 lo = cvt_e4m3x2_to_half2(static_cast<uint16_t>(s4 & 0xFFFFu));
                const __half2 hi = cvt_e4m3x2_to_half2(static_cast<uint16_t>(s4 >> 16));
                sc[nt][0] = __low2half2(lo);
                sc[nt][1] = __high2half2(lo);
                sc[nt][2] = __low2half2(hi);
                sc[nt][3] = __high2half2(hi);
            } else if constexpr (FMT == 2) {
                const uint32_t* sw = reinterpret_cast<const uint32_t*>(
                    w_smem + ((ks / C::R) % C::WS) * C::W_BYTES + row * C::WSTRIDE + 32) + (ks % C::R) * 2;
                const uint16_t d = *reinterpret_cast<const uint16_t*>(sc_smem + row * C::SCALE_ROW + (ks >> 2) * 2);
                sc[nt][0] = sc[nt][1] = iq3_scale(d, sw[0]);
                sc[nt][2] = sc[nt][3] = iq3_scale(d, sw[1]);
            } else {
                // Convert UE8M0 power-of-2 scales once to FP16 (clamped at 65504 to stay 100% NaN-free!)
                const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
                const __half2 h0 = __float2half2_rn(fminf(cvt_ue8m0_to_float(ms[0]), 65504.0f));
                const __half2 h1 = __float2half2_rn(fminf(cvt_ue8m0_to_float(ms[1]), 65504.0f));
                sc[nt][0] = h0;
                sc[nt][1] = h0;
                sc[nt][2] = h1;
                sc[nt][3] = h1;
            }
        }
    };

    auto decode = [&](int ks, int kk, const __half2 (&sc)[2][4], uint32_t (&b)[WG_BK / 16][2][2]) {
        const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
        const int kc = (ks % C::R) * WG_BK + kk * 16;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int row = warp * 16 + nt * 8 + (lane >> 2);
            const __half2 s = sc[nt][kk];
            if constexpr (FMT == 0) {
                const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + (kc >> 1));
                const uint32_t b0 = (wd.x >> (8 * c)) & 0xFFu;
                const uint32_t b1 = (wd.y >> (8 * c)) & 0xFFu;
                __half2 h0 = __hmul2(cvt_e2m1x2_to_half2(b0), s);
                __half2 h1 = __hmul2(cvt_e2m1x2_to_half2(b1), s);
                b[kk][nt][0] = *reinterpret_cast<const uint32_t*>(&h0);
                b[kk][nt][1] = *reinterpret_cast<const uint32_t*>(&h1);
            } else if constexpr (FMT == 2) {
                // This lane takes weights 2c..2c+1 and 8+2c..9+2c of the 16: one codebook half2 from each group of 8.
                const unsigned char* rp = ws + row * C::WSTRIDE;
                const uint32_t q = *reinterpret_cast<const uint32_t*>(rp + (kc >> 2));
                const uint32_t ss = *reinterpret_cast<const uint32_t*>(rp + 32 + (kc >> 5) * 4);
                #pragma unroll
                for (int g = 0; g < 2; ++g) {
                    const uint32_t idx = (q >> (8 * (2 * g + (c >> 1)))) & 0xFFu;
                    const uint32_t n = iq3_signs(ss, ((kc >> 3) & 3) + g) >> (2 * c);
                    const uint32_t v = reinterpret_cast<const uint32_t*>(iq3_grid + idx)[c & 1]
                                       ^ ((n & 1u) << 15) ^ ((n & 2u) << 30);
                    __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&v), s);
                    b[kk][nt][g] = *reinterpret_cast<const uint32_t*>(&h);
                }
            } else {
                // Load ONLY the 2 uint16_t pairs this lane owns (4x less SMEM traffic than uint4!) + 1-inst HMUL2!
                const unsigned char* rp = ws + row * C::WSTRIDE + kc;
                const uint16_t u0 = *reinterpret_cast<const uint16_t*>(rp + c * 2);
                const uint16_t u1 = *reinterpret_cast<const uint16_t*>(rp + 8 + c * 2);
                __half2 h0 = __hmul2(cvt_e4m3x2_to_half2(u0), s);
                __half2 h1 = __hmul2(cvt_e4m3x2_to_half2(u1), s);
                b[kk][nt][0] = *reinterpret_cast<const uint32_t*>(&h0);
                b[kk][nt][1] = *reinterpret_cast<const uint32_t*>(&h1);
            }
            asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
        }
    };

    uint32_t bcur[WG_BK / 16][2][2], bnext[WG_BK / 16][2][2];
    #pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
        wg_cp_wait<WG_AS - 2>();
        __syncthreads();
        const int chunk = ks / C::R;
        if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
        issue(ks + WG_AS - 1);
        if (ks == 0) {
            __half2 sc0[2][4];
            scales_of(0, sc0);
            #pragma unroll
            for (int kk = 0; kk < WG_BK / 16; ++kk) decode(0, kk, sc0, bcur);
        }
        const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % WG_AS) * C::A_BYTES);
        const bool more = (ks + 1 < KS);
        __half2 scn[2][4];
        if (more) scales_of(ks + 1, scn);
        auto after = [&](int kk) {
            if (more) decode(ks + 1, kk, scn, bnext);
        };
        switch (mact) {
            case 1: if constexpr (1 <= TM / 16) wg_mma_step<1, TM>(acc, A, bcur, lane, after); break;
            case 2: if constexpr (2 <= TM / 16) wg_mma_step<2, TM>(acc, A, bcur, lane, after); break;
            case 3: if constexpr (3 <= TM / 16) wg_mma_step<3, TM>(acc, A, bcur, lane, after); break;
            case 4: if constexpr (4 <= TM / 16) wg_mma_step<4, TM>(acc, A, bcur, lane, after); break;
            case 5: if constexpr (5 <= TM / 16) wg_mma_step<5, TM>(acc, A, bcur, lane, after); break;
            case 6: if constexpr (6 <= TM / 16) wg_mma_step<6, TM>(acc, A, bcur, lane, after); break;
            case 7: if constexpr (7 <= TM / 16) wg_mma_step<7, TM>(acc, A, bcur, lane, after); break;
            default: wg_mma_step<TM / 16, TM>(acc, A, bcur, lane, after); break;
        }
        #pragma unroll
        for (int kk = 0; kk < WG_BK / 16; ++kk) {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                bcur[kk][nt][0] = bnext[kk][nt][0];
                bcur[kk][nt][1] = bnext[kk][nt][1];
            }
        }
    }
    wg_cp_wait<0>();

    #pragma unroll
    for (int mt = 0; mt < TM / 16; ++mt) {
        if (mt >= mact) continue;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = mt * 16 + (lane >> 2) + half * 8;
            if (row >= n) continue;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int cl = warp * 16 + nt * 8 + c * 2;
                const float c0 = acc[mt][nt][2 * half] * factor;
                const float c1 = acc[mt][nt][2 * half + 1] * factor;
                if constexpr (MODE == WgemmMode::Up) {
                    const float g_cap = fast_tanh(c0 * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    const float u_cap = fast_tanh(c1 * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                    out_act_h[(size_t)(pb + row) * INTERMEDIATE + (n0 + cl) / 2] =
                        __float2half((g_cap * fast_sigmoid(c0)) * u_cap);
                } else {
                    const int dst_slot = sorted_tok[pb + row] * TOP_K + sorted_slot[pb + row];
                    const float rw = sorted_w[pb + row];
                    out_slots[(size_t)dst_slot * HIDDEN + n0 + cl + 0] = c0 * rw;
                    out_slots[(size_t)dst_slot * HIDDEN + n0 + cl + 1] = c1 * rw;
                }
            }
        }
    }
}

template <int FMT, WgemmMode MODE, int K, int TM>
static void dispatch_wgemm_tm(
    const ExpertBank& eb, const __half* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot, const float* sorted_w,
    __half* out_act_h, float* out_slots, cudaStream_t stream) {
    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    constexpr int SMEM = WgCfg<FMT, K, TM>::SMEM;
    static const cudaError_t attr = cudaFuncSetAttribute(
        p_wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    (void)attr;
    // Max 256 active tiles by pigeonhole bound (B<=512 w/ TR=32 or B<=2048 w/ TR=128)
    int grid = 256 * NTILES_N;
    p_wgemm_kernel<FMT, MODE, K, TM><<<grid, WG_NT, SMEM, stream>>>(
        eb.data, eb.scales, (FMT != 1) ? eb.expert_scale : nullptr,
        x_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, out_act_h, out_slots);
}

// ============================================================================
// Native Blackwell sm_120a Block-Scaled Tensor Core Prefill MoE!
// - NVFP4 (FMT=0): mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (4x TFLOPS!)
// - MXFP8 (FMT=1): mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 (2x TFLOPS!)
// Quantizes activations once into idle p_w_f16 scratch -> ZERO weight decode, ZERO extra VRAM!
// ============================================================================

__device__ __forceinline__ uint8_t p_cvt_f32x2_to_e2m1x2(float lo, float hi) {
    uint16_t out;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; mov.b16 %0, {b, b}; }"
        : "=h"(out) : "f"(lo), "f"(hi));
    return static_cast<uint8_t>(out & 0xFFu);
}

__global__ void __launch_bounds__(256)
p_quant_f16_to_nvfp4_kernel(
    const __half* __restrict__ x_h,
    uint8_t* __restrict__ out_code,
    uint8_t* __restrict__ out_scale,
    size_t total_blocks16) {
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks16) return;
    const uint4* src = reinterpret_cast<const uint4*>(x_h) + bidx * 2;
    uint4 q0 = __ldg(src), q1 = __ldg(src + 1);
    const __half2* h0 = reinterpret_cast<const __half2*>(&q0);
    const __half2* h1 = reinterpret_cast<const __half2*>(&q1);
    float2 f[8];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[i] = __half22float2(h0[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f[i].x), fabsf(f[i].y)));
        f[4 + i] = __half22float2(h1[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f[4 + i].x), fabsf(f[4 + i].y)));
    }
    uint8_t sc_u8 = float_to_e4m3(fminf(fmaxf(amax * (1.0f / 6.0f), 1e-5f), 448.0f));
    float inv_sc = 1.0f / cvt_e4m3_to_float(sc_u8);
    uint8_t b[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b[i] = p_cvt_f32x2_to_e2m1x2(f[i].x * inv_sc, f[i].y * inv_sc);
    }
    uint32_t w0 = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    uint32_t w1 = uint32_t(b[4]) | (uint32_t(b[5]) << 8) | (uint32_t(b[6]) << 16) | (uint32_t(b[7]) << 24);
    reinterpret_cast<uint2*>(out_code)[bidx] = make_uint2(w0, w1);
    out_scale[bidx] = sc_u8;
}

__global__ void __launch_bounds__(256)
p_quant_f16_to_mxfp8_kernel(
    const __half* __restrict__ x_h,
    uint8_t* __restrict__ out_code,
    uint8_t* __restrict__ out_scale,
    size_t total_blocks32) {
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks32) return;
    const uint4* src = reinterpret_cast<const uint4*>(x_h) + bidx * 4;
    float2 f[16];
    float amax = 0.0f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint4 q = __ldg(src + c);
        const __half2* h = reinterpret_cast<const __half2*>(&q);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            f[c * 4 + i] = __half22float2(h[i]);
            amax = fmaxf(amax, fmaxf(fabsf(f[c * 4 + i].x), fabsf(f[c * 4 + i].y)));
        }
    }
    float need = fmaxf(amax * (1.0f / 448.0f), 1e-30f);
    uint32_t bits = __float_as_uint(need);
    uint8_t exp_u8 = static_cast<uint8_t>(min(254u, (bits + 0x007FFFFFu) >> 23));
    float inv_sc = 1.0f / cvt_ue8m0_to_float(exp_u8);
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint16_t p0 = __nv_cvt_float2_to_fp8x2(make_float2(f[2 * i + 0].x * inv_sc, f[2 * i + 0].y * inv_sc), __NV_SATFINITE, __NV_E4M3);
        uint16_t p1 = __nv_cvt_float2_to_fp8x2(make_float2(f[2 * i + 1].x * inv_sc, f[2 * i + 1].y * inv_sc), __NV_SATFINITE, __NV_E4M3);
        w[i] = uint32_t(p0) | (uint32_t(p1) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out_code) + bidx * 2;
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    out_scale[bidx] = exp_u8;
}

template <int FMT, int K, int TM>
struct WgBsCfg {
    static constexpr int BK = 128; // K=2304 -> 18, K=1152 -> 9 EXACT (64B/row NVFP4 = full memory sector!)
    static constexpr int BK_BYTES = (FMT == 0) ? 64 : 128;
    static constexpr int BK_SC = (FMT == 0) ? 8 : 4;
    static constexpr int STRIDE = BK_BYTES + 16; // 80 B (NVFP4) or 144 B (MXFP8), 16B aligned, 0 bank conflicts!
    static constexpr int CODE_ROW = (FMT == 0) ? (K >> 1) : K;
    static constexpr int SCALE_ROW = (FMT == 0) ? (K >> 4) : (K >> 5);
    static constexpr int KK_STEPS = (FMT == 0) ? 2 : 4;
    static constexpr int A_CODE_BYTES = TM * STRIDE;
    static constexpr int A_SC_BYTES = TM * BK_SC;
    static constexpr int W_CODE_BYTES = WG_TN * STRIDE;
    static constexpr int W_SC_BYTES = WG_TN * BK_SC;
    static constexpr int STAGE_BYTES = ((A_CODE_BYTES + A_SC_BYTES + W_CODE_BYTES + W_SC_BYTES + 15) & ~15);
    static constexpr int SMEM = 2 * STAGE_BYTES;
};

template <int FMT, WgemmMode MODE, int K, int TM>
__global__ void __launch_bounds__(WG_NT, (FMT == 0 || TM <= 32) ? 2 : 1)
p_wgemm_blkscale_kernel(
    const uint8_t* __restrict__ w_codes,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ w_factors,
    const uint8_t* __restrict__ a_codes,
    const uint8_t* __restrict__ a_scales,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    const float* __restrict__ sorted_w,
    __half* __restrict__ out_act_h,
    float* __restrict__ out_slots) {
    using C = WgBsCfg<FMT, K, TM>;
    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    constexpr int KS = K / C::BK;

    const int ntile = blockIdx.x % NTILES_N;
    const int tile_idx = blockIdx.x / NTILES_N;
    if (tile_idx >= expert_offsets[129]) return;

    const int e  = expert_offsets[130 + tile_idx];
    const int pb = expert_offsets[386 + tile_idx];
    const int en = expert_offsets[642 + tile_idx];
    const int n  = en - pb;
    const int n0 = ntile * WG_TN;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    // Cache sorted_tok[pb..pb+n-1] once in SMEM for Up mode (eliminates 36x global loads in load_stage!)
    __shared__ int s_tok[TM];
    if constexpr (MODE == WgemmMode::Up) {
        if (tid < TM) {
            s_tok[tid] = sorted_tok[pb + ((tid < n) ? tid : 0)];
        }
        __syncthreads();
    }

    extern __shared__ __align__(128) unsigned char smem[];
    const uint8_t* w_code_base = w_codes + ((size_t)e * NE + n0) * C::CODE_ROW;
    const uint8_t* w_sc_base   = w_scales + ((size_t)e * NE + n0) * C::SCALE_ROW;

    auto load_stage = [&](int ks) {
        uint8_t* st = smem + (ks & 1) * C::STAGE_BYTES;
        uint8_t* sm_ac = st;
        uint8_t* sm_as = sm_ac + C::A_CODE_BYTES;
        uint8_t* sm_wc = sm_as + C::A_SC_BYTES;
        uint8_t* sm_ws = sm_wc + C::W_CODE_BYTES;

        // A_code: TM rows x ACPR uint4 chunks
        constexpr int ACPR = C::BK_BYTES / 16; // 4 (NVFP4) or 8 (MXFP8)
        #pragma unroll
        for (int i = 0; i < (TM * ACPR + WG_NT - 1) / WG_NT; ++i) {
            const int id = tid + i * WG_NT;
            if (id < TM * ACPR) {
                const int r = id / ACPR;
                const int ch = id % ACPR;
                const int row_idx = (MODE == WgemmMode::Up) ? s_tok[r] : (pb + ((r < n) ? r : 0));
                wg_cp_async16(sm_ac + r * C::STRIDE + ch * 16,
                              a_codes + (size_t)row_idx * C::CODE_ROW + ks * C::BK_BYTES + ch * 16,
                              (r < n) ? 16 : 0);
            }
        }
        // A_sc: TM rows x ASPR uint32 chunks
        constexpr int ASPR = C::BK_SC / 4; // 2 (NVFP4) or 1 (MXFP8)
        if (tid < TM * ASPR) {
            const int r = tid / ASPR;
            const int ch = tid % ASPR;
            const int row_idx = (MODE == WgemmMode::Up) ? s_tok[r] : (pb + ((r < n) ? r : 0));
            wg_cp_async4(sm_as + r * C::BK_SC + ch * 4,
                         a_scales + (size_t)row_idx * C::SCALE_ROW + ks * C::BK_SC + ch * 4,
                         (r < n) ? 4 : 0);
        }
        // W_code: 128 rows x ACPR uint4 chunks
        #pragma unroll
        for (int i = 0; i < (WG_TN * ACPR) / WG_NT; ++i) {
            const int id = tid + i * WG_NT;
            const int r = id / ACPR;
            const int ch = id % ACPR;
            wg_cp_async16(sm_wc + r * C::STRIDE + ch * 16,
                          w_code_base + (size_t)r * C::CODE_ROW + ks * C::BK_BYTES + ch * 16);
        }
        // W_sc: 128 rows x ASPR uint32 chunks
        if (tid < WG_TN * ASPR) {
            const int r = tid / ASPR;
            const int ch = tid % ASPR;
            wg_cp_async4(sm_ws + r * C::BK_SC + ch * 4,
                         w_sc_base + (size_t)r * C::SCALE_ROW + ks * C::BK_SC + ch * 4);
        }
        wg_cp_commit();
    };

    load_stage(0);

    const float factor = (FMT == 0) ? __ldg(w_factors + e) : 1.0f;
    const int mact = min((n + 15) / 16, TM / 16);
    float acc[TM / 16][2][4];
    #pragma unroll
    for (int a = 0; a < TM / 16; ++a) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.0f;
    }

    #pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
        if (ks + 1 < KS) load_stage(ks + 1);
        if (ks + 1 < KS) wg_cp_wait<1>();
        else             wg_cp_wait<0>();
        __syncthreads();

        const uint8_t* st = smem + (ks & 1) * C::STAGE_BYTES;
        const uint8_t* sm_ac = st;
        const uint8_t* sm_as = sm_ac + C::A_CODE_BYTES;
        const uint8_t* sm_wc = sm_as + C::A_SC_BYTES;
        const uint8_t* sm_ws = sm_wc + C::W_CODE_BYTES;

        #pragma unroll
        for (int kk = 0; kk < C::KK_STEPS; ++kk) {
            const int k_off = kk * 32;
            uint32_t b0[2], b1[2], sb[2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int wrow = warp * 16 + nt * 8 + g;
                b0[nt] = *reinterpret_cast<const uint32_t*>(sm_wc + wrow * C::STRIDE + k_off + t * 4);
                b1[nt] = *reinterpret_cast<const uint32_t*>(sm_wc + wrow * C::STRIDE + k_off + 16 + t * 4);
                if constexpr (FMT == 0) {
                    sb[nt] = *reinterpret_cast<const uint32_t*>(sm_ws + wrow * C::BK_SC + kk * 4);
                } else {
                    sb[nt] = static_cast<uint32_t>(sm_ws[wrow * C::BK_SC + kk]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < TM / 16; ++mt) {
                if (mt >= mact) break;
                const int r0 = mt * 16 + g;
                const int r1 = mt * 16 + g + 8;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sm_ac + r0 * C::STRIDE + k_off + t * 4);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sm_ac + r1 * C::STRIDE + k_off + t * 4);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sm_ac + r0 * C::STRIDE + k_off + 16 + t * 4);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sm_ac + r1 * C::STRIDE + k_off + 16 + t * 4);
                uint32_t sa;
                if constexpr (FMT == 0) {
                    sa = *reinterpret_cast<const uint32_t*>(sm_as + (mt * 16 + g + (t & 1) * 8) * C::BK_SC + kk * 4);
                } else {
                    sa = static_cast<uint32_t>(sm_as[(mt * 16 + g + (t & 1) * 8) * C::BK_SC + kk]);
                }
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    if constexpr (FMT == 0) {
                        asm volatile(
                            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                            : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nt]), "r"(b1[nt]), "r"(sa), "r"(sb[nt]));
                    } else {
                        asm volatile(
                            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                            : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nt]), "r"(b1[nt]), "r"(sa), "r"(sb[nt]));
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < TM / 16; ++mt) {
        if (mt >= mact) continue;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = mt * 16 + g + half * 8;
            if (row >= n) continue;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int cl = warp * 16 + nt * 8 + t * 2;
                const float c0 = acc[mt][nt][2 * half] * factor;
                const float c1 = acc[mt][nt][2 * half + 1] * factor;
                if constexpr (MODE == WgemmMode::Up) {
                    const float g_cap = fast_tanh(c0 * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    const float u_cap = fast_tanh(c1 * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                    out_act_h[(size_t)(pb + row) * INTERMEDIATE + (n0 + cl) / 2] =
                        __float2half((g_cap * fast_sigmoid(c0)) * u_cap);
                } else {
                    const int tok = sorted_tok[pb + row];
                    const float rw = sorted_w[pb + row];
                    atomicAdd(&out_slots[(size_t)tok * HIDDEN + n0 + cl + 0], c0 * rw);
                    atomicAdd(&out_slots[(size_t)tok * HIDDEN + n0 + cl + 1], c1 * rw);
                }
            }
        }
    }
}

template <int FMT, WgemmMode MODE, int K, int TM>
static void dispatch_wgemm_blkscale_tm(
    const ExpertBank& eb, const __half* x_h, int M_rows,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot, const float* sorted_w,
    __half* out_act_h, float* out_slots, void* scratch_q, cudaStream_t stream) {
    using C = WgBsCfg<FMT, K, TM>;
    uint8_t* a_codes = static_cast<uint8_t*>(scratch_q);
    size_t code_bytes = (((size_t)M_rows * C::CODE_ROW) + 255u) & ~size_t(255u);
    uint8_t* a_scales = a_codes + code_bytes;

    if constexpr (FMT == 0) {
        size_t n_blk = (size_t)M_rows * (K >> 4);
        p_quant_f16_to_nvfp4_kernel<<<(int)((n_blk + 255) / 256), 256, 0, stream>>>(x_h, a_codes, a_scales, n_blk);
    } else {
        size_t n_blk = (size_t)M_rows * (K >> 5);
        p_quant_f16_to_mxfp8_kernel<<<(int)((n_blk + 255) / 256), 256, 0, stream>>>(x_h, a_codes, a_scales, n_blk);
    }

    constexpr int NE = (MODE == WgemmMode::Up) ? (2 * INTERMEDIATE) : HIDDEN;
    constexpr int NTILES_N = NE / WG_TN;
    static const cudaError_t attr = cudaFuncSetAttribute(
        p_wgemm_blkscale_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    (void)attr;
    int grid = 256 * NTILES_N;
    p_wgemm_blkscale_kernel<FMT, MODE, K, TM><<<grid, WG_NT, C::SMEM, stream>>>(
        eb.data, eb.scales, (FMT == 0) ? eb.expert_scale : nullptr,
        a_codes, a_scales, expert_offsets, sorted_tok, sorted_slot, sorted_w, out_act_h, out_slots);
}

// Commit the final <=16 raw K & V tokens into their 16-slot sconv rings (100% race-free after fused prep_store!).
__global__ void p_kv_ring_commit_kernel(
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    float* __restrict__ k_ring,
    float* __restrict__ v_ring,
    int B,
    const int32_t* __restrict__ d_pos) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= KV_DIM) return;
    int is_v = blockIdx.y;
    const float* x = is_v ? v_proj : k_proj;
    float* ring = is_v ? v_ring : k_ring;
    int pos0 = *d_pos;
    int b0 = max(0, B - 16);
    for (int b = b0; b < B; ++b) {
        ring[((pos0 + b) & 15) * KV_DIM + c] = x[(size_t)b * KV_DIM + c];
    }
}

// 1-launch FUSED 2D-parallel Local static sconv + Q/R/K clip + V RMSNorm + BF16 KV store (ALL 170 SMs!)
// Eliminates both 4-SM serial static_sconv_kernel launches AND DRAM round-trips of k_proj/v_proj!
__global__ void __launch_bounds__(128)
p_local_sconv_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const __nv_bfloat16* __restrict__ k_sconv_w,
    const __nv_bfloat16* __restrict__ v_sconv_w,
    const float* __restrict__ k_ring,
    const float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int kv_cap,
    const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int pos0 = *d_pos;
    int p = pos0 + b;
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f);
    } else {
        bool is_v = (kind_id >= 56);
        int h = is_v ? (kind_id - 56) : (kind_id - 48);
        int c = h * HEAD_DIM + tid;
        const float* x = is_v ? v_proj : k_proj;
        const float* ring = is_v ? v_ring : k_ring;
        const __nv_bfloat16* sw = is_v ? v_sconv_w : k_sconv_w;

        auto get_hist = [&](int bi) -> float {
            if (bi >= 0) return x[(size_t)bi * KV_DIM + c];
            int ph = pos0 + bi;
            return (ph >= 0) ? ring[(ph & 15) * KV_DIM + c] : 0.0f;
        };
        float h0 = get_hist(b - 3);
        float h1 = get_hist(b - 2);
        float h2 = get_hist(b - 1);
        float cur = x[(size_t)b * KV_DIM + c];
        float wc0 = __bfloat162float(sw[c * 4 + 0]);
        float wc1 = __bfloat162float(sw[c * 4 + 1]);
        float wc2 = __bfloat162float(sw[c * 4 + 2]);
        float wc3 = __bfloat162float(sw[c * 4 + 3]);
        float val = (wc0 * h0 + wc1 * h1 + wc2 * h2 + wc3 * cur + cur) * INV_SQRT_5;

        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) {
            float tot = sm[0] + sm[1] + sm[2] + sm[3];
            factor = is_v ? rsqrtf(tot / float(HEAD_DIM) + RMS_EPS)
                          : fminf(QK_CLIP / (sqrtf(tot) + 1e-6f), 1.0f);
        }
        __syncthreads();
        int slot = p % kv_cap;
        __nv_bfloat16* dst = is_v ? v_cache : k_cache;
        dst[(size_t)slot * KV_DIM + c] = __float2bfloat16(val * factor);
    }
}

// 1-launch FUSED 2D-parallel Global dynamic sconv + Q/R/K clip (with log-pos tau) + V RMSNorm + KV store (BF16/FP8/Q4)!
// Eliminates the 8-SM 2048-step serial p_dynamic_sconv_kv_inplace_kernel completely!
template <int KV_FMT>
__global__ void __launch_bounds__(128)
p_global_attn_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const __nv_bfloat16* __restrict__ k_sw,
    const __nv_bfloat16* __restrict__ v_sw,
    const float* __restrict__ k_taps,
    const float* __restrict__ v_taps,
    const float* __restrict__ k_ring,
    const float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int kv_cap,
    const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int pos0 = *d_pos;
    int p = pos0 + b;
    float ratio = fmaxf(float(p + 1) / float(LOG_SCALE_FLOOR), 1.0f);
    float tau = 1.0f + LOG_SCALE_ALPHA * logf(ratio);
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f) * tau;
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f) * tau;
    } else {
        bool is_v = (kind_id >= 56);
        int h = is_v ? (kind_id - 56) : (kind_id - 48);
        int c = h * HEAD_DIM + tid;
        const float* x = is_v ? v_proj : k_proj;
        const float* ring = is_v ? v_ring : k_ring;
        const __nv_bfloat16* sw = is_v ? v_sw : k_sw;
        const float* dt = (is_v ? v_taps : k_taps) + (size_t)b * 32 + h * 4;

        auto get_hist = [&](int bi) -> float {
            if (bi >= 0) return x[(size_t)bi * KV_DIM + c];
            int ph = pos0 + bi;
            return (ph >= 0) ? ring[(ph & 15) * KV_DIM + c] : 0.0f;
        };
        float cur = x[(size_t)b * KV_DIM + c];
        float h1 = get_hist(b - 1);
        float h2 = get_hist(b - 2);
        float h3 = get_hist(b - 3);
        float sw0 = __bfloat162float(sw[0 * KV_DIM + c]);
        float sw1 = __bfloat162float(sw[1 * KV_DIM + c]);
        float sw2 = __bfloat162float(sw[2 * KV_DIM + c]);
        float sw3 = __bfloat162float(sw[3 * KV_DIM + c]);
        float acc = cur * (sw0 + dt[0]) + h1 * (sw1 + dt[1]) + h2 * (sw2 + dt[2]) + h3 * (sw3 + dt[3]);
        float val = (acc + cur) * INV_SQRT_9;

        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) {
            float tot = sm[0] + sm[1] + sm[2] + sm[3];
            factor = is_v ? rsqrtf(tot / float(HEAD_DIM) + RMS_EPS)
                          : fminf(QK_CLIP / (sqrtf(tot) + 1e-6f), 1.0f);
        }
        __syncthreads();
        store_kv_head_128<KV_FMT>(
            val * factor, p % kv_cap, h, tid,
            is_v ? v_cache : k_cache,
            is_v ? v_scale : k_scale, sm);
    }
}

// Batched DSpark Prefill Commit: mixes [4, B, HIDDEN] -> FP16 mix_h[0..2][B, HIDDEN] + last 16 f_ring slots!
__global__ void __launch_bounds__(256)
p_dspark_mix_h_kernel(
    const float* __restrict__ taps,       // [4, B, HIDDEN]
    int B,
    const float* __restrict__ fuse_alpha, // [4, 4]
    const __nv_bfloat16* __restrict__ n0,
    const __nv_bfloat16* __restrict__ n1,
    const __nv_bfloat16* __restrict__ n2,
    const __nv_bfloat16* __restrict__ n3,
    __half* __restrict__ mix_h0,
    __half* __restrict__ mix_h1,
    __half* __restrict__ mix_h2,
    float* __restrict__ f_ring,           // [16, HIDDEN]
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int m = blockIdx.y; // 0..3
    if (m == 3 && b < B - 16) return; // only last 16 tokens survive in 16-slot f_ring!
    int tid = threadIdx.x;

    const float a0 = fuse_alpha[m * 4 + 0];
    const float a1 = fuse_alpha[m * 4 + 1];
    const float a2 = fuse_alpha[m * 4 + 2];
    const float a3 = fuse_alpha[m * 4 + 3];

    const float* t0 = taps + ((size_t)0 * B + b) * HIDDEN;
    const float* t1 = taps + ((size_t)1 * B + b) * HIDDEN;
    const float* t2 = taps + ((size_t)2 * B + b) * HIDDEN;
    const float* t3 = taps + ((size_t)3 * B + b) * HIDDEN;

    float vals[9];
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        float v = a0 * t0[d] + a1 * t1[d] + a2 * t2[d] + a3 * t3[d];
        vals[i] = v;
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = sum_sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    const __nv_bfloat16* nw = (m == 0) ? n0 : (m == 1 ? n1 : (m == 2 ? n2 : n3));
    float s = inv_rms;
    if (m < 3) {
        __half* dst = ((m == 0) ? mix_h0 : (m == 1 ? mix_h1 : mix_h2)) + (size_t)b * HIDDEN;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int d = tid + i * 256;
            dst[d] = __float2half(vals[i] * s * __bfloat162float(nw[d]));
        }
    } else {
        float* dst = f_ring + (size_t)(((*d_pos) + b) & 15) * HIDDEN;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int d = tid + i * 256;
            dst[d] = vals[i] * s * __bfloat162float(nw[d]);
        }
    }
}

template <int KV_FMT>
__global__ void __launch_bounds__(128)
p_dspark_store_ctx_kv_kernel(
    const float* __restrict__ kv,
    const __nv_bfloat16* __restrict__ k_norm,
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int cap,
    int B,
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int pos = (*d_pos) + b;
    // For local DSpark ring (cap < B): only the last `cap` tokens survive — skip overwritten early writes!
    if (B > cap && b < B - cap) return;
    int slot = pos % cap;

    const float* kp = kv + (size_t)b * (2 * KV_DIM) + h * HEAD_DIM;
    const float* vp = kv + (size_t)b * (2 * KV_DIM) + KV_DIM + h * HEAD_DIM;

    __shared__ float sm[4];
    store_kv_head_128<KV_FMT>(vp[tid], slot, h, tid, v_cache, v_scale, sm);
    __syncthreads();

    float kv_val = kp[tid];
    float sq = warp_sum(kv_val * kv_val);
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    __syncthreads();

    __shared__ float kn[HEAD_DIM];
    kn[tid] = kv_val * inv_rms * __bfloat162float(k_norm[tid]);
    __syncthreads();

    if (tid < 64) {
        float freq = 1.0f / powf(MTP_ROPE_THETA, float(2 * tid) / float(HEAD_DIM));
        float ang = float(pos) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = kn[tid], x1 = kn[tid + 64];
        kn[tid]      = x0 * c - x1 * s;
        kn[tid + 64] = x0 * s + x1 * c;
    }
    __syncthreads();
    store_kv_head_128<KV_FMT>(kn[tid], slot, h, tid, k_cache, k_scale, sm);
}

// BLACKWELL TENSOR CORE FLASHATTENTION PREFILL (Global Q4/FP8/BF16 & Local SWA BF16!)
// Each CTA owns 1 KV head x Q_TOK tokens (GQA_RATIO * Q_TOK queries = NUM_WARPS x 16 exact!).
// Every KV tile (32 tokens) is loaded ONCE into SMEM and shared across 48 (local) or 96 (global) queries!
// Bulk global tiles (dist >= 1024) skip rel-bias + causal masking completely (pure TC FlashAttention!).
// Zero SMEM output reduction: each warp owns 16 queries end-to-end and writes directly to out_gated_h!
template <int KV_FMT, int Q_TOK>
__global__ void __launch_bounds__((Q_TOK * GQA_RATIO / 16) * 32, 2)
p_flash_attn_prefill_tc_kernel(
    const float* __restrict__ qrgs,
    const __nv_bfloat16* __restrict__ rel_proj,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    __half* __restrict__ out_gated_h,
    int kv_cap,
    int window,
    int B,
    const int32_t* __restrict__ d_pos) {
    constexpr int Q_MAX = Q_TOK * GQA_RATIO;     // 96
    constexpr int NUM_WARPS = Q_MAX / 16;        // 6
    constexpr int THREADS = NUM_WARPS * 32;      // 192
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    constexpr bool USE_BF16_ASYNC = (KV_FMT == 0);
    constexpr int KV_STAGES = USE_BF16_ASYNC ? 2 : 1;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float Q_SCALE_LOG2 = INV_SQRT_HEAD_DIM * LOG2E;

    // Reverse blockIdx.y so longest causal CTAs launch first (eliminates causal tail wave!)
    int kvh = blockIdx.x;
    int by = (gridDim.y - 1) - blockIdx.y;
    int b_start = by * Q_TOK;
    int q_tok_cnt = min(Q_TOK, B - b_start);
    if (q_tok_cnt <= 0) return;
    int q_cnt = q_tok_cnt * GQA_RATIO;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;

    int pos0 = *d_pos;
    int q_pos_min = pos0 + b_start;
    int q_pos_max = q_pos_min + q_tok_cnt - 1;

    __shared__ __align__(16) uint16_t s_k[KV_STAGES][32][HEAD_DIM + 8];
    __shared__ __align__(16) uint16_t s_v[KV_STAGES][32][HEAD_DIM + 8];
    __shared__ __align__(16) __half s_r[Q_MAX][D_REL];       // 3,072 B (16B aligned!)
    __shared__ __align__(16) __half s_rp[64][D_REL + 8];     // 3,072 B (48B stride = 12 banks -> 0 bank conflicts for uint4!)

    // 1. Load R[0..q_cnt-1][0..15] into SMEM once for all KV tiles
    for (int i = tid; i < q_cnt * D_REL; i += THREADS) {
        int qi = i >> 4;
        int k = i & 15;
        int h = kvh * GQA_RATIO + (qi % GQA_RATIO);
        int b = b_start + (qi / GQA_RATIO);
        s_r[qi][k] = __float2half(qrgs[(size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL + k]);
    }

    // 2. Load this warp's 16 queries (q0, q1) into registers + initialize softmax (base-2) with sink!
    int q0 = warp * 16 + grp;
    int q1 = warp * 16 + grp + 8;
    const float* qp0 = nullptr;
    const float* qp1 = nullptr;
    const float* gp0 = nullptr;
    const float* gp1 = nullptr;
    int b0 = 0, h0 = 0, pos_q0 = q_pos_min;
    int b1 = 0, h1 = 0, pos_q1 = q_pos_min;
    float m0 = 0.0f, m1 = 0.0f;
    float d0 = 1.0f, d1 = 1.0f;

    if (q0 < q_cnt) {
        h0 = kvh * GQA_RATIO + (q0 % GQA_RATIO);
        b0 = b_start + (q0 / GQA_RATIO);
        pos_q0 = pos0 + b0;
        const float* base0 = qrgs + (size_t)b0 * QRGS_STRIDE;
        qp0 = base0 + h0 * HEAD_DIM;
        gp0 = base0 + Q_DIM + R_DIM + h0 * HEAD_DIM;
        m0 = base0[Q_DIM + R_DIM + Q_DIM + h0] * LOG2E;
    }
    if (q1 < q_cnt) {
        h1 = kvh * GQA_RATIO + (q1 % GQA_RATIO);
        b1 = b_start + (q1 / GQA_RATIO);
        pos_q1 = pos0 + b1;
        const float* base1 = qrgs + (size_t)b1 * QRGS_STRIDE;
        qp1 = base1 + h1 * HEAD_DIM;
        gp1 = base1 + Q_DIM + R_DIM + h1 * HEAD_DIM;
        m1 = base1[Q_DIM + R_DIM + Q_DIM + h1] * LOG2E;
    }

    auto pack2 = [](float a, float b) -> uint32_t {
        if constexpr (USE_BF16_ASYNC) {
            __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
            return *reinterpret_cast<const uint32_t*>(&v);
        } else {
            __half2 v = __floats2half2_rn(a, b);
            return *reinterpret_cast<const uint32_t*>(&v);
        }
    };

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        int k_off = kt * 16;
        if (qp0) {
            q_reg[kt][0] = pack2(qp0[k_off + tid4 * 2] * Q_SCALE_LOG2, qp0[k_off + tid4 * 2 + 1] * Q_SCALE_LOG2);
            q_reg[kt][2] = pack2(qp0[k_off + tid4 * 2 + 8] * Q_SCALE_LOG2, qp0[k_off + tid4 * 2 + 9] * Q_SCALE_LOG2);
        } else {
            q_reg[kt][0] = 0u;
            q_reg[kt][2] = 0u;
        }
        if (qp1) {
            q_reg[kt][1] = pack2(qp1[k_off + tid4 * 2] * Q_SCALE_LOG2, qp1[k_off + tid4 * 2 + 1] * Q_SCALE_LOG2);
            q_reg[kt][3] = pack2(qp1[k_off + tid4 * 2 + 8] * Q_SCALE_LOG2, qp1[k_off + tid4 * 2 + 9] * Q_SCALE_LOG2);
        } else {
            q_reg[kt][1] = 0u;
            q_reg[kt][3] = 0u;
        }
    }

    float o[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }

    int c0 = tid4 * 2;
    int c1 = tid4 * 2 + 1;
    int c4 = tid4 * 2 + 8;
    int c5 = tid4 * 2 + 9;
    int k_ld_col = (lane & 8);

    int t_start = (window > 0) ? max(0, q_pos_min + 1 - window) : 0;
    int t_end = q_pos_max + 1;
    int num_tiles = (t_end > t_start) ? ((t_end - t_start + 31) >> 5) : 0;

    auto issue_bf16_tile = [&](int ti) {
        int t0 = t_start + (ti << 5);
        int tile_len = min(32, t_end - t0);
        int st = ti & 1;
        int s_base = (window > 0) ? (t0 % kv_cap) : t0;
        const __nv_bfloat16* kb = static_cast<const __nv_bfloat16*>(k_cache);
        const __nv_bfloat16* vb = static_cast<const __nv_bfloat16*>(v_cache);
        #pragma unroll
        for (int idx = tid; idx < 512; idx += THREADS) {
            int t_loc = idx >> 4;
            int sub = idx & 15;
            int slot = s_base + t_loc;
            if (window > 0 && slot >= kv_cap) slot -= kv_cap;
            int nb = (t_loc < tile_len) ? 16 : 0;
            size_t off = (size_t)slot * KV_DIM + kvh * HEAD_DIM + sub * 8;
            wg_cp_async16(&s_k[st][t_loc][sub * 8], kb + off, nb);
            wg_cp_async16(&s_v[st][t_loc][sub * 8], vb + off, nb);
        }
        wg_cp_commit();
    };

    if constexpr (USE_BF16_ASYNC) {
        if (num_tiles > 0) issue_bf16_tile(0);
    }

    for (int ti = 0; ti < num_tiles; ++ti) {
        int t0 = t_start + (ti << 5);
        int tile_len = min(32, t_end - t0);
        bool is_bulk = (window == 0) && (tile_len == 32) && (t0 + 31 <= q_pos_min - REL_EXTENT);
        int st = USE_BF16_ASYNC ? (ti & 1) : 0;

        if constexpr (USE_BF16_ASYNC) {
            if (ti + 1 < num_tiles) issue_bf16_tile(ti + 1);
        } else {
            // Zero integer modulo inside inner loop!
            int s_base = (window > 0) ? (t0 % kv_cap) : t0;
            #pragma unroll
            for (int idx = tid; idx < 512; idx += THREADS) {
                int t_loc = idx >> 4;
                int sub = idx & 15;
                if (t_loc < tile_len) {
                    int slot = s_base + t_loc;
                    if (window > 0 && slot >= kv_cap) slot -= kv_cap;
                    *reinterpret_cast<uint4*>(&s_k[0][t_loc][sub * 8]) =
                        load_kv_8elems_h2x4<KV_FMT>(k_cache, k_scale, slot, kvh, sub);
                    *reinterpret_cast<uint4*>(&s_v[0][t_loc][sub * 8]) =
                        load_kv_8elems_h2x4<KV_FMT>(v_cache, v_scale, slot, kvh, sub);
                } else {
                    *reinterpret_cast<uint4*>(&s_k[0][t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&s_v[0][t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        int d_min = 0;
        if (!is_bulk) {
            d_min = max(0, q_pos_min - (t0 + tile_len - 1));
            int d_max = min(REL_EXTENT - 1, q_pos_max - t0);
            int d_span = (d_max >= d_min) ? (d_max - d_min + 1) : 0;
            for (int i = tid; i < d_span * D_REL; i += THREADS) {
                int di = i >> 4;
                int k = i & 15;
                int dist = d_min + di;
                s_rp[di][k] = __float2half(__bfloat162float(__ldg(&rel_proj[k * REL_EXTENT + dist])) * LOG2E);
            }
        }

        if constexpr (USE_BF16_ASYNC) {
            if (ti + 1 < num_tiles) wg_cp_wait<1>();
            else wg_cp_wait<0>();
        }
        __syncthreads();

        // Compute BOTH 16-key Q@K^T sub-tiles together -> 1 unified 32-key softmax + conditional o rescale!
        float s[2][8];
        #pragma unroll
        for (int kv_sub = 0; kv_sub < 2; ++kv_sub) {
            int t_base = kv_sub * 16;
            int w_len = min(16, max(0, tile_len - t_base));
            s[kv_sub][0] = s[kv_sub][1] = s[kv_sub][2] = s[kv_sub][3] = 0.0f;
            s[kv_sub][4] = s[kv_sub][5] = s[kv_sub][6] = s[kv_sub][7] = 0.0f;
            if (w_len <= 0) {
                s[kv_sub][0] = s[kv_sub][1] = s[kv_sub][2] = s[kv_sub][3] = -1e30f;
                s[kv_sub][4] = s[kv_sub][5] = s[kv_sub][6] = s[kv_sub][7] = -1e30f;
                continue;
            }

            int k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                uint32_t kb0, kb1, kb2, kb3;
                ldmatrix_x4(kb0, kb1, kb2, kb3, &s_k[st][k_ld_row][kt * 16 + k_ld_col]);
                if constexpr (USE_BF16_ASYNC) {
                    mma_m16n8k16_f32_bf16(s[kv_sub][0], s[kv_sub][1], s[kv_sub][2], s[kv_sub][3],
                                          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                          kb0, kb1, s[kv_sub][0], s[kv_sub][1], s[kv_sub][2], s[kv_sub][3]);
                    mma_m16n8k16_f32_bf16(s[kv_sub][4], s[kv_sub][5], s[kv_sub][6], s[kv_sub][7],
                                          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                          kb2, kb3, s[kv_sub][4], s[kv_sub][5], s[kv_sub][6], s[kv_sub][7]);
                } else {
                    mma_m16n8k16_f32_f16(s[kv_sub][0], s[kv_sub][1], s[kv_sub][2], s[kv_sub][3],
                                         q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                         kb0, kb1, s[kv_sub][0], s[kv_sub][1], s[kv_sub][2], s[kv_sub][3]);
                    mma_m16n8k16_f32_f16(s[kv_sub][4], s[kv_sub][5], s[kv_sub][6], s[kv_sub][7],
                                         q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                         kb2, kb3, s[kv_sub][4], s[kv_sub][5], s[kv_sub][6], s[kv_sub][7]);
                }
            }
        }

        if (!is_bulk) {
            // Fast branchless interior path (~94% of Local SWA tiles): all 32 keys strictly valid & within REL_EXTENT!
            bool is_interior_rel = (tile_len == 32)
                && (t0 + 31 <= q_pos_min)
                && (window == 0 || t0 >= q_pos_max + 1 - window)
                && (q_pos_max - t0 < REL_EXTENT)
                && (q1 < q_cnt);

            auto rel_dot_u4 = [&](const uint4& rq0, const uint4& rq1, int di) -> float {
                uint4 pq0 = *reinterpret_cast<const uint4*>(&s_rp[di][0]);
                uint4 pq1 = *reinterpret_cast<const uint4*>(&s_rp[di][8]);
                const __half2* r0 = reinterpret_cast<const __half2*>(&rq0);
                const __half2* r1 = reinterpret_cast<const __half2*>(&rq1);
                const __half2* p0 = reinterpret_cast<const __half2*>(&pq0);
                const __half2* p1 = reinterpret_cast<const __half2*>(&pq1);
                __half2 a0 = __hmul2(r0[0], p0[0]), a1 = __hmul2(r0[1], p0[1]);
                a0 = __hfma2(r0[2], p0[2], a0); a1 = __hfma2(r0[3], p0[3], a1);
                a0 = __hfma2(r1[0], p1[0], a0); a1 = __hfma2(r1[1], p1[1], a1);
                a0 = __hfma2(r1[2], p1[2], a0); a1 = __hfma2(r1[3], p1[3], a1);
                float2 f = __half22float2(__hadd2(a0, a1));
                return f.x + f.y;
            };

            if (is_interior_rel) {
                auto apply_fast = [&](int qi, int pos_q, int row_off) {
                    uint4 rq0 = *reinterpret_cast<const uint4*>(&s_r[qi][0]);
                    uint4 rq1 = *reinterpret_cast<const uint4*>(&s_r[qi][8]);
                    int base_di = (pos_q - t0) - d_min;
                    s[0][row_off + 0] += rel_dot_u4(rq0, rq1, base_di - c0);
                    s[0][row_off + 1] += rel_dot_u4(rq0, rq1, base_di - c1);
                    s[0][row_off + 4] += rel_dot_u4(rq0, rq1, base_di - c4);
                    s[0][row_off + 5] += rel_dot_u4(rq0, rq1, base_di - c5);
                    s[1][row_off + 0] += rel_dot_u4(rq0, rq1, base_di - 16 - c0);
                    s[1][row_off + 1] += rel_dot_u4(rq0, rq1, base_di - 16 - c1);
                    s[1][row_off + 4] += rel_dot_u4(rq0, rq1, base_di - 16 - c4);
                    s[1][row_off + 5] += rel_dot_u4(rq0, rq1, base_di - 16 - c5);
                };
                apply_fast(q0, pos_q0, 0);
                apply_fast(q1, pos_q1, 2);
            } else {
                auto apply_rel_row = [&](int qi, int pos_q, int row_off) {
                    if (qi >= q_cnt) {
                        #pragma unroll
                        for (int kv_sub = 0; kv_sub < 2; ++kv_sub) {
                            s[kv_sub][row_off + 0] = -1e30f; s[kv_sub][row_off + 1] = -1e30f;
                            s[kv_sub][row_off + 4] = -1e30f; s[kv_sub][row_off + 5] = -1e30f;
                        }
                        return;
                    }
                    uint4 rq0 = *reinterpret_cast<const uint4*>(&s_r[qi][0]);
                    uint4 rq1 = *reinterpret_cast<const uint4*>(&s_r[qi][8]);
                    #pragma unroll
                    for (int kv_sub = 0; kv_sub < 2; ++kv_sub) {
                        int t_base = kv_sub * 16;
                        int w_len = min(16, max(0, tile_len - t_base));
                        auto one = [&](float& sv, int col) {
                            if (col >= w_len) { sv = -1e30f; return; }
                            int t = t0 + t_base + col;
                            if (t > pos_q || (window > 0 && t < pos_q + 1 - window)) { sv = -1e30f; return; }
                            int dist = pos_q - t;
                            if (dist < REL_EXTENT) sv += rel_dot_u4(rq0, rq1, dist - d_min);
                        };
                        one(s[kv_sub][row_off + 0], c0);
                        one(s[kv_sub][row_off + 1], c1);
                        one(s[kv_sub][row_off + 4], c4);
                        one(s[kv_sub][row_off + 5], c5);
                    }
                };
                apply_rel_row(q0, pos_q0, 0);
                apply_rel_row(q1, pos_q1, 2);
            }
        }

        // Unified 32-key online softmax + skip o-rescale completely when max didn't grow!
        float rmax0 = fmaxf(
            fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[0][4], s[0][5])),
            fmaxf(fmaxf(s[1][0], s[1][1]), fmaxf(s[1][4], s[1][5])));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
        float rmax1 = fmaxf(
            fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[0][6], s[0][7])),
            fmaxf(fmaxf(s[1][2], s[1][3]), fmaxf(s[1][6], s[1][7])));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

        float m0_new = fmaxf(m0, rmax0);
        float m1_new = fmaxf(m1, rmax1);
        bool grow0 = (m0_new > m0);
        bool grow1 = (m1_new > m1);
        float alpha0 = grow0 ? exp2f(m0 - m0_new) : 1.0f;
        float alpha1 = grow1 ? exp2f(m1 - m1_new) : 1.0f;

        if (__any_sync(0xffffffffu, grow0 || grow1)) {
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                o[nt][0] *= alpha0;
                o[nt][1] *= alpha0;
                o[nt][2] *= alpha1;
                o[nt][3] *= alpha1;
            }
        }
        d0 *= alpha0; m0 = m0_new;
        d1 *= alpha1; m1 = m1_new;

        float rsum0 = 0.0f, rsum1 = 0.0f;
        #pragma unroll
        for (int kv_sub = 0; kv_sub < 2; ++kv_sub) {
            int t_base = kv_sub * 16;
            if (tile_len <= t_base) break;

            float ps0 = (is_bulk || s[kv_sub][0] > -1e29f) ? exp2f(s[kv_sub][0] - m0_new) : 0.0f;
            float ps1 = (is_bulk || s[kv_sub][1] > -1e29f) ? exp2f(s[kv_sub][1] - m0_new) : 0.0f;
            float ps4 = (is_bulk || s[kv_sub][4] > -1e29f) ? exp2f(s[kv_sub][4] - m0_new) : 0.0f;
            float ps5 = (is_bulk || s[kv_sub][5] > -1e29f) ? exp2f(s[kv_sub][5] - m0_new) : 0.0f;
            float ps2 = (is_bulk || s[kv_sub][2] > -1e29f) ? exp2f(s[kv_sub][2] - m1_new) : 0.0f;
            float ps3 = (is_bulk || s[kv_sub][3] > -1e29f) ? exp2f(s[kv_sub][3] - m1_new) : 0.0f;
            float ps6 = (is_bulk || s[kv_sub][6] > -1e29f) ? exp2f(s[kv_sub][6] - m1_new) : 0.0f;
            float ps7 = (is_bulk || s[kv_sub][7] > -1e29f) ? exp2f(s[kv_sub][7] - m1_new) : 0.0f;

            rsum0 += (ps0 + ps1 + ps4 + ps5);
            rsum1 += (ps2 + ps3 + ps6 + ps7);

            uint32_t pa0 = pack2(ps0, ps1);
            uint32_t pa1 = pack2(ps2, ps3);
            uint32_t pa2 = pack2(ps4, ps5);
            uint32_t pa3 = pack2(ps6, ps7);

            int v_ld_row = t_base + (lane & 15);
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                uint32_t vb0, vb1;
                ldmatrix_x2_trans(vb0, vb1, &s_v[st][v_ld_row][nt * 8]);
                if constexpr (USE_BF16_ASYNC) {
                    mma_m16n8k16_f32_bf16(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                                          pa0, pa1, pa2, pa3,
                                          vb0, vb1,
                                          o[nt][0], o[nt][1], o[nt][2], o[nt][3]);
                } else {
                    mma_m16n8k16_f32_f16(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                                         pa0, pa1, pa2, pa3,
                                         vb0, vb1,
                                         o[nt][0], o[nt][1], o[nt][2], o[nt][3]);
                }
            }
        }
        rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
        rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
        rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
        rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
        d0 += rsum0;
        d1 += rsum1;
        __syncthreads();
    }

    // 3. Direct gated FP16 writeback (zero SMEM reduction!)
    if (q0 < q_cnt) {
        float inv_d0 = 1.0f / d0;
        __half* dst0 = out_gated_h + (size_t)b0 * Q_DIM + h0 * HEAD_DIM;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            int col0 = nt * 8 + c0;
            int col1 = nt * 8 + c1;
            dst0[col0] = __float2half((o[nt][0] * inv_d0) * fast_sigmoid(gp0[col0]));
            dst0[col1] = __float2half((o[nt][1] * inv_d0) * fast_sigmoid(gp0[col1]));
        }
    }
    if (q1 < q_cnt) {
        float inv_d1 = 1.0f / d1;
        __half* dst1 = out_gated_h + (size_t)b1 * Q_DIM + h1 * HEAD_DIM;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            int col0 = nt * 8 + c0;
            int col1 = nt * 8 + c1;
            dst1[col0] = __float2half((o[nt][2] * inv_d1) * fast_sigmoid(gp1[col0]));
            dst1[col1] = __float2half((o[nt][3] * inv_d1) * fast_sigmoid(gp1[col1]));
        }
    }
}

} // namespace

static int cache_fp8_algo_entry(int M, int N, int K, const void* dA, const void* dB, void* dC, cudaStream_t stream) {
    cublasLtHandle_t lt = get_cublaslt();
    for (int i = 0; i < g_fp8_algo_count; ++i) {
        if (g_fp8_algo_cache[i].m == M && g_fp8_algo_cache[i].n == N && g_fp8_algo_cache[i].k == K) {
            return i;
        }
    }
    if (g_fp8_algo_count >= FP8_ALGO_CACHE_CAP) return -1;

    Fp8AlgoEntry ent{};
    ent.m = M;
    ent.n = N;
    ent.k = K;
    cublasLtMatmulDescCreate(&ent.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(ent.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(ent.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayoutCreate(&ent.lA, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&ent.lB, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&ent.lC, CUDA_R_32F, M, N, M);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_b = CUBLASLT_WS_BYTES;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_b, sizeof(ws_b));
    cublasLtMatmulHeuristicResult_t res[8];
    int found = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, ent.desc, ent.lA, ent.lB, ent.lC, ent.lC, pref, 8, res, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    ent.valid = (st == CUBLAS_STATUS_SUCCESS && found > 0);
    if (ent.valid) {
        ent.algo = res[0].algo;
        if (dA && dB && dC && found > 1) {
            const float alpha = 1.0f, beta = 0.0f;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            float best_ms = 1e9f;
            for (int a = 0; a < found; ++a) {
                cublasStatus_t s0 = cublasLtMatmul(lt, ent.desc, &alpha, dA, ent.lA, dB, ent.lB, &beta, dC, ent.lC, dC, ent.lC, &res[a].algo, g_cublaslt_ws, CUBLASLT_WS_BYTES, stream);
                if (s0 != CUBLAS_STATUS_SUCCESS) continue;
                cudaEventRecord(e0, stream);
                for (int it = 0; it < 3; ++it) {
                    cublasLtMatmul(lt, ent.desc, &alpha, dA, ent.lA, dB, ent.lB, &beta, dC, ent.lC, dC, ent.lC, &res[a].algo, g_cublaslt_ws, CUBLASLT_WS_BYTES, stream);
                }
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 1e9f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best_ms) {
                    best_ms = ms;
                    ent.algo = res[a].algo;
                }
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    } else {
        cublasLtMatrixLayoutDestroy(ent.lA);
        cublasLtMatrixLayoutDestroy(ent.lB);
        cublasLtMatrixLayoutDestroy(ent.lC);
        cublasLtMatmulDescDestroy(ent.desc);
        ent.desc = nullptr;
        ent.lA = ent.lB = ent.lC = nullptr;
    }
    int idx = g_fp8_algo_count;
    g_fp8_algo_cache[g_fp8_algo_count++] = ent;
    return idx;
}

void launch_prefill_warmup_cublas(cudaStream_t stream) {
    cublasHandle_t h = get_cublas(stream);
    (void)get_cublaslt();
    // Allocate a temporary scratch buffer to warm cuBLAS + auto-tune the 4 FP8 Blackwell prefill shapes once at startup!
    void* d_tmp = nullptr;
    size_t bytes = (size_t)6552 * 2048 * 4; // 53.7 MiB (freed immediately after warmup)
    if (cudaMalloc(&d_tmp, bytes) == cudaSuccess && d_tmp) {
        cudaMemsetAsync(d_tmp, 0, 2048, stream);
        float alpha = 1.0f, beta = 0.0f;
        const __half* a = reinterpret_cast<const __half*>(d_tmp);
        const __half* b = a + 256;
        float* c = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(d_tmp) + 1024);
        cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            16, 16, 16,
            &alpha,
            a, CUDA_R_16F, 16,
            b, CUDA_R_16F, 16,
            &beta,
            c, CUDA_R_32F, 16,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        const int fp8_shapes[][3] = {
            {6552, 2048, 2304},
            {2304, 2048, 3072},
            {1024, 2048, 2304},
            {2048, 2048, 2304},
        };
        for (const auto& s : fp8_shapes) {
            cache_fp8_algo_entry(s[0], s[1], s[2], d_tmp, d_tmp, d_tmp, stream);
        }
        cudaStreamSynchronize(stream);
        cudaFree(d_tmp);
    }
}

void launch_prefill_f32_to_f16(const float* src, __half* dst, int count, cudaStream_t stream) {
    p_f32_to_f16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(src, dst, (size_t)count);
}

void launch_prefill_gemm_h(
    const LinearWeight& w,
    const __half* x_h,
    float* y,
    __half* w_f16_scratch,
    int B,
    cudaStream_t stream) {
    // 1. Native Blackwell FP8 Tensor Core cublasLtMatmul (2x TFLOPS, ZERO weight dequant, pre-tuned cached descriptors!)
    if (w.kind == WeightKind::FP8 && B >= 32 && (B & 15) == 0 && (w.cols & 15) == 0 && (w.rows & 3) == 0) {
        cublasLtHandle_t lt = get_cublaslt();
        int cache_idx = cache_fp8_algo_entry(w.rows, B, w.cols, nullptr, nullptr, nullptr, stream);
        if (cache_idx >= 0 && g_fp8_algo_cache[cache_idx].valid) {
            const Fp8AlgoEntry& ent = g_fp8_algo_cache[cache_idx];
            uint8_t* x_e4m3 = reinterpret_cast<uint8_t*>(w_f16_scratch);
            size_t x_bytes = (((size_t)B * w.cols) + 255u) & ~size_t(255u);
            float* x_scale = reinterpret_cast<float*>(x_e4m3 + x_bytes);
            p_quant_f16_to_e4m3_rowwise_kernel<<<B, 256, 0, stream>>>(x_h, x_e4m3, x_scale, w.cols);

            const float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t st = cublasLtMatmul(
                lt, ent.desc, &alpha,
                w.data, ent.lA,
                x_e4m3, ent.lB,
                &beta,
                y, ent.lC,
                y, ent.lC,
                &ent.algo,
                g_cublaslt_ws, CUBLASLT_WS_BYTES, stream);
            if (st == CUBLAS_STATUS_SUCCESS) {
                size_t total_f4 = ((size_t)B * w.rows) >> 2;
                p_scale_y_fp8_kernel<<<(int)((total_f4 + 255) / 256), 256, 0, stream>>>(y, w.row_scales, x_scale, B, w.rows);
                return;
            }
        }
    }

    if (w.kind == WeightKind::NVFP4) {
        size_t total_blocks = (size_t)w.rows * (w.cols >> 4);
        p_dequant_nvfp4_to_f16_kernel<<<(int)((total_blocks + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w_f16_scratch, total_blocks);
    } else if (w.kind == WeightKind::FP8) {
        size_t total_chunks16 = (size_t)w.rows * (w.cols >> 4);
        p_dequant_fp8_to_f16_kernel<<<(int)((total_chunks16 + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, w_f16_scratch, w.rows, w.cols);
    } else {
        size_t total_vec8 = ((size_t)w.rows * w.cols) >> 3;
        p_dequant_bf16_to_f16_kernel<<<(int)((total_vec8 + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), w_f16_scratch, total_vec8);
    }

    float alpha = (w.kind == WeightKind::NVFP4) ? w.global_scale : 1.0f, beta = 0.0f;
    cublasHandle_t h = get_cublas(stream);
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        w.rows, B, w.cols,
        &alpha,
        w_f16_scratch, CUDA_R_16F, w.cols,
        x_h, CUDA_R_16F, w.cols,
        &beta,
        y, CUDA_R_32F, w.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %d (N=%d B=%d K=%d)\n", (int)st, w.rows, B, w.cols);
        std::exit(1);
    }
}

void launch_prefill_sconv_post_res(
    const float* src_or_slots, int is_slotted, const __nv_bfloat16* sconv_w, float* ring,
    const float* residual, float* out_res, float* pre_tap_out,
    float* blocks, const float* partial, int block_idx, int is_boundary,
    int B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, cudaStream_t stream) {
    int write_conv_only = (abl_r != nullptr) ? 1 : 0;
    int ty = (B + SCONV_T_BLK - 1) / SCONV_T_BLK;
    int update_ring = (ty == 1) ? 1 : 0;
    p_sconv_post_res_kernel<<<dim3(9, ty), 64, 0, stream>>>(
        src_or_slots, is_slotted, sconv_w, ring, residual, out_res, pre_tap_out,
        blocks, partial, block_idx, is_boundary, write_conv_only, update_ring, B, d_pos);
    if (!update_ring) {
        p_sconv_ring_commit_kernel<<<9, 64, 0, stream>>>(
            src_or_slots, is_slotted, ring, B, d_pos);
    }
    if (abl_r != nullptr) {
        p_ablate_post_res_kernel<<<B, 576, 0, stream>>>(
            out_res, residual, pre_tap_out, blocks, partial, block_idx, is_boundary,
            abl_r, abl_d, abl_p);
    }
}

void launch_prefill_copy_taps(const float* src, float* dst_taps, int tap_slot, int B, cudaStream_t stream) {
    int total_f4 = (B * HIDDEN) >> 2;
    p_copy_taps_chunked_kernel<<<(total_f4 + 255) / 256, 256, 0, stream>>>(src, dst_taps, tap_slot, B);
}

void launch_prefill_local_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int kv_cap, int B, const int32_t* d_pos, cudaStream_t stream) {
    p_local_sconv_prep_store_kernel<<<dim3(64, B), 128, 0, stream>>>(
        qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_ring, v_ring,
        q_clip_scale, rel_clip_scale, k_cache, v_cache, kv_cap, d_pos);
    p_kv_ring_commit_kernel<<<dim3(KV_DIM / 256, 2), 256, 0, stream>>>(
        k_proj, v_proj, k_ring, v_ring, B, d_pos);
}

void launch_prefill_global_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const __half* in_k_h, const __half* in_v_h,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    const __nv_bfloat16* k_sconv_proj, const __nv_bfloat16* v_sconv_proj,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    void* k_cache, void* v_cache, __half* k_scale, __half* v_scale,
    KVFormat kv_fmt, int kv_cap,
    float* scratch_taps, __half* w_f16_scratch,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    // 1. Batched cuBLAS FP16 TC dynamic tap projections ([32, 2304] x [B, 2304])
    float* k_taps = scratch_taps;
    float* v_taps = scratch_taps + (size_t)B * 32;
    LinearWeight kw = {WeightKind::BF16, 32, HIDDEN, const_cast<__nv_bfloat16*>(k_sconv_proj), nullptr, nullptr, 1.0f};
    LinearWeight vw = {WeightKind::BF16, 32, HIDDEN, const_cast<__nv_bfloat16*>(v_sconv_proj), nullptr, nullptr, 1.0f};
    launch_prefill_gemm_h(kw, in_k_h, k_taps, w_f16_scratch, B, stream);
    launch_prefill_gemm_h(vw, in_v_h, v_taps, w_f16_scratch, B, stream);

    // 2. 1-launch fused 2D-parallel dynamic sconv + Q/R/K clip (with log-pos tau) + V RMSNorm + quantized KV store!
    dim3 grid(64, B);
    if (kv_fmt == KVFormat::FP8) {
        p_global_attn_prep_store_kernel<1><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else if (kv_fmt == KVFormat::Q4) {
        p_global_attn_prep_store_kernel<2><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else {
        p_global_attn_prep_store_kernel<0><<<grid, 128, 0, stream>>>(
            qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_taps, v_taps, k_ring, v_ring,
            q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    }
    p_kv_ring_commit_kernel<<<dim3(KV_DIM / 256, 2), 256, 0, stream>>>(
        k_proj, v_proj, k_ring, v_ring, B, d_pos);
}

void launch_prefill_dspark_commit(
    const DsparkWeights& dw,
    const float* taps,
    void* ctx_k[MTP_LAYERS],
    void* ctx_v[MTP_LAYERS],
    __half* ctx_ks[MTP_LAYERS],
    __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS],
    KVFormat global_kv_fmt,
    float* f_ring,
    __half* mix_h0,
    __half* mix_h1,
    __half* mix_h2,
    float* scratch_kv,
    __half* w_f16_scratch,
    int B,
    const int32_t* d_pos,
    cudaStream_t stream) {
    p_dspark_mix_h_kernel<<<dim3(B, 4), 256, 0, stream>>>(
        taps, B, dw.fuse_alpha,
        dw.layers[0].dflare_norm, dw.layers[1].dflare_norm, dw.layers[2].dflare_norm,
        dw.markov_dflare_norm, mix_h0, mix_h1, mix_h2, f_ring, d_pos);
    __half* mix_h[3] = {mix_h0, mix_h1, mix_h2};
    for (int l = 0; l < MTP_LAYERS; ++l) {
        launch_prefill_gemm_h(dw.layers[l].w_kv_ctx, mix_h[l], scratch_kv, w_f16_scratch, B, stream);
        KVFormat lfmt = dw.layers[l].is_local ? KVFormat::BF16 : global_kv_fmt;
        if (lfmt == KVFormat::FP8) {
            p_dspark_store_ctx_kv_kernel<1><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], B, d_pos);
        } else if (lfmt == KVFormat::Q4) {
            p_dspark_store_ctx_kv_kernel<2><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], B, d_pos);
        } else {
            p_dspark_store_ctx_kv_kernel<0><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], B, d_pos);
        }
    }
}

void launch_prefill_flash_attn_tc(
    const float* qrgs, const __nv_bfloat16* rel_proj,
    const void* k_cache, const void* v_cache,
    const __half* k_scale, const __half* v_scale,
    KVFormat kv_fmt, __half* out_gated_h, __half* scratch_rel,
    int kv_cap, int window, int B, const int32_t* d_pos, cudaStream_t stream) {
    (void)scratch_rel;
    // Q_TOK=32 (96 queries/CTA, 6 warps=192 threads, 22.5 KB SMEM -> 4 blocks/SM, 2x KV reuse!)
    dim3 grid(NUM_KV_HEADS, (B + 31) / 32);
    if (window > 0) {
        p_flash_attn_prefill_tc_kernel<0, 32><<<grid, 192, 0, stream>>>(
            qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, window, B, d_pos);
    } else {
        if (kv_fmt == KVFormat::FP8) {
            p_flash_attn_prefill_tc_kernel<1, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        } else if (kv_fmt == KVFormat::Q4) {
            p_flash_attn_prefill_tc_kernel<2, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        } else {
            p_flash_attn_prefill_tc_kernel<0, 32><<<grid, 192, 0, stream>>>(
                qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, out_gated_h, kv_cap, 0, B, d_pos);
        }
    }
}

void launch_prefill_controller_one(
    const float* x, const float* saved, const float* prior_last, const float* cp_q,
    const ControllerWeights& cw, const __nv_bfloat16* post_norm,
    float* out, __half* out_h,
    float* scratch_msg, __half* scratch_cat_h, float* scratch_d, __half* scratch_d_h,
    float* scratch_f, float* scratch_g, __half* w_f16_scratch,
    int B, cudaStream_t stream) {
    p_ctrl_prep_h_kernel<<<B, 576, 0, stream>>>(x, saved, prior_last, cp_q, scratch_msg, scratch_cat_h);

    LinearWeight down_w = cw.down_nvfp4;
    LinearWeight forget_w = cw.forget_nvfp4;
    LinearWeight gain_w = cw.gain_nvfp4;
    if (down_w.data == nullptr) {
        down_w = {WeightKind::BF16, GATE_RANK, 2 * HIDDEN, cw.down_w, nullptr, nullptr, 1.0f};
        forget_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.forget_w, nullptr, nullptr, 1.0f};
        gain_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.gain_w, nullptr, nullptr, 1.0f};
    }

    launch_prefill_gemm_h(down_w, scratch_cat_h, scratch_d, w_f16_scratch, B, stream);
    launch_prefill_f32_to_f16(scratch_d, scratch_d_h, B * GATE_RANK, stream);
    launch_prefill_gemm_h(forget_w, scratch_d_h, scratch_f, w_f16_scratch, B, stream);
    launch_prefill_gemm_h(gain_w, scratch_d_h, scratch_g, w_f16_scratch, B, stream);

    p_ctrl_finish_kernel<<<B, 576, 0, stream>>>(x, scratch_msg, scratch_f, scratch_g, post_norm, out, out_h);
}

void launch_prefill_route_and_sort(
    const float* router_logits, const float* router_bias,
    int32_t* route_ids, float* route_w,
    int32_t* expert_offsets, int32_t* sorted_tok, int32_t* sorted_slot, float* sorted_w,
    int B, cudaStream_t stream) {
    int TR = (B <= 512) ? 32 : 128;
    p_route_topk_only_kernel<<<B, 32, 0, stream>>>(router_logits, router_bias, route_ids, route_w);
    p_sort_experts_kernel<<<1, NUM_EXPERTS, 0, stream>>>(
        route_ids, route_w, B, TR, expert_offsets, sorted_tok, sorted_slot, sorted_w);
}

void launch_prefill_moe_gate_up_h(
    const ExpertBank& w13, const __half* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    __half* out_act_h, void* scratch_q, int B, cudaStream_t stream) {
    if (w13.is_iq3) {
        if (B <= 512) dispatch_wgemm_tm<2, WgemmMode::Up, HIDDEN, 32>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        else          dispatch_wgemm_tm<2, WgemmMode::Up, HIDDEN, 128>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        return;
    }
    if (scratch_q != nullptr) {
        if (w13.is_nvfp4) {
            if (B <= 512) dispatch_wgemm_blkscale_tm<0, WgemmMode::Up, HIDDEN, 32>(w13, x_h, B, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, scratch_q, stream);
            else          dispatch_wgemm_blkscale_tm<0, WgemmMode::Up, HIDDEN, 128>(w13, x_h, B, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, scratch_q, stream);
        } else {
            if (B <= 512) dispatch_wgemm_blkscale_tm<1, WgemmMode::Up, HIDDEN, 32>(w13, x_h, B, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, scratch_q, stream);
            else          dispatch_wgemm_blkscale_tm<1, WgemmMode::Up, HIDDEN, 128>(w13, x_h, B, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, scratch_q, stream);
        }
        return;
    }
    if (w13.is_nvfp4) {
        if (B <= 512) dispatch_wgemm_tm<0, WgemmMode::Up, HIDDEN, 32>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        else          dispatch_wgemm_tm<0, WgemmMode::Up, HIDDEN, 128>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
    } else {
        if (B <= 512) dispatch_wgemm_tm<1, WgemmMode::Up, HIDDEN, 32>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
        else          dispatch_wgemm_tm<1, WgemmMode::Up, HIDDEN, 128>(w13, x_h, expert_offsets, sorted_tok, sorted_slot, nullptr, out_act_h, nullptr, stream);
    }
}

void launch_prefill_moe_down_h(
    const ExpertBank& w2, const __half* act_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    const float* sorted_w, float* out_slots, void* scratch_q, int B, cudaStream_t stream) {
    if (scratch_q != nullptr) {
        // L2-resident 18.8 MB atomic accumulation (8x less DRAM traffic than 151 MB slotted write/read!)
        cudaMemsetAsync(out_slots, 0, (size_t)B * HIDDEN * sizeof(float), stream);
        int M_rows = B * TOP_K;
        if (w2.is_nvfp4) {
            if (B <= 512) dispatch_wgemm_blkscale_tm<0, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, M_rows, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, scratch_q, stream);
            else          dispatch_wgemm_blkscale_tm<0, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, M_rows, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, scratch_q, stream);
        } else {
            if (B <= 512) dispatch_wgemm_blkscale_tm<1, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, M_rows, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, scratch_q, stream);
            else          dispatch_wgemm_blkscale_tm<1, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, M_rows, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, scratch_q, stream);
        }
        return;
    }
    if (w2.is_nvfp4) {
        if (B <= 512) dispatch_wgemm_tm<0, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, stream);
        else          dispatch_wgemm_tm<0, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, stream);
    } else {
        if (B <= 512) dispatch_wgemm_tm<1, WgemmMode::Down, INTERMEDIATE, 32>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, stream);
        else          dispatch_wgemm_tm<1, WgemmMode::Down, INTERMEDIATE, 128>(w2, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots, stream);
    }
}

} // namespace squid
