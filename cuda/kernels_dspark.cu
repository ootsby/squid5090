#include "kernels.hpp"
#include "kernels_common.cuh"

namespace squid {

namespace {

// Vectorized 576-thread (float4, zero SMEM buf, PDL) DSpark tap mixer:
// Mix 4 target taps [4, B, HIDDEN] using fuse_alpha[m, 0..3], RMSNorm + dflare_norm
// m = 0,1,2 -> draft layers 0..2 (writes __half to out_mix_h when non-null, else float to out_mix)
// m = 3 -> markov f stored in FP32 into f_ring[(pos+b) & 15]
__global__ void __launch_bounds__(576)
dspark_mix_kernel(
    const float* __restrict__ taps,       // [4, B, HIDDEN]
    int B,
    const float* __restrict__ fuse_alpha, // [4, 4]
    const __nv_bfloat16* __restrict__ n0,
    const __nv_bfloat16* __restrict__ n1,
    const __nv_bfloat16* __restrict__ n2,
    const __nv_bfloat16* __restrict__ n3,
    float* __restrict__ out_mix,          // [3, B, HIDDEN] (nullptr when out_mix_h used)
    __half* __restrict__ out_mix_h,       // [3, B, HIDDEN] (nullptr for BF16 weights)
    float* __restrict__ f_ring,           // [16, HIDDEN]
    const int32_t* __restrict__ d_pos) {
    const int b = blockIdx.x;
    const int m = blockIdx.y; // 0..3
    const int tid = threadIdx.x; // 0..575
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c4 = tid << 2;

    // Prefetch read-only weights before PDL wait
    const float4 af = *reinterpret_cast<const float4*>(fuse_alpha + m * 4);
    const __nv_bfloat16* nw = (m == 0) ? n0 : (m == 1 ? n1 : (m == 2 ? n2 : n3));
    const uint2 nw4 = __ldg(reinterpret_cast<const uint2*>(nw + c4));
    const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
    const float2 nf0 = __bfloat1622float2(nb[0]), nf1 = __bfloat1622float2(nb[1]);

    pdl_wait_fenced(0u);
    pdl_trigger();

    const float4 t0 = ld_dep_f4(taps + ((size_t)0 * B + b) * HIDDEN + c4);
    const float4 t1 = ld_dep_f4(taps + ((size_t)1 * B + b) * HIDDEN + c4);
    const float4 t2 = ld_dep_f4(taps + ((size_t)2 * B + b) * HIDDEN + c4);
    const float4 t3 = ld_dep_f4(taps + ((size_t)3 * B + b) * HIDDEN + c4);

    float4 v;
    v.x = fmaf(af.x, t0.x, fmaf(af.y, t1.x, fmaf(af.z, t2.x, af.w * t3.x)));
    v.y = fmaf(af.x, t0.y, fmaf(af.y, t1.y, fmaf(af.z, t2.y, af.w * t3.y)));
    v.z = fmaf(af.x, t0.z, fmaf(af.y, t1.z, fmaf(af.z, t2.z, af.w * t3.z)));
    v.w = fmaf(af.x, t0.w, fmaf(af.y, t1.w, fmaf(af.z, t2.w, af.w * t3.w)));

    float sum_sq = warp_sum(fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w))));
    __shared__ float sm[18];
    __shared__ float inv_rms;
    if (lane == 0) sm[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    const float s = inv_rms;
    v.x *= s * nf0.x;
    v.y *= s * nf0.y;
    v.z *= s * nf1.x;
    v.w *= s * nf1.y;

    if (m == 3) {
        *reinterpret_cast<float4*>(f_ring + (size_t)(((*d_pos) + b) & 15) * HIDDEN + c4) = v;
    } else if (out_mix_h) {
        __half2 h0 = __floats2half2_rn(v.x, v.y);
        __half2 h1 = __floats2half2_rn(v.z, v.w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&h0);
        packed.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(out_mix_h + ((size_t)m * B + b) * HIDDEN + c4) = packed;
    } else {
        *reinterpret_cast<float4*>(out_mix + ((size_t)m * B + b) * HIDDEN + c4) = v;
    }
}

// NeoX RoPE (theta = 1e6) + per-head RMSNorm -> store ctx K & V into draft KV cache (BF16/FP8/Q4!)
// Input kv [B, 2*KV_DIM] (K at 0..1023, V at 1024..2047)
template <int KV_FMT>
__global__ void __launch_bounds__(128)
dspark_store_ctx_kv_kernel(
    const float* __restrict__ kv,
    const __nv_bfloat16* __restrict__ k_norm, // [128]
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int cap,
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int h = blockIdx.y; // 0..7
    int tid = threadIdx.x; // 0..127 (full head_dim!)
    float knw = __bfloat162float(__ldg(k_norm + tid));
    pdl_wait_fenced(0u);
    pdl_trigger();
    int pos = (*d_pos) + b;
    int slot = pos % cap;

    const float* kp = kv + (size_t)b * (2 * KV_DIM) + h * HEAD_DIM;
    const float* vp = kv + (size_t)b * (2 * KV_DIM) + KV_DIM + h * HEAD_DIM;

    __shared__ float sm[4];
    // Store V
    store_kv_head_128<KV_FMT>(ld_dep_f32(vp + tid), slot, h, tid, v_cache, v_scale, sm);
    __syncthreads();

    // RMSNorm K per head
    float kv_val = ld_dep_f32(kp + tid);
    float sq = warp_sum(kv_val * kv_val);
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    }
    __syncthreads();

    __shared__ float kn[HEAD_DIM];
    kn[tid] = kv_val * inv_rms * knw;
    __syncthreads();

    // NeoX RoPE: pair (i, i + 64) for i in 0..63
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

// Initialize DSpark 8-row block: row 0 = embed_norm(anchor), rows 1..7 = mask_embed
__global__ void __launch_bounds__(576)
dspark_init_block_kernel(
    const __nv_bfloat16* __restrict__ embed,
    const __nv_bfloat16* __restrict__ embed_norm,
    const __nv_bfloat16* __restrict__ mask_embed,
    const int32_t* __restrict__ d_anchor,
    float* __restrict__ block_x) { // [8, HIDDEN]
    int r = blockIdx.x; // 0..7
    int tid = threadIdx.x; // 0..575
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;
    float* dst = block_x + (size_t)r * HIDDEN + c4;
    if (r == 0) {
        uint2 nw4 = __ldg(reinterpret_cast<const uint2*>(embed_norm + c4));
        const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
        float2 nf0 = __bfloat1622float2(nb[0]), nf1 = __bfloat1622float2(nb[1]);
        pdl_wait_fenced(0u);
        pdl_trigger();
        int tok = ld_dep_i32(d_anchor);
        uint2 ew4 = __ldg(reinterpret_cast<const uint2*>(embed + (size_t)tok * HIDDEN + c4));
        const __nv_bfloat162* eb = reinterpret_cast<const __nv_bfloat162*>(&ew4);
        float2 ef0 = __bfloat1622float2(eb[0]), ef1 = __bfloat1622float2(eb[1]);
        float sum_sq = warp_sum(fmaf(ef0.x, ef0.x, fmaf(ef0.y, ef0.y, fmaf(ef1.x, ef1.x, ef1.y * ef1.y))));
        __shared__ float sm[18];
        __shared__ float inv_rms;
        if (lane == 0) sm[warp] = sum_sq;
        __syncthreads();
        if (warp == 0) {
            float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
            if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
        }
        __syncthreads();
        float s = inv_rms;
        *reinterpret_cast<float4*>(dst) = make_float4(
            ef0.x * s * nf0.x, ef0.y * s * nf0.y, ef1.x * s * nf1.x, ef1.y * s * nf1.y);
    } else {
        uint2 mw4 = __ldg(reinterpret_cast<const uint2*>(mask_embed + c4));
        const __nv_bfloat162* mb = reinterpret_cast<const __nv_bfloat162*>(&mw4);
        float2 mf0 = __bfloat1622float2(mb[0]), mf1 = __bfloat1622float2(mb[1]);
        pdl_wait_fenced(0u);
        pdl_trigger();
        *reinterpret_cast<float4*>(dst) = make_float4(mf0.x, mf0.y, mf1.x, mf1.y);
    }
}

// DSpark Q/K RMSNorm + NeoX RoPE on the 8-row block (positions p..p+7)
// qkv layout: [8, 5120] (Q: 0..3071, K: 3072..4095, V: 4096..5119)
__global__ void __launch_bounds__(128)
dspark_block_qk_rope_kernel(
    float* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm,
    const int32_t* __restrict__ d_pos) {
    int r = blockIdx.x; // 0..7
    int h = blockIdx.y; // 0..23: Q, 24..31: K
    int tid = threadIdx.x; // 0..127
    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;

    float* vec = (h < 24)
        ? (qkv + (size_t)r * QKV_STRIDE + h * HEAD_DIM)
        : (qkv + (size_t)r * QKV_STRIDE + Q_DIM + (h - 24) * HEAD_DIM);
    float nw = __bfloat162float(__ldg(((h < 24) ? q_norm : k_norm) + tid));
    pdl_wait_fenced(0u);
    pdl_trigger();
    int pos = (*d_pos) + r;

    float val = ld_dep_f32(vec + tid);
    float sq = warp_sum(val * val);
    __shared__ float sm[4];
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    __syncthreads();

    __shared__ float vn[HEAD_DIM];
    vn[tid] = val * inv_rms * nw;
    __syncthreads();

    if (tid < 64) {
        float freq = 1.0f / powf(MTP_ROPE_THETA, float(2 * tid) / float(HEAD_DIM));
        float ang = float(pos) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = vn[tid], x1 = vn[tid + 64];
        vec[tid]      = x0 * c - x1 * s;
        vec[tid + 64] = x0 * s + x1 * c;
    }
}

// Fast 576-thread vectorized float4 PDL RMSNorm producing __half directly for DSpark's linear_h layers!
__global__ void __launch_bounds__(576)
dspark_rmsnorm_h_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out_f,
    __half* __restrict__ out_h) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;

    uint2 nw4 = __ldg(reinterpret_cast<const uint2*>(w + c4));
    const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
    float2 nf0 = __bfloat1622float2(nb[0]), nf1 = __bfloat1622float2(nb[1]);

    pdl_wait_fenced(0u);
    pdl_trigger();

    float4 xv = ld_dep_f4(x + (size_t)r * HIDDEN + c4);
    float sq = warp_sum(fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, xv.w * xv.w))));
    __shared__ float sm[18];
    __shared__ float inv_rms;
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ir = inv_rms;
    float4 ov = make_float4(xv.x * ir * nf0.x, xv.y * ir * nf0.y, xv.z * ir * nf1.x, xv.w * ir * nf1.y);
    __half2 h0 = __floats2half2_rn(ov.x, ov.y);
    __half2 h1 = __floats2half2_rn(ov.z, ov.w);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&h0);
    packed.y = *reinterpret_cast<uint32_t*>(&h1);
    *reinterpret_cast<uint2*>(out_h + (size_t)r * HIDDEN + c4) = packed;
    if (out_f) *reinterpret_cast<float4*>(out_f + (size_t)r * HIDDEN + c4) = ov;
}

// 256-CTA Blackwell Tensor Core (m16n8k16) Split-K FlashDecoding for DSpark Global Layer 1!
// Reads each global context KV token ONCE across all 24 DSpark queries (24x less BW, ~350 TFLOPS TC!).
template <int KV_FMT>
__global__ void __launch_bounds__(128)
dspark_global_attn_bulk_splitk_kernel(
    const float* __restrict__ qkv, // [8, 5120]
    const void* __restrict__ ck,
    const void* __restrict__ cv,
    const __half* __restrict__ cks,
    const __half* __restrict__ cvs,
    int cap,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ split_buf) {
    pdl_wait();
    pdl_trigger();
    int p = *d_pos;
    if (p <= 0) return;

    // Reserve MTP_MAX_DRAFTS (7) slots of ring slack so rejected future draft tokens
    // written ahead of *d_pos are never read as valid history when p > cap.
    int max_win = max(0, cap - MTP_MAX_DRAFTS);
    int first = (p > max_win) ? (p - max_win) : 0;
    int total_len = p - first;
    int split = blockIdx.y;
    int chunk = ((total_len + KV_SPLIT_K - 1) / KV_SPLIT_K + 31) & ~31;
    int t_start = first + split * chunk;
    if (t_start >= p) return;
    int t_end = min(p, t_start + chunk);

    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;
    run_tc_bulk_splitk<KV_FMT, 24, QKV_STRIDE, 128>(
        qkv, ck, cv, cks, cvs, cap, blockIdx.x, split, t_start, t_end, split_buf);
}

constexpr int DSPARK_LOCAL_SPLITS = 8;

// 64-CTA Blackwell Tensor Core (m16n8k16) Split-K FlashDecoding for DSpark Local Layers 0 & 2!
// Reads each of the <=1024 BF16 local KV tokens ONCE across all 24 DSpark queries (24x less BW!).
__global__ void __launch_bounds__(128)
dspark_local_attn_bulk_splitk_kernel(
    const float* __restrict__ qkv, // [8, 5120]
    const void* __restrict__ ck,
    const void* __restrict__ cv,
    int cap,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ split_buf) {
    pdl_wait();
    pdl_trigger();
    int p = *d_pos;
    if (p <= 0) return;

    int first = (p > SWA_WINDOW) ? (p - SWA_WINDOW) : 0;
    int total_len = p - first;
    int split = blockIdx.y;
    int chunk = ((total_len + DSPARK_LOCAL_SPLITS - 1) / DSPARK_LOCAL_SPLITS + 31) & ~31;
    int t_start = first + split * chunk;
    if (t_start >= p) return;
    int t_end = min(p, t_start + chunk);

    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;
    run_tc_bulk_splitk<0, 24, QKV_STRIDE, 128>(
        qkv, ck, cv, nullptr, nullptr, cap, blockIdx.x, split, t_start, t_end, split_buf);
}

// DSpark Block Attention writing __half directly for w_o linear_h!
// Both local (8 TC splits) and global (64 TC splits) layers attend to the 8 bidirectional block rows
// and merge ONLY the active 32-aligned TC bulk Split-K partials in one fast PDL launch!
__global__ void __launch_bounds__(256)
dspark_block_attn_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ split_buf,
    int num_splits,
    int max_win,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ out,
    __half* __restrict__ out_h) {
    int h = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int kvh = h / GQA_RATIO;
    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;

    pdl_wait();
    pdl_trigger();

    int p = *d_pos;
    int total_len = (p > max_win) ? max_win : p;
    int chunk = (total_len > 0 && num_splits > 0) ? (((total_len + num_splits - 1) / num_splits + 31) & ~31) : 32;
    int active_splits = (split_buf != nullptr && total_len > 0 && num_splits > 0)
        ? min(num_splits, (total_len + chunk - 1) / chunk)
        : 0;
    bool has_bulk = (active_splits > 0);

    int d_off = lane * 4;
    const float* qp = qkv + (size_t)r * QKV_STRIDE + h * HEAD_DIM;
    float4 qv = *reinterpret_cast<const float4*>(qp + d_off);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float m_val = -1e30f, d_val = 0.0f;

    // Bidirectional block KV (all 8 rows: warp `w` handles block row `w`!)
    {
        float4 bkv = *reinterpret_cast<const float4*>(qkv + (size_t)warp * QKV_STRIDE + Q_DIM + kvh * HEAD_DIM + d_off);
        float4 bvv = *reinterpret_cast<const float4*>(qkv + (size_t)warp * QKV_STRIDE + Q_DIM + KV_DIM + kvh * HEAD_DIM + d_off);
        float dot = fmaf(qv.x, bkv.x, fmaf(qv.y, bkv.y, fmaf(qv.z, bkv.z, qv.w * bkv.w)));
        float score = warp_sum(dot) * INV_SQRT_HEAD_DIM;
        m_val = score;
        d_val = 1.0f;
        acc0 = bvv.x;
        acc1 = bvv.y;
        acc2 = bvv.z;
        acc3 = bvv.w;
    }

    __shared__ float sm_m[8], sm_d[8], sm_acc[8][HEAD_DIM];
    if (lane == 0) { sm_m[warp] = m_val; sm_d[warp] = d_val; }
    sm_acc[warp][d_off + 0] = acc0;
    sm_acc[warp][d_off + 1] = acc1;
    sm_acc[warp][d_off + 2] = acc2;
    sm_acc[warp][d_off + 3] = acc3;
    __syncthreads();

    if (tid < HEAD_DIM) {
        float g_m = sm_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) g_m = fmaxf(g_m, sm_m[w]);

        int base_hr = (h * MAX_VERIFY_B + r) * KV_TOTAL_SPLITS;
        const float* s_m_in = split_buf + SPLIT_M_OFF;
        const float* s_d_in = split_buf + SPLIT_D_OFF;
        const float* s_o_in = split_buf + SPLIT_O_OFF;
        if (has_bulk) {
            for (int s = 0; s < active_splits; ++s) {
                g_m = fmaxf(g_m, s_m_in[base_hr + s]);
            }
        }

        float g_d = 0.0f, g_o = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (sm_d[w] > 0.0f) {
                float sc = __expf(sm_m[w] - g_m);
                g_d += sm_d[w] * sc;
                g_o += sm_acc[w][tid] * sc;
            }
        }
        if (has_bulk) {
            for (int s = 0; s < active_splits; ++s) {
                float sd = s_d_in[base_hr + s];
                if (sd > 0.0f) {
                    float sc = __expf(s_m_in[base_hr + s] - g_m);
                    g_d += sd * sc;
                    g_o += s_o_in[(size_t)(base_hr + s) * HEAD_DIM + tid] * sc;
                }
            }
        }
        float val = g_o / g_d;
        size_t oidx = (size_t)r * Q_DIM + h * HEAD_DIM + tid;
        if (out) out[oidx] = val;
        if (out_h) out_h[oidx] = __float2half(val);
    }
}

// Standard SwiGLU for DSpark MLP writing __half directly for w_down linear_h!
__global__ void dspark_swiglu_kernel(const float* __restrict__ gu, float* __restrict__ out, __half* __restrict__ out_h, int B) {
    pdl_wait();
    pdl_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * MTP_INTERMEDIATE;
    if (idx >= total) return;
    int b = idx / MTP_INTERMEDIATE;
    int j = idx % MTP_INTERMEDIATE;
    float g = ld_dep_f32(gu + (size_t)b * (2 * MTP_INTERMEDIATE) + j);
    float u = ld_dep_f32(gu + (size_t)b * (2 * MTP_INTERMEDIATE) + MTP_INTERMEDIATE + j);
    float v = (g * fast_sigmoid(g)) * u;
    if (out) out[idx] = v;
    if (out_h) out_h[idx] = __float2half(v);
}

// Coalesced 24-block Markov RNN projection:
// raw[768] = rnn_bias + c_h[i] + c_f + rnn_w_sp @ [state; w1[prev_tok]]
// Grid: 24 blocks, Block: 128 threads (4 warps/block -> 8 rows/warp = 768 rows, 100% coalesced uint4!)
__global__ void __launch_bounds__(128)
markov_rnn_proj_kernel(
    const __nv_bfloat16* __restrict__ rnn_w_sp,  // [768, 512]
    const __nv_bfloat16* __restrict__ rnn_bias,  // [768]
    const __nv_bfloat16* __restrict__ markov_w1, // [201024, 256]
    const float* __restrict__ c_h_i,             // [768]
    const float* __restrict__ c_f,               // [768]
    const int32_t* __restrict__ prev_tok_ptr,    // [1]
    const float* __restrict__ state,             // [256]
    float* __restrict__ out_raw) {               // [768]
    pdl_wait();
    pdl_trigger();
    __shared__ float sp[512];
    int tid = threadIdx.x;
    int prev_tok = ld_dep_i32(prev_tok_ptr);
    const __nv_bfloat16* w1_row = markov_w1 + (size_t)prev_tok * MTP_MARKOV_RANK;
    sp[tid]       = ld_dep_f32(state + tid);
    sp[128 + tid] = ld_dep_f32(state + 128 + tid);
    sp[256 + tid] = __bfloat162float(w1_row[tid]);
    sp[384 + tid] = __bfloat162float(w1_row[128 + tid]);
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int base_r = blockIdx.x * 32 + warp * 8;
    #pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
        int r = base_r + rr;
        const uint4* wr = reinterpret_cast<const uint4*>(rnn_w_sp + (size_t)r * 512);
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            int v = lane + iter * 32; // 0..63 (64 * 8 = 512)
            uint4 w8 = __ldg(wr + v);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
            const float4* xf = reinterpret_cast<const float4*>(sp + (v << 3));
            float4 x0 = xf[0], x1 = xf[1];
            acc = fmaf(wf0.x, x0.x, acc); acc = fmaf(wf0.y, x0.y, acc);
            acc = fmaf(wf1.x, x0.z, acc); acc = fmaf(wf1.y, x0.w, acc);
            acc = fmaf(wf2.x, x1.x, acc); acc = fmaf(wf2.y, x1.y, acc);
            acc = fmaf(wf3.x, x1.z, acc); acc = fmaf(wf3.y, x1.w, acc);
        }
        float val = warp_sum(acc);
        if (lane == 0) {
            out_raw[r] = val + __bfloat162float(rnn_bias[r]) + c_h_i[r] + c_f[r];
        }
    }
}

// Dynamic context shortlist gather: copies `count` rows from main unembed into unembed_draft[dst_offset..]
// Works for FP8 (KIND=1), NVFP4 (KIND=2), and BF16 (KIND=0) with zero re-quantization!
template <int KIND>
__global__ void __launch_bounds__(256)
dspark_gather_shortlist_kernel(
    const void* __restrict__ src_data,
    const uint8_t* __restrict__ src_scales,
    const float* __restrict__ src_rscales,
    void* __restrict__ dst_data,
    uint8_t* __restrict__ dst_scales,
    float* __restrict__ dst_rscales,
    int32_t* __restrict__ draft_vocab_ids,
    int32_t* __restrict__ d_draft_vocab_size,
    const int32_t* __restrict__ d_new_ids,
    int dst_offset,
    int count) {
    int idx = blockIdx.x;
    if (idx >= count) return;
    int tid = threadIdx.x;
    int tok = d_new_ids[idx];
    int dst_row = dst_offset + idx;

    if constexpr (KIND == 1) {
        // FP8: 2304 bytes = 144 uint4s + 1 float row_scale
        if (tid < (HIDDEN >> 4)) {
            const uint4* s = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(src_data) + (size_t)tok * HIDDEN);
            uint4* d = reinterpret_cast<uint4*>(static_cast<uint8_t*>(dst_data) + (size_t)dst_row * HIDDEN);
            d[tid] = s[tid];
        }
        if (tid == 0) dst_rscales[dst_row] = src_rscales[tok];
    } else if constexpr (KIND == 2) {
        // NVFP4: 1152 code bytes = 72 uint4s + 144 scale bytes = 9 uint4s
        if (tid < (HIDDEN >> 5)) {
            const uint4* s = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(src_data) + (size_t)tok * (HIDDEN >> 1));
            uint4* d = reinterpret_cast<uint4*>(static_cast<uint8_t*>(dst_data) + (size_t)dst_row * (HIDDEN >> 1));
            d[tid] = s[tid];
        }
        if (tid < (HIDDEN >> 8)) {
            const uint4* ss = reinterpret_cast<const uint4*>(src_scales + (size_t)tok * (HIDDEN >> 4));
            uint4* ds = reinterpret_cast<uint4*>(dst_scales + (size_t)dst_row * (HIDDEN >> 4));
            ds[tid] = ss[tid];
        }
    } else {
        // BF16: 4608 bytes = 288 uint4s
        const uint4* s = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(src_data) + (size_t)tok * HIDDEN);
        uint4* d = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(dst_data) + (size_t)dst_row * HIDDEN);
        for (int i = tid; i < (HIDDEN >> 3); i += 256) d[i] = s[i];
    }

    if (tid == 0) {
        if (draft_vocab_ids) draft_vocab_ids[dst_row] = tok;
        if (idx == 0 && d_draft_vocab_size) *d_draft_vocab_size = dst_offset + count;
    }
}

__global__ void dspark_set_vocab_size_kernel(int32_t* d_draft_vocab_size, int new_size) {
    if (threadIdx.x == 0 && d_draft_vocab_size) *d_draft_vocab_size = new_size;
}

// Persistent 680-block SMEM NVFP4 / FP8 base_logits[K, V_stride] = unembed_draft @ normed_h (dynamic V!)
constexpr int DSPARK_UNEMBED_BLOCKS = 680;

template <int K_CONST>
__global__ void __launch_bounds__(256)
dspark_base_logits_nvfp4_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float gscale,
    const __half* __restrict__ normed_h,
    float* __restrict__ out_base,
    int V_stride,
    const int32_t* __restrict__ d_active_v) {
    __shared__ __half s_xh[K_CONST * HIDDEN];
    int tid = threadIdx.x;
    int total_x = K_CONST * HIDDEN;
    for (int i = tid; i < total_x; i += 256) s_xh[i] = normed_h[i];
    __syncthreads();

    const int V = d_active_v ? *d_active_v : V_stride;
    int hw = tid >> 4;
    int lane = tid & 15;
    int global_r = blockIdx.x * 16 + hw;
    int total_r = gridDim.x * 16;

    for (int row = global_r; row < V; row += total_r) {
        const uint2* wr = reinterpret_cast<const uint2*>(w_data + (size_t)row * (HIDDEN >> 1));
        const uint8_t* sr = w_scales + (size_t)row * (HIDDEN >> 4);
        float acc[K_CONST];
        #pragma unroll
        for (int k = 0; k < K_CONST; ++k) acc[k] = 0.0f;

        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 16;
            uint2 w8 = __ldg(wr + blk);
            float sc = cvt_e4m3_to_float(__ldg(sr + blk));
            __half2 wh[8];
            cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
            cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
            int koff = blk << 4;
            #pragma unroll
            for (int k = 0; k < K_CONST; ++k) {
                const __half2* xb = reinterpret_cast<const __half2*>(s_xh + k * HIDDEN + koff);
                __half2 p0 = __hmul2(wh[0], xb[0]), p1 = __hmul2(wh[1], xb[1]);
                p0 = __hfma2(wh[2], xb[2], p0); p1 = __hfma2(wh[3], xb[3], p1);
                p0 = __hfma2(wh[4], xb[4], p0); p1 = __hfma2(wh[5], xb[5], p1);
                p0 = __hfma2(wh[6], xb[6], p0); p1 = __hfma2(wh[7], xb[7], p1);
                float2 pf = __half22float2(__hadd2(p0, p1));
                acc[k] = fmaf(pf.x + pf.y, sc, acc[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < K_CONST; ++k) {
            float val = half_warp_sum(acc[k]) * gscale;
            if (lane == 0) out_base[(size_t)k * V_stride + row] = val;
        }
    }
}

template <int K_CONST>
__global__ void __launch_bounds__(192)
dspark_base_logits_nvfp4_mma_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float gscale,
    const __half* __restrict__ normed_h,
    float* __restrict__ out_base,
    int V_stride,
    const int32_t* __restrict__ d_active_v,
    uint32_t zero) {
    constexpr int SPLITK = 6;
    constexpr int U = 3;
    constexpr int CB = HIDDEN >> 1;
    constexpr int SB = HIDDEN >> 4;
    __shared__ float cs[SPLITK][16][8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c_base = warp * (4 * U) + t;
    const bool lx = (g < K_CONST);
    const __half* xg = lx ? (normed_h + (size_t)g * HIDDEN) : nullptr;
    const int V = d_active_v ? *d_active_v : V_stride;

    auto decode32 = [](uint4 q, uint16_t s, uint32_t (&o)[16]) {
        const __half2 sc = cvt_e4m3x2_to_half2(s);
        const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
        const uint32_t words[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            __half2 h0, h1, h2, h3;
            cvt_e2m1x8_to_half2x4(words[w], h0, h1, h2, h3);
            const __half2 sm = (w < 2) ? slo : shi;
            __half2 p0 = __hmul2(h0, sm), p1 = __hmul2(h1, sm), p2 = __hmul2(h2, sm), p3 = __hmul2(h3, sm);
            o[4 * w + 0] = *reinterpret_cast<uint32_t*>(&p0);
            o[4 * w + 1] = *reinterpret_cast<uint32_t*>(&p1);
            o[4 * w + 2] = *reinterpret_cast<uint32_t*>(&p2);
            o[4 * w + 3] = *reinterpret_cast<uint32_t*>(&p3);
        }
    };

    bool waited = false;
    const int n_tiles = (V + 15) / 16;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int r0 = tile * 16;
        const int ra = r0 + g, rb = r0 + g + 8;
        const bool la = (ra < V), lb = (rb < V);

        uint4 aq[U], bq[U];
        uint16_t as[U], bs[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int c = c_base + 4 * u;
            if (la) {
                aq[u] = ld_stream_u4(w_data + (size_t)ra * CB + (size_t)c * 16);
                as[u] = ld_scale_u16(w_scales + (size_t)ra * SB + (size_t)c * 2);
            } else { aq[u] = make_uint4(0, 0, 0, 0); as[u] = 0; }
            if (lb) {
                bq[u] = ld_stream_u4(w_data + (size_t)rb * CB + (size_t)c * 16);
                bs[u] = ld_scale_u16(w_scales + (size_t)rb * SB + (size_t)c * 2);
            } else { bq[u] = make_uint4(0, 0, 0, 0); bs[u] = 0; }
        }
        if (!waited) {
            pdl_wait_fenced(zero);
            waited = true;
        }
        uint32_t xw[U][16];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int col = (c_base + 4 * u) * 32;
            if (lx) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    uint4 q = ld_dep_u4(xg + col + 8 * v);
                    xw[u][4 * v + 0] = q.x; xw[u][4 * v + 1] = q.y;
                    xw[u][4 * v + 2] = q.z; xw[u][4 * v + 3] = q.w;
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) xw[u][v] = 0;
            }
        }
        uint32_t dep = 0;
        #pragma unroll
        for (int u = 0; u < U; ++u) dep ^= aq[u].x ^ aq[u].w ^ bq[u].x ^ bq[u].w ^ xw[u][0];
        dep &= zero;

        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            uint32_t a[16], b[16];
            decode32(aq[u], as[u], a);
            decode32(bq[u], bs[u], b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                     a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                     xw[u][2 * i] ^ dep, xw[u][2 * i + 1],
                                     d[0], d[1], d[2], d[3]);
            }
        }
        cs[warp][g][2 * t]     = d[0];
        cs[warp][g][2 * t + 1] = d[1];
        cs[warp][g + 8][2 * t]     = d[2];
        cs[warp][g + 8][2 * t + 1] = d[3];
        __syncthreads();

        for (int idx = tid; idx < 16 * K_CONST; idx += 192) {
            int r = idx / K_CONST;
            int k = idx % K_CONST;
            int row = r0 + r;
            if (row < V) {
                float s = cs[0][r][k] + cs[1][r][k] + cs[2][r][k]
                        + cs[3][r][k] + cs[4][r][k] + cs[5][r][k];
                out_base[(size_t)k * V_stride + row] = s * gscale;
            }
        }
        __syncthreads();
    }
    if (!waited) pdl_wait_fenced(zero);
    pdl_trigger();
}

// Persistent Half-Warp FP8 Draft Unembed for K = 1
__global__ void __launch_bounds__(256)
dspark_base_logits_fp8_b1_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ normed_h,
    float* __restrict__ out_base,
    int V_stride,
    const int32_t* __restrict__ d_active_v) {
    __shared__ __half s_xh[HIDDEN];
    int tid = threadIdx.x;
    for (int i = tid; i < HIDDEN; i += 256) s_xh[i] = normed_h[i];
    __syncthreads();

    const int V = d_active_v ? *d_active_v : V_stride;
    int hw = tid >> 4;
    int lane = tid & 15;
    int global_r = blockIdx.x * 16 + hw;
    int total_r = gridDim.x * 16;

    for (int row = global_r; row < V; row += total_r) {
        const uint4* wr = reinterpret_cast<const uint4*>(w_data + (size_t)row * HIDDEN);
        uint4 wv[9];
        #pragma unroll
        for (int p = 0; p < 9; ++p) wv[p] = ld_stream_u4(wr + lane + p * 16);
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int koff = (lane + iter * 16) << 4;
            const __half2* xb = reinterpret_cast<const __half2*>(s_xh + koff);
            const uint32_t ws[4] = {wv[iter].x, wv[iter].y, wv[iter].z, wv[iter].w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf0 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] & 0xFFFFu)));
                float2 wf1 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] >> 16)));
                float2 xf0 = __half22float2(xb[2 * j + 0]);
                float2 xf1 = __half22float2(xb[2 * j + 1]);
                acc = fmaf(wf0.x, xf0.x, acc); acc = fmaf(wf0.y, xf0.y, acc);
                acc = fmaf(wf1.x, xf1.x, acc); acc = fmaf(wf1.y, xf1.y, acc);
            }
        }
        float val = half_warp_sum(acc) * __ldg(row_scales + row);
        if (lane == 0) out_base[row] = val;
    }
}

// High-Occupancy 3 KB SMEM L1-Cached 16-Col U=3 FP8 Tensor Core Draft Unembed for K = 2..7!
template <int K_CONST>
__global__ void __launch_bounds__(192)
dspark_base_logits_fp8_mma_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ normed_h,
    float* __restrict__ out_base,
    int V_stride,
    const int32_t* __restrict__ d_active_v,
    uint32_t zero) {
    constexpr int SPLITK = 6;
    constexpr int CPL = 16;
    constexpr int SPAN = 4 * CPL;       // 64
    constexpr int KS = HIDDEN / SPLITK; // 384
    constexpr int NJ = KS / SPAN;       // 6 (2 batches of U=3)
    constexpr int U = 3;
    __shared__ float cs[SPLITK][16][8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = warp * KS + t * CPL;
    const bool lx = (g < K_CONST);
    const __half* xg = lx ? (normed_h + (size_t)g * HIDDEN) : nullptr;
    const int V = d_active_v ? *d_active_v : V_stride;

    bool waited = false;
    const int n_tiles = (V + 15) / 16;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int r0 = tile * 16;
        const int ra = r0 + g, rb = r0 + g + 8;
        const bool la = (ra < V), lb = (rb < V);
        const uint8_t* wa = w_data + (size_t)(la ? ra : 0) * HIDDEN;
        const uint8_t* wb = w_data + (size_t)(lb ? rb : 0) * HIDDEN;

        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 1
        for (int jb = 0; jb < NJ; jb += U) {
            uint4 A[U], B[U], X[U][2];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                int col = c0 + (jb + u) * SPAN;
                A[u] = la ? ld_stream_u4(wa + col) : make_uint4(0, 0, 0, 0);
                B[u] = lb ? ld_stream_u4(wb + col) : make_uint4(0, 0, 0, 0);
            }
            if (!waited) {
                pdl_wait_fenced(zero);
                waited = true;
            }
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                int col = c0 + (jb + u) * SPAN;
                X[u][0] = lx ? ld_dep_u4(xg + col)     : make_uint4(0, 0, 0, 0);
                X[u][1] = lx ? ld_dep_u4(xg + col + 8) : make_uint4(0, 0, 0, 0);
            }
            uint32_t dep = 0u;
            #pragma unroll
            for (int u = 0; u < U; ++u) dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w ^ X[u][0].x;
            dep &= zero;
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w};
                const uint32_t bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
                const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                        X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    __half2 a0 = cvt_e4m3x2_to_half2(uint16_t(aw[s] & 0xFFFFu));
                    __half2 a1 = cvt_e4m3x2_to_half2(uint16_t(aw[s] >> 16));
                    __half2 b0 = cvt_e4m3x2_to_half2(uint16_t(bw[s] & 0xFFFFu));
                    __half2 b1 = cvt_e4m3x2_to_half2(uint16_t(bw[s] >> 16));
                    mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                         *reinterpret_cast<uint32_t*>(&a0), *reinterpret_cast<uint32_t*>(&b0),
                                         *reinterpret_cast<uint32_t*>(&a1), *reinterpret_cast<uint32_t*>(&b1),
                                         xw[2 * s] ^ dep, xw[2 * s + 1],
                                         d[0], d[1], d[2], d[3]);
                }
            }
        }
        cs[warp][g][2 * t]     = d[0];
        cs[warp][g][2 * t + 1] = d[1];
        cs[warp][g + 8][2 * t]     = d[2];
        cs[warp][g + 8][2 * t + 1] = d[3];
        __syncthreads();

        for (int idx = tid; idx < 16 * K_CONST; idx += 192) {
            int r = idx / K_CONST;
            int k = idx % K_CONST;
            int row = r0 + r;
            if (row < V) {
                float s = cs[0][r][k] + cs[1][r][k] + cs[2][r][k]
                        + cs[3][r][k] + cs[4][r][k] + cs[5][r][k];
                out_base[(size_t)k * V_stride + row] = s * __ldg(row_scales + row);
            }
        }
        __syncthreads();
    }
    if (!waited) pdl_wait_fenced(zero);
    pdl_trigger();
}

// ============================================================================
// Top-K 1024 Radix Select + 1024-Candidate Exact BF16 Markov Head
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine markov.cu (Apache-2.0).
// Cuts 7 sequential Markov fold passes from 32K-200K rows -> ONLY 1,024 rows (~70 us total, T=0 & T=1.0)!
// ============================================================================

constexpr int DSPARK_CANDS = 1024;
constexpr int TOPK_SLICES = 32;
constexpr int TOPK_THREADS = 256;

struct DsparkTopkItem {
    float val;
    int32_t idx;
};

struct DsparkTopkRowWs {
    uint32_t total_hi[256];
    uint32_t total_lo[256];
    int32_t count_gt[TOPK_SLICES];
    int32_t count_tie[TOPK_SLICES];
    DsparkTopkItem above[TOPK_SLICES][DSPARK_CANDS];
    DsparkTopkItem ties[TOPK_SLICES][DSPARK_CANDS];
};

__device__ __forceinline__ uint32_t dspark_key16(float v) {
    uint32_t u = __float_as_uint(v);
    uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return k >> 16;
}

__device__ __forceinline__ void dspark_slice_range(int V, int s, int& b, int& e) {
    int per = (V + TOPK_SLICES - 1) / TOPK_SLICES;
    b = min(V, s * per);
    e = min(V, b + per);
}

__device__ __forceinline__ int dspark_count_above(int c, int* warp_tot) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int NW = TOPK_THREADS / 32;
    int v = c;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int u = __shfl_down_sync(0xffffffffu, v, o);
        if (lane + o < 32) v += u;
    }
    if (lane == 0) warp_tot[warp] = v;
    __syncthreads();
    int above = v - c;
    for (int w = warp + 1; w < NW; ++w) above += warp_tot[w];
    __syncthreads();
    return above;
}

__device__ __forceinline__ void dspark_select_bin(
    const uint32_t* hist, int k, int above0, int* sh_bin, int* sh_above, int* warp_tot) {
    int c = (int)hist[threadIdx.x];
    int above = above0 + dspark_count_above(c, warp_tot);
    if (above < k && above + c >= k) {
        *sh_bin = threadIdx.x;
        *sh_above = above;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(TOPK_THREADS)
dspark_topk_hist_hi_kernel(
    const float* __restrict__ base,
    int V_stride,
    const int32_t* __restrict__ d_active_v,
    DsparkTopkRowWs* __restrict__ ws) {
    __shared__ uint32_t hist[256];
    hist[threadIdx.x] = 0u;
    pdl_wait();
    pdl_trigger();
    __syncthreads();
    const int V = d_active_v ? *d_active_v : V_stride;
    const int s = blockIdx.x, row = blockIdx.y;
    const float* x = base + (size_t)row * V_stride;
    int b, e;
    dspark_slice_range(V, s, b, e);
    for (int i = b + threadIdx.x; i < e; i += TOPK_THREADS) {
        atomicAdd(&hist[dspark_key16(ld_dep_f32(x + i)) >> 8], 1u);
    }
    __syncthreads();
    if (hist[threadIdx.x] != 0u) atomicAdd(&ws[row].total_hi[threadIdx.x], hist[threadIdx.x]);
}

__global__ void __launch_bounds__(TOPK_THREADS)
dspark_topk_hist_lo_kernel(
    const float* __restrict__ base,
    int V_stride,
    const int32_t* __restrict__ d_active_v,
    DsparkTopkRowWs* __restrict__ ws) {
    __shared__ uint32_t hist[256];
    __shared__ int sh_bin, sh_above, warp_tot[TOPK_THREADS / 32];
    hist[threadIdx.x] = 0u;
    pdl_wait();
    pdl_trigger();
    const int V = d_active_v ? *d_active_v : V_stride;
    const int s = blockIdx.x, row = blockIdx.y;
    DsparkTopkRowWs& w = ws[row];
    dspark_select_bin(w.total_hi, DSPARK_CANDS, 0, &sh_bin, &sh_above, warp_tot);
    const uint32_t b1 = (uint32_t)sh_bin;
    const float* x = base + (size_t)row * V_stride;
    int b, e;
    dspark_slice_range(V, s, b, e);
    for (int i = b + threadIdx.x; i < e; i += TOPK_THREADS) {
        uint32_t k = dspark_key16(x[i]);
        if ((k >> 8) == b1) atomicAdd(&hist[k & 255u], 1u);
    }
    __syncthreads();
    if (hist[threadIdx.x] != 0u) atomicAdd(&w.total_lo[threadIdx.x], hist[threadIdx.x]);
}

__global__ void __launch_bounds__(TOPK_THREADS)
dspark_topk_collect_kernel(
    const float* __restrict__ base,
    int V_stride,
    const int32_t* __restrict__ d_active_v,
    DsparkTopkRowWs* __restrict__ ws) {
    __shared__ int sh_bin, sh_above, warp_tot[TOPK_THREADS / 32];
    __shared__ uint32_t warp_cnt[TOPK_THREADS / 32];
    pdl_wait();
    pdl_trigger();
    const int V = d_active_v ? *d_active_v : V_stride;
    const int s = blockIdx.x, row = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    DsparkTopkRowWs& w = ws[row];
    dspark_select_bin(w.total_hi, DSPARK_CANDS, 0, &sh_bin, &sh_above, warp_tot);
    uint32_t b1 = (uint32_t)sh_bin;
    int above1 = sh_above;
    __syncthreads();
    dspark_select_bin(w.total_lo, DSPARK_CANDS, above1, &sh_bin, &sh_above, warp_tot);
    const uint32_t T = (b1 << 8) | (uint32_t)sh_bin;

    const float* x = base + (size_t)row * V_stride;
    int b, e;
    dspark_slice_range(V, s, b, e);
    int n_above = 0, n_tie = 0;
    for (int i0 = b; i0 < e; i0 += TOPK_THREADS) {
        int i = i0 + threadIdx.x;
        float v = 0.f;
        uint32_t k = 0u;
        if (i < e) {
            v = x[i];
            k = dspark_key16(v);
        }
        bool gt = (i < e && k > T);
        bool tie = (i < e && k == T);
        unsigned bg = __ballot_sync(0xffffffffu, gt);
        unsigned bt = __ballot_sync(0xffffffffu, tie);
        if (lane == 0) warp_cnt[warp] = uint32_t(__popc(bg)) | (uint32_t(__popc(bt)) << 16);
        __syncthreads();
        int rg = n_above + __popc(bg & ((1u << lane) - 1u));
        int rt = n_tie + __popc(bt & ((1u << lane) - 1u));
        #pragma unroll
        for (int u = 0; u < TOPK_THREADS / 32; ++u) {
            uint32_t c = warp_cnt[u];
            if (u < warp) {
                rg += int(c & 0xffffu);
                rt += int(c >> 16);
            }
            n_above += int(c & 0xffffu);
            n_tie += int(c >> 16);
        }
        if (gt && rg < DSPARK_CANDS) w.above[s][rg] = DsparkTopkItem{v, i};
        if (tie && rt < DSPARK_CANDS) w.ties[s][rt] = DsparkTopkItem{v, i};
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        w.count_gt[s] = min(n_above, DSPARK_CANDS);
        w.count_tie[s] = min(n_tie, DSPARK_CANDS);
    }
}

__global__ void __launch_bounds__(DSPARK_CANDS)
dspark_topk_gather_kernel(
    DsparkTopkRowWs* __restrict__ ws,
    const int32_t* __restrict__ draft_vocab_ids,
    float* __restrict__ cand_val,
    int32_t* __restrict__ cand_tok) {
    __shared__ int pre_gt[TOPK_SLICES + 1], pre_tie[TOPK_SLICES + 1];
    pdl_wait();
    pdl_trigger();
    const int tid = threadIdx.x;
    DsparkTopkRowWs& w = ws[blockIdx.x];
    if (tid < TOPK_SLICES) {
        int cg = w.count_gt[tid], ct = w.count_tie[tid];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int ug = __shfl_up_sync(0xffffffffu, cg, o);
            int ut = __shfl_up_sync(0xffffffffu, ct, o);
            if (tid >= o) { cg += ug; ct += ut; }
        }
        pre_gt[tid + 1] = cg;
        pre_tie[tid + 1] = ct;
        if (tid == 0) { pre_gt[0] = 0; pre_tie[0] = 0; }
    }
    if (tid < 256) {
        w.total_hi[tid] = 0u;
        w.total_lo[tid] = 0u;
    }
    __syncthreads();
    const int n_gt = pre_gt[TOPK_SLICES];
    const bool above = (tid < n_gt);
    const int* pre = above ? pre_gt : pre_tie;
    const int j = above ? tid : (tid - n_gt);
    int s = 0;
    #pragma unroll
    for (int step = TOPK_SLICES / 2; step > 0; step >>= 1) {
        if (pre[s + step] <= j) s += step;
    }
    DsparkTopkItem it = above ? w.above[s][j - pre[s]] : w.ties[s][j - pre[s]];
    size_t at = (size_t)blockIdx.x * DSPARK_CANDS + tid;
    cand_val[at] = it.val;
    cand_tok[at] = draft_vocab_ids ? draft_vocab_ids[it.idx] : it.idx;
}

// 128 blocks x 256 threads (1 warp per candidate, 1024 candidates total): exact BF16 fold_w[tok] + fold_b[tok] + softcap!
__global__ void __launch_bounds__(256)
markov_cand1024_logits_kernel(
    const __nv_bfloat16* __restrict__ fold_w, // [VOCAB_UNPADDED, 256] BF16
    const float* __restrict__ fold_b,         // [VOCAB_UNPADDED] F32
    const float* __restrict__ cand_val,       // [1024]
    const int32_t* __restrict__ cand_tok,     // [1024]
    const float* __restrict__ raw_rnn,        // [768]
    float* __restrict__ state,                // [256] updated by block 0
    float* __restrict__ out_cand_logits) {    // [1024]
    pdl_wait();
    pdl_trigger();
    __shared__ float s_lf[256];
    int tid = threadIdx.x;
    s_lf[tid] = fast_tanh(ld_dep_f32(raw_rnn + 512 + tid));
    if (blockIdx.x == 0) {
        float gate = fast_sigmoid(ld_dep_f32(raw_rnn + tid));
        float cand = fast_tanh(ld_dep_f32(raw_rnn + 256 + tid));
        state[tid] = gate * state[tid] + (1.0f - gate) * cand;
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int k = blockIdx.x * 8 + warp; // 0..1023
    int tok = cand_tok[k];

    const uint4* wr = reinterpret_cast<const uint4*>(fold_w + (size_t)tok * MTP_MARKOV_RANK);
    uint4 w8 = __ldg(wr + lane);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
    float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
    float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
    const float4* xf = reinterpret_cast<const float4*>(s_lf + (lane << 3));
    float4 x0 = xf[0], x1 = xf[1];
    float acc = 0.0f;
    acc = fmaf(wf0.x, x0.x, acc); acc = fmaf(wf0.y, x0.y, acc);
    acc = fmaf(wf1.x, x0.z, acc); acc = fmaf(wf1.y, x0.w, acc);
    acc = fmaf(wf2.x, x1.x, acc); acc = fmaf(wf2.y, x1.y, acc);
    acc = fmaf(wf3.x, x1.z, acc); acc = fmaf(wf3.y, x1.w, acc);
    acc = warp_sum(acc);

    if (lane == 0) {
        float total = cand_val[k] + __ldg(fold_b + tok) + acc;
        out_cand_logits[k] = fast_tanh(total * (1.0f / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
    }
}

// 1 block x 1024 threads (1 thread per candidate): greedy or exact top-p/top-k Gumbel-Max + sparse full-vocab draft probs!
__global__ void __launch_bounds__(DSPARK_CANDS)
markov_cand1024_pick_kernel(
    const float* __restrict__ cand_logits,    // [1024]
    const int32_t* __restrict__ cand_tok,     // [1024]
    int32_t* __restrict__ prev_cand_tok,      // [1024] (for O(1024) sparse zeroing of out_step_probs!)
    int32_t* __restrict__ out_tok,
    float* __restrict__ out_conf,
    float* __restrict__ out_step_probs,       // optional [VOCAB_UNPADDED] sparse full-vocab q(v)
    float* __restrict__ out_q_sel,
    int32_t* __restrict__ d_draft_is_delta,
    const SampleConfig* __restrict__ sample_cfg,
    const int32_t* __restrict__ d_pos,
    int step_idx) {
    pdl_wait();
    pdl_trigger();
    const int tid = threadIdx.x; // 0..1023
    const int warp = tid >> 5;   // 0..31
    const int lane = tid & 31;

    const float l = ld_dep_f32(cand_logits + tid);
    const int32_t tok = cand_tok[tid];

    // 1. Greedy argmax + max_cap
    float bv = l;
    int32_t bi = tok;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int32_t si[32];
    __shared__ float sq[32];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();

    __shared__ float s_max_cap;
    __shared__ int32_t s_greedy_tok;
    if (warp == 0) {
        float mv = sv[lane];
        int32_t mi = si[lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, mv, off);
            int32_t oi = __shfl_xor_sync(0xffffffffu, mi, off);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if (lane == 0) { s_max_cap = mv; s_greedy_tok = mi; }
    }
    __syncthreads();

    // 2. Confidence = 1 / sum(exp(l - max_cap))
    const float max_cap = s_max_cap;
    float e_conf = warp_sum(__expf(l - max_cap));
    if (lane == 0) sv[warp] = e_conf;
    __syncthreads();
    __shared__ float s_conf;
    if (warp == 0) {
        float tot = warp_sum(sv[lane]);
        if (lane == 0) s_conf = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
    }
    __syncthreads();

    const bool do_sample = (sample_cfg != nullptr && sample_cfg->inv_temp > 0.0f && out_step_probs != nullptr);
    if (!do_sample) {
        if (tid == 0) {
            *out_tok = s_greedy_tok;
            if (out_conf) *out_conf = s_conf;
            if (step_idx == 0 && d_draft_is_delta) *d_draft_is_delta = 0;
        }
        return;
    }

    // Sparse zero previous call's 1,024 entries in out_step_probs[VOCAB_UNPADDED]!
    {
        int32_t old_t = prev_cand_tok[tid];
        if (old_t >= 0 && old_t < VOCAB_UNPADDED) out_step_probs[old_t] = 0.0f;
    }
    __syncthreads();

    const int32_t top_k = sample_cfg->top_k;
    if (top_k == 1) {
        prev_cand_tok[tid] = tok;
        if (tid == 0) {
            int32_t gt = s_greedy_tok;
            out_step_probs[gt] = 1.0f;
            *out_tok = gt;
            if (out_q_sel) *out_q_sel = 1.0f;
            if (out_conf) *out_conf = s_conf;
            if (step_idx == 0 && d_draft_is_delta) *d_draft_is_delta = 0;
        }
        return;
    }

    const float inv_temp = sample_cfg->inv_temp;
    const float top_p = sample_cfg->top_p;
    const uint64_t draft_seed = sample_cfg->seed ^ 0xD5A2C1B39E3779B9ULL;
    const int32_t pos = (d_pos ? (*d_pos) : 0) + step_idx;

    constexpr float HIST_RANGE = 20.0f;
    constexpr float HIST_SCALE = 256.0f / HIST_RANGE;
    __shared__ unsigned long long hist_sum[256];
    __shared__ uint32_t hist_cnt[256];
    __shared__ float s_tau;
    __shared__ float s_inv_z;

    if ((top_p > 0.0f && top_p < 0.9999f) || top_k > 0) {
        if (tid < 256) { hist_sum[tid] = 0ULL; hist_cnt[tid] = 0u; }
        __syncthreads();
        float d = (max_cap - l) * inv_temp;
        if (d < HIST_RANGE) {
            int bin = (int)(d * HIST_SCALE);
            if (bin < 0) bin = 0;
            else if (bin > 255) bin = 255;
            unsigned long long fp = (unsigned long long)(__expf(-d) * 16777216.0f + 0.5f);
            atomicAdd(&hist_sum[bin], fp);
            atomicAdd(&hist_cnt[bin], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            double total_z = 0.0;
            for (int k = 0; k < 256; ++k) total_z += (double)hist_sum[k];
            double cum_z = 0.0;
            uint32_t cum_c = 0u;
            int cut_bin = 255;
            const double target_z = (top_p > 0.0f && top_p < 1.0f) ? ((double)top_p * total_z) : total_z;
            for (int k = 0; k < 256; ++k) {
                cum_z += (double)hist_sum[k];
                cum_c += hist_cnt[k];
                const bool hit_p = (top_p > 0.0f && top_p < 0.9999f && cum_z >= target_z);
                const bool hit_k = (top_k > 0 && cum_c >= (uint32_t)top_k);
                if (hit_p || hit_k) { cut_bin = k; break; }
            }
            s_tau = max_cap - ((float)(cut_bin + 1) / HIST_SCALE) / inv_temp;
        }
        __syncthreads();
    } else {
        if (tid == 0) s_tau = max_cap - HIST_RANGE / inv_temp;
        __syncthreads();
    }

    const float tau = s_tau;
    const bool kept = (l >= tau);
    const float e = kept ? __expf((l - max_cap) * inv_temp) : 0.0f;
    float wz = warp_sum(e);
    if (lane == 0) sv[warp] = wz;
    __syncthreads();
    if (warp == 0) {
        float tot_z = warp_sum(sv[lane]);
        if (lane == 0) s_inv_z = (tot_z > 0.0f) ? (1.0f / tot_z) : 0.0f;
    }
    __syncthreads();

    const float q = e * s_inv_z;
    out_step_probs[tok] = q;
    prev_cand_tok[tid] = tok;

    float best_s = kept ? ((l - max_cap) * inv_temp + hash_gumbel(draft_seed, pos, tok)) : -1e30f;
    int32_t best_t = tok;
    float best_q = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float os = __shfl_xor_sync(0xffffffffu, best_s, off);
        int32_t ot = __shfl_xor_sync(0xffffffffu, best_t, off);
        float oq = __shfl_xor_sync(0xffffffffu, best_q, off);
        if (os > best_s || (os == best_s && ot < best_t)) { best_s = os; best_t = ot; best_q = oq; }
    }
    if (lane == 0) { sv[warp] = best_s; si[warp] = best_t; sq[warp] = best_q; }
    __syncthreads();
    if (warp == 0) {
        float ms = sv[lane];
        int32_t mt = si[lane];
        float mq = sq[lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float os = __shfl_xor_sync(0xffffffffu, ms, off);
            int32_t ot = __shfl_xor_sync(0xffffffffu, mt, off);
            float oq = __shfl_xor_sync(0xffffffffu, mq, off);
            if (os > ms || (os == ms && ot < mt)) { ms = os; mt = ot; mq = oq; }
        }
        if (lane == 0) {
            *out_tok = mt;
            if (out_q_sel) *out_q_sel = mq;
            if (out_conf) *out_conf = s_conf;
            if (step_idx == 0 && d_draft_is_delta) *d_draft_is_delta = 0;
        }
    }
}

__global__ void copy_f_prev_zero_state_kernel(
    const float* __restrict__ f_ring, const int32_t* __restrict__ d_pos,
    float* __restrict__ out_f, float* __restrict__ state) {
    pdl_wait();
    pdl_trigger();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN) {
        int prev_p = (*d_pos) - 1;
        out_f[i] = (prev_p >= 0) ? f_ring[(size_t)(prev_p & 15) * HIDDEN + i] : 0.0f;
    }
    if (i < MTP_MARKOV_RANK) state[i] = 0.0f;
}

__global__ void spec_verify_kernel(
    const int32_t* __restrict__ target_preds,
    const int32_t* __restrict__ draft_tokens,
    int B,
    int32_t* __restrict__ out_accepted,
    int32_t* __restrict__ out_num_accepted,
    int32_t* __restrict__ d_pos,
    int32_t* __restrict__ d_next_anchor) {
    if (threadIdx.x != 0) return;
    int n_acc = 0;
    for (int i = 0; i < B - 1; ++i) {
        int32_t tp = target_preds[i];
        out_accepted[n_acc++] = tp;
        if (tp != draft_tokens[i]) {
            *out_num_accepted = n_acc;
            *d_pos += n_acc;
            *d_next_anchor = tp;
            return;
        }
    }
    // All B-1 drafts matched! Emit bonus token target_preds[B-1]!
    int32_t bonus = target_preds[B - 1];
    out_accepted[n_acc++] = bonus;
    *out_num_accepted = n_acc;
    *d_pos += n_acc;
    *d_next_anchor = bonus;
}

} // namespace

void launch_dspark_commit(
    const DsparkWeights& dw, const float* taps,
    void* ctx_k[MTP_LAYERS], void* ctx_v[MTP_LAYERS],
    __half* ctx_ks[MTP_LAYERS], __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS], KVFormat global_kv_fmt,
    float* f_ring, float* scratch_mix, float* scratch_kv,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    const bool use_half = (dw.layers[0].w_kv_ctx.kind != WeightKind::BF16) && (B <= 8);
    __half* scratch_mix_h = use_half ? reinterpret_cast<__half*>(scratch_mix) : nullptr;
    launch_pdl(dspark_mix_kernel, dim3(B, 4), 576, stream,
        taps, B, dw.fuse_alpha,
        dw.layers[0].dflare_norm, dw.layers[1].dflare_norm, dw.layers[2].dflare_norm,
        dw.markov_dflare_norm, use_half ? nullptr : scratch_mix, scratch_mix_h, f_ring, d_pos);
    for (int l = 0; l < MTP_LAYERS; ++l) {
        if (use_half) {
            const __half* mix_lh = scratch_mix_h + (size_t)l * B * HIDDEN;
            launch_linear_h(dw.layers[l].w_kv_ctx, mix_lh, scratch_kv, B, false, stream);
        } else {
            const float* mix_l = scratch_mix + (size_t)l * B * HIDDEN;
            launch_linear(dw.layers[l].w_kv_ctx, mix_l, scratch_kv, B, false, stream);
        }
        KVFormat lfmt = dw.layers[l].is_local ? KVFormat::BF16 : global_kv_fmt;
        if (lfmt == KVFormat::FP8) {
            launch_pdl(dspark_store_ctx_kv_kernel<1>, dim3(B, NUM_KV_HEADS), HEAD_DIM, stream,
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        } else if (lfmt == KVFormat::Q4) {
            launch_pdl(dspark_store_ctx_kv_kernel<2>, dim3(B, NUM_KV_HEADS), HEAD_DIM, stream,
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        } else {
            launch_pdl(dspark_store_ctx_kv_kernel<0>, dim3(B, NUM_KV_HEADS), HEAD_DIM, stream,
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        }
    }
}

void launch_dspark_gather_shortlist(
    const LinearWeight& unembed,
    LinearWeight& unembed_draft,
    int32_t* draft_vocab_ids,
    int32_t* d_draft_vocab_size,
    const int32_t* d_new_ids,
    int dst_offset,
    int count,
    cudaStream_t stream) {
    if (count <= 0) {
        if (d_draft_vocab_size) {
            dspark_set_vocab_size_kernel<<<1, 1, 0, stream>>>(d_draft_vocab_size, dst_offset);
        }
        return;
    }
    if (unembed.kind == WeightKind::FP8) {
        dspark_gather_shortlist_kernel<1><<<count, 256, 0, stream>>>(
            unembed.data, nullptr, unembed.row_scales,
            unembed_draft.data, nullptr, unembed_draft.row_scales,
            draft_vocab_ids, d_draft_vocab_size, d_new_ids, dst_offset, count);
    } else if (unembed.kind == WeightKind::NVFP4) {
        dspark_gather_shortlist_kernel<2><<<count, 256, 0, stream>>>(
            unembed.data, unembed.scales, nullptr,
            unembed_draft.data, unembed_draft.scales, nullptr,
            draft_vocab_ids, d_draft_vocab_size, d_new_ids, dst_offset, count);
    } else {
        dspark_gather_shortlist_kernel<0><<<count, 256, 0, stream>>>(
            unembed.data, nullptr, nullptr,
            unembed_draft.data, nullptr, nullptr,
            draft_vocab_ids, d_draft_vocab_size, d_new_ids, dst_offset, count);
    }
}

void launch_dspark_draft(
    const ModelWeights& mw, const int32_t* d_anchor_token, const int32_t* d_pos, int num_drafts,
    void* ctx_k[MTP_LAYERS], void* ctx_v[MTP_LAYERS],
    const __half* ctx_ks[MTP_LAYERS], const __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS], KVFormat global_kv_fmt, float* split_buf,
    const float* f_ring, float* ws, int32_t* out_draft_tokens, float* out_draft_conf,
    const SampleConfig* sample_cfg, float* out_draft_probs, float* out_draft_q_sel,
    int32_t* out_draft_is_delta, cudaStream_t stream) {
    const DsparkWeights& dw = mw.dspark;
    const int V_stride = (dw.draft_vocab_cap > 0) ? dw.draft_vocab_cap : dw.draft_vocab_size;

    // Partition preallocated DSpark workspace (in floats; ds_ws = 3,200,000 floats, zeroed at init/reset!):
    float* block_x      = ws;
    float* normed       = block_x + MTP_BLOCK_SIZE * HIDDEN;
    __half* normed_h    = reinterpret_cast<__half*>(normed + MTP_BLOCK_SIZE * HIDDEN);
    float* qkv          = reinterpret_cast<float*>(normed_h + MTP_BLOCK_SIZE * HIDDEN);
    float* attn_out     = qkv + MTP_BLOCK_SIZE * (Q_DIM + 2 * KV_DIM);
    __half* attn_out_h  = reinterpret_cast<__half*>(attn_out + MTP_BLOCK_SIZE * Q_DIM);
    float* gu           = reinterpret_cast<float*>(attn_out_h + MTP_BLOCK_SIZE * Q_DIM);
    float* swiglu       = gu + MTP_BLOCK_SIZE * (2 * MTP_INTERMEDIATE);
    __half* swiglu_h    = reinterpret_cast<__half*>(swiglu + MTP_BLOCK_SIZE * MTP_INTERMEDIATE);
    float* f_prev       = reinterpret_cast<float*>(swiglu_h + MTP_BLOCK_SIZE * MTP_INTERMEDIATE);
    float* c_f          = f_prev + HIDDEN;
    float* c_h          = c_f + 3 * MTP_MARKOV_RANK;
    float* state        = c_h + MTP_MAX_DRAFTS * (3 * MTP_MARKOV_RANK);
    float* raw_rnn      = state + MTP_MARKOV_RANK;
    float* cand_val     = raw_rnn + 3 * MTP_MARKOV_RANK;
    int32_t* cand_tok   = reinterpret_cast<int32_t*>(cand_val + MTP_MAX_DRAFTS * DSPARK_CANDS);
    float* cand_logits  = reinterpret_cast<float*>(cand_tok + MTP_MAX_DRAFTS * DSPARK_CANDS);
    int32_t* prev_ctok  = reinterpret_cast<int32_t*>(cand_logits + MTP_MAX_DRAFTS * DSPARK_CANDS);
    // Align topk_ws up to 16 bytes
    uintptr_t topk_addr = (reinterpret_cast<uintptr_t>(prev_ctok + MTP_MAX_DRAFTS * DSPARK_CANDS) + 15u) & ~uintptr_t(15u);
    DsparkTopkRowWs* topk_ws = reinterpret_cast<DsparkTopkRowWs*>(topk_addr);
    float* base_logits  = reinterpret_cast<float*>(topk_ws + MTP_MAX_DRAFTS);

    bool use_half = (dw.layers[0].w_qkv.kind != WeightKind::BF16);

    // 1. Build 8-row block [anchor, mask x7]
    launch_pdl(dspark_init_block_kernel, dim3(MTP_BLOCK_SIZE), 576, stream,
        mw.embed, mw.embed_norm, dw.mask_embed, d_anchor_token, block_x);

    // 2. Run 3 dense DSpark layers over all 8 rows (100% PDL-chained!)
    for (int l = 0; l < MTP_LAYERS; ++l) {
        const DsparkLayerWeights& dl = dw.layers[l];
        if (use_half) {
            launch_pdl(dspark_rmsnorm_h_kernel, dim3(MTP_BLOCK_SIZE), 576, stream, block_x, dl.input_norm, nullptr, normed_h);
            launch_linear_h(dl.w_qkv, normed_h, qkv, MTP_BLOCK_SIZE, false, stream);
        } else {
            launch_rmsnorm(block_x, dl.input_norm, normed, MTP_BLOCK_SIZE, HIDDEN, HIDDEN, RMS_EPS, stream);
            launch_linear(dl.w_qkv, normed, qkv, MTP_BLOCK_SIZE, false, stream);
        }
        launch_pdl(dspark_block_qk_rope_kernel, dim3(MTP_BLOCK_SIZE, NUM_HEADS + NUM_KV_HEADS), HEAD_DIM, stream,
            qkv, dl.q_norm, dl.k_norm, d_pos);
        if (dl.is_local) {
            launch_pdl(dspark_local_attn_bulk_splitk_kernel, dim3(NUM_KV_HEADS, DSPARK_LOCAL_SPLITS), 128, stream,
                qkv, ctx_k[l], ctx_v[l], ctx_cap[l], d_pos, split_buf);
        } else {
            dim3 sgrid(NUM_KV_HEADS, KV_SPLIT_K);
            if (global_kv_fmt == KVFormat::FP8) {
                launch_pdl(dspark_global_attn_bulk_splitk_kernel<1>, sgrid, 128, stream,
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            } else if (global_kv_fmt == KVFormat::Q4) {
                launch_pdl(dspark_global_attn_bulk_splitk_kernel<2>, sgrid, 128, stream,
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            } else {
                launch_pdl(dspark_global_attn_bulk_splitk_kernel<0>, sgrid, 128, stream,
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            }
        }
        const int max_win = dl.is_local ? SWA_WINDOW : max(0, ctx_cap[l] - MTP_MAX_DRAFTS);
        launch_pdl(dspark_block_attn_kernel, dim3(NUM_HEADS, MTP_BLOCK_SIZE), 256, stream,
            qkv, split_buf, dl.is_local ? DSPARK_LOCAL_SPLITS : KV_SPLIT_K, max_win, d_pos,
            use_half ? nullptr : attn_out, use_half ? attn_out_h : nullptr);
        if (use_half) launch_linear_h(dl.w_o, attn_out_h, block_x, MTP_BLOCK_SIZE, true, stream);
        else          launch_linear(dl.w_o, attn_out, block_x, MTP_BLOCK_SIZE, true, stream);

        if (use_half) {
            launch_pdl(dspark_rmsnorm_h_kernel, dim3(MTP_BLOCK_SIZE), 576, stream, block_x, dl.post_norm, nullptr, normed_h);
            launch_linear_h(dl.w_gu, normed_h, gu, MTP_BLOCK_SIZE, false, stream);
            launch_pdl(dspark_swiglu_kernel, dim3((MTP_BLOCK_SIZE * MTP_INTERMEDIATE + 255) / 256), 256, stream,
                gu, nullptr, swiglu_h, MTP_BLOCK_SIZE);
            launch_linear_h(dl.w_down, swiglu_h, block_x, MTP_BLOCK_SIZE, true, stream);
        } else {
            launch_rmsnorm(block_x, dl.post_norm, normed, MTP_BLOCK_SIZE, HIDDEN, HIDDEN, RMS_EPS, stream);
            launch_linear(dl.w_gu, normed, gu, MTP_BLOCK_SIZE, false, stream);
            launch_pdl(dspark_swiglu_kernel, dim3((MTP_BLOCK_SIZE * MTP_INTERMEDIATE + 255) / 256), 256, stream,
                gu, swiglu, nullptr, MTP_BLOCK_SIZE);
            launch_linear(dl.w_down, swiglu, block_x, MTP_BLOCK_SIZE, true, stream);
        }
    }

    // 3. Final norm on rows 1..num_drafts -> normed (F32 for rnn_w_h) + normed_h (F16 for unembed_draft)
    launch_pdl(dspark_rmsnorm_h_kernel, dim3(num_drafts), 576, stream, block_x + HIDDEN, dw.norm_w, normed, normed_h);

    // 4. Precompute Markov inputs + zero RNN state
    launch_pdl(copy_f_prev_zero_state_kernel, dim3((HIDDEN + 255) / 256), 256, stream, f_ring, d_pos, f_prev, state);
    launch_bf16_linear(dw.rnn_w_f, f_prev, c_f, 1, 3 * MTP_MARKOV_RANK, HIDDEN, stream);
    launch_bf16_linear(dw.rnn_w_h, normed, c_h, num_drafts, 3 * MTP_MARKOV_RANK, HIDDEN, stream);

    // 5. Precompute base_logits[0..num_drafts-1, 0..active_V-1] (stride = V_stride)
    if (dw.unembed_draft.kind == WeightKind::NVFP4) {
        const auto* wd = static_cast<const uint8_t*>(dw.unembed_draft.data);
        const auto* ws_sc = dw.unembed_draft.scales;
        float gs = dw.unembed_draft.global_scale;
        switch (num_drafts) {
            case 1: dspark_base_logits_nvfp4_kernel<1><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size); break;
            case 2: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<2>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 3: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<3>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 4: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<4>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 5: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<5>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 6: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<6>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 7: launch_pdl(dspark_base_logits_nvfp4_mma_kernel<7>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, ws_sc, gs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
        }
    } else if (dw.unembed_draft.kind == WeightKind::FP8) {
        const auto* wd = static_cast<const uint8_t*>(dw.unembed_draft.data);
        const float* rs = dw.unembed_draft.row_scales;
        switch (num_drafts) {
            case 1: dspark_base_logits_fp8_b1_kernel<<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size); break;
            case 2: launch_pdl(dspark_base_logits_fp8_mma_kernel<2>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 3: launch_pdl(dspark_base_logits_fp8_mma_kernel<3>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 4: launch_pdl(dspark_base_logits_fp8_mma_kernel<4>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 5: launch_pdl(dspark_base_logits_fp8_mma_kernel<5>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 6: launch_pdl(dspark_base_logits_fp8_mma_kernel<6>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
            case 7: launch_pdl(dspark_base_logits_fp8_mma_kernel<7>, dim3(DSPARK_UNEMBED_BLOCKS), 192, stream, wd, rs, normed_h, base_logits, V_stride, dw.d_draft_vocab_size, 0u); break;
        }
    } else {
        launch_linear(dw.unembed_draft, normed, base_logits, num_drafts, false, stream);
    }

    // 6. Parallel Top-K 1024 radix select across all num_drafts rows!
    launch_pdl(dspark_topk_hist_hi_kernel, dim3(TOPK_SLICES, num_drafts), TOPK_THREADS, stream,
        base_logits, V_stride, dw.d_draft_vocab_size, topk_ws);
    launch_pdl(dspark_topk_hist_lo_kernel, dim3(TOPK_SLICES, num_drafts), TOPK_THREADS, stream,
        base_logits, V_stride, dw.d_draft_vocab_size, topk_ws);
    launch_pdl(dspark_topk_collect_kernel, dim3(TOPK_SLICES, num_drafts), TOPK_THREADS, stream,
        base_logits, V_stride, dw.d_draft_vocab_size, topk_ws);
    launch_pdl(dspark_topk_gather_kernel, dim3(num_drafts), DSPARK_CANDS, stream,
        topk_ws, dw.draft_vocab_ids, cand_val, cand_tok);

    // 7. Run num_drafts sequential Markov steps over ONLY 1,024 exact BF16 fold_w candidates!
    const __nv_bfloat16* fold_w_bf16 = static_cast<const __nv_bfloat16*>(dw.fold_w.data);
    for (int i = 0; i < num_drafts; ++i) {
        const int32_t* prev_ptr = (i == 0) ? d_anchor_token : (out_draft_tokens + i - 1);
        launch_pdl(markov_rnn_proj_kernel, dim3(24), 128, stream,
            dw.rnn_w_sp, dw.rnn_bias, dw.markov_w1,
            c_h + i * (3 * MTP_MARKOV_RANK), c_f, prev_ptr, state, raw_rnn);

        const float* cv_i = cand_val + (size_t)i * DSPARK_CANDS;
        const int32_t* ct_i = cand_tok + (size_t)i * DSPARK_CANDS;
        float* cl_i = cand_logits + (size_t)i * DSPARK_CANDS;
        int32_t* pc_i = prev_ctok + (size_t)i * DSPARK_CANDS;
        float* dp_i = out_draft_probs ? (out_draft_probs + (size_t)i * VOCAB_UNPADDED) : nullptr;

        launch_pdl(markov_cand1024_logits_kernel, dim3(128), 256, stream,
            fold_w_bf16, dw.fold_b, cv_i, ct_i, raw_rnn, state, cl_i);
        launch_pdl(markov_cand1024_pick_kernel, dim3(1), DSPARK_CANDS, stream,
            cl_i, ct_i, pc_i,
            out_draft_tokens + i,
            out_draft_conf ? (out_draft_conf + i) : nullptr,
            dp_i,
            out_draft_q_sel ? (out_draft_q_sel + i) : nullptr,
            out_draft_is_delta,
            sample_cfg, d_pos, i);
    }
}

void launch_spec_verify(
    const int32_t* target_preds, const int32_t* draft_tokens, int B,
    int32_t* out_accepted_tokens, int32_t* out_num_accepted,
    int32_t* d_pos, int32_t* d_next_anchor, cudaStream_t stream) {
    spec_verify_kernel<<<1, 1, 0, stream>>>(
        target_preds, draft_tokens, B, out_accepted_tokens, out_num_accepted, d_pos, d_next_anchor);
}

} // namespace squid
