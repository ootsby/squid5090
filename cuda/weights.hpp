#pragma once

#include "config.hpp"
#include <cuda_bf16.h>
#include <cstdint>

namespace squid {

enum class WeightKind : uint8_t {
    BF16 = 0,
    FP8  = 1, // F8_E4M3 weights + per-row F32 scale
    NVFP4 = 2 // Packed U8 (2x e2m1) + F8_E4M3 block-16 scales + F32 global scale
};

struct LinearWeight {
    WeightKind kind = WeightKind::BF16;
    int rows = 0; // out_features (N)
    int cols = 0; // in_features (K)
    void* data = nullptr;       // BF16 [N, K] or FP8 [N, K] or U8 [N, K/2]
    uint8_t* scales = nullptr;  // NVFP4 F8_E4M3 [N, K/16]
    float* row_scales = nullptr;// FP8 per-row F32 [N]
    float global_scale = 1.0f;  // NVFP4 tensor global scale (already divided by denom)
};

struct ExpertBank {
    bool is_nvfp4 = true; // true: NVFP4, false: MXFP8
    bool is_iq3 = false;  // w13 only: IQ3_XXS, data/scales laid out by tools/mkiq3.py
    int n_experts = NUM_EXPERTS;
    int rows = 0; // w13: 2*INTERMEDIATE=2304, w2: HIDDEN=2304
    int cols = 0; // w13: HIDDEN=2304, w2: INTERMEDIATE=1152
    uint8_t* data = nullptr;    // NVFP4: U8 [E, rows, cols/2], MXFP8: F8_E4M3 [E, rows, cols]
    uint8_t* scales = nullptr;  // NVFP4: F8_E4M3 [E, rows, cols/16], MXFP8: U8 UE8M0 [E, rows, cols/32]
    float* expert_scale = nullptr; // NVFP4: precomputed F32 [E] = amax[e] / denom[0]
};

struct ControllerWeights {
    __nv_bfloat16* down_w = nullptr;   // [64, 4608] (Exact/FP8)
    __nv_bfloat16* forget_w = nullptr; // [2304, 64]
    __nv_bfloat16* gain_w = nullptr;   // [2304, 64]
    // NVFP4 packed controllers for single-kernel fused L2-resident execution!
    LinearWeight down_nvfp4;
    LinearWeight forget_nvfp4;
    LinearWeight gain_nvfp4;
};

struct LayerWeights {
    bool is_global = false;

    // Input norms (BF16 [2304])
    __nv_bfloat16* attn_norm = nullptr;   // local layers
    __nv_bfloat16* q_in_norm = nullptr;   // global layers
    __nv_bfloat16* k_in_norm = nullptr;   // global layers
    __nv_bfloat16* v_in_norm = nullptr;   // global layers
    __nv_bfloat16* mlp_norm = nullptr;

    // Fused Q(3072) + R(384) + Gate(3072) + Sink(24) = 6552 x 2304
    LinearWeight w_qrgs;
    // K(1024) x 2304 and V(1024) x 2304 (also contiguous as w_kv [2048, 2304])
    LinearWeight w_kv;
    LinearWeight w_k;
    LinearWeight w_v;
    // O(2304) x 3072
    LinearWeight w_o;

    // Short convolutions
    __nv_bfloat16* k_sconv_w = nullptr;      // local: [1024, 4], global static_w: [4, 1024]
    __nv_bfloat16* v_sconv_w = nullptr;      // local: [1024, 4], global static_w: [4, 1024]
    __nv_bfloat16* k_sconv_proj = nullptr;   // global only: [32, 2304]
    __nv_bfloat16* v_sconv_proj = nullptr;   // global only: [32, 2304]
    __nv_bfloat16* attn_sconv_w = nullptr;   // [2304, 4]
    __nv_bfloat16* mlp_sconv_w = nullptr;    // [2304, 4]

    // Clipping & relative bias
    __nv_bfloat16* q_clip_scale = nullptr;   // [24, 128]
    __nv_bfloat16* rel_clip_scale = nullptr; // [24, 16]
    __nv_bfloat16* rel_logits_proj = nullptr;// [16, 1024]

    // Cross-pass controllers (physical layers 6..17 only: 0=attn, 1=k, 2=v, 3=mlp)
    bool has_controllers = false;
    ControllerWeights ctrl[4];

    // MoE router & experts
    __nv_bfloat16* router_w = nullptr; // [128, 2304] (Exact/FP8)
    LinearWeight router_nvfp4;         // [128, 2304] NVFP4 (only 166 KB/layer -> 4 MB total in L2!)
    float* router_bias = nullptr;      // [128] F32
    ExpertBank w13;
    ExpertBank w2;
};

struct DsparkLayerWeights {
    bool is_local = false;
    __nv_bfloat16* input_norm = nullptr;  // [2304]
    __nv_bfloat16* post_norm = nullptr;   // [2304]
    __nv_bfloat16* dflare_norm = nullptr; // [2304]
    __nv_bfloat16* q_norm = nullptr;      // [128]
    __nv_bfloat16* k_norm = nullptr;      // [128]
    __nv_bfloat16* k_norm_ctx = nullptr;  // [128]

    LinearWeight w_qkv;   // fused [3072 + 1024 + 1024 = 5120, 2304]
    LinearWeight w_o;     // [2304, 3072]
    LinearWeight w_gu;    // fused [9216 + 9216 = 18432, 2304] (gate then up)
    LinearWeight w_down;  // [2304, 9216]
    LinearWeight w_kv_ctx;// fused [1024 + 1024 = 2048, 2304]
};

struct DsparkWeights {
    bool loaded = false;
    float* fuse_alpha = nullptr; // [4, 4] F32 softmaxed rows
    __nv_bfloat16* mask_embed = nullptr;   // [2304]
    __nv_bfloat16* norm_w = nullptr;       // [2304]
    __nv_bfloat16* markov_dflare_norm = nullptr; // [2304]
    __nv_bfloat16* markov_w1 = nullptr;    // [201024, 256] (kept on GPU: 102.9 MB!)
    __nv_bfloat16* rnn_w_sp = nullptr;     // [768, 512] (state + w1[prev])
    __nv_bfloat16* rnn_w_h = nullptr;      // [768, 2304]
    __nv_bfloat16* rnn_w_f = nullptr;      // [768, 2304]
    __nv_bfloat16* rnn_bias = nullptr;     // [768]
    __nv_bfloat16* hidden_out_w = nullptr; // [2304, 256]
    __nv_bfloat16* hidden_out_b = nullptr; // [2304]

    // Precomputed folded draft projections for blazing fast Markov RNN argmax:
    // Optional pruned draft vocabulary (32K base + 4096 dynamic context rows -> 100% lossless!)
    int draft_vocab_size = VOCAB_UNPADDED;
    int draft_vocab_cap = VOCAB_UNPADDED;
    int32_t* d_draft_vocab_size = nullptr;  // device-side active count (nullptr = static draft_vocab_size)
    int32_t* draft_vocab_ids = nullptr;     // [draft_vocab_cap] maps local idx -> real vocab ID (nullptr = identity)
    int32_t* inv_draft_vocab_ids = nullptr; // unused with full-vocab sparse draft probs (kept for ABI compat)
    LinearWeight unembed_draft;             // [draft_vocab_cap, 2304]
    LinearWeight fold_w;                    // [VOCAB_UNPADDED, 256] BF16 (indexed by real token ID!)
    float* fold_b = nullptr;                // [VOCAB_UNPADDED] F32 (indexed by real token ID!)

    DsparkLayerWeights layers[MTP_LAYERS];
};

struct VisionLayerWeights {
    const float* ln1_w = nullptr;         // [1024]
    const float* ln1_b = nullptr;         // [1024]
    const float* ln2_w = nullptr;         // [1024]
    const float* ln2_b = nullptr;         // [1024]
    const float* ls1 = nullptr;           // [1024]
    const float* ls2 = nullptr;           // [1024]
    const __nv_bfloat16* wqkv = nullptr;  // [3072, 1024] stacked Q, K, V
    const float* bqkv = nullptr;          // [3072]
    const __nv_bfloat16* wo = nullptr;    // [1024, 1024]
    const float* bo = nullptr;            // [1024]
    const __nv_bfloat16* fc = nullptr;    // [4096, 1024]
    const float* bfc = nullptr;           // [4096]
    const __nv_bfloat16* proj = nullptr;  // [1024, 4096]
    const float* bproj = nullptr;         // [1024]
};

struct VisionWeights {
    bool loaded = false;
    const __nv_bfloat16* patch_w = nullptr;  // [1024, 1200]
    const float* pos = nullptr;              // [32*32, 1024]
    const float* cls = nullptr;              // [1024] class_embedding + cls_position_embedding
    const float* ln_pre_w = nullptr;         // [1024]
    const float* ln_pre_b = nullptr;         // [1024]
    const __nv_bfloat16* proj_in = nullptr;  // [4096, 4096]
    const __nv_bfloat16* proj_out = nullptr; // [2304, 4096]
    const float* inv_freq = nullptr;         // [16]
    VisionLayerWeights layers[VISION_LAYERS];
};

struct ModelWeights {
    QuantMode mode = QuantMode::Exact;

    __nv_bfloat16* embed = nullptr;      // [201024, 2304] BF16
    __nv_bfloat16* embed_norm = nullptr; // [2304] BF16
    __nv_bfloat16* final_norm = nullptr; // [2304] BF16
    LinearWeight unembed;                // [200058, 2304]

    // AttnRes depth queries (F32)
    // attn_queries[l-1] for l=1..23: [3, 2304] if global else [2304]
    float* attn_queries[NUM_LAYERS - 1] = {};
    float* mlp_queries[NUM_LAYERS] = {};
    float* output_query = nullptr; // [2304] F32

    // Cross-pass queries (F32 [24, 2304])
    float* cross_pass_queries = nullptr;
    float* cross_pass_kv_queries = nullptr;

    LayerWeights layers[NUM_LAYERS];
    DsparkWeights dspark;
};

} // namespace squid
