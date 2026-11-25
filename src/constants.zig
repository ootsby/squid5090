// Comptime-verified Inkling-2-Mini & DSpark MTP architectural constants.
// Every geometry invariant is checked at compile time per TigerStyle.

const tiger = @import("tiger.zig");

pub const hidden_dim: u32 = 2304;
pub const layer_count: u32 = 24;
pub const visit_count: u32 = 48;
pub const loop_start_layer: u32 = 6;
pub const loop_end_layer: u32 = 18;
pub const loop_pass_count: u32 = 3;
pub const block_layer_count: u32 = 6;
pub const gate_rank: u32 = 64;

pub const head_count: u32 = 24;
pub const kv_head_count: u32 = 8;
pub const head_dim: u32 = 128;
pub const query_dim: u32 = head_count * head_dim; // 3072
pub const kv_dim: u32 = kv_head_count * head_dim; // 1024
pub const gqa_ratio: u32 = head_count / kv_head_count; // 3

pub const rel_dim: u32 = 16;
pub const r_dim: u32 = head_count * rel_dim; // 384
pub const rel_extent: u32 = 1024;
pub const swa_window_tokens: u32 = 1024;
pub const conv_tap_count: u32 = 4;
pub const sconv_ring_slots: u32 = 16;

pub const expert_count: u32 = 128;
pub const top_k_experts: u32 = 8;
pub const intermediate_dim: u32 = 1152;
// IQ3_XXS w13 bank (tools/mkiq3.py): 864 code + 18 scale bytes per row
pub const iq3_codes_bytes: u64 = @as(u64, expert_count) * 2 * intermediate_dim * 864;
pub const iq3_d_bytes: u64 = @as(u64, expert_count) * 2 * intermediate_dim * 18;
pub const vocab_padded_count: u32 = 201024;
pub const vocab_unpadded_count: u32 = 200058;
pub const eos_token_id: i32 = 200006;

pub const rms_eps: f32 = 1e-6;

// DSpark MTP constants
pub const mtp_layer_count: u32 = 3;
pub const mtp_intermediate_dim: u32 = 9216;
pub const mtp_block_size_max: u32 = 8; // anchor + 7 drafts
pub const mtp_draft_count_max: u32 = 7;
pub const mtp_markov_rank: u32 = 256;
pub const mtp_target_visits: [4]u32 = .{ 5, 23, 41, 48 };

// Inkling-2-Mini Vision Tower constants (model.visual.*)
pub const vision_layer_count: u32 = 23;
pub const vision_hidden_dim: u32 = 1024;
pub const vision_patch_dim: u32 = 1200;
pub const vision_mlp_dim: u32 = 4096;
pub const vision_merged_dim: u32 = 4096;
pub const vision_pos_grid: u32 = 32;
pub const vision_rope_freqs: u32 = 16;
pub const vision_tile_px: u32 = 40;
pub const vision_max_tokens_default: u32 = 4096;
pub const vision_max_images_default: u32 = 16;

// FlashDecoding Split-K and Q4 KV constants
pub const kv_q4_group_size: u32 = 64;
pub const kv_q4_groups_per_head: u32 = head_dim / kv_q4_group_size; // 2
pub const kv_q4_scales_per_token: u32 = kv_head_count * kv_q4_groups_per_head; // 16
pub const split_buffer_floats: u64 = 2_050_000;
pub const dspark_workspace_floats: u64 = 3_200_000;
pub const dspark_dynamic_ctx_rows: u32 = 16384;

pub const vram_alignment_bytes: u64 = 256;
pub const qrgs_out_dim: u32 = query_dim + r_dim + query_dim + head_count; // 6552
pub const dspark_qkv_out_dim: u32 = query_dim + 2 * kv_dim; // 5120

pub const QuantMode = enum(u8) {
    exact = 0,
    fp8 = 1,
    nvfp4 = 2,
    /// fp8 dense attention + BF16 router/controllers, but NVFP4 experts (L0-23) and lm_head. Fits 16 GB cards.
    mix = 3,

    pub fn expert_mode(self: QuantMode) QuantMode {
        return if (self == .mix) .nvfp4 else self;
    }

    pub fn name(self: QuantMode) []const u8 {
        return switch (self) {
            .exact => "exact",
            .fp8 => "fp8",
            .nvfp4 => "nvfp4",
            .mix => "mix",
        };
    }
};

pub const WeightKind = enum(u8) {
    bf16 = 0,
    fp8 = 1,
    nvfp4 = 2,
};

pub const KVFormat = enum(u8) {
    bf16 = 0,
    fp8 = 1,
    q4 = 2,

    pub fn name(self: KVFormat) []const u8 {
        return switch (self) {
            .bf16 => "bf16",
            .fp8 => "fp8",
            .q4 => "q4",
        };
    }
};

pub const VisitInfo = struct {
    visit_index: u32,
    physical_layer: u32,
    pass_index: u32,
    is_loop: bool,
    is_global: bool,
};

pub inline fn is_global_layer(physical_layer: u32) bool {
    tiger.assert(physical_layer < layer_count);
    return (physical_layer + 1) % block_layer_count == 0;
}

pub inline fn is_nvfp4_expert_layer(physical_layer: u32) bool {
    tiger.assert(physical_layer < layer_count);
    return physical_layer < 20;
}

pub inline fn is_mtp_local_layer(layer_index: u32) bool {
    tiger.assert(layer_index < mtp_layer_count);
    return layer_index == 0 or layer_index == 2;
}

pub fn build_visit_schedule() [visit_count]VisitInfo {
    var schedule: [visit_count]VisitInfo = undefined;
    var index: u32 = 0;

    var layer: u32 = 0;
    while (layer < loop_start_layer) : (layer += 1) {
        schedule[index] = .{
            .visit_index = index,
            .physical_layer = layer,
            .pass_index = 0,
            .is_loop = false,
            .is_global = is_global_layer(layer),
        };
        index += 1;
    }

    var pass: u32 = 0;
    while (pass < loop_pass_count) : (pass += 1) {
        layer = loop_start_layer;
        while (layer < loop_end_layer) : (layer += 1) {
            schedule[index] = .{
                .visit_index = index,
                .physical_layer = layer,
                .pass_index = pass,
                .is_loop = true,
                .is_global = is_global_layer(layer),
            };
            index += 1;
        }
    }

    layer = loop_end_layer;
    while (layer < layer_count) : (layer += 1) {
        schedule[index] = .{
            .visit_index = index,
            .physical_layer = layer,
            .pass_index = 0,
            .is_loop = false,
            .is_global = is_global_layer(layer),
        };
        index += 1;
    }

    tiger.assert(index == visit_count);
    return schedule;
}

/// Compute aligned VRAM byte requirement for a single LinearWeight matrix.
pub fn linear_weight_vram_bytes(rows: u32, cols: u32, kind: WeightKind) u64 {
    tiger.assert(rows > 0);
    tiger.assert(cols > 0);
    tiger.assert(cols % 16 == 0);
    return switch (kind) {
        .bf16 => tiger.align_up_u64(@as(u64, rows) * @as(u64, cols) * 2, vram_alignment_bytes),
        .fp8 => tiger.align_up_u64(@as(u64, rows) * @as(u64, cols), vram_alignment_bytes) +
            tiger.align_up_u64(@as(u64, rows) * 4, vram_alignment_bytes),
        .nvfp4 => tiger.align_up_u64(@as(u64, rows) * @as(u64, cols / 2), vram_alignment_bytes) +
            tiger.align_up_u64(@as(u64, rows) * @as(u64, cols / 16), vram_alignment_bytes),
    };
}

/// Compute aligned VRAM byte requirement for a K+V cache pair of `capacity_tokens`.
pub fn kv_cache_pair_vram_bytes(capacity_tokens: u32, kv_format: KVFormat) u64 {
    tiger.assert(capacity_tokens > 0);
    const cap: u64 = capacity_tokens;
    return switch (kv_format) {
        .bf16 => 2 * tiger.align_up_u64(cap * kv_dim * 2, vram_alignment_bytes),
        .fp8 => 2 * tiger.align_up_u64(cap * kv_dim, vram_alignment_bytes) +
            2 * tiger.align_up_u64(cap * kv_head_count * 2, vram_alignment_bytes),
        .q4 => 2 * tiger.align_up_u64(cap * (kv_dim / 2), vram_alignment_bytes) +
            2 * tiger.align_up_u64(cap * kv_q4_scales_per_token * 2, vram_alignment_bytes),
    };
}

// Compile-time invariants verifying all architectural equations hold identically.
comptime {
    if (query_dim != 3072) @compileError("query_dim must be 3072");
    if (kv_dim != 1024) @compileError("kv_dim must be 1024");
    if (qrgs_out_dim != 6552) @compileError("qrgs_out_dim must be 6552");
    if (hidden_dim % 32 != 0) @compileError("hidden_dim must be divisible by 32");
    if (intermediate_dim % 32 != 0) @compileError("intermediate_dim must be divisible by 32");
    if (mtp_intermediate_dim % 32 != 0) @compileError("mtp_intermediate_dim must be divisible by 32");
    if (loop_start_layer + (loop_end_layer - loop_start_layer) * loop_pass_count + (layer_count - loop_end_layer) != visit_count) {
        @compileError("visit schedule length must equal 48");
    }
    if (mtp_block_size_max != mtp_draft_count_max + 1) {
        @compileError("mtp_block_size_max must equal mtp_draft_count_max + 1");
    }
}
