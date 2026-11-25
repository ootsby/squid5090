// TigerStyle two-pass ModelWeights loader.
// Pass 1 computes the exact 256-byte-aligned VRAM requirement.
// Pass 2 allocates ONE contiguous VRAMArena (1 cudaMalloc instead of ~600) and packs every tensor,
// asserting exact fill (`arena.offset_bytes == arena.capacity_bytes`) on completion.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const safetensors = @import("safetensors.zig");
const tiger = @import("tiger.zig");
const vram_arena = @import("vram_arena.zig");

pub const LayerWeights = struct {
    is_global: bool = false,
    has_controllers: bool = false,

    attn_norm: ?*anyopaque = null,
    q_in_norm: ?*anyopaque = null,
    k_in_norm: ?*anyopaque = null,
    v_in_norm: ?*anyopaque = null,
    mlp_norm: ?*anyopaque = null,

    w_qrgs: cuda.CLinearWeight = .{},
    w_kv: cuda.CLinearWeight = .{},
    w_k: cuda.CLinearWeight = .{},
    w_v: cuda.CLinearWeight = .{},
    w_o: cuda.CLinearWeight = .{},

    k_sconv_w: ?*anyopaque = null,
    v_sconv_w: ?*anyopaque = null,
    k_sconv_proj: ?*anyopaque = null,
    v_sconv_proj: ?*anyopaque = null,
    attn_sconv_w: ?*anyopaque = null,
    mlp_sconv_w: ?*anyopaque = null,

    q_clip_scale: ?*anyopaque = null,
    rel_clip_scale: ?*anyopaque = null,
    rel_logits_proj: ?*anyopaque = null,

    ctrl: [4]cuda.CControllerWeights = .{ .{}, .{}, .{}, .{} },

    router_w: ?*anyopaque = null,
    router_nvfp4: cuda.CLinearWeight = .{},
    router_bias: ?[*]f32 = null,
    w13: cuda.CExpertBank = .{},
    w2: cuda.CExpertBank = .{},
};

pub const AblationWeights = struct {
    active: bool = false,
    has_detectors: bool = false,
    attn_dirs: ?[*]f32 = null, // [48 * 2304] unit directions r_v
    mlp_dirs: ?[*]f32 = null, // [48 * 2304] unit directions r_v
    attn_det: ?[*]f32 = null, // [48 * 2304] Fisher-LDA detectors d_v (optional)
    mlp_det: ?[*]f32 = null, // [48 * 2304] Fisher-LDA detectors d_v (optional)
    attn_params: ?[*]f32 = null, // [48 * 4] {weight, offset, floor, threshold}
    mlp_params: ?[*]f32 = null, // [48 * 4] {weight, offset, floor, threshold}
    attn_nonzero: [constants.visit_count]bool = @splat(false),
    mlp_nonzero: [constants.visit_count]bool = @splat(false),
};

pub const ModelWeights = struct {
    mode: constants.QuantMode = .nvfp4,
    arena: vram_arena.VRAMArena = .{},
    pinned_host_embed: ?*anyopaque = null,

    embed: ?*anyopaque = null,
    embed_norm: ?*anyopaque = null,
    final_norm: ?*anyopaque = null,
    unembed: cuda.CLinearWeight = .{},

    attn_queries: [constants.layer_count - 1]?[*]f32 = @splat(null),
    mlp_queries: [constants.layer_count]?[*]f32 = @splat(null),
    output_query: ?[*]f32 = null,
    cross_pass_queries: ?[*]f32 = null,
    cross_pass_kv_queries: ?[*]f32 = null,

    layers: [constants.layer_count]LayerWeights = @splat(.{}),
    dspark: cuda.CDsparkWeights = .{},
    vision: cuda.CVisionWeights = .{},
    draft_vocab_base_ids: []i32 = &.{},
    ablation: AblationWeights = .{},

    pub fn deinit(self: *ModelWeights) void {
        self.arena.deinit();
        if (self.pinned_host_embed) |host_ptr| {
            tiger.assert_cuda(cuda.squid_cuda_free_host(host_ptr));
            self.pinned_host_embed = null;
        }
        if (self.draft_vocab_base_ids.len > 0) {
            tiger.free_host_slice(i32, self.draft_vocab_base_ids);
            self.draft_vocab_base_ids = &.{};
        }
        self.* = .{};
    }

    pub fn load(
        self: *ModelWeights,
        model_directory: []const u8,
        quant_mode: constants.QuantMode,
        load_mtp: bool,
        embed_on_host: bool,
        ablation_path: []const u8,
        enable_vision: bool,
    ) bool {
        tiger.assert(model_directory.len > 0);
        tiger.assert(self.arena.base_ptr == null);

        const start_ns: u64 = tiger.clock_monotonic_ns();
        self.mode = quant_mode;

        var main_path_buf: [1024]u8 = undefined;
        const main_path_z = tiger.format_path_z(&main_path_buf, model_directory, "/model-00001-of-00001.safetensors");
        var sf_main: safetensors.SafetensorsFile = .{};
        if (!sf_main.open_file(main_path_z)) {
            tiger.print_stderr("[weights] Failed to open {s}/model-00001-of-00001.safetensors\n", .{model_directory});
            return false;
        }
        defer sf_main.close_file();

        var mtp_path_buf: [1024]u8 = undefined;
        const mtp_path_z = tiger.format_path_z(&mtp_path_buf, model_directory, "/mtp.safetensors");
        var sf_mtp: safetensors.SafetensorsFile = .{};
        const has_mtp: bool = load_mtp and sf_mtp.open_file(mtp_path_z);
        defer if (has_mtp) sf_mtp.close_file();

        var abl_path_buf: [1024]u8 = undefined;
        var sf_abl: safetensors.SafetensorsFile = .{};
        var has_ablation: bool = false;
        var has_ablation_detectors: bool = false;
        if (ablation_path.len > 0) {
            tiger.assert(ablation_path.len < abl_path_buf.len);
            @memcpy(abl_path_buf[0..ablation_path.len], ablation_path);
            abl_path_buf[ablation_path.len] = 0;
            const abl_path_z: [*:0]const u8 = @ptrCast(abl_path_buf[0..ablation_path.len :0]);
            if (!sf_abl.open_file(abl_path_z)) {
                tiger.print_stderr("[weights] Failed to open ablation safetensors: {s}\n", .{ablation_path});
                return false;
            }
            has_ablation = true;
            const plain_layout: bool = sf_abl.lookup_optional("attn_w") != null;
            has_ablation_detectors = !plain_layout and sf_abl.lookup_optional("attn_det") != null;
        }
        defer if (has_ablation) sf_abl.close_file();

        // Optional IQ3_XXS w13 banks (tools/mkiq3.py), mix mode only. SQUID_IQ3 unset: every layer in iq3.safetensors
        // (mix D) if present. 0: none (mix B). N: layers 0..N-1 from iq3f.safetensors plus an FP8 lm_head (mix F<N>).
        var f_layers: u32 = 0;
        var want_iq3: bool = quant_mode == .mix;
        if (want_iq3) {
            if (tiger.getenv("SQUID_IQ3")) |v| {
                f_layers = std.fmt.parseInt(u32, std.mem.span(v), 10) catch constants.layer_count + 1;
                if (f_layers > constants.layer_count) {
                    tiger.print_stderr("[weights] SQUID_IQ3 must be 0..{d}\n", .{constants.layer_count});
                    return false;
                }
                want_iq3 = f_layers > 0;
            }
        }
        const iq3_file: []const u8 = if (f_layers > 0) "/iq3f.safetensors" else "/iq3.safetensors";
        var iq3_path_buf: [1024]u8 = undefined;
        const iq3_path_z = tiger.format_path_z(&iq3_path_buf, model_directory, iq3_file);
        var sf_iq3: safetensors.SafetensorsFile = .{};
        const has_iq3: bool = want_iq3 and sf_iq3.open_file(iq3_path_z);
        defer if (has_iq3) sf_iq3.close_file();
        const iq3: ?*const safetensors.SafetensorsFile = if (has_iq3) &sf_iq3 else null;
        const iq3_limit: u32 = if (f_layers > 0) f_layers else constants.layer_count;
        var iq3_count: u32 = 0;
        for (0..constants.layer_count) |l| iq3_count += @intFromBool(iq3_bank_present(iq3, @intCast(l), iq3_limit));
        if (f_layers > 0 and iq3_count != f_layers) {
            tiger.print_stderr("[weights] SQUID_IQ3={d} needs layers 0..{d} in {s}{s}\n", .{ f_layers, f_layers - 1, model_directory, iq3_file });
            return false;
        }
        if (has_iq3) tiger.print_stdout("[weights] IQ3_XXS w13 banks: {d} layers from {s}\n", .{ iq3_count, iq3_file[1..] });

        const dense_kind: constants.WeightKind = switch (quant_mode) {
            .exact => .bf16,
            .fp8, .mix => .fp8,
            .nvfp4 => .nvfp4,
        };
        const unembed_kind: constants.WeightKind = if (quant_mode == .mix and f_layers == 0) .nvfp4 else dense_kind;
        const env_mtp_bf16 = tiger.getenv("SQUID_MTP_BF16");
        const force_mtp_bf16: bool = (env_mtp_bf16 != null and env_mtp_bf16.?[0] == '1');
        const mtp_kind: constants.WeightKind = switch (quant_mode) {
            .exact => .bf16,
            .fp8 => if (force_mtp_bf16) .bf16 else .fp8,
            .nvfp4, .mix => if (force_mtp_bf16) .bf16 else .nvfp4,
        };

        // Check optional pruned draft vocab file before Pass 1 so VRAM byte accounting is exact!
        var draft_vocab_host: []i32 = &.{};
        var draft_vocab_count: u32 = constants.vocab_unpadded_count;
        var draft_vocab_cap: u32 = constants.vocab_unpadded_count;
        var has_pruned_draft_vocab: bool = false;
        if (has_mtp) {
            if (try_load_draft_vocab(model_directory, quant_mode)) |loaded_slice| {
                draft_vocab_host = loaded_slice;
                draft_vocab_count = @intCast(loaded_slice.len);
                draft_vocab_cap = @min(constants.vocab_unpadded_count, draft_vocab_count + constants.dspark_dynamic_ctx_rows);
                has_pruned_draft_vocab = true;
            }
        }
        self.draft_vocab_base_ids = draft_vocab_host;

        const has_vision: bool = enable_vision and (sf_main.lookup_optional("model.visual.proj_out.weight") != null);

        // --- PASS 1: Compute Exact 256-Byte-Aligned VRAM Arena Capacity ---
        const total_vram_bytes: u64 = compute_total_vram_bytes(
            quant_mode,
            dense_kind,
            unembed_kind,
            mtp_kind,
            embed_on_host,
            has_mtp,
            has_pruned_draft_vocab,
            draft_vocab_cap,
            has_ablation,
            has_ablation_detectors,
            iq3,
            iq3_limit,
            has_vision,
        );
        self.arena = vram_arena.VRAMArena.init(total_vram_bytes);

        // Temporary staging buffers on GPU for BF16 upload & quantization (freed before returning)
        const stage_bytes: u64 = @as(u64, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
        var d_stage_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_stage_opt, @intCast(stage_bytes)));
        const d_stage: *anyopaque = d_stage_opt.?;
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_stage));

        var d_amax_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_amax_opt, @sizeOf(f32)));
        const d_amax: *f32 = @ptrCast(@alignCast(d_amax_opt.?));
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_amax_opt));

        // --- PASS 2: Upload & Pack Directly Into Contiguous VRAMArena ---

        // Embeddings & top-level norms
        if (embed_on_host) {
            const ei = sf_main.lookup("model.llm.embed.weight");
            const embed_bytes: usize = @as(usize, constants.vocab_padded_count) * constants.hidden_dim * 2;
            tiger.assert(ei.byte_length == embed_bytes);
            var h_ptr: ?*anyopaque = null;
            tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_ptr, embed_bytes, 1));
            tiger.assert(h_ptr != null);
            self.pinned_host_embed = h_ptr;
            const dst_slice = @as([*]u8, @ptrCast(h_ptr.?))[0..embed_bytes];
            @memcpy(dst_slice, sf_main.data_slice(ei));
            var dev_ptr: ?*anyopaque = null;
            tiger.assert_cuda(cuda.squid_cuda_host_get_device_pointer(&dev_ptr, h_ptr.?));
            self.embed = dev_ptr;
            tiger.print_stdout(
                "[weights] Embed table pinned in zero-copy host RAM (saved {d:.2} GiB VRAM)\n",
                .{@as(f64, @floatFromInt(embed_bytes)) / (1024.0 * 1024.0 * 1024.0)},
            );
        } else {
            self.embed = self.upload_bf16(&sf_main, "model.llm.embed.weight", @as(u64, constants.vocab_padded_count) * constants.hidden_dim);
        }
        self.embed_norm = self.upload_bf16(&sf_main, "model.llm.embed_norm.weight", constants.hidden_dim);
        self.final_norm = self.upload_bf16(&sf_main, "model.llm.norm.weight", constants.hidden_dim);

        // Unembed (only first vocab_unpadded_count = 200,058 rows are usable)
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const usable_bytes: usize = @as(usize, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
            tiger.assert(ui.byte_length >= usable_bytes);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, sf_main.data_ptr(ui), usable_bytes));
            self.unembed = self.pack_linear(d_stage, constants.vocab_unpadded_count, constants.hidden_dim, unembed_kind, d_amax);
        }

        // AttnRes & cross-pass queries
        var name_buf: [128]u8 = undefined;
        var layer_idx: u32 = 1;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const elems: u64 = if (constants.is_global_layer(layer_idx)) 3 * constants.hidden_dim else constants.hidden_dim;
            const tname = std.fmt.bufPrint(&name_buf, "model.llm.attn_res.attn_queries.{d}", .{layer_idx - 1}) catch unreachable;
            self.attn_queries[layer_idx - 1] = self.upload_f32(&sf_main, tname, elems);
        }
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const tname = std.fmt.bufPrint(&name_buf, "model.llm.attn_res.mlp_queries.{d}", .{layer_idx}) catch unreachable;
            self.mlp_queries[layer_idx] = self.upload_f32(&sf_main, tname, constants.hidden_dim);
        }
        self.output_query = self.upload_f32(&sf_main, "model.llm.attn_res.output_query", constants.hidden_dim);
        self.cross_pass_queries = self.upload_f32(&sf_main, "model.llm.middle_cycle_cross_pass_queries", 24 * constants.hidden_dim);
        self.cross_pass_kv_queries = self.upload_f32(&sf_main, "model.llm.middle_cycle_cross_pass_kv_queries", 24 * constants.hidden_dim);

        // 24 physical layers
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const lw: *LayerWeights = &self.layers[layer_idx];
            lw.is_global = constants.is_global_layer(layer_idx);

            if (lw.is_global) {
                lw.q_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "q_input_norm.weight", constants.hidden_dim);
                lw.k_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "k_input_norm.weight", constants.hidden_dim);
                lw.v_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "v_input_norm.weight", constants.hidden_dim);
            } else {
                lw.attn_norm = self.upload_layer_bf16(&sf_main, layer_idx, "attn_norm.weight", constants.hidden_dim);
            }
            lw.mlp_norm = self.upload_layer_bf16(&sf_main, layer_idx, "mlp_norm.weight", constants.hidden_dim);

            // Concatenate [wq(3072), wr(384), gate(3072), sink(24)] -> 6552 x 2304
            {
                const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);
                var offset: usize = 0;
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.wq_du.weight", stage_bytes_ptr + offset, @as(u64, constants.query_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.wr_du.weight", stage_bytes_ptr + offset, @as(u64, constants.r_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.gate_attn.weight", stage_bytes_ptr + offset, @as(u64, constants.query_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.attn_sink_proj.weight", stage_bytes_ptr + offset, @as(u64, constants.head_count) * constants.hidden_dim * 2);
                tiger.assert(offset == @as(usize, constants.qrgs_out_dim) * constants.hidden_dim * 2);
                lw.w_qrgs = self.pack_linear(d_stage, constants.qrgs_out_dim, constants.hidden_dim, dense_kind, d_amax);
            }

            // Concatenate [wk(1024), wv(1024)] -> 2048 x 2304, then slice w_k & w_v
            {
                const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);
                const kv_bytes: u64 = @as(u64, constants.kv_dim) * constants.hidden_dim * 2;
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wk_dv.weight", stage_bytes_ptr, kv_bytes);
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wv_dv.weight", stage_bytes_ptr + @as(usize, @intCast(kv_bytes)), kv_bytes);
                lw.w_kv = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, dense_kind, d_amax);

                lw.w_k = lw.w_kv;
                lw.w_k.rows = constants.kv_dim;
                lw.w_v = lw.w_kv;
                lw.w_v.rows = constants.kv_dim;
                const base_data: [*]u8 = @ptrCast(lw.w_kv.data.?);
                switch (dense_kind) {
                    .bf16 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim * 2);
                    },
                    .fp8 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim);
                        lw.w_v.row_scales = lw.w_kv.row_scales.? + constants.kv_dim;
                    },
                    .nvfp4 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * (constants.hidden_dim / 2));
                        lw.w_v.scales = lw.w_kv.scales.? + @as(usize, constants.kv_dim) * (constants.hidden_dim / 16);
                    },
                }
            }

            // w_o [2304, 3072]
            {
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wo_ud.weight", @ptrCast(d_stage), @as(u64, constants.hidden_dim) * constants.query_dim * 2);
                lw.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, dense_kind, d_amax);
            }

            // Short convolutions & clip/rel projections
            if (lw.is_global) {
                lw.k_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.static_w", constants.conv_tap_count * constants.kv_dim);
                lw.v_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.static_w", constants.conv_tap_count * constants.kv_dim);
                lw.k_sconv_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim);
                lw.v_sconv_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim);
            } else {
                lw.k_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.weight", constants.kv_dim * constants.conv_tap_count);
                lw.v_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.weight", constants.kv_dim * constants.conv_tap_count);
            }
            lw.attn_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn_sconv.weight", constants.hidden_dim * constants.conv_tap_count);
            lw.mlp_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "mlp_sconv.weight", constants.hidden_dim * constants.conv_tap_count);
            lw.q_clip_scale = self.upload_layer_bf16(&sf_main, layer_idx, "attn.q_l2_clip.scale", constants.head_count * constants.head_dim);
            lw.rel_clip_scale = self.upload_layer_bf16(&sf_main, layer_idx, "attn.rel_l2_clip.scale", constants.head_count * constants.rel_dim);
            lw.rel_logits_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.rel_logits_proj.proj", constants.rel_dim * constants.rel_extent);

            // Cross-pass controllers (layers 6..17)
            if (layer_idx >= constants.loop_start_layer and layer_idx < constants.loop_end_layer) {
                lw.has_controllers = true;
                const stems: [4][]const u8 = .{ "attn", "k", "v", "mlp" };
                var s_idx: u32 = 0;
                while (s_idx < 4) : (s_idx += 1) {
                    if (quant_mode == .nvfp4) {
                        var tb: [128]u8 = undefined;
                        const dn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.down_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, dn, d_stage, @as(u64, constants.gate_rank) * 2 * constants.hidden_dim * 2);
                        lw.ctrl[s_idx].down_nvfp4 = self.pack_linear(d_stage, constants.gate_rank, 2 * constants.hidden_dim, .nvfp4, d_amax);

                        const fn_name = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, fn_name, d_stage, @as(u64, constants.hidden_dim) * constants.gate_rank * 2);
                        lw.ctrl[s_idx].forget_nvfp4 = self.pack_linear(d_stage, constants.hidden_dim, constants.gate_rank, .nvfp4, d_amax);

                        const gn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, gn, d_stage, @as(u64, constants.hidden_dim) * constants.gate_rank * 2);
                        lw.ctrl[s_idx].gain_nvfp4 = self.pack_linear(d_stage, constants.hidden_dim, constants.gate_rank, .nvfp4, d_amax);
                    } else {
                        var tb: [128]u8 = undefined;
                        const dn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.down_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].down_w = self.upload_bf16(&sf_main, dn, @as(u64, constants.gate_rank) * 2 * constants.hidden_dim);

                        const fn_name = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].forget_w = self.upload_bf16(&sf_main, fn_name, @as(u64, constants.hidden_dim) * constants.gate_rank);

                        const gn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].gain_w = self.upload_bf16(&sf_main, gn, @as(u64, constants.hidden_dim) * constants.gate_rank);
                    }
                }
            }

            // Router
            if (quant_mode == .nvfp4) {
                _ = stage_layer_raw(&sf_main, layer_idx, "mlp.gate.weight", @ptrCast(d_stage), @as(u64, constants.expert_count) * constants.hidden_dim * 2);
                lw.router_nvfp4 = self.pack_linear(d_stage, constants.expert_count, constants.hidden_dim, .nvfp4, d_amax);
            } else {
                lw.router_w = self.upload_layer_bf16(&sf_main, layer_idx, "mlp.gate.weight", @as(u64, constants.expert_count) * constants.hidden_dim);
            }
            const rb_name = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.mlp.gate.bias", .{layer_idx}) catch unreachable;
            lw.router_bias = self.upload_f32(&sf_main, rb_name, constants.expert_count);

            // MoE Expert Banks (w13 and w2)
            lw.w13 = if (iq3_bank_present(iq3, layer_idx, iq3_limit)) self.load_iq3_bank(iq3.?, layer_idx) else self.load_expert_bank(&sf_main, layer_idx, "w13", 2 * constants.intermediate_dim, constants.hidden_dim, quant_mode.expert_mode(), d_stage);
            lw.w2 = self.load_expert_bank(&sf_main, layer_idx, "w2", constants.hidden_dim, constants.intermediate_dim, quant_mode.expert_mode(), d_stage);
        }

        // Load DSpark MTP weights if available
        if (has_mtp) {
            self.load_dspark(
                &sf_main,
                &sf_mtp,
                unembed_kind,
                mtp_kind,
                d_stage,
                d_amax,
                has_pruned_draft_vocab,
                draft_vocab_host,
                draft_vocab_count,
                draft_vocab_cap,
            );
        }

        if (has_ablation) {
            self.load_ablation(&sf_abl, has_ablation_detectors);
        }

        if (has_vision) {
            self.load_vision(&sf_main);
        }

        tiger.assert_cuda(cuda.squid_cuda_device_sync());
        // Paired TigerStyle invariant: Pass 1 predicted byte total == Pass 2 exact arena usage!
        self.arena.assert_exact_fill();

        const elapsed_sec: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - start_ns)) / 1_000_000_000.0;
        const vram_gib: f64 = @as(f64, @floatFromInt(self.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[weights] Loaded mode={s} in {d:.2}s | VRAM={d:.2} GiB (1 arena, {d} tensors) | MTP={s} | Vision={s} | Ablation={s}\n",
            .{
                quant_mode.name(),
                elapsed_sec,
                vram_gib,
                self.arena.allocation_count,
                if (self.dspark.loaded != 0) "on" else "off",
                if (self.vision.loaded != 0) "on" else "off",
                if (!self.ablation.active) "off" else if (self.ablation.has_detectors) "gated-lda" else "plain",
            },
        );
        return true;
    }

    fn upload_bf16(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, elems: u64) *anyopaque {
        const bytes: u64 = elems * 2;
        const dst: *anyopaque = self.arena.alloc_raw(bytes);
        upload_raw(sf, name, dst, bytes);
        return dst;
    }

    fn upload_layer_bf16(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, elems: u64) *anyopaque {
        var name_buf: [128]u8 = undefined;
        const tname = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        return self.upload_bf16(sf, tname, elems);
    }

    fn upload_f32(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, elems: u64) [*]f32 {
        const bytes: u64 = elems * 4;
        const dst: [*]f32 = self.arena.alloc_many(f32, elems);
        upload_raw(sf, name, @ptrCast(dst), bytes);
        return dst;
    }

    fn pack_linear(
        self: *ModelWeights,
        src_bf16_dev: *const anyopaque,
        rows: u32,
        cols: u32,
        kind: constants.WeightKind,
        d_amax: *f32,
    ) cuda.CLinearWeight {
        tiger.assert(rows > 0 and cols > 0 and cols % 16 == 0);
        var lw: cuda.CLinearWeight = .{
            .kind = @intFromEnum(kind),
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .global_scale = 1.0,
        };
        switch (kind) {
            .bf16 => {
                const bytes: u64 = @as(u64, rows) * cols * 2;
                lw.data = self.arena.alloc_raw(bytes);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2d_async(lw.data.?, src_bf16_dev, @intCast(bytes), null));
            },
            .fp8 => {
                lw.data = self.arena.alloc_raw(@as(u64, rows) * cols);
                lw.row_scales = self.arena.alloc_many(f32, rows);
                cuda.squid_pack_bf16_to_fp8(src_bf16_dev, lw.data.?, lw.row_scales.?, @intCast(rows), @intCast(cols));
            },
            .nvfp4 => {
                lw.data = self.arena.alloc_raw(@as(u64, rows) * (cols / 2));
                lw.scales = self.arena.alloc_many(u8, @as(u64, rows) * (cols / 16));
                lw.global_scale = cuda.squid_pack_bf16_to_nvfp4(
                    src_bf16_dev,
                    lw.data.?,
                    lw.scales.?,
                    d_amax,
                    @intCast(rows),
                    @intCast(cols),
                );
            },
        }
        return lw;
    }

    fn load_iq3_bank(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, layer_idx: u32) cuda.CExpertBank {
        var eb: cuda.CExpertBank = .{
            .is_iq3 = 1,
            .n_experts = constants.expert_count,
            .rows = 2 * constants.intermediate_dim,
            .cols = constants.hidden_dim,
        };
        var nbuf: [128]u8 = undefined;
        eb.data = self.arena.alloc_many(u8, constants.iq3_codes_bytes);
        eb.scales = self.arena.alloc_many(u8, constants.iq3_d_bytes);
        upload_raw(sf, iq3_name(&nbuf, layer_idx, "codes"), @ptrCast(eb.data.?), constants.iq3_codes_bytes);
        upload_raw(sf, iq3_name(&nbuf, layer_idx, "d"), @ptrCast(eb.scales.?), constants.iq3_d_bytes);
        eb.expert_scale = self.upload_f32(sf, iq3_name(&nbuf, layer_idx, "scale"), constants.expert_count);
        return eb;
    }

    fn load_expert_bank(
        self: *ModelWeights,
        sf: *const safetensors.SafetensorsFile,
        layer_idx: u32,
        wname: []const u8,
        rows: u32,
        cols: u32,
        quant_mode: constants.QuantMode,
        d_stage: *anyopaque,
    ) cuda.CExpertBank {
        var eb: cuda.CExpertBank = .{
            .n_experts = constants.expert_count,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
        };
        var nbuf: [128]u8 = undefined;
        const ckpt_nvfp4: bool = constants.is_nvfp4_expert_layer(layer_idx);
        if (ckpt_nvfp4) {
            eb.is_nvfp4 = 1;
            const w_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 2);
            const s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 16);
            eb.data = self.arena.alloc_many(u8, w_bytes);
            eb.scales = self.arena.alloc_many(u8, s_bytes);

            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(eb.data.?), w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(eb.scales.?), s_bytes);

            const an = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_amax", .{ layer_idx, wname }) catch unreachable;
            const amax_slice = sf.data_slice(sf.lookup(an));
            const dn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_outer_scale_denominator", .{ layer_idx, wname }) catch unreachable;
            const denom_slice = sf.data_slice(sf.lookup(dn));
            tiger.assert(amax_slice.len == constants.expert_count * 4);
            tiger.assert(denom_slice.len >= 4);

            const denom: f32 = @bitCast(std.mem.readInt(u32, denom_slice[0..4], .little));
            const inv_denom: f32 = 1.0 / denom;
            var h_scales: [constants.expert_count]f32 = undefined;
            var e: u32 = 0;
            while (e < constants.expert_count) : (e += 1) {
                const amax_val: f32 = @bitCast(std.mem.readInt(u32, amax_slice[e * 4 ..][0..4], .little));
                h_scales[e] = amax_val * inv_denom;
            }
            eb.expert_scale = self.arena.alloc_many(f32, constants.expert_count);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(eb.expert_scale.?), @ptrCast(&h_scales), @sizeOf(@TypeOf(h_scales))));
        } else if (quant_mode != .nvfp4) {
            eb.is_nvfp4 = 0;
            const w_bytes: u64 = @as(u64, constants.expert_count) * rows * cols;
            const s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 32);
            eb.data = self.arena.alloc_many(u8, w_bytes);
            eb.scales = self.arena.alloc_many(u8, s_bytes);
            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(eb.data.?), w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(eb.scales.?), s_bytes);
        } else {
            eb.is_nvfp4 = 1;
            const mx_w_bytes: u64 = @as(u64, constants.expert_count) * rows * cols;
            const mx_s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 32);
            const tmp_w: [*]u8 = @ptrCast(d_stage);
            const tmp_s: [*]u8 = tmp_w + @as(usize, @intCast(mx_w_bytes));
            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(tmp_w), mx_w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(tmp_s), mx_s_bytes);

            eb.expert_scale = self.arena.alloc_many(f32, constants.expert_count);
            eb.data = self.arena.alloc_many(u8, @as(u64, constants.expert_count) * rows * (cols / 2));
            eb.scales = self.arena.alloc_many(u8, @as(u64, constants.expert_count) * rows * (cols / 16));
            cuda.squid_convert_mxfp8_to_nvfp4_experts(
                tmp_w,
                tmp_s,
                eb.data.?,
                eb.scales.?,
                eb.expert_scale.?,
                constants.expert_count,
                @intCast(rows),
                @intCast(cols),
            );
        }
        return eb;
    }

    fn load_dspark(
        self: *ModelWeights,
        sf_main: *const safetensors.SafetensorsFile,
        sf_mtp: *const safetensors.SafetensorsFile,
        dense_kind: constants.WeightKind,
        mtp_kind: constants.WeightKind,
        d_stage: *anyopaque,
        d_amax: *f32,
        has_pruned_draft_vocab: bool,
        draft_vocab_host: []const i32,
        draft_vocab_count: u32,
        draft_vocab_cap: u32,
    ) void {
        // fuse_weight [4, 4] BF16 -> row-wise softmax F32 [4, 4]
        const fw_slice = sf_mtp.data_slice(sf_mtp.lookup("fuse_weight"));
        tiger.assert(fw_slice.len == 16 * 2);
        var alpha: [16]f32 = undefined;
        var row: u32 = 0;
        while (row < 4) : (row += 1) {
            var vals: [4]f64 = undefined;
            var max_val: f64 = -1e30;
            var col: u32 = 0;
            while (col < 4) : (col += 1) {
                const bf16_bits: u16 = std.mem.readInt(u16, fw_slice[(row * 4 + col) * 2 ..][0..2], .little);
                const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                const val_f32: f32 = @bitCast(f32_bits);
                vals[col] = @floatCast(val_f32);
                if (vals[col] > max_val) max_val = vals[col];
            }
            var sum_exp: f64 = 0.0;
            col = 0;
            while (col < 4) : (col += 1) {
                vals[col] = @exp(vals[col] - max_val);
                sum_exp += vals[col];
            }
            col = 0;
            while (col < 4) : (col += 1) {
                alpha[row * 4 + col] = @floatCast(vals[col] / sum_exp);
            }
        }
        self.dspark.fuse_alpha = self.arena.alloc_many(f32, 16);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.dspark.fuse_alpha.?), @ptrCast(&alpha), @sizeOf(@TypeOf(alpha))));

        self.dspark.mask_embed = self.upload_bf16(sf_mtp, "mask_embed", constants.hidden_dim);
        self.dspark.norm_w = self.upload_bf16(sf_mtp, "norm.weight", constants.hidden_dim);
        self.dspark.markov_dflare_norm = self.upload_bf16(sf_mtp, "markov_head.dflare_norm.weight", constants.hidden_dim);
        self.dspark.markov_w1 = self.upload_bf16(sf_mtp, "markov_head.markov_w1.weight", @as(u64, constants.vocab_padded_count) * constants.mtp_markov_rank);
        self.dspark.rnn_bias = self.upload_bf16(sf_mtp, "markov_head.rnn_proj.bias", 3 * constants.mtp_markov_rank);
        self.dspark.hidden_out_w = self.upload_bf16(sf_mtp, "markov_head.hidden_out.weight", @as(u64, constants.hidden_dim) * constants.mtp_markov_rank);
        self.dspark.hidden_out_b = self.upload_bf16(sf_mtp, "markov_head.hidden_out.bias", constants.hidden_dim);

        // Split rnn_proj.weight [768, 5120] -> w_sp [768, 512], w_h [768, 2304], w_f [768, 2304]
        {
            const rnn_bytes: u64 = (3 * constants.mtp_markov_rank) * (2 * constants.mtp_markov_rank + 2 * constants.hidden_dim) * 2;
            upload_raw(sf_mtp, "markov_head.rnn_proj.weight", d_stage, rnn_bytes);
            self.dspark.rnn_w_sp = self.arena.alloc_raw(768 * 512 * 2);
            self.dspark.rnn_w_h = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            self.dspark.rnn_w_f = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            cuda.squid_split_dspark_rnn(d_stage, self.dspark.rnn_w_sp.?, self.dspark.rnn_w_h.?, self.dspark.rnn_w_f.?);
        }

        // Folded Markov head (full-vocab BF16 fold_w/fold_b + dynamic-context shortlist unembed_draft!)
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const h_unembed = sf_main.data_slice(ui);
            if (has_pruned_draft_vocab) {
                tiger.assert(draft_vocab_host.len == draft_vocab_count);
                tiger.assert(draft_vocab_cap >= draft_vocab_count);
                self.dspark.draft_vocab_size = @intCast(draft_vocab_count);
                self.dspark.draft_vocab_cap = @intCast(draft_vocab_cap);
                self.dspark.d_draft_vocab_size = self.arena.alloc_many(i32, 1);
                self.dspark.draft_vocab_ids = self.arena.alloc_many(i32, draft_vocab_cap);
                self.dspark.inv_draft_vocab_ids = null;

                // Allocate unembed_draft with exact same format as main unembed so dynamic context rows gather losslessly!
                var ud: cuda.CLinearWeight = .{
                    .kind = @intFromEnum(dense_kind),
                    .rows = @intCast(draft_vocab_cap),
                    .cols = constants.hidden_dim,
                    .global_scale = self.unembed.global_scale,
                };
                switch (dense_kind) {
                    .bf16 => {
                        ud.data = self.arena.alloc_raw(@as(u64, draft_vocab_cap) * constants.hidden_dim * 2);
                    },
                    .fp8 => {
                        ud.data = self.arena.alloc_raw(@as(u64, draft_vocab_cap) * constants.hidden_dim);
                        ud.row_scales = self.arena.alloc_many(f32, draft_vocab_cap);
                    },
                    .nvfp4 => {
                        ud.data = self.arena.alloc_raw(@as(u64, draft_vocab_cap) * (constants.hidden_dim / 2));
                        ud.scales = self.arena.alloc_many(u8, @as(u64, draft_vocab_cap) * (constants.hidden_dim / 16));
                    },
                }
                self.dspark.unembed_draft = ud;

                // Gather initial shortlist rows straight from self.unembed on GPU
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, @ptrCast(draft_vocab_host.ptr), draft_vocab_host.len * @sizeOf(i32)));
                cuda.squid_launch_dspark_gather_shortlist(
                    &self.unembed,
                    &self.dspark.unembed_draft,
                    self.dspark.draft_vocab_ids.?,
                    &self.dspark.d_draft_vocab_size.?[0],
                    @ptrCast(@alignCast(d_stage)),
                    0,
                    @intCast(draft_vocab_count),
                    null,
                );
                tiger.assert_cuda(cuda.squid_cuda_device_sync());
            } else {
                self.dspark.draft_vocab_size = constants.vocab_unpadded_count;
                self.dspark.draft_vocab_cap = constants.vocab_unpadded_count;
                self.dspark.d_draft_vocab_size = null;
                self.dspark.draft_vocab_ids = null;
                self.dspark.inv_draft_vocab_ids = null;
                self.dspark.unembed_draft = self.unembed;
            }

            // Always compute full-vocab BF16 fold_w [200058, 256] & fold_b [200058] indexed by real token ID!
            const full_v: u32 = constants.vocab_unpadded_count;
            const usable_bytes: usize = @as(usize, full_v) * constants.hidden_dim * 2;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, @ptrCast(h_unembed.ptr), usable_bytes));
            self.dspark.fold_b = self.arena.alloc_many(f32, full_v);
            const fold_w_ptr = self.arena.alloc_raw(@as(u64, full_v) * constants.mtp_markov_rank * 2);
            self.dspark.fold_w = .{
                .kind = @intFromEnum(constants.WeightKind.bf16),
                .rows = @intCast(full_v),
                .cols = constants.mtp_markov_rank,
                .data = fold_w_ptr,
            };
            cuda.squid_compute_dspark_fold(
                d_stage,
                self.dspark.hidden_out_w.?,
                self.dspark.hidden_out_b.?,
                self.dspark.fold_b.?,
                fold_w_ptr,
                @intCast(full_v),
                constants.hidden_dim,
                constants.mtp_markov_rank,
            );
            tiger.assert_cuda(cuda.squid_cuda_device_sync());
            tiger.print_stdout(
                "[dspark] Draft vocab size={d}/{d} ({s}, mtp={s}, fold=bf16, topk=1024)\n",
                .{
                    draft_vocab_count,
                    draft_vocab_cap,
                    if (has_pruned_draft_vocab) "pruned+dyn" else "full",
                    @tagName(mtp_kind),
                },
            );
        }

        // 3 DSpark layers
        var l: u32 = 0;
        while (l < constants.mtp_layer_count) : (l += 1) {
            const dl: *cuda.CDsparkLayerWeights = &self.dspark.layers[l];
            dl.is_local = if (constants.is_mtp_local_layer(l)) 1 else 0;
            var nbuf: [128]u8 = undefined;
            const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);

            dl.input_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.post_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.dflare_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.dflare_norm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.q_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, constants.head_dim);
            dl.k_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, constants.head_dim);
            dl.k_norm_ctx = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm_ctx.weight", .{l}) catch unreachable, constants.head_dim);

            // w_qkv [5120, 2304]
            const qb: u64 = @as(u64, constants.query_dim) * constants.hidden_dim * 2;
            const kb: u64 = @as(u64, constants.kv_dim) * constants.hidden_dim * 2;
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), qb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(qb))), kb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(qb + kb))), kb);
            dl.w_qkv = self.pack_linear(d_stage, constants.dspark_qkv_out_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_o [2304, 3072]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, d_stage, @as(u64, constants.hidden_dim) * constants.query_dim * 2);
            dl.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, mtp_kind, d_amax);

            // w_gu [18432, 2304]
            const gb: u64 = @as(u64, constants.mtp_intermediate_dim) * constants.hidden_dim * 2;
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), gb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(gb))), gb);
            dl.w_gu = self.pack_linear(d_stage, 2 * constants.mtp_intermediate_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_down [2304, 9216]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, d_stage, @as(u64, constants.hidden_dim) * constants.mtp_intermediate_dim * 2);
            dl.w_down = self.pack_linear(d_stage, constants.hidden_dim, constants.mtp_intermediate_dim, mtp_kind, d_amax);

            // w_kv_ctx [2048, 2304]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), kb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(kb))), kb);
            dl.w_kv_ctx = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, mtp_kind, d_amax);
        }
        self.dspark.loaded = 1;
    }

    fn load_ablation(self: *ModelWeights, sf_abl: *const safetensors.SafetensorsFile, has_detectors: bool) void {
        const visits: usize = constants.visit_count;
        const hidden: usize = constants.hidden_dim;
        const dir_elems: usize = visits * hidden;
        const plain_layout: bool = sf_abl.lookup_optional("attn_w") != null;

        const dir_host = tiger.alloc_host_zeroed(f32, dir_elems);
        defer tiger.free_host_slice(f32, dir_host);

        var w_host: [constants.visit_count]f32 = @splat(0.0);
        var c_host: [constants.visit_count]f32 = @splat(0.0);
        var tau_host: [constants.visit_count]f32 = @splat(0.0);
        var params_host: [constants.visit_count * 4]f32 = @splat(0.0);

        var gate_val: [1]f32 = .{0.0};
        if (!plain_layout) {
            read_f32_or_bf16(sf_abl, "gate", &gate_val);
        }
        const floor_val: f32 = if (!plain_layout and gate_val[0] != 0.0) 0.0 else -std.math.floatMax(f32);

        // Attention ablation
        read_f32_or_bf16(sf_abl, "attn_dirs", dir_host);
        normalize_rows_f32(dir_host, visits, hidden);
        self.ablation.attn_dirs = self.arena.alloc_many(f32, dir_elems);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_dirs.?), @ptrCast(dir_host.ptr), dir_elems * 4));

        read_f32_or_bf16(sf_abl, if (plain_layout) "attn_w" else "attn_weight", &w_host);
        if (plain_layout) @memset(&c_host, 0.0) else read_f32_or_bf16(sf_abl, "attn_offset", &c_host);
        if (has_detectors) {
            read_f32_or_bf16(sf_abl, "attn_det", dir_host);
            self.ablation.attn_det = self.arena.alloc_many(f32, dir_elems);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_det.?), @ptrCast(dir_host.ptr), dir_elems * 4));
            read_f32_or_bf16(sf_abl, "attn_threshold", &tau_host);
        } else {
            @memset(&tau_host, 0.0);
        }
        for (0..visits) |v| {
            self.ablation.attn_nonzero[v] = (w_host[v] != 0.0);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        self.ablation.attn_params = self.arena.alloc_many(f32, visits * 4);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_params.?), @ptrCast(&params_host), visits * 4 * 4));

        // MLP ablation
        read_f32_or_bf16(sf_abl, "mlp_dirs", dir_host);
        normalize_rows_f32(dir_host, visits, hidden);
        self.ablation.mlp_dirs = self.arena.alloc_many(f32, dir_elems);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_dirs.?), @ptrCast(dir_host.ptr), dir_elems * 4));

        read_f32_or_bf16(sf_abl, if (plain_layout) "mlp_w" else "mlp_weight", &w_host);
        if (plain_layout) @memset(&c_host, 0.0) else read_f32_or_bf16(sf_abl, "mlp_offset", &c_host);
        if (has_detectors) {
            read_f32_or_bf16(sf_abl, "mlp_det", dir_host);
            self.ablation.mlp_det = self.arena.alloc_many(f32, dir_elems);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_det.?), @ptrCast(dir_host.ptr), dir_elems * 4));
            read_f32_or_bf16(sf_abl, "mlp_threshold", &tau_host);
        } else {
            @memset(&tau_host, 0.0);
        }
        for (0..visits) |v| {
            self.ablation.mlp_nonzero[v] = (w_host[v] != 0.0);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        self.ablation.mlp_params = self.arena.alloc_many(f32, visits * 4);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_params.?), @ptrCast(&params_host), visits * 4 * 4));

        self.ablation.active = true;
        self.ablation.has_detectors = has_detectors;
    }

    fn load_vision(self: *ModelWeights, sf: *const safetensors.SafetensorsFile) void {
        const vh: u64 = constants.vision_hidden_dim;
        const vmlp: u64 = constants.vision_mlp_dim;
        const vmerg: u64 = constants.vision_merged_dim;
        const pos_elems: u64 = @as(u64, constants.vision_pos_grid) * constants.vision_pos_grid * vh;

        const host_f32 = tiger.alloc_host_zeroed(f32, @intCast(pos_elems));
        defer tiger.free_host_slice(f32, host_f32);

        const upload_bf16_as_f32 = struct {
            fn run(w: *ModelWeights, sff: *const safetensors.SafetensorsFile, name: []const u8, n: usize, buf: []f32) [*]const f32 {
                read_f32_or_bf16(sff, name, buf[0..n]);
                const dst = w.arena.alloc_many(f32, n);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(dst), @ptrCast(buf.ptr), n * 4));
                return dst;
            }
        }.run;

        self.vision.patch_w = self.upload_bf16(sf, "model.visual.backbone.embeddings.patch_embedding.weight", vh * constants.vision_patch_dim);
        self.vision.pos = upload_bf16_as_f32(self, sf, "model.visual.backbone.embeddings.position_embedding", @intCast(pos_elems), host_f32);

        // cls = class_embedding + cls_position_embedding
        {
            var cls_a: [constants.vision_hidden_dim]f32 = undefined;
            var cls_b: [constants.vision_hidden_dim]f32 = undefined;
            read_f32_or_bf16(sf, "model.visual.backbone.embeddings.class_embedding", &cls_a);
            read_f32_or_bf16(sf, "model.visual.backbone.embeddings.cls_position_embedding", &cls_b);
            for (0..constants.vision_hidden_dim) |c| cls_a[c] += cls_b[c];
            const cls_dev = self.arena.alloc_many(f32, constants.vision_hidden_dim);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(cls_dev), @ptrCast(&cls_a), constants.vision_hidden_dim * 4));
            self.vision.cls = cls_dev;
        }

        self.vision.ln_pre_w = upload_bf16_as_f32(self, sf, "model.visual.backbone.ln_pre.weight", constants.vision_hidden_dim, host_f32);
        self.vision.ln_pre_b = upload_bf16_as_f32(self, sf, "model.visual.backbone.ln_pre.bias", constants.vision_hidden_dim, host_f32);

        var nbuf: [128]u8 = undefined;
        var l: u32 = 0;
        while (l < constants.vision_layer_count) : (l += 1) {
            const vl: *cuda.CVisionLayerWeights = &self.vision.layers[l];
            vl.ln1_w = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ln_1.weight", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.ln1_b = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ln_1.bias", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.ln2_w = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ln_2.weight", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.ln2_b = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ln_2.bias", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.ls1 = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ls_1.gamma", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.ls2 = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.ls_2.gamma", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);

            // Stack Q, K, V weights [3072, 1024] BF16 and biases [3072] F32
            const wqkv_dev = self.arena.alloc_raw(3 * vh * vh * 2);
            const wqkv_bytes: [*]u8 = @ptrCast(wqkv_dev);
            const part_bytes: usize = @intCast(vh * vh * 2);
            const names = [_][]const u8{ "q_proj", "k_proj", "v_proj" };
            for (names, 0..) |nm, pi| {
                const wn = std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.self_attn.{s}.weight", .{ l, nm }) catch unreachable;
                upload_raw(sf, wn, @ptrCast(wqkv_bytes + pi * part_bytes), @intCast(part_bytes));
                const bn = std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.self_attn.{s}.bias", .{ l, nm }) catch unreachable;
                read_f32_or_bf16(sf, bn, host_f32[pi * constants.vision_hidden_dim .. (pi + 1) * constants.vision_hidden_dim]);
            }
            vl.wqkv = wqkv_dev;
            const bqkv_dev = self.arena.alloc_many(f32, 3 * constants.vision_hidden_dim);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(bqkv_dev), @ptrCast(host_f32.ptr), 3 * constants.vision_hidden_dim * 4));
            vl.bqkv = bqkv_dev;

            vl.wo = self.upload_bf16(sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.self_attn.out_proj.weight", .{l}) catch unreachable, vh * vh);
            vl.bo = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.self_attn.out_proj.bias", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
            vl.fc = self.upload_bf16(sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.mlp.c_fc.weight", .{l}) catch unreachable, vmlp * vh);
            vl.bfc = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.mlp.c_fc.bias", .{l}) catch unreachable, constants.vision_mlp_dim, host_f32);
            vl.proj = self.upload_bf16(sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.mlp.c_proj.weight", .{l}) catch unreachable, vh * vmlp);
            vl.bproj = upload_bf16_as_f32(self, sf, std.fmt.bufPrint(&nbuf, "model.visual.backbone.layers.{d}.mlp.c_proj.bias", .{l}) catch unreachable, constants.vision_hidden_dim, host_f32);
        }

        self.vision.proj_in = self.upload_bf16(sf, "model.visual.proj_in.weight", vmerg * vmerg);
        self.vision.proj_out = self.upload_bf16(sf, "model.visual.proj_out.weight", @as(u64, constants.hidden_dim) * vmerg);

        var inv_freq_h: [constants.vision_rope_freqs]f32 = undefined;
        for (0..constants.vision_rope_freqs) |fi| {
            const exp_v: f32 = @as(f32, @floatFromInt(2 * fi)) / @as(f32, @floatFromInt(2 * constants.vision_rope_freqs));
            inv_freq_h[fi] = 1.0 / std.math.pow(f32, 10000.0, exp_v);
        }
        const inv_freq_dev = self.arena.alloc_many(f32, constants.vision_rope_freqs);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(inv_freq_dev), @ptrCast(&inv_freq_h), @sizeOf(@TypeOf(inv_freq_h))));
        self.vision.inv_freq = inv_freq_dev;
        self.vision.loaded = 1;
    }
};

fn read_f32_or_bf16(sf: *const safetensors.SafetensorsFile, name: []const u8, dst: []f32) void {
    const info = sf.lookup(name);
    const raw = sf.data_slice(info);
    if (info.byte_length == dst.len * 4) {
        for (0..dst.len) |i| {
            const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
            dst[i] = @bitCast(bits);
        }
    } else if (info.byte_length == dst.len * 2) {
        for (0..dst.len) |i| {
            const b16: u32 = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
            dst[i] = @bitCast(b16 << 16);
        }
    } else {
        tiger.print_stderr("[ablation] Unexpected byte_length for {s}: got {d}, expected {d} or {d}\n", .{ name, info.byte_length, dst.len * 4, dst.len * 2 });
        @panic("Invalid ablation tensor size");
    }
}

fn normalize_rows_f32(buf: []f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        const row = buf[r * cols .. (r + 1) * cols];
        var ss: f64 = 0.0;
        for (row) |v| {
            const d: f64 = v;
            ss += d * d;
        }
        const norm: f32 = @max(@as(f32, @floatCast(@sqrt(ss))), 1e-12);
        for (row) |*v| v.* /= norm;
    }
}

fn iq3_name(buf: *[128]u8, layer_idx: u32, part: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "model.llm.layers.{d}.mlp.experts.w13_iq3_{s}", .{ layer_idx, part }) catch unreachable;
}

fn iq3_bank_present(iq3: ?*const safetensors.SafetensorsFile, layer_idx: u32, limit: u32) bool {
    if (layer_idx >= limit) return false;
    var nbuf: [128]u8 = undefined;
    const sf = iq3 orelse return false;
    return sf.lookup_optional(iq3_name(&nbuf, layer_idx, "codes")) != null;
}

fn upload_raw(sf: *const safetensors.SafetensorsFile, name: []const u8, dst_dev: *anyopaque, expected_bytes: u64) void {
    const info = sf.lookup(name);
    tiger.assert(info.byte_length == expected_bytes);
    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(dst_dev, sf.data_ptr(info), @intCast(expected_bytes)));
}

fn stage_layer_raw(sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, dst_dev: [*]u8, expected_bytes: u64) usize {
    var name_buf: [128]u8 = undefined;
    const tname = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
    upload_raw(sf, tname, @ptrCast(dst_dev), expected_bytes);
    return @intCast(expected_bytes);
}

const embedded_draft_shortlist_bytes = @embedFile("draft_shortlist.bin");

fn try_load_draft_vocab(model_directory: []const u8, quant_mode: constants.QuantMode) ?[]i32 {
    if (quant_mode == .exact) return null;
    _ = model_directory;
    var shortlist_rows: usize = 32768;
    if (tiger.getenv("SQUID_DRAFT_VOCAB")) |ptr| {
        const s = std.mem.span(ptr);
        if (std.mem.eql(u8, s, "full")) return null;
        if (std.mem.eql(u8, s, "49k")) shortlist_rows = 49152;
        if (std.mem.eql(u8, s, "64k")) shortlist_rows = 65536;
    }
    const vocab_n: usize = constants.vocab_unpadded_count;
    const present = tiger.alloc_host_zeroed(u8, vocab_n);
    defer tiger.free_host_slice(u8, present);

    // Corpus-frequency-ranked top-K shortlist + all special tokens (199990..V), sorted ascending by token ID
    // (tie-breaking matches full unembed; adapted from satellitedown/inkling-2-mini-abliterated-ingine, Apache-2.0).
    const n_ranks: usize = embedded_draft_shortlist_bytes.len / 4;
    var taken: usize = 0;
    var ri: usize = 0;
    while (ri < n_ranks and taken < shortlist_rows) : (ri += 1) {
        const tid = std.mem.readInt(u32, embedded_draft_shortlist_bytes[ri * 4 ..][0..4], .little);
        if (tid < vocab_n and present[tid] == 0) {
            present[tid] = 1;
            taken += 1;
        }
    }
    var sp: usize = 199990;
    while (sp < vocab_n) : (sp += 1) {
        present[sp] = 1;
    }

    var total: usize = 0;
    for (present) |p| total += p;
    if (total == 0) return null;

    const ids = tiger.alloc_host_zeroed(i32, total);
    var out_i: usize = 0;
    for (present, 0..) |p, tid| {
        if (p != 0) {
            ids[out_i] = @intCast(tid);
            out_i += 1;
        }
    }
    return ids;
}

/// Pass 1: exact 256-byte-aligned VRAM byte accounting matching Pass 2 allocation-for-allocation.
fn compute_total_vram_bytes(
    quant_mode: constants.QuantMode,
    dense_kind: constants.WeightKind,
    unembed_kind: constants.WeightKind,
    mtp_kind: constants.WeightKind,
    embed_on_host: bool,
    has_mtp: bool,
    has_pruned_draft_vocab: bool,
    draft_vocab_count: u32,
    has_ablation: bool,
    has_ablation_detectors: bool,
    iq3: ?*const safetensors.SafetensorsFile,
    iq3_limit: u32,
    has_vision: bool,
) u64 {
    const a = constants.vram_alignment_bytes;
    var total: u64 = 0;

    if (!embed_on_host) {
        total += tiger.align_up_u64(@as(u64, constants.vocab_padded_count) * constants.hidden_dim * 2, a);
    }
    total += tiger.align_up_u64(constants.hidden_dim * 2, a); // embed_norm
    total += tiger.align_up_u64(constants.hidden_dim * 2, a); // final_norm
    total += constants.linear_weight_vram_bytes(constants.vocab_unpadded_count, constants.hidden_dim, unembed_kind); // unembed

    var l: u32 = 1;
    while (l < constants.layer_count) : (l += 1) {
        const elems: u64 = if (constants.is_global_layer(l)) 3 * constants.hidden_dim else constants.hidden_dim;
        total += tiger.align_up_u64(elems * 4, a);
    }
    l = 0;
    while (l < constants.layer_count) : (l += 1) {
        total += tiger.align_up_u64(constants.hidden_dim * 4, a);
    }
    total += tiger.align_up_u64(constants.hidden_dim * 4, a); // output_query
    total += tiger.align_up_u64(24 * constants.hidden_dim * 4, a); // cross_pass_queries
    total += tiger.align_up_u64(24 * constants.hidden_dim * 4, a); // cross_pass_kv_queries

    l = 0;
    while (l < constants.layer_count) : (l += 1) {
        const is_glob = constants.is_global_layer(l);
        if (is_glob) {
            total += 3 * tiger.align_up_u64(constants.hidden_dim * 2, a);
        } else {
            total += tiger.align_up_u64(constants.hidden_dim * 2, a);
        }
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // mlp_norm

        total += constants.linear_weight_vram_bytes(constants.qrgs_out_dim, constants.hidden_dim, dense_kind);
        total += constants.linear_weight_vram_bytes(2 * constants.kv_dim, constants.hidden_dim, dense_kind);
        total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.query_dim, dense_kind);

        if (is_glob) {
            total += 2 * tiger.align_up_u64(constants.conv_tap_count * constants.kv_dim * 2, a);
            total += 2 * tiger.align_up_u64(constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim * 2, a);
        } else {
            total += 2 * tiger.align_up_u64(constants.kv_dim * constants.conv_tap_count * 2, a);
        }
        total += 2 * tiger.align_up_u64(constants.hidden_dim * constants.conv_tap_count * 2, a); // attn_sconv, mlp_sconv
        total += tiger.align_up_u64(constants.head_count * constants.head_dim * 2, a); // q_clip_scale
        total += tiger.align_up_u64(constants.head_count * constants.rel_dim * 2, a); // rel_clip_scale
        total += tiger.align_up_u64(constants.rel_dim * constants.rel_extent * 2, a); // rel_logits_proj

        if (l >= constants.loop_start_layer and l < constants.loop_end_layer) {
            var s: u32 = 0;
            while (s < 4) : (s += 1) {
                if (quant_mode == .nvfp4) {
                    total += constants.linear_weight_vram_bytes(constants.gate_rank, 2 * constants.hidden_dim, .nvfp4);
                    total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.gate_rank, .nvfp4);
                    total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.gate_rank, .nvfp4);
                } else {
                    total += tiger.align_up_u64(@as(u64, constants.gate_rank) * 2 * constants.hidden_dim * 2, a);
                    total += 2 * tiger.align_up_u64(@as(u64, constants.hidden_dim) * constants.gate_rank * 2, a);
                }
            }
        }

        if (quant_mode == .nvfp4) {
            total += constants.linear_weight_vram_bytes(constants.expert_count, constants.hidden_dim, .nvfp4);
        } else {
            total += tiger.align_up_u64(@as(u64, constants.expert_count) * constants.hidden_dim * 2, a);
        }
        total += tiger.align_up_u64(constants.expert_count * 4, a); // router_bias

        // Expert banks w13 [2304, 2304] and w2 [2304, 1152]
        const dims: [2][2]u32 = .{
            .{ 2 * constants.intermediate_dim, constants.hidden_dim },
            .{ constants.hidden_dim, constants.intermediate_dim },
        };
        const is_nvfp4_bank: bool = constants.is_nvfp4_expert_layer(l) or quant_mode.expert_mode() == .nvfp4;
        for (dims, 0..) |d, bank| {
            const rows: u64 = d[0];
            const cols: u64 = d[1];
            if (bank == 0 and iq3_bank_present(iq3, l, iq3_limit)) {
                total += tiger.align_up_u64(constants.iq3_codes_bytes, a);
                total += tiger.align_up_u64(constants.iq3_d_bytes, a);
                total += tiger.align_up_u64(constants.expert_count * 4, a);
                continue;
            }
            if (is_nvfp4_bank) {
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 2), a);
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 16), a);
                total += tiger.align_up_u64(constants.expert_count * 4, a);
            } else {
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * cols, a);
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 32), a);
            }
        }
    }

    if (has_mtp) {
        total += tiger.align_up_u64(16 * 4, a); // fuse_alpha
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // mask_embed
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // norm_w
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // markov_dflare_norm
        total += tiger.align_up_u64(@as(u64, constants.vocab_padded_count) * constants.mtp_markov_rank * 2, a); // markov_w1
        total += tiger.align_up_u64(3 * constants.mtp_markov_rank * 2, a); // rnn_bias
        total += tiger.align_up_u64(@as(u64, constants.hidden_dim) * constants.mtp_markov_rank * 2, a); // hidden_out_w
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // hidden_out_b
        total += tiger.align_up_u64(768 * 512 * 2, a); // rnn_w_sp
        total += 2 * tiger.align_up_u64(768 * @as(u64, constants.hidden_dim) * 2, a); // rnn_w_h, rnn_w_f

        if (has_pruned_draft_vocab) {
            total += tiger.align_up_u64(4, a); // d_draft_vocab_size
            total += tiger.align_up_u64(@as(u64, draft_vocab_count) * 4, a); // draft_vocab_ids (cap)
            total += constants.linear_weight_vram_bytes(draft_vocab_count, constants.hidden_dim, unembed_kind); // unembed_draft (cap)
        }
        total += tiger.align_up_u64(@as(u64, constants.vocab_unpadded_count) * 4, a); // fold_b (full)
        total += constants.linear_weight_vram_bytes(constants.vocab_unpadded_count, constants.mtp_markov_rank, .bf16); // fold_w (full BF16)

        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            total += 3 * tiger.align_up_u64(constants.hidden_dim * 2, a); // input/post/dflare norms
            total += 3 * tiger.align_up_u64(constants.head_dim * 2, a); // q/k/k_ctx norms
            total += constants.linear_weight_vram_bytes(constants.dspark_qkv_out_dim, constants.hidden_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.query_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(2 * constants.mtp_intermediate_dim, constants.hidden_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.mtp_intermediate_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(2 * constants.kv_dim, constants.hidden_dim, mtp_kind);
        }
    }

    if (has_ablation) {
        const dir_bytes: u64 = @as(u64, constants.visit_count) * constants.hidden_dim * 4;
        const param_bytes: u64 = @as(u64, constants.visit_count) * 4 * 4;
        // attn_dirs, attn_det?, attn_params, mlp_dirs, mlp_det?, mlp_params
        total += 2 * tiger.align_up_u64(dir_bytes, a);
        total += 2 * tiger.align_up_u64(param_bytes, a);
        if (has_ablation_detectors) {
            total += 2 * tiger.align_up_u64(dir_bytes, a);
        }
    }

    if (has_vision) {
        const vh: u64 = constants.vision_hidden_dim;
        const vmlp: u64 = constants.vision_mlp_dim;
        const vmerg: u64 = constants.vision_merged_dim;
        const pos_elems: u64 = @as(u64, constants.vision_pos_grid) * constants.vision_pos_grid * vh;
        total += tiger.align_up_u64(vh * constants.vision_patch_dim * 2, a); // patch_w
        total += tiger.align_up_u64(pos_elems * 4, a); // pos
        total += 3 * tiger.align_up_u64(vh * 4, a); // cls, ln_pre_w, ln_pre_b
        var vl: u32 = 0;
        while (vl < constants.vision_layer_count) : (vl += 1) {
            total += 6 * tiger.align_up_u64(vh * 4, a); // ln1/2 w/b, ls1/2
            total += tiger.align_up_u64(3 * vh * vh * 2, a); // wqkv
            total += tiger.align_up_u64(3 * vh * 4, a); // bqkv
            total += tiger.align_up_u64(vh * vh * 2, a); // wo
            total += tiger.align_up_u64(vh * 4, a); // bo
            total += tiger.align_up_u64(vmlp * vh * 2, a); // fc
            total += tiger.align_up_u64(vmlp * 4, a); // bfc
            total += tiger.align_up_u64(vh * vmlp * 2, a); // proj
            total += tiger.align_up_u64(vh * 4, a); // bproj
        }
        total += tiger.align_up_u64(vmerg * vmerg * 2, a); // proj_in
        total += tiger.align_up_u64(@as(u64, constants.hidden_dim) * vmerg * 2, a); // proj_out
        total += tiger.align_up_u64(@as(u64, constants.vision_rope_freqs) * 4, a); // inv_freq
    }

    return total;
}
