// TigerStyle Zero-Malloc-Per-Request OpenAI HTTP/1.1 + SSE Server.
// Every buffer (HTTP request, chat template render, token IDs, SSE frame, JSON response,
// UTF-8 stream decoder, and tokenizer scratch) is allocated ONCE at startup in ServerBuffers.
// During request handling and token streaming: ZERO malloc/free calls and ZERO recursion.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const engine_mod = @import("engine.zig");
const tiger = @import("tiger.zig");
const tokenizer_mod = @import("tokenizer.zig");

const request_buffer_bytes: usize = 32 * 1024 * 1024;
const prompt_render_bytes: usize = 32 * 1024 * 1024;
const response_buffer_bytes: usize = 16 * 1024 * 1024;
const content_buffer_bytes: usize = 8 * 1024 * 1024;
const reasoning_buffer_bytes: usize = 8 * 1024 * 1024;
const scratch_string_bytes: usize = 32 * 1024 * 1024;
const sse_frame_bytes: usize = 64 * 1024;
const prompt_tokens_max: usize = 1_048_576;
const stop_sequences_max: usize = 8;
const stop_sequence_max_bytes: usize = 64;
const tool_calls_max: usize = 16;

var g_stop_requested: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var g_listen_fd: std.atomic.Value(i32) = std.atomic.Value(i32).init(-1);

fn handle_signal(_: i32) callconv(.c) void {
    g_stop_requested.store(true, .release);
    const fd = g_listen_fd.swap(-1, .acq_rel);
    if (fd >= 0) _ = tiger.close(fd);
}

pub const ServerConfig = struct {
    host: []const u8 = "0.0.0.0",
    port: u16 = 8000,
    model_name: []const u8 = "inkling-2-mini",
    api_key: []const u8 = "",
    default_effort: []const u8 = "none",
    default_mtp: u32 = 3,
    default_temperature: f32 = 1.0,
    default_top_p: f32 = 0.95,
    default_top_k: i32 = 0,
    default_loopcd_omega: f32 = 0.0,
    default_loopcd_ref: i32 = 1,
    context_tokens_max: u32 = 65536,
    raw_thinking: bool = false,
    keep_history_thinking: bool = false,
    vision_enabled: bool = false,
    image_max_tokens: u32 = constants.vision_max_tokens_default,
    max_images: u32 = constants.vision_max_images_default,
};

const StopList = struct {
    count: usize = 0,
    lengths: [stop_sequences_max]usize = @splat(0),
    storage: [stop_sequences_max][stop_sequence_max_bytes]u8 = undefined,
    max_len: usize = 0,

    fn add(self: *StopList, text: []const u8) void {
        if (text.len == 0 or self.count >= stop_sequences_max or text.len > stop_sequence_max_bytes) return;
        @memcpy(self.storage[self.count][0..text.len], text);
        self.lengths[self.count] = text.len;
        if (text.len > self.max_len) self.max_len = text.len;
        self.count += 1;
    }

    fn find_first(self: *const StopList, haystack: []const u8) ?usize {
        var best: ?usize = null;
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const needle = self.storage[i][0..self.lengths[i]];
            if (std.mem.indexOf(u8, haystack, needle)) |pos| {
                if (best == null or pos < best.?) best = pos;
            }
        }
        return best;
    }
};

const ToolCallRecord = struct {
    id_hex: [12]u8 = undefined,
    name_offset: usize = 0,
    name_len: usize = 0,
    args_offset: usize = 0,
    args_len: usize = 0,
};

const ToolCallIdEntry = struct {
    gen: u32 = 0,
    hash: u64 = 0,
    id_raw: []const u8 = "",
    name_raw: []const u8 = "",
};

const ToolCallIdMap = struct {
    const cap: usize = 16384;
    const mask: usize = cap - 1;
    entries: []ToolCallIdEntry,
    gen: u32 = 1,

    fn init() ToolCallIdMap {
        return .{ .entries = tiger.alloc_host_zeroed(ToolCallIdEntry, cap), .gen = 1 };
    }

    fn deinit(self: *ToolCallIdMap) void {
        tiger.free_host_slice(ToolCallIdEntry, self.entries);
    }

    fn clear(self: *ToolCallIdMap) void {
        self.gen +%= 1;
        if (self.gen == 0) {
            for (self.entries) |*e| e.gen = 0;
            self.gen = 1;
        }
    }

    fn insert(self: *ToolCallIdMap, id_unescaped: []const u8, id_raw: []const u8, name_raw: []const u8) void {
        if (id_unescaped.len == 0 or name_raw.len == 0) return;
        const h = tiger.hash_bytes_u64(id_unescaped);
        var slot: usize = @intCast(h & mask);
        var probe: usize = 0;
        var tmp_buf: [128]u8 = undefined;
        while (probe < 64) : (probe += 1) {
            const e = &self.entries[slot];
            if (e.gen != self.gen) {
                e.* = .{ .gen = self.gen, .hash = h, .id_raw = id_raw, .name_raw = name_raw };
                return;
            }
            if (e.hash == h) {
                var p: usize = 0;
                if (tokenizer_mod.parse_json_string(e.id_raw, &p, &tmp_buf)) |existing_id| {
                    if (std.mem.eql(u8, existing_id, id_unescaped)) {
                        e.name_raw = name_raw;
                        return;
                    }
                }
            }
            slot = (slot + 1) & mask;
        }
    }

    fn lookup(self: *const ToolCallIdMap, target_id: []const u8, out_buf: []u8) ?[]const u8 {
        if (target_id.len == 0) return null;
        const h = tiger.hash_bytes_u64(target_id);
        var slot: usize = @intCast(h & mask);
        var probe: usize = 0;
        var tmp_buf: [128]u8 = undefined;
        while (probe < 64) : (probe += 1) {
            const e = self.entries[slot];
            if (e.gen != self.gen) return null;
            if (e.hash == h) {
                var p: usize = 0;
                if (tokenizer_mod.parse_json_string(e.id_raw, &p, &tmp_buf)) |existing_id| {
                    if (std.mem.eql(u8, existing_id, target_id)) {
                        var np: usize = 0;
                        return tokenizer_mod.parse_json_string(e.name_raw, &np, out_buf);
                    }
                }
            }
            slot = (slot + 1) & mask;
        }
        return null;
    }
};

/// Static preallocated host workspace owned by the server for its entire lifetime.
const ServerBuffers = struct {
    request_buf: []u8,
    render_buf: []u8,
    response_buf: []u8,
    content_buf: []u8,
    reasoning_buf: []u8,
    tool_raw_buf: []u8,
    tool_store_buf: []u8,
    scratch_str_buf: []u8,
    sse_buf: []u8,
    prompt_tokens: []i32,
    tool_id_map: ToolCallIdMap,
    tok_scratch: tokenizer_mod.TokenizerScratch,

    fn init() ServerBuffers {
        return .{
            .request_buf = tiger.alloc_host_zeroed(u8, request_buffer_bytes),
            .render_buf = tiger.alloc_host_zeroed(u8, prompt_render_bytes),
            .response_buf = tiger.alloc_host_zeroed(u8, response_buffer_bytes),
            .content_buf = tiger.alloc_host_zeroed(u8, content_buffer_bytes),
            .reasoning_buf = tiger.alloc_host_zeroed(u8, reasoning_buffer_bytes),
            .tool_raw_buf = tiger.alloc_host_zeroed(u8, 65536),
            .tool_store_buf = tiger.alloc_host_zeroed(u8, 131072),
            .scratch_str_buf = tiger.alloc_host_zeroed(u8, scratch_string_bytes),
            .sse_buf = tiger.alloc_host_zeroed(u8, sse_frame_bytes),
            .prompt_tokens = tiger.alloc_host_zeroed(i32, prompt_tokens_max),
            .tool_id_map = ToolCallIdMap.init(),
            .tok_scratch = tokenizer_mod.TokenizerScratch.init(),
        };
    }

    fn deinit(self: *ServerBuffers) void {
        tiger.free_host_slice(u8, self.request_buf);
        tiger.free_host_slice(u8, self.render_buf);
        tiger.free_host_slice(u8, self.response_buf);
        tiger.free_host_slice(u8, self.content_buf);
        tiger.free_host_slice(u8, self.reasoning_buf);
        tiger.free_host_slice(u8, self.tool_raw_buf);
        tiger.free_host_slice(u8, self.tool_store_buf);
        tiger.free_host_slice(u8, self.scratch_str_buf);
        tiger.free_host_slice(u8, self.sse_buf);
        tiger.free_host_slice(i32, self.prompt_tokens);
        self.tool_id_map.deinit();
        self.tok_scratch.deinit();
    }
};

const InklingSection = enum(u8) { none, header, thinking, text, tool };

const InklingStreamParser = struct {
    tokenizer: *const tokenizer_mod.Tokenizer,
    raw_thinking: bool,
    section: InklingSection = .none,
    pending_utf8: [16]u8 = undefined,
    pending_utf8_len: usize = 0,
    content_writer: tiger.ByteWriter,
    reasoning_writer: tiger.ByteWriter,
    tool_raw_writer: tiger.ByteWriter,
    tool_store_writer: tiger.ByteWriter,
    tool_calls: [tool_calls_max]ToolCallRecord = undefined,
    tool_call_count: usize = 0,
    prng_state: u64,

    fn init(
        tokenizer: *const tokenizer_mod.Tokenizer,
        raw_thinking: bool,
        buffers: *ServerBuffers,
        seed: u64,
    ) InklingStreamParser {
        return .{
            .tokenizer = tokenizer,
            .raw_thinking = raw_thinking,
            .content_writer = tiger.ByteWriter.init(buffers.content_buf),
            .reasoning_writer = tiger.ByteWriter.init(buffers.reasoning_buf),
            .tool_raw_writer = tiger.ByteWriter.init(buffers.tool_raw_buf),
            .tool_store_writer = tiger.ByteWriter.init(buffers.tool_store_buf),
            .prng_state = seed,
        };
    }

    const FeedDelta = struct {
        reasoning: []const u8 = &.{},
        content: []const u8 = &.{},
    };

    fn feed(self: *InklingStreamParser, token_id: i32) FeedDelta {
        if (token_id == self.tokenizer.id_eos) return .{};

        if (self.raw_thinking) {
            const bytes = self.tokenizer.decode_token_bytes(token_id, false);
            return self.push_utf8_bytes(bytes, false);
        }

        if (token_id == self.tokenizer.id_message_model) {
            self.section = .header;
            self.pending_utf8_len = 0;
            self.tool_raw_writer.reset();
            return .{};
        }
        if (token_id == self.tokenizer.id_thinking) {
            self.section = .thinking;
            self.pending_utf8_len = 0;
            return .{};
        }
        if (token_id == self.tokenizer.id_text) {
            self.section = .text;
            self.pending_utf8_len = 0;
            return .{};
        }
        if (self.tokenizer.id_tool_json >= 0 and token_id == self.tokenizer.id_tool_json) {
            self.section = .tool;
            self.tool_raw_writer.reset();
            return .{};
        }
        if (token_id == self.tokenizer.id_end_message) {
            if (self.section == .tool and self.tool_raw_writer.length > 0) {
                self.parse_and_record_tool_call();
                self.tool_raw_writer.reset();
            }
            self.section = .none;
            self.pending_utf8_len = 0;
            return .{};
        }

        if (self.section == .header) return .{};
        if (self.section == .tool) {
            const bytes = self.tokenizer.decode_token_bytes(token_id, true);
            _ = self.tool_raw_writer.try_append_slice(bytes);
            return .{};
        }

        const bytes = self.tokenizer.decode_token_bytes(token_id, false);
        return self.push_utf8_bytes(bytes, self.section == .thinking);
    }

    fn push_utf8_bytes(self: *InklingStreamParser, bytes: []const u8, is_reasoning: bool) FeedDelta {
        if (bytes.len == 0) return .{};
        var combined: [256]u8 = undefined;
        var comb_len: usize = 0;
        if (self.pending_utf8_len > 0) {
            @memcpy(combined[0..self.pending_utf8_len], self.pending_utf8[0..self.pending_utf8_len]);
            comb_len = self.pending_utf8_len;
        }
        const copy_len = @min(bytes.len, combined.len - comb_len);
        @memcpy(combined[comb_len .. comb_len + copy_len], bytes[0..copy_len]);
        comb_len += copy_len;

        const valid_len = valid_utf8_prefix_len(combined[0..comb_len]);
        const rem_len = comb_len - valid_len;
        if (rem_len <= self.pending_utf8.len) {
            @memcpy(self.pending_utf8[0..rem_len], combined[valid_len..comb_len]);
            self.pending_utf8_len = rem_len;
        } else {
            self.pending_utf8_len = 0;
        }
        if (valid_len == 0) return .{};

        const ready = combined[0..valid_len];
        if (is_reasoning) {
            const start = self.reasoning_writer.length;
            if (!self.reasoning_writer.try_append_slice(ready)) return .{};
            return .{ .reasoning = self.reasoning_writer.buffer[start..self.reasoning_writer.length] };
        } else {
            const start = self.content_writer.length;
            if (!self.content_writer.try_append_slice(ready)) return .{};
            return .{ .content = self.content_writer.buffer[start..self.content_writer.length] };
        }
    }

    fn parse_and_record_tool_call(self: *InklingStreamParser) void {
        if (self.tool_call_count >= tool_calls_max) return;
        const raw = self.tool_raw_writer.slice();
        var pos: usize = 0;
        tokenizer_mod.skip_ws(raw, &pos);
        if (pos >= raw.len or raw[pos] != '{') return;
        pos += 1;

        var key_buf: [128]u8 = undefined;
        var name_buf: [256]u8 = undefined;
        var name_slice: []const u8 = "";
        var args_slice: []const u8 = "{}";

        var iter: u32 = 0;
        while (pos < raw.len and iter < 32) : (iter += 1) {
            tokenizer_mod.skip_ws(raw, &pos);
            if (pos < raw.len and raw[pos] == '}') break;
            const k = tokenizer_mod.parse_json_string(raw, &pos, &key_buf) orelse break;
            tokenizer_mod.skip_ws(raw, &pos);
            if (pos >= raw.len or raw[pos] != ':') break;
            pos += 1;
            tokenizer_mod.skip_ws(raw, &pos);

            if (std.mem.eql(u8, k, "name")) {
                name_slice = tokenizer_mod.parse_json_string(raw, &pos, &name_buf) orelse "";
            } else if (std.mem.eql(u8, k, "args") or std.mem.eql(u8, k, "arguments")) {
                const start = pos;
                tokenizer_mod.skip_json_value(raw, &pos);
                args_slice = raw[start..pos];
            } else {
                tokenizer_mod.skip_json_value(raw, &pos);
            }
            tokenizer_mod.skip_ws(raw, &pos);
            if (pos < raw.len and raw[pos] == ',') pos += 1;
        }

        if (name_slice.len == 0) return;

        const name_off = self.tool_store_writer.length;
        if (!self.tool_store_writer.try_append_slice(name_slice)) return;
        const args_off = self.tool_store_writer.length;
        if (!self.tool_store_writer.try_append_slice(args_slice)) return;

        var rec: ToolCallRecord = .{
            .name_offset = name_off,
            .name_len = name_slice.len,
            .args_offset = args_off,
            .args_len = args_slice.len,
        };
        fill_random_hex(&self.prng_state, &rec.id_hex);
        self.tool_calls[self.tool_call_count] = rec;
        self.tool_call_count += 1;
    }
};

pub fn run_openai_server(
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    config: ServerConfig,
) u8 {
    tiger.assert(config.port > 0);
    tiger.assert(config.context_tokens_max >= 64);

    _ = tiger.signal(tiger.SIGPIPE, tiger.SIG_IGN);
    _ = tiger.signal(tiger.SIGINT, handle_signal);
    _ = tiger.signal(tiger.SIGTERM, handle_signal);

    const listen_fd: i32 = tiger.socket(tiger.AF_INET, tiger.SOCK_STREAM, 0);
    if (listen_fd < 0) {
        tiger.print_stderr("[server] Failed to create socket\n", .{});
        return 1;
    }

    var reuse: i32 = 1;
    _ = tiger.setsockopt(listen_fd, tiger.SOL_SOCKET, tiger.SO_REUSEADDR, @ptrCast(&reuse), @sizeOf(i32));

    var addr: tiger.SockaddrIn = .{
        .sin_family = tiger.AF_INET,
        .sin_port = tiger.htons(config.port),
        .sin_addr = 0,
    };
    if (!std.mem.eql(u8, config.host, "0.0.0.0")) {
        var host_z_buf: [128]u8 = undefined;
        tiger.assert(config.host.len < host_z_buf.len);
        @memcpy(host_z_buf[0..config.host.len], config.host);
        host_z_buf[config.host.len] = 0;
        const rc = tiger.inet_pton(tiger.AF_INET, @ptrCast(&host_z_buf), &addr.sin_addr);
        if (rc != 1) {
            tiger.print_stderr("[server] Invalid bind host '{s}'\n", .{config.host});
            _ = tiger.close(listen_fd);
            return 1;
        }
    }

    if (tiger.bind(listen_fd, &addr, @sizeOf(tiger.SockaddrIn)) < 0) {
        tiger.print_stderr("[server] Failed to bind {s}:{d}\n", .{ config.host, config.port });
        _ = tiger.close(listen_fd);
        return 1;
    }
    if (tiger.listen(listen_fd, 128) < 0) {
        tiger.print_stderr("[server] Failed to listen\n", .{});
        _ = tiger.close(listen_fd);
        return 1;
    }

    g_listen_fd.store(listen_fd, .release);
    engine.enable_prefix_cache();
    var buffers = ServerBuffers.init();
    defer buffers.deinit();

    var prng_state: u64 = tiger.clock_monotonic_ns() ^ 0x9e3779b97f4a7c15;
    if (prng_state == 0) prng_state = 0x9e3779b97f4a7c15;

    tiger.print_stdout(
        "[server] TigerStyle Zig OpenAI API listening on http://{s}:{d}/v1 (model={s}, ctx={d}, mtp={d}, zero-malloc=on)\n",
        .{ config.host, config.port, config.model_name, config.context_tokens_max, config.default_mtp },
    );

    const sock_timeout: tiger.Timeval = .{ .tv_sec = 30, .tv_usec = 0 };
    while (!g_stop_requested.load(.acquire)) {
        const client_fd: i32 = tiger.accept(listen_fd, null, null);
        if (client_fd < 0) {
            if (g_stop_requested.load(.acquire)) break;
            if (tiger.get_errno() == tiger.EINTR) continue;
            continue;
        }
        var nodelay: i32 = 1;
        _ = tiger.setsockopt(client_fd, tiger.IPPROTO_TCP, tiger.TCP_NODELAY, @ptrCast(&nodelay), @sizeOf(i32));
        _ = tiger.setsockopt(client_fd, tiger.SOL_SOCKET, tiger.SO_RCVTIMEO, @ptrCast(&sock_timeout), @sizeOf(tiger.Timeval));
        _ = tiger.setsockopt(client_fd, tiger.SOL_SOCKET, tiger.SO_SNDTIMEO, @ptrCast(&sock_timeout), @sizeOf(tiger.Timeval));

        handle_connection(client_fd, engine, tokenizer, &config, &buffers, &prng_state);
        _ = tiger.close(client_fd);
    }

    const rem_fd = g_listen_fd.swap(-1, .acq_rel);
    if (rem_fd >= 0) _ = tiger.close(rem_fd);
    tiger.print_stdout("[server] Shutdown complete.\n", .{});
    return 0;
}

const ParsedHttp = struct {
    method: []const u8,
    path: []const u8,
    auth_header: []const u8,
    body: []const u8,
};

const HttpReadResult = union(enum) {
    ok: ParsedHttp,
    bad_request: []const u8,
    payload_too_large,
    not_implemented: []const u8,
    closed,
};

fn read_http_request(fd: i32, buffer: []u8) HttpReadResult {
    tiger.assert(fd >= 0 and buffer.len > 4096);
    var received: usize = 0;
    var header_end: ?usize = null;

    var recv_iter: u32 = 0;
    while (header_end == null and recv_iter < 4096) : (recv_iter += 1) {
        if (received >= 65536) return .{ .bad_request = "HTTP headers exceed 64 KiB limit" };
        const n: isize = tiger.recv(fd, buffer[received..].ptr, @min(8192, buffer.len - received), 0);
        if (n <= 0) {
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            return .closed;
        }
        received += @intCast(n);
        header_end = std.mem.indexOf(u8, buffer[0..received], "\r\n\r\n");
    }
    const hdr_end = header_end orelse return .closed;
    const hdr_block = buffer[0..hdr_end];

    const line_end = std.mem.indexOf(u8, hdr_block, "\r\n") orelse hdr_block.len;
    const req_line = hdr_block[0..line_end];
    const sp1 = std.mem.indexOfScalar(u8, req_line, ' ') orelse return .{ .bad_request = "Malformed HTTP request line" };
    const sp2_rel = std.mem.indexOfScalar(u8, req_line[sp1 + 1 ..], ' ') orelse return .{ .bad_request = "Malformed HTTP request line" };
    const method = req_line[0..sp1];
    const full_path = req_line[sp1 + 1 .. sp1 + 1 + sp2_rel];
    const path = if (std.mem.indexOfScalar(u8, full_path, '?')) |q| full_path[0..q] else full_path;

    var content_len: usize = 0;
    var has_content_len: bool = false;
    var auth_header: []const u8 = "";
    var expect_continue: bool = false;
    var chunked_te: bool = false;

    var pos: usize = if (line_end + 2 <= hdr_block.len) line_end + 2 else hdr_block.len;
    var hdr_iter: u32 = 0;
    while (pos < hdr_block.len and hdr_iter < 128) : (hdr_iter += 1) {
        const next_rel = std.mem.indexOf(u8, hdr_block[pos..], "\r\n");
        const line = if (next_rel) |nr| hdr_block[pos .. pos + nr] else hdr_block[pos..];
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const hname = trim_ascii_ws(line[0..colon]);
            const hval = trim_ascii_ws(line[colon + 1 ..]);
            if (eql_ignore_case(hname, "content-length")) {
                const parsed_len = std.fmt.parseInt(usize, hval, 10) catch {
                    return .{ .bad_request = "Invalid Content-Length header" };
                };
                if (has_content_len and parsed_len != content_len) {
                    return .{ .bad_request = "Conflicting Content-Length headers" };
                }
                content_len = parsed_len;
                has_content_len = true;
            } else if (eql_ignore_case(hname, "transfer-encoding")) {
                if (contains_ignore_case(hval, "chunked")) {
                    chunked_te = true;
                }
            } else if (eql_ignore_case(hname, "authorization")) {
                auth_header = hval;
            } else if (eql_ignore_case(hname, "expect") and eql_ignore_case(hval, "100-continue")) {
                expect_continue = true;
            }
        }
        if (next_rel) |nr| {
            pos = pos + nr + 2;
        } else {
            break;
        }
    }

    if (chunked_te) {
        return .{ .not_implemented = "Transfer-Encoding: chunked is not supported" };
    }

    const body_start = hdr_end + 4;
    if (content_len > buffer.len - body_start) {
        return .payload_too_large;
    }

    if (expect_continue) {
        _ = send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n");
    }

    var body_iter: u32 = 0;
    while (received < body_start + content_len and body_iter < 262144) : (body_iter += 1) {
        const need = body_start + content_len - received;
        const n: isize = tiger.recv(fd, buffer[received..].ptr, @min(65536, need), 0);
        if (n <= 0) {
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            return .closed;
        }
        received += @intCast(n);
    }
    if (received < body_start + content_len) {
        return .closed;
    }

    return .{ .ok = .{
        .method = method,
        .path = path,
        .auth_header = auth_header,
        .body = buffer[body_start .. body_start + content_len],
    } };
}

fn handle_connection(
    fd: i32,
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    config: *const ServerConfig,
    buffers: *ServerBuffers,
    prng_state: *u64,
) void {
    const req = switch (read_http_request(fd, buffers.request_buf)) {
        .ok => |parsed| parsed,
        .bad_request => |msg| {
            send_error_json(fd, 400, msg, "invalid_request_error", buffers.response_buf);
            return;
        },
        .payload_too_large => {
            send_error_json(fd, 413, "Request body exceeds 32 MiB static HTTP buffer", "invalid_request_error", buffers.response_buf);
            return;
        },
        .not_implemented => |msg| {
            send_error_json(fd, 501, msg, "invalid_request_error", buffers.response_buf);
            return;
        },
        .closed => return,
    };

    if (std.mem.eql(u8, req.method, "OPTIONS")) {
        _ = send_all(fd, "HTTP/1.1 204 No Content\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Headers: Authorization, Content-Type\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Content-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }

    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/health")) {
        var bw = tiger.ByteWriter.init(buffers.response_buf);
        bw.append_slice("{\"status\":\"ok\",\"model\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"max_context\":{d},\"mtp\":{d},\"vision\":{s},\"engine\":\"squid5090\"}}", .{ config.context_tokens_max, config.default_mtp, if (config.vision_enabled) "true" else "false" });
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (config.api_key.len > 0) {
        const prefix = "Bearer ";
        const authorized: bool = std.mem.startsWith(u8, req.auth_header, prefix) and
            constant_time_eql(req.auth_header[prefix.len..], config.api_key);
        if (!authorized) {
            send_error_json(fd, 401, "Invalid API key", "authentication_error", buffers.response_buf);
            return;
        }
    }

    if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, req.path, "/v1/models") or std.mem.eql(u8, req.path, "/models"))) {
        var bw = tiger.ByteWriter.init(buffers.response_buf);
        bw.append_slice("{\"object\":\"list\",\"data\":[{\"id\":");
        bw.append_json_escaped(config.model_name);
        bw.append_fmt(",\"object\":\"model\",\"created\":{d},\"owned_by\":\"squid5090\",\"max_model_len\":{d}}}]}}", .{ tiger.epoch_seconds(), config.context_tokens_max });
        send_json_response(fd, 200, bw.slice());
        return;
    }

    if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/v1/chat/completions") or std.mem.eql(u8, req.path, "/chat/completions"))) {
        handle_chat_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/v1/completions") or std.mem.eql(u8, req.path, "/completions"))) {
        handle_text_completions(fd, req.body, engine, tokenizer, config, buffers, prng_state);
        return;
    }

    send_error_json(fd, 404, "Unknown endpoint", "invalid_request_error", buffers.response_buf);
}

fn handle_chat_completions(
    fd: i32,
    body: []const u8,
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    config: *const ServerConfig,
    buffers: *ServerBuffers,
    prng_state: *u64,
) void {
    // Iterative single-pass JSON request extractor (zero DOM allocation!)
    var pos: usize = 0;
    tokenizer_mod.skip_ws(body, &pos);
    if (pos >= body.len or body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf);
        return;
    }
    pos += 1;

    var messages_json: []const u8 = "";
    var tools_json: []const u8 = "";
    var req_model_buf: [128]u8 = undefined;
    var req_model: []const u8 = config.model_name;
    var effort_buf: [32]u8 = undefined;
    var effort: []const u8 = config.default_effort;
    var max_tokens: i32 = 16384;
    var has_max_completion_tokens: bool = false;
    var req_temp: f32 = config.default_temperature;
    var req_top_p: f32 = config.default_top_p;
    var req_top_k: i32 = config.default_top_k;
    var req_loopcd_omega: f32 = config.default_loopcd_omega;
    var req_loopcd_ref: i32 = config.default_loopcd_ref;
    var has_seed: bool = false;
    var req_seed: u64 = 0;
    var stream: bool = false;
    var include_usage: bool = false;
    var enable_thinking: bool = true;
    var req_mtp: i32 = @intCast(config.default_mtp);
    var stops: StopList = .{};
    var key_buf: [128]u8 = undefined;

    var field_iter: u32 = 0;
    while (pos < body.len and field_iter < 256) : (field_iter += 1) {
        tokenizer_mod.skip_ws(body, &pos);
        if (pos < body.len and body[pos] == '}') break;
        const k = tokenizer_mod.parse_json_string(body, &pos, &key_buf) orelse {
            send_error_json(fd, 400, "Malformed JSON key", "invalid_request_error", buffers.response_buf);
            return;
        };
        tokenizer_mod.skip_ws(body, &pos);
        if (pos >= body.len or body[pos] != ':') {
            send_error_json(fd, 400, "Expected ':' in JSON", "invalid_request_error", buffers.response_buf);
            return;
        }
        pos += 1;
        tokenizer_mod.skip_ws(body, &pos);

        if (std.mem.eql(u8, k, "messages")) {
            const start = pos;
            tokenizer_mod.skip_json_value(body, &pos);
            messages_json = body[start..pos];
        } else if (std.mem.eql(u8, k, "tools")) {
            const start = pos;
            tokenizer_mod.skip_json_value(body, &pos);
            tools_json = body[start..pos];
        } else if (std.mem.eql(u8, k, "model")) {
            if (tokenizer_mod.parse_json_string(body, &pos, &req_model_buf)) |ms| req_model = ms else tokenizer_mod.skip_json_value(body, &pos);
        } else if (std.mem.eql(u8, k, "reasoning_effort")) {
            if (pos < body.len and body[pos] == '"') {
                if (tokenizer_mod.parse_json_string(body, &pos, &effort_buf)) |es| effort = es;
            } else {
                const start = pos;
                tokenizer_mod.skip_json_value(body, &pos);
                const raw_num = trim_ascii_ws(body[start..pos]);
                if (raw_num.len > 0 and raw_num.len <= effort_buf.len and !std.mem.eql(u8, raw_num, "null")) {
                    @memcpy(effort_buf[0..raw_num.len], raw_num);
                    effort = effort_buf[0..raw_num.len];
                }
            }
        } else if (std.mem.eql(u8, k, "max_completion_tokens")) {
            if (parse_json_i64_opt(body, &pos)) |v| {
                max_tokens = clamp_i64_to_i32(v, 1, @intCast(config.context_tokens_max));
                has_max_completion_tokens = true;
            }
        } else if (std.mem.eql(u8, k, "max_tokens")) {
            if (parse_json_i64_opt(body, &pos)) |v| {
                if (!has_max_completion_tokens) max_tokens = clamp_i64_to_i32(v, 1, @intCast(config.context_tokens_max));
            }
        } else if (std.mem.eql(u8, k, "temperature")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_temp = @floatCast(v);
        } else if (std.mem.eql(u8, k, "top_p")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_top_p = @floatCast(v);
        } else if (std.mem.eql(u8, k, "top_k")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_top_k = clamp_i64_to_i32(v, 0, @intCast(constants.vocab_unpadded_count));
        } else if (std.mem.eql(u8, k, "seed")) {
            if (parse_json_i64_opt(body, &pos)) |v| {
                req_seed = @bitCast(v);
                has_seed = true;
            }
        } else if (std.mem.eql(u8, k, "stream")) {
            stream = parse_json_bool(body, &pos);
        } else if (std.mem.eql(u8, k, "enable_thinking")) {
            if (parse_json_bool_opt(body, &pos)) |b| enable_thinking = b;
        } else if (std.mem.eql(u8, k, "stream_options")) {
            const start = pos;
            tokenizer_mod.skip_json_value(body, &pos);
            include_usage = extract_bool_field(body[start..pos], "include_usage");
        } else if (std.mem.eql(u8, k, "mtp")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_mtp = clamp_i64_to_i32(v, 0, @intCast(constants.mtp_draft_count_max));
        } else if (std.mem.eql(u8, k, "loopcd_omega")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_loopcd_omega = @floatCast(v);
        } else if (std.mem.eql(u8, k, "loopcd_ref")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_loopcd_ref = if (v == 2) 2 else 1;
        } else if (std.mem.eql(u8, k, "stop")) {
            parse_stop_field(body, &pos, &stops);
        } else {
            tokenizer_mod.skip_json_value(body, &pos);
        }
        tokenizer_mod.skip_ws(body, &pos);
        if (pos < body.len and body[pos] == ',') pos += 1;
    }

    if (messages_json.len < 2 or messages_json[0] != '[') {
        send_error_json(fd, 400, "'messages' must be a non-empty array", "invalid_request_error", buffers.response_buf);
        return;
    }
    if (max_tokens < 1) max_tokens = 1;
    if (!std.math.isFinite(req_temp) or req_temp < 0.0) req_temp = 0.0;
    if (!std.math.isFinite(req_top_p) or req_top_p <= 0.0 or req_top_p > 1.0) req_top_p = 1.0;
    if (req_top_k < 0) req_top_k = 0;
    if (!std.math.isFinite(req_loopcd_omega) or req_loopcd_omega < -1.0 or req_loopcd_omega > 2.0) req_loopcd_omega = config.default_loopcd_omega;
    if (req_loopcd_ref != 1 and req_loopcd_ref != 2) req_loopcd_ref = config.default_loopcd_ref;
    req_mtp = tiger.max_i32(0, tiger.min_i32(@intCast(constants.mtp_draft_count_max), req_mtp));

    var decoded_imgs: [constants.vision_max_images_default]cuda.CDecodedImage = @splat(.{});
    var decoded_pos: [constants.vision_max_images_default]u32 = @splat(0);
    var decoded_count: usize = 0;
    defer {
        for (decoded_imgs[0..decoded_count]) |*di| cuda.squid_image_free(di);
    }

    // Encode Inkling-2-Mini chat template directly into prompt_tokens (with optional multimodal image parts!):
    // server-emitted control tags use parse_special=true, while all untrusted user/tool/assistant text
    // uses parse_special=false (blocking control-token spoofing AND avoiding 32 MiB intermediate overflow!).
    const enc_res = encode_chat_template_zero_alloc(
        tokenizer,
        messages_json,
        tools_json,
        effort,
        config,
        enable_thinking,
        buffers,
        &decoded_imgs,
        &decoded_pos,
        &decoded_count,
    );
    if (enc_res.img_error.len > 0) {
        send_error_json(fd, 400, enc_res.img_error, "invalid_request_error", buffers.response_buf);
        return;
    }
    if (enc_res.overflow) {
        send_error_json(fd, 413, "Rendered prompt exceeds static buffer or max_context", "invalid_request_error", buffers.response_buf);
        return;
    }
    var prompt_imgs_buf: [constants.vision_max_images_default]engine_mod.PromptImage = undefined;
    for (0..decoded_count) |ii| {
        prompt_imgs_buf[ii] = .{
            .pos = decoded_pos[ii],
            .height = decoded_imgs[ii].height,
            .width = decoded_imgs[ii].width,
            .tokens = @intCast(decoded_imgs[ii].tokens),
            .rgb = decoded_imgs[ii].rgb.?,
        };
    }
    const prompt_images: []const engine_mod.PromptImage = prompt_imgs_buf[0..decoded_count];
    const system_prefix_tokens: u32 = enc_res.system_prefix_tokens;
    const prompt_len: usize = enc_res.prompt_len;
    if (prompt_len == 0 or prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt empty or exceeds max_context", "invalid_request_error", buffers.response_buf);
        return;
    }
    const max_room: u32 = config.context_tokens_max - @as(u32, @intCast(prompt_len));
    const gen_limit: u32 = @min(@as(u32, @intCast(max_tokens)), max_room);

    var req_id_buf: [33]u8 = undefined;
    @memcpy(req_id_buf[0..9], "chatcmpl-");
    fill_random_hex(prng_state, req_id_buf[9..33]);
    const req_id = req_id_buf[0..33];
    const created: i64 = tiger.epoch_seconds();

    var parser = InklingStreamParser.init(tokenizer, config.raw_thinking, buffers, prng_state.*);
    if (!enable_thinking) parser.section = .text;
    prng_state.* = prng_state.* *% 6364136223846793005 +% 1;
    if (!has_seed) {
        req_seed = prng_state.* ^ tiger.clock_monotonic_ns();
        prng_state.* = prng_state.* *% 6364136223846793005 +% 1;
    }
    engine.sample_params = .{
        .temperature = req_temp,
        .top_p = req_top_p,
        .top_k = req_top_k,
        .seed = req_seed,
        .loopcd_omega = req_loopcd_omega,
        .loopcd_ref = req_loopcd_ref,
    };

    if (!stream) {
        const NonStreamCtx = struct {
            parser: *InklingStreamParser,
            stops: *const StopList,
            scanned_len: usize = 0,
            hit_stop: bool = false,

            fn on_token(ctx_ptr: *anyopaque, tid: i32) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
                _ = self.parser.feed(tid);
                if (self.stops.count > 0) {
                    const full = self.parser.content_writer.slice();
                    const overlap: usize = if (self.stops.max_len > 1) self.stops.max_len - 1 else 0;
                    const scan_from: usize = if (self.scanned_len > overlap) self.scanned_len - overlap else 0;
                    if (self.stops.find_first(full[scan_from..])) |rel_idx| {
                        self.parser.content_writer.length = scan_from + rel_idx;
                        self.hit_stop = true;
                        return false;
                    }
                    self.scanned_len = full.len;
                }
                return true;
            }
        };
        var ctx: NonStreamCtx = .{ .parser = &parser, .stops = &stops };
        const stats = engine.generate_with_images(
            buffers.prompt_tokens[0..prompt_len],
            prompt_images,
            system_prefix_tokens,
            gen_limit,
            @intCast(req_mtp),
            &.{},
            NonStreamCtx.on_token,
            @ptrCast(&ctx),
        );

        const hit_eos: bool = (stats.last_token_id == constants.eos_token_id);
        const finish_reason: []const u8 = if (parser.tool_call_count > 0) "tool_calls" else if (ctx.hit_stop or hit_eos) "stop" else "length";
        const dec_toks: f64 = @floatFromInt(if (stats.generated_tokens > 0) stats.generated_tokens - 1 else 0);
        const tps: f64 = if (stats.decode_ms > 0.0) (dec_toks * 1000.0 / stats.decode_ms) else 0.0;
        const acc_pct: f64 = if (stats.total_drafts > 0) (100.0 * @as(f64, @floatFromInt(stats.total_accepted)) / @as(f64, @floatFromInt(stats.total_drafts))) else 0.0;
        const dl: f64 = if (stats.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(stats.spec_steps))) else 1.0;

        var bw = tiger.ByteWriter.init(buffers.response_buf);
        var ok: bool = true;
        ok = ok and bw.try_append_slice("{\"id\":");
        ok = ok and bw.try_append_json_escaped(req_id);
        ok = ok and bw.try_append_fmt(",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{created});
        ok = ok and bw.try_append_json_escaped(req_model);
        ok = ok and bw.try_append_slice(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (parser.content_writer.length > 0 or parser.tool_call_count == 0) {
            ok = ok and bw.try_append_json_escaped(parser.content_writer.slice());
        } else {
            ok = ok and bw.try_append_slice("null");
        }
        if (parser.reasoning_writer.length > 0) {
            ok = ok and bw.try_append_slice(",\"reasoning_content\":");
            ok = ok and bw.try_append_json_escaped(parser.reasoning_writer.slice());
        }
        if (parser.tool_call_count > 0) {
            ok = ok and bw.try_append_slice(",\"tool_calls\":[");
            var ti: usize = 0;
            while (ti < parser.tool_call_count and ok) : (ti += 1) {
                if (ti > 0) ok = ok and bw.try_append_byte(',');
                const tc = parser.tool_calls[ti];
                const tname = parser.tool_store_writer.buffer[tc.name_offset .. tc.name_offset + tc.name_len];
                const targs = parser.tool_store_writer.buffer[tc.args_offset .. tc.args_offset + tc.args_len];
                ok = ok and bw.try_append_slice("{\"id\":\"call_");
                ok = ok and bw.try_append_slice(&tc.id_hex);
                ok = ok and bw.try_append_slice("\",\"type\":\"function\",\"function\":{\"name\":");
                ok = ok and bw.try_append_json_escaped(tname);
                ok = ok and bw.try_append_slice(",\"arguments\":");
                ok = ok and bw.try_append_json_escaped(targs);
                ok = ok and bw.try_append_slice("}}");
            }
            ok = ok and bw.try_append_byte(']');
        }
        ok = ok and bw.try_append_slice("},\"finish_reason\":");
        ok = ok and bw.try_append_json_escaped(finish_reason);
        ok = ok and bw.try_append_fmt("}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}}}},\"timings\":{{\"prefill_ms\":{d:.2},\"decode_ms\":{d:.2},\"tokens_per_second\":{d:.2},\"cached_tokens\":{d},\"spec_acc_pct\":{d:.1},\"spec_dl\":{d:.2}}}}}", .{
            stats.prompt_tokens,
            stats.generated_tokens,
            stats.prompt_tokens + stats.generated_tokens,
            stats.cached_tokens,
            stats.prefill_ms,
            stats.decode_ms,
            tps,
            stats.cached_tokens,
            acc_pct,
            dl,
        });
        if (!ok) {
            send_error_json(fd, 500, "Response exceeded 16 MiB static response buffer", "server_error", buffers.response_buf);
            return;
        }

        if (stats.image_tokens > 0) {
            tiger.print_stdout(
                "[http] POST /v1/chat/completions prompt={d} (cached={d}, img={d}/{d}tok vis={d:.1}ms) gen={d} temp={d:.2} prefill={d:.1}ms decode={d:.1}ms ({d:.1} tok/s, dl={d:.2})\n",
                .{ stats.prompt_tokens, stats.cached_tokens, stats.images_encoded, stats.image_tokens, stats.vision_ms, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl },
            );
        } else {
            tiger.print_stdout(
                "[http] POST /v1/chat/completions prompt={d} (cached={d}) gen={d} temp={d:.2} prefill={d:.1}ms decode={d:.1}ms ({d:.1} tok/s, dl={d:.2})\n",
                .{ stats.prompt_tokens, stats.cached_tokens, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl },
            );
        }
        send_json_response(fd, 200, bw.slice());
        engine.save_post_decode_checkpoint();
        return;
    }

    // --- STREAMING SSE RESPONSE (ZERO MALLOC PER TOKEN!) ---
    if (!send_sse_headers(fd)) return;

    if (!send_chat_sse_delta(fd, buffers.sse_buf, req_id, created, req_model, .role, "assistant", null, null)) return;

    const StreamCtx = struct {
        fd: i32,
        sse_buf: []u8,
        req_id: []const u8,
        created: i64,
        req_model: []const u8,
        parser: *InklingStreamParser,
        stops: *const StopList,
        emitted_len: usize = 0,
        scanned_len: usize = 0,
        hit_stop: bool = false,
        conn_ok: bool = true,

        fn on_token(ctx_ptr: *anyopaque, tid: i32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            const delta = self.parser.feed(tid);
            if (delta.reasoning.len > 0) {
                if (!send_chat_sse_delta(self.fd, self.sse_buf, self.req_id, self.created, self.req_model, .reasoning, delta.reasoning, null, null)) {
                    self.conn_ok = false;
                    return false;
                }
            }
            if (delta.content.len > 0) {
                if (self.stops.count > 0) {
                    const full = self.parser.content_writer.slice();
                    const hold: usize = if (self.stops.max_len > 1) self.stops.max_len - 1 else 0;
                    const scan_from: usize = if (self.scanned_len > hold) self.scanned_len - hold else 0;
                    if (self.stops.find_first(full[scan_from..])) |rel_idx| {
                        const idx = scan_from + rel_idx;
                        if (idx > self.emitted_len) {
                            _ = send_chat_sse_delta(self.fd, self.sse_buf, self.req_id, self.created, self.req_model, .content, full[self.emitted_len..idx], null, null);
                        }
                        self.emitted_len = idx;
                        self.hit_stop = true;
                        return false;
                    }
                    self.scanned_len = full.len;
                    if (full.len > self.emitted_len + hold) {
                        const send_end = full.len - hold;
                        if (!send_chat_sse_delta(self.fd, self.sse_buf, self.req_id, self.created, self.req_model, .content, full[self.emitted_len..send_end], null, null)) {
                            self.conn_ok = false;
                            return false;
                        }
                        self.emitted_len = send_end;
                    }
                } else {
                    if (!send_chat_sse_delta(self.fd, self.sse_buf, self.req_id, self.created, self.req_model, .content, delta.content, null, null)) {
                        self.conn_ok = false;
                        return false;
                    }
                    self.emitted_len = self.parser.content_writer.length;
                }
            }
            return true;
        }
    };

    var sctx: StreamCtx = .{
        .fd = fd,
        .sse_buf = buffers.sse_buf,
        .req_id = req_id,
        .created = created,
        .req_model = req_model,
        .parser = &parser,
        .stops = &stops,
    };

    const stats = engine.generate_with_images(
        buffers.prompt_tokens[0..prompt_len],
        prompt_images,
        system_prefix_tokens,
        gen_limit,
        @intCast(req_mtp),
        &.{},
        StreamCtx.on_token,
        @ptrCast(&sctx),
    );
    if (!sctx.conn_ok) {
        engine.save_post_decode_checkpoint();
        return;
    }

    if (stops.count > 0 and !sctx.hit_stop and sctx.emitted_len < parser.content_writer.length) {
        const rem = parser.content_writer.slice()[sctx.emitted_len..];
        if (!send_chat_sse_delta(fd, buffers.sse_buf, req_id, created, req_model, .content, rem, null, null)) {
            engine.save_post_decode_checkpoint();
            return;
        }
    }

    // Emit streaming OpenAI tool_calls delta chunk(s) before final finish_reason="tool_calls"!
    if (parser.tool_call_count > 0) {
        if (!send_chat_sse_tool_calls(fd, buffers.sse_buf, req_id, created, req_model, &parser)) {
            engine.save_post_decode_checkpoint();
            return;
        }
    }

    const hit_eos: bool = (stats.last_token_id == constants.eos_token_id);
    const finish_reason: []const u8 = if (parser.tool_call_count > 0) "tool_calls" else if (sctx.hit_stop or hit_eos) "stop" else "length";
    const usage_opt: ?engine_mod.GenerateStats = if (include_usage) stats else null;
    _ = send_chat_sse_delta(fd, buffers.sse_buf, req_id, created, req_model, .empty, "", finish_reason, usage_opt);
    _ = send_all(fd, "data: [DONE]\n\n");
    engine.save_post_decode_checkpoint();

    const dec_toks: f64 = @floatFromInt(if (stats.generated_tokens > 0) stats.generated_tokens - 1 else 0);
    const tps: f64 = if (stats.decode_ms > 0.0) (dec_toks * 1000.0 / stats.decode_ms) else 0.0;
    const dl: f64 = if (stats.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(stats.spec_steps))) else 1.0;
    if (stats.image_tokens > 0) {
        tiger.print_stdout(
            "[http] POST /v1/chat/completions (stream) prompt={d} (cached={d}, img={d}/{d}tok vis={d:.1}ms) gen={d} temp={d:.2} prefill={d:.1}ms decode={d:.1}ms ({d:.1} tok/s, dl={d:.2})\n",
            .{ stats.prompt_tokens, stats.cached_tokens, stats.images_encoded, stats.image_tokens, stats.vision_ms, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl },
        );
    } else {
        tiger.print_stdout(
            "[http] POST /v1/chat/completions (stream) prompt={d} (cached={d}) gen={d} temp={d:.2} prefill={d:.1}ms decode={d:.1}ms ({d:.1} tok/s, dl={d:.2})\n",
            .{ stats.prompt_tokens, stats.cached_tokens, stats.generated_tokens, req_temp, stats.prefill_ms, stats.decode_ms, tps, dl },
        );
    }
}

fn handle_text_completions(
    fd: i32,
    body: []const u8,
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    config: *const ServerConfig,
    buffers: *ServerBuffers,
    prng_state: *u64,
) void {
    var pos: usize = 0;
    tokenizer_mod.skip_ws(body, &pos);
    if (pos >= body.len or body[pos] != '{') {
        send_error_json(fd, 400, "Invalid JSON body", "invalid_request_error", buffers.response_buf);
        return;
    }
    pos += 1;

    var prompt: []const u8 = "";
    var max_tokens: i32 = 2048;
    var req_temp: f32 = config.default_temperature;
    var req_top_p: f32 = config.default_top_p;
    var req_top_k: i32 = config.default_top_k;
    var req_loopcd_omega: f32 = config.default_loopcd_omega;
    var req_loopcd_ref: i32 = config.default_loopcd_ref;
    var has_seed: bool = false;
    var req_seed: u64 = 0;
    var stream: bool = false;
    var echo: bool = false;
    var req_mtp: i32 = @intCast(config.default_mtp);
    var req_model_buf: [128]u8 = undefined;
    var req_model: []const u8 = config.model_name;
    var stops: StopList = .{};
    var key_buf: [128]u8 = undefined;

    var field_iter: u32 = 0;
    while (pos < body.len and field_iter < 256) : (field_iter += 1) {
        tokenizer_mod.skip_ws(body, &pos);
        if (pos < body.len and body[pos] == '}') break;
        const k = tokenizer_mod.parse_json_string(body, &pos, &key_buf) orelse break;
        tokenizer_mod.skip_ws(body, &pos);
        if (pos >= body.len or body[pos] != ':') break;
        pos += 1;
        tokenizer_mod.skip_ws(body, &pos);

        if (std.mem.eql(u8, k, "prompt")) {
            if (pos < body.len and body[pos] == '"') {
                prompt = tokenizer_mod.parse_json_string(body, &pos, buffers.render_buf) orelse "";
            } else {
                tokenizer_mod.skip_json_value(body, &pos);
            }
        } else if (std.mem.eql(u8, k, "model")) {
            if (tokenizer_mod.parse_json_string(body, &pos, &req_model_buf)) |ms| req_model = ms else tokenizer_mod.skip_json_value(body, &pos);
        } else if (std.mem.eql(u8, k, "max_tokens")) {
            if (parse_json_i64_opt(body, &pos)) |v| max_tokens = clamp_i64_to_i32(v, 1, @intCast(config.context_tokens_max));
        } else if (std.mem.eql(u8, k, "temperature")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_temp = @floatCast(v);
        } else if (std.mem.eql(u8, k, "top_p")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_top_p = @floatCast(v);
        } else if (std.mem.eql(u8, k, "top_k")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_top_k = clamp_i64_to_i32(v, 0, @intCast(constants.vocab_unpadded_count));
        } else if (std.mem.eql(u8, k, "seed")) {
            if (parse_json_i64_opt(body, &pos)) |v| {
                req_seed = @bitCast(v);
                has_seed = true;
            }
        } else if (std.mem.eql(u8, k, "stream")) {
            stream = parse_json_bool(body, &pos);
        } else if (std.mem.eql(u8, k, "echo")) {
            echo = parse_json_bool(body, &pos);
        } else if (std.mem.eql(u8, k, "mtp")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_mtp = clamp_i64_to_i32(v, 0, @intCast(constants.mtp_draft_count_max));
        } else if (std.mem.eql(u8, k, "loopcd_omega")) {
            if (parse_json_f64_opt(body, &pos)) |v| req_loopcd_omega = @floatCast(v);
        } else if (std.mem.eql(u8, k, "loopcd_ref")) {
            if (parse_json_i64_opt(body, &pos)) |v| req_loopcd_ref = if (v == 2) 2 else 1;
        } else if (std.mem.eql(u8, k, "stop")) {
            parse_stop_field(body, &pos, &stops);
        } else {
            tokenizer_mod.skip_json_value(body, &pos);
        }
        tokenizer_mod.skip_ws(body, &pos);
        if (pos < body.len and body[pos] == ',') pos += 1;
    }

    if (prompt.len == 0) {
        send_error_json(fd, 400, "'prompt' must be a non-empty string", "invalid_request_error", buffers.response_buf);
        return;
    }
    if (max_tokens < 1) max_tokens = 1;
    if (!std.math.isFinite(req_temp) or req_temp < 0.0) req_temp = 0.0;
    if (!std.math.isFinite(req_top_p) or req_top_p <= 0.0 or req_top_p > 1.0) req_top_p = 1.0;
    if (req_top_k < 0) req_top_k = 0;
    if (!std.math.isFinite(req_loopcd_omega) or req_loopcd_omega < -1.0 or req_loopcd_omega > 2.0) req_loopcd_omega = config.default_loopcd_omega;
    if (req_loopcd_ref != 1 and req_loopcd_ref != 2) req_loopcd_ref = config.default_loopcd_ref;
    req_mtp = tiger.max_i32(0, tiger.min_i32(@intCast(constants.mtp_draft_count_max), req_mtp));

    const prompt_len = tokenizer.encode_into(prompt, true, &buffers.tok_scratch, buffers.prompt_tokens);
    if (prompt_len == 0 or prompt_len >= config.context_tokens_max) {
        send_error_json(fd, 400, "Prompt exceeds max_context", "invalid_request_error", buffers.response_buf);
        return;
    }
    const gen_limit: u32 = @min(@as(u32, @intCast(max_tokens)), config.context_tokens_max - @as(u32, @intCast(prompt_len)));

    var req_id_buf: [29]u8 = undefined;
    @memcpy(req_id_buf[0..5], "cmpl-");
    fill_random_hex(prng_state, req_id_buf[5..29]);
    const req_id = req_id_buf[0..29];
    const created: i64 = tiger.epoch_seconds();

    var parser = InklingStreamParser.init(tokenizer, true, buffers, prng_state.*);
    if (!has_seed) {
        req_seed = prng_state.* ^ tiger.clock_monotonic_ns();
        prng_state.* = prng_state.* *% 6364136223846793005 +% 1;
    }
    engine.sample_params = .{
        .temperature = req_temp,
        .top_p = req_top_p,
        .top_k = req_top_k,
        .seed = req_seed,
        .loopcd_omega = req_loopcd_omega,
        .loopcd_ref = req_loopcd_ref,
    };

    if (!stream) {
        const CmplCtx = struct {
            parser: *InklingStreamParser,
            stops: *const StopList,
            scanned_len: usize = 0,
            hit_stop: bool = false,
            fn on_token(ctx_ptr: *anyopaque, tid: i32) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
                if (tid == self.parser.tokenizer.id_eos) return false;
                _ = self.parser.feed(tid);
                if (self.stops.count > 0) {
                    const full = self.parser.content_writer.slice();
                    const overlap: usize = if (self.stops.max_len > 1) self.stops.max_len - 1 else 0;
                    const scan_from: usize = if (self.scanned_len > overlap) self.scanned_len - overlap else 0;
                    if (self.stops.find_first(full[scan_from..])) |rel_idx| {
                        self.parser.content_writer.length = scan_from + rel_idx;
                        self.hit_stop = true;
                        return false;
                    }
                    self.scanned_len = full.len;
                }
                return true;
            }
        };
        var cctx: CmplCtx = .{ .parser = &parser, .stops = &stops };
        const stats = engine.generate(buffers.prompt_tokens[0..prompt_len], gen_limit, @intCast(req_mtp), &.{}, CmplCtx.on_token, @ptrCast(&cctx));
        const hit_eos: bool = (stats.last_token_id == constants.eos_token_id);
        const dec_toks: f64 = @floatFromInt(if (stats.generated_tokens > 0) stats.generated_tokens - 1 else 0);
        const tps: f64 = if (stats.decode_ms > 0.0) (dec_toks * 1000.0 / stats.decode_ms) else 0.0;

        var bw = tiger.ByteWriter.init(buffers.response_buf);
        var ok: bool = true;
        ok = ok and bw.try_append_slice("{\"id\":");
        ok = ok and bw.try_append_json_escaped(req_id);
        ok = ok and bw.try_append_fmt(",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{created});
        ok = ok and bw.try_append_json_escaped(req_model);
        ok = ok and bw.try_append_slice(",\"choices\":[{\"index\":0,\"text\":");
        if (echo) {
            var tw = tiger.ByteWriter.init(buffers.scratch_str_buf);
            _ = tw.try_append_slice(prompt);
            _ = tw.try_append_slice(parser.content_writer.slice());
            ok = ok and bw.try_append_json_escaped(tw.slice());
        } else {
            ok = ok and bw.try_append_json_escaped(parser.content_writer.slice());
        }
        ok = ok and bw.try_append_slice(",\"logprobs\":null,\"finish_reason\":");
        ok = ok and bw.try_append_json_escaped(if (cctx.hit_stop or hit_eos) "stop" else "length");
        ok = ok and bw.try_append_fmt("}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}},\"timings\":{{\"prefill_ms\":{d:.2},\"decode_ms\":{d:.2},\"tokens_per_second\":{d:.2}}}}}", .{
            stats.prompt_tokens,
            stats.generated_tokens,
            stats.prompt_tokens + stats.generated_tokens,
            stats.prefill_ms,
            stats.decode_ms,
            tps,
        });
        if (!ok) {
            send_error_json(fd, 500, "Response exceeded 16 MiB static response buffer", "server_error", buffers.response_buf);
            return;
        }
        send_json_response(fd, 200, bw.slice());
        engine.save_post_decode_checkpoint();
        return;
    }

    if (!send_sse_headers(fd)) return;
    const StreamCmplCtx = struct {
        fd: i32,
        sse_buf: []u8,
        req_id: []const u8,
        created: i64,
        req_model: []const u8,
        parser: *InklingStreamParser,
        conn_ok: bool = true,
        fn on_token(ctx_ptr: *anyopaque, tid: i32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (tid == self.parser.tokenizer.id_eos) return false;
            const d = self.parser.feed(tid);
            if (d.content.len > 0) {
                var bw = tiger.ByteWriter.init(self.sse_buf);
                bw.append_slice("data: {\"id\":");
                bw.append_json_escaped(self.req_id);
                bw.append_fmt(",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{self.created});
                bw.append_json_escaped(self.req_model);
                bw.append_slice(",\"choices\":[{\"index\":0,\"text\":");
                bw.append_json_escaped(d.content);
                bw.append_slice(",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
                if (!send_all(self.fd, bw.slice())) {
                    self.conn_ok = false;
                    return false;
                }
            }
            return true;
        }
    };
    var scctx: StreamCmplCtx = .{ .fd = fd, .sse_buf = buffers.sse_buf, .req_id = req_id, .created = created, .req_model = req_model, .parser = &parser };
    const stats = engine.generate(buffers.prompt_tokens[0..prompt_len], gen_limit, @intCast(req_mtp), &.{}, StreamCmplCtx.on_token, @ptrCast(&scctx));
    if (!scctx.conn_ok) return;
    var bw = tiger.ByteWriter.init(buffers.sse_buf);
    bw.append_slice("data: {\"id\":");
    bw.append_json_escaped(req_id);
    bw.append_fmt(",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{created});
    bw.append_json_escaped(req_model);
    bw.append_slice(",\"choices\":[{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":");
    bw.append_json_escaped(if (stats.last_token_id == constants.eos_token_id) "stop" else "length");
    bw.append_slice("}]}\n\n");
    _ = send_all(fd, bw.slice());
    _ = send_all(fd, "data: [DONE]\n\n");
    engine.save_post_decode_checkpoint();
}

const DeltaKind = enum { role, content, reasoning, empty };

fn send_chat_sse_delta(
    fd: i32,
    sse_buf: []u8,
    req_id: []const u8,
    created: i64,
    req_model: []const u8,
    kind: DeltaKind,
    text: []const u8,
    finish_reason: ?[]const u8,
    usage: ?engine_mod.GenerateStats,
) bool {
    var bw = tiger.ByteWriter.init(sse_buf);
    bw.append_slice("data: {\"id\":");
    bw.append_json_escaped(req_id);
    bw.append_fmt(",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{created});
    bw.append_json_escaped(req_model);
    bw.append_slice(",\"choices\":[{\"index\":0,\"delta\":{");
    switch (kind) {
        .role => {
            bw.append_slice("\"role\":");
            bw.append_json_escaped(text);
        },
        .content => {
            bw.append_slice("\"content\":");
            bw.append_json_escaped(text);
        },
        .reasoning => {
            bw.append_slice("\"reasoning_content\":");
            bw.append_json_escaped(text);
        },
        .empty => {},
    }
    bw.append_slice("},\"finish_reason\":");
    if (finish_reason) |fr| {
        bw.append_json_escaped(fr);
    } else {
        bw.append_slice("null");
    }
    bw.append_slice("}]");
    if (usage) |st| {
        bw.append_fmt(",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}}}}", .{
            st.prompt_tokens,
            st.generated_tokens,
            st.prompt_tokens + st.generated_tokens,
            st.cached_tokens,
        });
    }
    bw.append_slice("}\n\n");
    return send_all(fd, bw.slice());
}

fn send_chat_sse_tool_calls(
    fd: i32,
    sse_buf: []u8,
    req_id: []const u8,
    created: i64,
    req_model: []const u8,
    parser: *const InklingStreamParser,
) bool {
    const arg_chunk_bytes: usize = 4096;
    var ti: usize = 0;
    while (ti < parser.tool_call_count) : (ti += 1) {
        const tc = parser.tool_calls[ti];
        const tname = parser.tool_store_writer.buffer[tc.name_offset .. tc.name_offset + tc.name_len];
        const targs = parser.tool_store_writer.buffer[tc.args_offset .. tc.args_offset + tc.args_len];

        // First chunk: tool call id, type, function name, and initial arguments slice (<= 4 KiB -> <= 24 KiB escaped).
        var first_end: usize = @min(targs.len, arg_chunk_bytes);
        if (first_end < targs.len) {
            var back: usize = 0;
            while (back < 4 and first_end > 1 and (targs[first_end] & 0xC0) == 0x80) : (back += 1) first_end -= 1;
        }
        var bw = tiger.ByteWriter.init(sse_buf);
        var ok: bool = true;
        ok = ok and bw.try_append_slice("data: {\"id\":");
        ok = ok and bw.try_append_json_escaped(req_id);
        ok = ok and bw.try_append_fmt(",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{created});
        ok = ok and bw.try_append_json_escaped(req_model);
        ok = ok and bw.try_append_fmt(",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{d},\"id\":\"call_", .{ti});
        ok = ok and bw.try_append_slice(&tc.id_hex);
        ok = ok and bw.try_append_slice("\",\"type\":\"function\",\"function\":{\"name\":");
        ok = ok and bw.try_append_json_escaped(tname);
        ok = ok and bw.try_append_slice(",\"arguments\":");
        ok = ok and bw.try_append_json_escaped(targs[0..first_end]);
        ok = ok and bw.try_append_slice("}}]},\"finish_reason\":null}]}\n\n");
        if (!ok or !send_all(fd, bw.slice())) return false;

        // Subsequent chunks (if arguments > 4 KiB): standard OpenAI incremental tool_calls arguments deltas.
        var off: usize = first_end;
        while (off < targs.len) {
            var end: usize = @min(targs.len, off + arg_chunk_bytes);
            if (end < targs.len) {
                var back: usize = 0;
                while (back < 4 and end > off + 1 and (targs[end] & 0xC0) == 0x80) : (back += 1) end -= 1;
            }
            bw.reset();
            ok = true;
            ok = ok and bw.try_append_slice("data: {\"id\":");
            ok = ok and bw.try_append_json_escaped(req_id);
            ok = ok and bw.try_append_fmt(",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{created});
            ok = ok and bw.try_append_json_escaped(req_model);
            ok = ok and bw.try_append_fmt(",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{d},\"function\":{{\"arguments\":", .{ti});
            ok = ok and bw.try_append_json_escaped(targs[off..end]);
            ok = ok and bw.try_append_slice("}}]},\"finish_reason\":null}]}\n\n");
            if (!ok or !send_all(fd, bw.slice())) return false;
            off = end;
        }
    }
    return true;
}

const ChatEncodeResult = struct {
    prompt_len: usize = 0,
    system_prefix_tokens: u32 = 0,
    overflow: bool = false,
    img_error: []const u8 = "",
};

fn json_array_has_items(arr: []const u8) bool {
    if (arr.len < 2 or arr[0] != '[') return false;
    var p: usize = 1;
    tokenizer_mod.skip_ws(arr, &p);
    return p < arr.len and arr[p] != ']';
}

fn append_encoded_segment(
    tokenizer: *const tokenizer_mod.Tokenizer,
    text: []const u8,
    parse_special: bool,
    buffers: *ServerBuffers,
    out_len: *usize,
    overflow: *bool,
) void {
    if (text.len == 0 or overflow.*) return;
    const cap = buffers.prompt_tokens.len;
    if (out_len.* >= cap) {
        overflow.* = true;
        return;
    }
    const added = tokenizer.encode_into(text, parse_special, &buffers.tok_scratch, buffers.prompt_tokens[out_len.*..]);
    out_len.* += added;
    if (out_len.* >= cap) overflow.* = true;
}

/// Zero-allocation Inkling-2-Mini chat template encoder operating directly on JSON slices.
/// Encodes server-controlled template markers with `parse_special=true` and all untrusted
/// message/tool text with `parse_special=false`, blocking special-token spoofing attacks!
fn encode_chat_template_zero_alloc(
    tokenizer: *const tokenizer_mod.Tokenizer,
    messages_json: []const u8,
    tools_json: []const u8,
    effort: []const u8,
    config: *const ServerConfig,
    enable_thinking: bool,
    buffers: *ServerBuffers,
    decoded_imgs: []cuda.CDecodedImage,
    decoded_pos: []u32,
    decoded_count: *usize,
) ChatEncodeResult {
    const keep_history_thinking: bool = config.keep_history_thinking;
    var out_len: usize = 0;
    var overflow: bool = false;
    var img_error: []const u8 = "";
    const scratch = buffers.scratch_str_buf;
    buffers.tool_id_map.clear();

    if (json_array_has_items(tools_json)) {
        append_encoded_segment(tokenizer, "<|message_system|>tool_declare<|content_xml|>", true, buffers, &out_len, &overflow);
        var rw = tiger.ByteWriter.init(buffers.render_buf);
        var ok: bool = rw.try_append_byte('[');
        var tp: usize = 1;
        var t_iter: u32 = 0;
        var emitted_tools: u32 = 0;
        while (tp < tools_json.len and t_iter < 1024 and ok) : (t_iter += 1) {
            tokenizer_mod.skip_ws(tools_json, &tp);
            if (tp >= tools_json.len or tools_json[tp] == ']') break;
            const t_start = tp;
            tokenizer_mod.skip_json_value(tools_json, &tp);
            const tool_obj = tools_json[t_start..tp];
            if (tool_obj.len >= 2 and tool_obj[0] == '{') {
                const fn_obj = extract_raw_field(tool_obj, "function") orelse tool_obj;
                var type_buf: [64]u8 = undefined;
                const ttype = extract_string_field(tool_obj, "type", &type_buf) orelse "function";
                var tname_buf: [256]u8 = undefined;
                const tname = extract_string_field(fn_obj, "name", &tname_buf) orelse "";
                const tdesc = extract_string_field(fn_obj, "description", scratch) orelse "";
                const tparams = extract_raw_field(fn_obj, "parameters") orelse "{}";

                if (emitted_tools > 0) ok = ok and rw.try_append_byte(',');
                ok = ok and rw.try_append_slice("{\"description\":");
                ok = ok and rw.try_append_json_escaped(tdesc);
                ok = ok and rw.try_append_slice(",\"name\":");
                ok = ok and rw.try_append_json_escaped(tname);
                ok = ok and rw.try_append_slice(",\"parameters\":");
                ok = ok and rw.try_append_slice(tparams);
                ok = ok and rw.try_append_slice(",\"type\":");
                ok = ok and rw.try_append_json_escaped(ttype);
                ok = ok and rw.try_append_byte('}');
                emitted_tools += 1;
            }
            tokenizer_mod.skip_ws(tools_json, &tp);
            if (tp < tools_json.len and tools_json[tp] == ',') tp += 1;
        }
        ok = ok and rw.try_append_byte(']');
        if (!ok) {
            overflow = true;
        } else {
            append_encoded_segment(tokenizer, rw.slice(), false, buffers, &out_len, &overflow);
        }
        append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
    }

    var system_prefix_tokens: u32 = @intCast(@min(out_len, std.math.maxInt(u32)));
    var effort_emitted: bool = false;
    var effort_render: [128]u8 = undefined;
    var pos: usize = 1; // skip '['
    var msg_iter: u32 = 0;
    while (pos < messages_json.len and msg_iter < 65536 and !overflow) : (msg_iter += 1) {
        tokenizer_mod.skip_ws(messages_json, &pos);
        if (pos >= messages_json.len or messages_json[pos] == ']') break;
        if (messages_json[pos] != '{') {
            tokenizer_mod.skip_json_value(messages_json, &pos);
            tokenizer_mod.skip_ws(messages_json, &pos);
            if (pos < messages_json.len and messages_json[pos] == ',') pos += 1;
            continue;
        }
        const obj_start = pos;
        tokenizer_mod.skip_json_value(messages_json, &pos);
        const msg_obj = messages_json[obj_start..pos];

        var role_buf: [32]u8 = undefined;
        const role = extract_string_field(msg_obj, "role", &role_buf) orelse "user";
        const is_sys: bool = std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer");
        if (!effort_emitted and !is_sys) {
            var ew = tiger.ByteWriter.init(&effort_render);
            _ = tokenizer_mod.Tokenizer.try_emit_thinking_effort(&ew, effort);
            append_encoded_segment(tokenizer, ew.slice(), true, buffers, &out_len, &overflow);
            effort_emitted = true;
            system_prefix_tokens = @intCast(@min(out_len, std.math.maxInt(u32)));
        }

        const role_tok: []const u8 = if (std.mem.eql(u8, role, "user"))
            "<|message_user|>"
        else if (std.mem.eql(u8, role, "assistant"))
            "<|message_model|>"
        else if (is_sys)
            "<|message_system|>"
        else if (std.mem.eql(u8, role, "tool"))
            "<|message_tool|>"
        else
            "<|message_user|>";

        var name_buf: [128]u8 = undefined;
        var resolved_name: ?[]const u8 = extract_string_field(msg_obj, "name", &name_buf);
        if (std.mem.eql(u8, role, "tool") and (resolved_name == null or resolved_name.?.len == 0)) {
            var tcid_buf: [128]u8 = undefined;
            if (extract_string_field(msg_obj, "tool_call_id", &tcid_buf)) |tcid| {
                resolved_name = buffers.tool_id_map.lookup(tcid, &name_buf);
            }
        }
        const is_assistant: bool = std.mem.eql(u8, role, "assistant");
        var emitted_assistant_block: bool = false;

        var has_reasoning: bool = false;
        if (is_assistant and keep_history_thinking) {
            if (extract_string_field(msg_obj, "reasoning_content", scratch)) |rc| {
                if (rc.len > 0) {
                    has_reasoning = true;
                    emitted_assistant_block = true;
                    append_encoded_segment(tokenizer, role_tok, true, buffers, &out_len, &overflow);
                    if (resolved_name) |nm| append_encoded_segment(tokenizer, nm, false, buffers, &out_len, &overflow);
                    append_encoded_segment(tokenizer, "<|content_thinking|>", true, buffers, &out_len, &overflow);
                    append_encoded_segment(tokenizer, rc, false, buffers, &out_len, &overflow);
                    append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                }
            }
        }

        const tc_raw_opt = if (is_assistant) extract_raw_field(msg_obj, "tool_calls") else null;
        const has_tool_calls: bool = (tc_raw_opt != null and json_array_has_items(tc_raw_opt.?));

        // Content (string or array of text parts)
        if (extract_raw_field(msg_obj, "content")) |c_raw| {
            if (c_raw.len > 0 and c_raw[0] == '"') {
                var cp: usize = 0;
                if (tokenizer_mod.parse_json_string(c_raw, &cp, scratch)) |cs| {
                    if (cs.len > 0 or !(is_assistant and (has_tool_calls or has_reasoning))) {
                        emitted_assistant_block = true;
                        append_encoded_segment(tokenizer, role_tok, true, buffers, &out_len, &overflow);
                        if (resolved_name) |nm| append_encoded_segment(tokenizer, nm, false, buffers, &out_len, &overflow);
                        append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, &out_len, &overflow);
                        append_encoded_segment(tokenizer, cs, false, buffers, &out_len, &overflow);
                        append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                    }
                }
            } else if (c_raw.len > 0 and c_raw[0] == '[') {
                var ap: usize = 1;
                var part_iter: u32 = 0;
                while (ap < c_raw.len and part_iter < 1024 and !overflow) : (part_iter += 1) {
                    tokenizer_mod.skip_ws(c_raw, &ap);
                    if (ap >= c_raw.len or c_raw[ap] == ']') break;
                    if (c_raw[ap] == '"') {
                        if (tokenizer_mod.parse_json_string(c_raw, &ap, scratch)) |ps| {
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, &out_len, &overflow);
                            if (resolved_name) |nm| append_encoded_segment(tokenizer, nm, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, ps, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                        }
                    } else if (c_raw[ap] == '{') {
                        const ps_start = ap;
                        tokenizer_mod.skip_json_value(c_raw, &ap);
                        const part_obj = c_raw[ps_start..ap];
                        var ptype_buf: [64]u8 = undefined;
                        const ptype = extract_string_field(part_obj, "type", &ptype_buf) orelse "";
                        const is_img_part: bool = std.mem.eql(u8, ptype, "image_url") or
                            std.mem.eql(u8, ptype, "input_image") or
                            std.mem.eql(u8, ptype, "image");
                        if (is_img_part) {
                            if (!config.vision_enabled) {
                                img_error = "Vision tower is not loaded on this server";
                                return .{ .img_error = img_error };
                            }
                            if (!std.mem.eql(u8, role, "user") and !std.mem.eql(u8, role, "tool")) {
                                img_error = "Image content is only supported for user and tool messages";
                                return .{ .img_error = img_error };
                            }
                            const max_imgs: usize = @min(@as(usize, config.max_images), decoded_imgs.len);
                            if (decoded_count.* >= max_imgs) {
                                img_error = "Request exceeds maximum number of images";
                                return .{ .img_error = img_error };
                            }
                            var uri_slice: ?[]const u8 = null;
                            var b64_slice: ?[]const u8 = null;
                            if (extract_raw_field(part_obj, "image_url")) |iu_raw| {
                                if (iu_raw.len > 0 and iu_raw[0] == '{') {
                                    uri_slice = extract_string_field(iu_raw, "url", scratch);
                                } else if (iu_raw.len > 0 and iu_raw[0] == '"') {
                                    var iup: usize = 0;
                                    uri_slice = tokenizer_mod.parse_json_string(iu_raw, &iup, scratch);
                                }
                            }
                            if (uri_slice == null) uri_slice = extract_string_field(part_obj, "image", scratch);
                            if (uri_slice == null) uri_slice = extract_string_field(part_obj, "url", scratch);
                            if (uri_slice == null) {
                                if (extract_raw_field(part_obj, "source")) |src_obj| {
                                    if (src_obj.len > 0 and src_obj[0] == '{') {
                                        var sk_buf: [32]u8 = undefined;
                                        const sk = extract_string_field(src_obj, "type", &sk_buf) orelse "";
                                        if (std.mem.eql(u8, sk, "base64")) {
                                            b64_slice = extract_string_field(src_obj, "data", scratch);
                                        } else if (std.mem.eql(u8, sk, "url")) {
                                            uri_slice = extract_string_field(src_obj, "url", scratch);
                                        }
                                    }
                                }
                            }
                            if (uri_slice == null and b64_slice == null) {
                                img_error = "Image part requires a base64 data: URI (image_url.url) or base64 source.data";
                                return .{ .img_error = img_error };
                            }
                            const slot_idx: usize = decoded_count.*;
                            const max_tok_i32: i32 = @intCast(config.image_max_tokens);
                            const dec_rc: i32 = if (uri_slice) |u|
                                cuda.squid_image_decode_data_uri(u.ptr, u.len, max_tok_i32, &decoded_imgs[slot_idx])
                            else
                                cuda.squid_image_decode_base64(b64_slice.?.ptr, b64_slice.?.len, max_tok_i32, &decoded_imgs[slot_idx]);
                            if (dec_rc != 0 or decoded_imgs[slot_idx].rgb == null or decoded_imgs[slot_idx].tokens <= 0) {
                                img_error = if (dec_rc == -7)
                                    "Only base64 data: URIs (data:image/...;base64,...) are supported"
                                else if (dec_rc == -5)
                                    "WebP decoding not enabled in this build (use PNG/JPEG/GIF/BMP or build with -Dwebp=true)"
                                else
                                    "Failed to decode image payload";
                                return .{ .img_error = img_error };
                            }
                            const n_img_toks: usize = @intCast(decoded_imgs[slot_idx].tokens);
                            decoded_count.* = slot_idx + 1;

                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, &out_len, &overflow);
                            if (resolved_name) |nm| append_encoded_segment(tokenizer, nm, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|content_image|>", true, buffers, &out_len, &overflow);
                            if (out_len + n_img_toks >= buffers.prompt_tokens.len) {
                                overflow = true;
                            } else {
                                decoded_pos[slot_idx] = @intCast(out_len);
                                const img_hash: u64 = decoded_imgs[slot_idx].hash;
                                var ki: usize = 0;
                                while (ki < n_img_toks) : (ki += 1) {
                                    buffers.prompt_tokens[out_len + ki] = cuda.squid_image_token_id(img_hash, @intCast(ki));
                                }
                                out_len += n_img_toks;
                            }
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                        } else if (extract_string_field(part_obj, "text", scratch)) |ts| {
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, role_tok, true, buffers, &out_len, &overflow);
                            if (resolved_name) |nm| append_encoded_segment(tokenizer, nm, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|content_text|>", true, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, ts, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                        }
                    } else {
                        tokenizer_mod.skip_json_value(c_raw, &ap);
                    }
                    tokenizer_mod.skip_ws(c_raw, &ap);
                    if (ap < c_raw.len and c_raw[ap] == ',') ap += 1;
                }
            }
        }

        // Render prior assistant tool_calls in Inkling's native tool_invoke format
        if (has_tool_calls) {
            const tc_arr = tc_raw_opt.?;
            var tcp: usize = 1;
            var tc_iter: u32 = 0;
            while (tcp < tc_arr.len and tc_iter < 256 and !overflow) : (tc_iter += 1) {
                tokenizer_mod.skip_ws(tc_arr, &tcp);
                if (tcp >= tc_arr.len or tc_arr[tcp] == ']') break;
                const tc_start = tcp;
                tokenizer_mod.skip_json_value(tc_arr, &tcp);
                const tc_obj = tc_arr[tc_start..tcp];
                if (extract_raw_field(tc_obj, "function")) |fn_obj| {
                    var fname_buf: [256]u8 = undefined;
                    if (extract_raw_field(fn_obj, "name")) |fname_raw| {
                        var fnp: usize = 0;
                        if (tokenizer_mod.parse_json_string(fname_raw, &fnp, &fname_buf)) |fname| {
                            if (extract_raw_field(tc_obj, "id")) |id_raw| {
                                var id_buf: [128]u8 = undefined;
                                var idp: usize = 0;
                                if (tokenizer_mod.parse_json_string(id_raw, &idp, &id_buf)) |id_val| {
                                    buffers.tool_id_map.insert(id_val, id_raw, fname_raw);
                                }
                            }
                            emitted_assistant_block = true;
                            append_encoded_segment(tokenizer, "<|message_model|>", true, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, fname, false, buffers, &out_len, &overflow);
                            append_encoded_segment(tokenizer, "<|content_invoke_tool_json|>", true, buffers, &out_len, &overflow);
                            var tw = tiger.ByteWriter.init(buffers.render_buf);
                            var tok_ok: bool = tw.try_append_slice("{\"name\":");
                            tok_ok = tok_ok and tw.try_append_json_escaped(fname);
                            tok_ok = tok_ok and tw.try_append_slice(",\"args\":");
                            if (extract_raw_field(fn_obj, "arguments")) |args_raw| {
                                if (args_raw.len > 0 and args_raw[0] == '"') {
                                    var ap2: usize = 0;
                                    if (tokenizer_mod.parse_json_string(args_raw, &ap2, scratch)) |args_str| {
                                        const trimmed = trim_ascii_ws(args_str);
                                        tok_ok = tok_ok and tw.try_append_slice(if (trimmed.len > 0) trimmed else "{}");
                                    } else {
                                        tok_ok = tok_ok and tw.try_append_slice("{}");
                                    }
                                } else {
                                    tok_ok = tok_ok and tw.try_append_slice(args_raw);
                                }
                            } else {
                                tok_ok = tok_ok and tw.try_append_slice("{}");
                            }
                            tok_ok = tok_ok and tw.try_append_byte('}');
                            if (!tok_ok) {
                                overflow = true;
                            } else {
                                append_encoded_segment(tokenizer, tw.slice(), false, buffers, &out_len, &overflow);
                            }
                            append_encoded_segment(tokenizer, "<|end_message|>", true, buffers, &out_len, &overflow);
                        }
                    }
                }
                tokenizer_mod.skip_ws(tc_arr, &tcp);
                if (tcp < tc_arr.len and tc_arr[tcp] == ',') tcp += 1;
            }
        }

        if (is_assistant and emitted_assistant_block) {
            append_encoded_segment(tokenizer, "<|content_model_end_sampling|>", true, buffers, &out_len, &overflow);
        }
        if (is_sys and !effort_emitted) {
            system_prefix_tokens = @intCast(@min(out_len, std.math.maxInt(u32)));
        }

        tokenizer_mod.skip_ws(messages_json, &pos);
        if (pos < messages_json.len and messages_json[pos] == ',') pos += 1;
    }

    // Never silently truncate trailing messages if the loop cap was hit!
    if (!overflow) {
        tokenizer_mod.skip_ws(messages_json, &pos);
        if (pos < messages_json.len and messages_json[pos] != ']') overflow = true;
    }

    if (!effort_emitted and !overflow) {
        var ew = tiger.ByteWriter.init(&effort_render);
        _ = tokenizer_mod.Tokenizer.try_emit_thinking_effort(&ew, effort);
        append_encoded_segment(tokenizer, ew.slice(), true, buffers, &out_len, &overflow);
        system_prefix_tokens = @intCast(@min(out_len, std.math.maxInt(u32)));
    }
    if (!enable_thinking and !overflow) {
        append_encoded_segment(tokenizer, "<|message_model|><|content_text|>", true, buffers, &out_len, &overflow);
    }
    return .{
        .prompt_len = out_len,
        .system_prefix_tokens = system_prefix_tokens,
        .overflow = overflow,
    };
}

// --- Low-Level Socket, UTF-8 & Iterative JSON Field Helpers ---

fn send_all(fd: i32, data: []const u8) bool {
    var sent: usize = 0;
    var iter: usize = 0;
    const max_iter: usize = data.len + 64;
    while (sent < data.len and iter < max_iter) : (iter += 1) {
        const n: isize = tiger.send(fd, data[sent..].ptr, data.len - sent, tiger.MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            return false;
        }
        sent += @intCast(n);
    }
    return sent == data.len;
}

fn send_json_response(fd: i32, status_code: u16, body: []const u8) void {
    const status_text: []const u8 = switch (status_code) {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        else => "OK",
    };
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: application/json; charset=utf-8\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Content-Length: {d}\r\nConnection: close\r\n\r\n", .{ status_code, status_text, body.len }) catch return;
    _ = send_all(fd, hdr);
    _ = send_all(fd, body);
}

fn send_error_json(fd: i32, status_code: u16, message: []const u8, err_type: []const u8, buf: []u8) void {
    var bw = tiger.ByteWriter.init(buf);
    bw.append_slice("{\"error\":{\"message\":");
    bw.append_json_escaped(message);
    bw.append_slice(",\"type\":");
    bw.append_json_escaped(err_type);
    bw.append_slice("}}");
    send_json_response(fd, status_code, bw.slice());
}

fn send_sse_headers(fd: i32) bool {
    return send_all(fd, "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream; charset=utf-8\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Connection: close\r\n\r\n");
}

fn valid_utf8_prefix_len(s: []const u8) usize {
    const n = s.len;
    if (n == 0) return 0;
    const check = @min(n, 3);
    var back: usize = 1;
    while (back <= check) : (back += 1) {
        const c = s[n - back];
        if ((c & 0x80) == 0) return n;
        if ((c & 0xC0) == 0xC0) {
            const expected: usize = if ((c & 0xE0) == 0xC0) 2 else if ((c & 0xF0) == 0xE0) 3 else if ((c & 0xF8) == 0xF0) 4 else 1;
            return if (back < expected) (n - back) else n;
        }
    }
    return n;
}

fn fill_random_hex(state: *u64, out: []u8) void {
    if (state.* == 0) state.* = 0x9e3779b97f4a7c15;
    const hex = "0123456789abcdef";
    for (out) |*dst| {
        state.* ^= state.* >> 12;
        state.* ^= state.* << 25;
        state.* ^= state.* >> 27;
        const v = state.* *% 0x2545F4914F6CDD1D;
        dst.* = hex[@intCast((v >> 60) & 0x0f)];
    }
}

fn constant_time_eql(a: []const u8, b: []const u8) bool {
    var diff: u8 = if (a.len == b.len) 0 else 1;
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

fn contains_ignore_case(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (eql_ignore_case(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn clamp_i64_to_i32(v: i64, lo: i32, hi: i32) i32 {
    if (v < @as(i64, lo)) return lo;
    if (v > @as(i64, hi)) return hi;
    return @intCast(v);
}

fn trim_ascii_ws(s: []const u8) []const u8 {
    var a: usize = 0;
    var b: usize = s.len;
    while (a < b and (s[a] == ' ' or s[a] == '\t' or s[a] == '\r' or s[a] == '\n')) : (a += 1) {}
    while (b > a and (s[b - 1] == ' ' or s[b - 1] == '\t' or s[b - 1] == '\r' or s[b - 1] == '\n')) : (b -= 1) {}
    return s[a..b];
}

fn eql_ignore_case(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        const la = if (ca >= 'A' and ca <= 'Z') ca + 32 else ca;
        const lb = if (cb >= 'A' and cb <= 'Z') cb + 32 else cb;
        if (la != lb) return false;
    }
    return true;
}

fn parse_json_i64_opt(data: []const u8, pos: *usize) ?i64 {
    tokenizer_mod.skip_ws(data, pos);
    if (pos.* + 4 <= data.len and std.mem.eql(u8, data[pos.* .. pos.* + 4], "null")) {
        pos.* += 4;
        return null;
    }
    const start = pos.*;
    tokenizer_mod.skip_json_value(data, pos);
    const raw = trim_ascii_ws(data[start..pos.*]);
    if (raw.len == 0) return null;
    return std.fmt.parseInt(i64, raw, 10) catch blk: {
        const f = std.fmt.parseFloat(f64, raw) catch break :blk null;
        if (!std.math.isFinite(f)) break :blk null;
        if (f >= 9.0e18) break :blk std.math.maxInt(i64);
        if (f <= -9.0e18) break :blk std.math.minInt(i64);
        break :blk @intFromFloat(f);
    };
}

fn parse_json_f64_opt(data: []const u8, pos: *usize) ?f64 {
    tokenizer_mod.skip_ws(data, pos);
    if (pos.* + 4 <= data.len and std.mem.eql(u8, data[pos.* .. pos.* + 4], "null")) {
        pos.* += 4;
        return null;
    }
    const start = pos.*;
    tokenizer_mod.skip_json_value(data, pos);
    const raw = trim_ascii_ws(data[start..pos.*]);
    if (raw.len == 0) return null;
    const f = std.fmt.parseFloat(f64, raw) catch return null;
    if (!std.math.isFinite(f)) return null;
    return f;
}

fn parse_json_bool_opt(data: []const u8, pos: *usize) ?bool {
    tokenizer_mod.skip_ws(data, pos);
    if (pos.* + 4 <= data.len and std.mem.eql(u8, data[pos.* .. pos.* + 4], "true")) {
        pos.* += 4;
        return true;
    }
    if (pos.* + 5 <= data.len and std.mem.eql(u8, data[pos.* .. pos.* + 5], "false")) {
        pos.* += 5;
        return false;
    }
    tokenizer_mod.skip_json_value(data, pos);
    return null;
}

fn parse_json_bool(data: []const u8, pos: *usize) bool {
    return parse_json_bool_opt(data, pos) orelse false;
}

fn parse_stop_field(data: []const u8, pos: *usize, stops: *StopList) void {
    tokenizer_mod.skip_ws(data, pos);
    var tmp: [stop_sequence_max_bytes]u8 = undefined;
    if (pos.* < data.len and data[pos.*] == '"') {
        if (tokenizer_mod.parse_json_string(data, pos, &tmp)) |s| stops.add(s);
    } else if (pos.* < data.len and data[pos.*] == '[') {
        pos.* += 1;
        var iter: u32 = 0;
        while (pos.* < data.len and iter < 4096) : (iter += 1) {
            tokenizer_mod.skip_ws(data, pos);
            if (pos.* >= data.len) break;
            if (data[pos.*] == ']') {
                pos.* += 1;
                break;
            }
            if (data[pos.*] == '"') {
                if (tokenizer_mod.parse_json_string(data, pos, &tmp)) |s| stops.add(s);
            } else {
                tokenizer_mod.skip_json_value(data, pos);
            }
            tokenizer_mod.skip_ws(data, pos);
            if (pos.* < data.len and data[pos.*] == ',') pos.* += 1;
        }
    } else {
        tokenizer_mod.skip_json_value(data, pos);
    }
}

fn extract_raw_field(obj: []const u8, target_key: []const u8) ?[]const u8 {
    var pos: usize = 0;
    tokenizer_mod.skip_ws(obj, &pos);
    if (pos >= obj.len or obj[pos] != '{') return null;
    pos += 1;
    var kbuf: [128]u8 = undefined;
    var iter: u32 = 0;
    while (pos < obj.len and iter < 256) : (iter += 1) {
        tokenizer_mod.skip_ws(obj, &pos);
        if (pos >= obj.len or obj[pos] == '}') break;
        const k = tokenizer_mod.parse_json_string(obj, &pos, &kbuf) orelse break;
        tokenizer_mod.skip_ws(obj, &pos);
        if (pos >= obj.len or obj[pos] != ':') break;
        pos += 1;
        tokenizer_mod.skip_ws(obj, &pos);
        const start = pos;
        tokenizer_mod.skip_json_value(obj, &pos);
        if (std.mem.eql(u8, k, target_key)) return obj[start..pos];
        tokenizer_mod.skip_ws(obj, &pos);
        if (pos < obj.len and obj[pos] == ',') pos += 1;
    }
    return null;
}

fn extract_string_field(obj: []const u8, target_key: []const u8, out_buf: []u8) ?[]const u8 {
    const raw = extract_raw_field(obj, target_key) orelse return null;
    var p: usize = 0;
    return tokenizer_mod.parse_json_string(raw, &p, out_buf);
}

fn extract_bool_field(obj: []const u8, target_key: []const u8) bool {
    const raw = extract_raw_field(obj, target_key) orelse return false;
    var p: usize = 0;
    return parse_json_bool(raw, &p);
}
