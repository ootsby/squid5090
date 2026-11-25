// Native image decoder, 8x8 checkerboard alpha compositor, separable area-average downscaler,
// and content-addressed negative synthetic token ID generator for Inkling-2-Mini vision.
// Pipeline contract & algorithm adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
// Uses stb_image.h (MIT / Public Domain) for PNG/JPEG/GIF/BMP and optional libwebp (BSD-3-Clause) for WebP.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "../third_party/stb_image.h"

#ifdef SQUID_HAS_WEBP
#include <webp/decode.h>
#include <webp/demux.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int IMG_TILE = 40;
constexpr int IMG_MAX_SIDE = 16384;
constexpr int64_t IMG_MAX_PIXELS = 67108864; // 64 MP
constexpr int IMG_ABS_MAX_TOKENS = 4096;
constexpr uint8_t HASH_VERSION = 1;

static uint64_t splitmix64_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int img_tokens_for(int h, int w) {
    return ((h + IMG_TILE - 1) / IMG_TILE) * ((w + IMG_TILE - 1) / IMG_TILE);
}

static void img_fit(int height, int width, int max_tokens, int& out_h, int& out_w) {
    if (img_tokens_for(height, width) <= max_tokens) {
        out_h = height;
        out_w = width;
        return;
    }
    const double scale = std::sqrt(double(max_tokens) * IMG_TILE * IMG_TILE / (double(height) * double(width)));
    const bool wide = (width >= height);
    int longer = std::max(1, int(std::floor((wide ? width : height) * scale)));
    for (;; --longer) {
        const double ratio = double(wide ? height : width) / double(wide ? width : height);
        const int shorter = std::max(1, int(std::lround(longer * ratio)));
        const int h = wide ? shorter : longer;
        const int w = wide ? longer : shorter;
        if (img_tokens_for(h, w) <= max_tokens || longer == 1) {
            out_h = h;
            out_w = w;
            return;
        }
    }
}

struct Tap {
    int first = 0;
    std::vector<float> weights;
};

static std::vector<Tap> build_taps(int in_len, int out_len) {
    std::vector<Tap> result(static_cast<size_t>(out_len));
    const double step = double(in_len) / double(out_len);
    for (int o = 0; o < out_len; ++o) {
        const double lo = o * step;
        const double hi = (o + 1) * step;
        const int first = std::min(in_len - 1, int(std::floor(lo)));
        const int last = std::min(in_len - 1, std::max(first, int(std::ceil(hi)) - 1));
        Tap& tap = result[static_cast<size_t>(o)];
        tap.first = first;
        double sum = 0.0;
        for (int i = first; i <= last; ++i) {
            const double w = std::max(0.0, std::min(hi, double(i + 1)) - std::max(lo, double(i)));
            tap.weights.push_back(float(w));
            sum += w;
        }
        for (float& w : tap.weights) w = float(w / sum);
    }
    return result;
}

static std::vector<uint8_t> reduce_area_avg(const uint8_t* rgb, int h, int w, int out_h, int out_w) {
    const std::vector<Tap> tx = build_taps(w, out_w);
    const std::vector<Tap> ty = build_taps(h, out_h);
    std::vector<float> rows(static_cast<size_t>(h) * out_w * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgb + static_cast<size_t>(y) * w * 3;
        float* dst = rows.data() + static_cast<size_t>(y) * out_w * 3;
        for (int x = 0; x < out_w; ++x, dst += 3) {
            const Tap& tap = tx[static_cast<size_t>(x)];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            const uint8_t* p = src + static_cast<size_t>(tap.first) * 3;
            for (const float wt : tap.weights) {
                a0 += wt * p[0];
                a1 += wt * p[1];
                a2 += wt * p[2];
                p += 3;
            }
            dst[0] = a0; dst[1] = a1; dst[2] = a2;
        }
    }
    std::vector<uint8_t> out(static_cast<size_t>(out_h) * out_w * 3);
    std::vector<float> acc(static_cast<size_t>(out_w) * 3);
    for (int y = 0; y < out_h; ++y) {
        const Tap& tap = ty[static_cast<size_t>(y)];
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (size_t i = 0; i < tap.weights.size(); ++i) {
            const float wt = tap.weights[i];
            const float* src = rows.data() + (static_cast<size_t>(tap.first) + i) * out_w * 3;
            for (size_t j = 0; j < acc.size(); ++j) acc[j] += wt * src[j];
        }
        uint8_t* dst = out.data() + static_cast<size_t>(y) * out_w * 3;
        for (size_t j = 0; j < acc.size(); ++j) {
            long v = std::lround(acc[j]);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            dst[j] = static_cast<uint8_t>(v);
        }
    }
    return out;
}

static uint64_t hash_rgb_pixels(int width, int height, const uint8_t* rgb, size_t nbytes) {
    uint64_t h = splitmix64_mix((uint64_t(uint32_t(width)) << 32) | uint32_t(height)) ^ HASH_VERSION;
    const uint8_t* p = rgb;
    size_t n = nbytes;
    for (; n >= 8; p += 8, n -= 8) {
        uint64_t word = 0;
        std::memcpy(&word, p, 8);
        h = splitmix64_mix(h + 0x9E3779B97F4A7C15ULL + word);
    }
    if (n > 0) {
        uint64_t word = 0;
        std::memcpy(&word, p, n);
        h = splitmix64_mix(h + 0x9E3779B97F4A7C15ULL + word);
    }
    return splitmix64_mix(h ^ uint64_t(nbytes));
}

static bool is_webp_bytes(const uint8_t* data, size_t len) {
    return len >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

static bool decode_base64_vec(const char* text, size_t len, std::vector<uint8_t>& out) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t;
        t.fill(-1);
        const char* alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(alpha[i])] = static_cast<int8_t>(i);
        return t;
    }();
    out.clear();
    out.reserve((len / 4) * 3 + 3);
    uint32_t buf = 0;
    int bits = 0;
    size_t symbols = 0;
    bool padded = false;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        if (c == '=') { padded = true; continue; }
        int v = table[c];
        if (v < 0 || padded) return false;
        buf = (buf << 6) | uint32_t(v);
        bits += 6;
        ++symbols;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFFu));
        }
    }
    if ((symbols % 4) == 1) return false;
    return true;
}

static bool starts_with_nocase(const char* s, size_t len, const char* prefix) {
    size_t plen = std::strlen(prefix);
    if (len < plen) return false;
    for (size_t i = 0; i < plen; ++i) {
        char c = s[i];
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if (c != prefix[i]) return false;
    }
    return true;
}

} // namespace

extern "C" {

struct CDecodedImage {
    int32_t width;
    int32_t height;
    int32_t tokens;
    uint64_t hash;
    uint8_t* rgb; // malloc-owned [height * width * 3], free via squid_image_free
};

int32_t squid_image_token_id(uint64_t hash, int32_t k) {
    const uint64_t v = splitmix64_mix(hash ^ (uint64_t(uint32_t(k)) + 1ULL) * 0x9E3779B97F4A7C15ULL) >> 33;
    return static_cast<int32_t>(-1LL - static_cast<int64_t>(v));
}

void squid_image_free(CDecodedImage* img) {
    if (img && img->rgb) {
        std::free(img->rgb);
        img->rgb = nullptr;
    }
}

// Decode raw compressed image bytes (PNG, JPEG, GIF, BMP, WebP) into RGB with 8x8 checkerboard
// alpha compositing and aspect-preserving area-average downscale to <= max_tokens.
// Returns 0 on success, non-zero error code on failure.
int32_t squid_image_decode_bytes(
    const uint8_t* data,
    size_t len,
    int32_t max_tokens,
    CDecodedImage* out_img) {
    if (!data || len == 0 || len > 0x7FFFFFFFu || !out_img) return -1;
    std::memset(out_img, 0, sizeof(*out_img));

    int w = 0, h = 0, ch = 0;
    uint8_t* raw_pixels = nullptr;
    int raw_kind = 0; // 1 = stbi, 2 = webp_free, 3 = malloc

    if (is_webp_bytes(data, len)) {
#ifdef SQUID_HAS_WEBP
        WebPBitstreamFeatures feat;
        if (WebPGetFeatures(data, len, &feat) != VP8_STATUS_OK) return -2;
        if (feat.width < 1 || feat.height < 1 || feat.width > IMG_MAX_SIDE || feat.height > IMG_MAX_SIDE ||
            int64_t(feat.width) * feat.height > IMG_MAX_PIXELS) return -3;
        w = feat.width;
        h = feat.height;
        ch = feat.has_alpha ? 4 : 3;
        if (feat.has_animation) {
            WebPAnimDecoderOptions opts;
            if (!WebPAnimDecoderOptionsInit(&opts)) return -2;
            opts.color_mode = MODE_RGBA;
            opts.use_threads = 0;
            WebPData wd{data, len};
            WebPAnimDecoder* dec = WebPAnimDecoderNew(&wd, &opts);
            if (!dec) return -2;
            uint8_t* canvas = nullptr;
            int ts = 0;
            if (!WebPAnimDecoderGetNext(dec, &canvas, &ts)) {
                WebPAnimDecoderDelete(dec);
                return -2;
            }
            size_t npix = static_cast<size_t>(w) * h;
            raw_pixels = static_cast<uint8_t*>(std::malloc(npix * ch));
            if (!raw_pixels) { WebPAnimDecoderDelete(dec); return -4; }
            for (size_t i = 0; i < npix; ++i) std::memcpy(raw_pixels + i * ch, canvas + i * 4, size_t(ch));
            WebPAnimDecoderDelete(dec);
            raw_kind = 3;
        } else {
            int dw = 0, dh = 0;
            raw_pixels = feat.has_alpha ? WebPDecodeRGBA(data, len, &dw, &dh) : WebPDecodeRGB(data, len, &dw, &dh);
            if (!raw_pixels || dw != w || dh != h) {
                if (raw_pixels) WebPFree(raw_pixels);
                return -2;
            }
            raw_kind = 2;
        }
#else
        return -5; // WebP not compiled in
#endif
    } else {
        if (!stbi_info_from_memory(data, int(len), &w, &h, &ch)) return -2;
        if (w < 1 || h < 1 || w > IMG_MAX_SIDE || h > IMG_MAX_SIDE ||
            int64_t(w) * h > IMG_MAX_PIXELS) return -3;
        raw_pixels = stbi_load_from_memory(data, int(len), &w, &h, &ch, 0);
        if (!raw_pixels || ch < 1 || ch > 4) {
            if (raw_pixels) stbi_image_free(raw_pixels);
            return -2;
        }
        raw_kind = 1;
    }

    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    const uint8_t* src = raw_pixels;
    const bool grey = (ch <= 2);
    const bool alpha = (ch == 2 || ch == 4);
    for (int y = 0; y < h; ++y) {
        uint8_t* dst = rgb.data() + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x, src += ch, dst += 3) {
            int c0 = src[0];
            int c1 = grey ? src[0] : src[1];
            int c2 = grey ? src[0] : src[2];
            if (alpha) {
                const int a = src[ch - 1];
                const int bg = (((y >> 3) + (x >> 3)) & 1) == 0 ? 204 : 153;
                c0 = (c0 * a + bg * (255 - a) + 127) / 255;
                c1 = (c1 * a + bg * (255 - a) + 127) / 255;
                c2 = (c2 * a + bg * (255 - a) + 127) / 255;
            }
            dst[0] = static_cast<uint8_t>(c0);
            dst[1] = static_cast<uint8_t>(c1);
            dst[2] = static_cast<uint8_t>(c2);
        }
    }

    if (raw_kind == 1) stbi_image_free(raw_pixels);
#ifdef SQUID_HAS_WEBP
    else if (raw_kind == 2) WebPFree(raw_pixels);
#endif
    else if (raw_kind == 3) std::free(raw_pixels);

    const int mt = std::max(1, std::min(max_tokens, IMG_ABS_MAX_TOKENS));
    int out_h = h, out_w = w;
    img_fit(h, w, mt, out_h, out_w);
    if (out_h != h || out_w != w) {
        rgb = reduce_area_avg(rgb.data(), h, w, out_h, out_w);
    }

    uint8_t* out_buf = static_cast<uint8_t*>(std::malloc(rgb.size()));
    if (!out_buf) return -4;
    std::memcpy(out_buf, rgb.data(), rgb.size());

    out_img->width = out_w;
    out_img->height = out_h;
    out_img->tokens = img_tokens_for(out_h, out_w);
    out_img->hash = hash_rgb_pixels(out_w, out_h, out_buf, rgb.size());
    out_img->rgb = out_buf;
    return 0;
}

int32_t squid_image_decode_base64(
    const char* b64,
    size_t b64_len,
    int32_t max_tokens,
    CDecodedImage* out_img) {
    if (!b64 || b64_len == 0 || !out_img) return -1;
    std::vector<uint8_t> bytes;
    if (!decode_base64_vec(b64, b64_len, bytes)) return -6;
    return squid_image_decode_bytes(bytes.data(), bytes.size(), max_tokens, out_img);
}

int32_t squid_image_decode_data_uri(
    const char* uri,
    size_t uri_len,
    int32_t max_tokens,
    CDecodedImage* out_img) {
    if (!uri || uri_len == 0 || !out_img) return -1;
    if (!starts_with_nocase(uri, uri_len, "data:")) return -7;
    const char* comma = static_cast<const char*>(std::memchr(uri, ',', uri_len));
    if (!comma) return -7;
    const size_t hdr_len = static_cast<size_t>(comma - (uri + 5));
    if (hdr_len < 7 || !starts_with_nocase(uri + 5 + hdr_len - 7, 7, ";base64")) return -7;
    const char* payload = comma + 1;
    const size_t payload_len = uri_len - static_cast<size_t>(payload - uri);
    return squid_image_decode_base64(payload, payload_len, max_tokens, out_img);
}

} // extern "C"
