"""Extract IQ3_XXS w13 expert banks from a GGUF into iq3.safetensors, in the layout cuda/ reads (FMT 2).
Usage: mkiq3.py REF.gguf MODEL_DIR [NAME]   writes MODEL_DIR/NAME (default iq3.safetensors = mix D; mix F uses iq3f.safetensors)
and checks it against the official weights.
Row (2304 weights, 864 B) = 18 chunks of 128: 32 grid-index bytes + 4 u32 (4x7-bit signs, 4-bit scale).
d: one f16 per 256 weights, shifted by a per-expert power of two so products stay in FP16 range (undone in _scale)."""
import json, os, sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import IQ3_XXS

def convert(t, gs):
    b = np.asarray(t.data).reshape(128, 2304, 9, 98)
    p = np.empty(2304, np.int64); p[0::2] = np.arange(1152); p[1::2] = np.arange(1152, 2304)  # gate = 2p, up = 2p+1
    b = b[:, p]
    body = np.concatenate([b[..., 2:66].reshape(128, 2304, 18, 32), b[..., 66:98].reshape(128, 2304, 18, 16)], -1)
    d = np.ascontiguousarray(b[..., :2]).view('<f2')[..., 0].astype(np.float64)
    k = 1 - np.floor(np.log2(np.maximum(np.abs(d).reshape(128, -1).max(1), 2.0 ** -24)))  # max |d| in [2, 4) -> max |w| < 1922
    dn = (d * 2.0 ** k[:, None, None]).astype('<f2')
    assert np.array_equal(dn.astype(np.float64), d * 2.0 ** k[:, None, None])  # exact
    return body, dn, (gs.astype(np.float64) * 2.0 ** -k).astype('<f4')

def decode(body, d, s):  # one expert, as the kernels do it
    IQ3_XXS.init_grid(); G = IQ3_XXS.grid[0, 0].astype(np.float32)
    ss = np.ascontiguousarray(body[..., 32:]).view('<u4')                       # [2304, 18, 4]
    w = G[body[..., :32]].reshape(2304, 18, 4, 4, 8)
    s7 = (ss[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127
    bits = (s7[..., None] >> np.arange(7, dtype=np.uint32)) & 1
    bits = np.concatenate([bits, bits.sum(-1, keepdims=True) & 1], -1)
    db = np.repeat(d.astype(np.float32), 2, -1)[..., None] * (0.5 + (ss >> 28).astype(np.float32)) * 0.5
    return (w * (1 - 2.0 * bits) * db[..., None, None]).reshape(2304, 2304) * s

def official(mdir):
    f = np.memmap(f'{mdir}/model-00001-of-00001.safetensors', np.uint8, 'r'); n = int(f[:8].view('<u8')[0])
    h = json.loads(bytes(f[8:8 + n]))
    raw = lambda k: f[8 + n + h[k]['data_offsets'][0]:8 + n + h[k]['data_offsets'][1]]
    e4m3 = np.array([((1 + m / 8) * 2.0 ** (e - 7) if e else m * 2.0 ** -9) * s for s in (1, -1) for e in range(16) for m in range(8)], np.float32)
    e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    def weight(l, e):
        k = f'model.llm.layers.{l}.mlp.experts.w13_weight'
        w, s = raw(k), raw(k + '_scale')
        if l < 20:
            c = w[e * 2304 * 1152:(e + 1) * 2304 * 1152]; c = np.stack([c & 15, c >> 4], -1).reshape(-1, 16)
            gs = np.array(raw(k + '_amax')).view('<f4')[e] / np.array(raw(k + '_outer_scale_denominator')[:4]).view('<f4')[0]
            x = e2m1[c] * e4m3[s[e * 2304 * 144:(e + 1) * 2304 * 144]][:, None] * gs
        else:
            c = w[e * 2304 * 2304:(e + 1) * 2304 * 2304].reshape(-1, 32)
            x = e4m3[c] * (2.0 ** (s[e * 2304 * 72:(e + 1) * 2304 * 72].astype(np.float32) - 127))[:, None]
        return x.reshape(2304, 2304)
    return weight

def main(ref, mdir, name='iq3.safetensors'):
    R = GGUFReader(ref); T = {t.name: t for t in R.tensors}
    layers = [l for l in range(24) if T[f'blk.{l}.ffn_gate_up_exps.weight'].tensor_type.name == 'IQ3_XXS']
    sizes = {'codes': ('U8', [128, 2304, 864]), 'd': ('F16', [128, 2304, 9]), 'scale': ('F32', [128])}
    hdr, off = {}, 0
    for l in layers:
        for n, (dt, sh) in sizes.items():
            nb = int(np.prod(sh)) * {'U8': 1, 'F16': 2, 'F32': 4}[dt]
            hdr[f'model.llm.layers.{l}.mlp.experts.w13_iq3_{n}'] = {'dtype': dt, 'shape': sh, 'data_offsets': [off, off + nb]}; off += nb
    h = json.dumps(hdr, separators=(',', ':')).encode(); h += b' ' * (-len(h) % 8)
    weight = official(mdir); out = f'{mdir}/{name}'
    with open(out + '.tmp', 'wb') as f:
        f.write(len(h).to_bytes(8, 'little') + h)
        for l in layers:
            s = T.get(f'blk.{l}.ffn_gate_up_exps.scale')
            parts = convert(T[f'blk.{l}.ffn_gate_up_exps.weight'], np.array(s.data) if s else np.ones(128, np.float32))
            for e in (3, 100):
                x, y = weight(l, e), decode(parts[0][e], parts[1][e], parts[2][e])
                err = float(np.linalg.norm(x - y) / np.linalg.norm(x))
                print(f'layer {l:2d} expert {e:3d} relative error vs official {err:.4f}', flush=True)
                assert err < 0.5, 'layout or scale mismatch'
            for p in parts: f.write(p.tobytes())
        assert f.tell() == 8 + len(h) + off
    os.replace(out + '.tmp', out); print(out, 8 + len(h) + off)

if __name__ == '__main__': main(*sys.argv[1:4])
