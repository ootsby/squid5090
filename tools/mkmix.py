"""Build fake-quantised GGUFs: copy the BF16/NVFP4 reference and overwrite tensors with values rounded
exactly as squid5090 rounds them (cuda/cuda_abi.cu z_quant_bf16_to_{nvfp4,fp8}_kernel), still stored BF16/F32.
Usage: mkmix.py REF.gguf OUT.gguf A|B|B2      self-check: mkmix.py REF.gguf --check
A = squid nvfp4 mode; B = A with FP8 attention and BF16 router/controllers; B2 = B with FP8 lm_head."""
import shutil, sys
import numpy as np
from gguf import GGUFReader

POS = np.array([0] + [(1 + m / 8) * 2.0 ** (e - 7) if e else m * 2.0 ** -9
                for e in range(16) for m in range(8) if (e or m) and not (e == 15 and m == 7)], np.float32)  # E4M3, max 448
MID = (POS[1:] + POS[:-1]) / 2
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
E2MID = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)

def e4m3(a):  # magnitude -> nearest E4M3 (ties differ from hardware by at most one code)
    return POS[np.searchsorted(MID, np.minimum(a, 448.0), side='right')]

def nvfp4(x, gs):
    b = x.reshape(-1, 16)
    s = (e4m3(np.abs(b).max(1) / (6.0 * gs)) * np.float32(gs))[:, None]
    q = np.divide(b, s, out=np.zeros_like(b), where=s > 1e-30)
    return (np.sign(q) * E2M1[np.searchsorted(E2MID, np.abs(q), side='right')] * s).reshape(x.shape)

def fp8(x):  # per-row scale
    s = np.maximum(np.abs(x).max(-1, keepdims=True) / 448.0, 1e-12)
    return np.sign(x) * e4m3(np.abs(x) / s) * s

def load(t):
    if t.tensor_type.name == 'F32': return np.array(t.data, np.float32)
    assert t.tensor_type.name == 'BF16', (t.name, t.tensor_type.name)
    u = np.ascontiguousarray(t.data).view(np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)

def store(f, t, x, off=0):
    if t.tensor_type.name == 'F32': raw = x.astype('<f4')
    else:
        u = x.astype('<f4').view(np.uint32)
        raw = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype('<u2')  # round-to-nearest-even
    f.seek(t.data_offset + off); f.write(raw.tobytes())

def main(ref, out, mix):
    R = GGUFReader(ref); T = {t.name: t for t in R.tensors}
    shutil.copyfile(ref, out)
    gs_of = lambda am: max(am / (448.0 * 6.0), 1e-12)
    with open(out, 'r+b') as f:
        for l in range(24):
            # squid fuses [q, rel, gate, sink], then [k, v], then o: one global scale per fused matrix
            for grp in (('q', 'rel', 'gate', 'sink_proj'), ('k', 'v'), ('output',)):
                ts = [T[f'blk.{l}.attn_{n}.weight'] for n in grp]; xs = [load(t) for t in ts]
                gs = gs_of(max(float(np.abs(x).max()) for x in xs))
                for t, x in zip(ts, xs): store(f, t, nvfp4(x, gs) if mix == 'A' else fp8(x))
            if mix == 'A':
                small = [f'blk.{l}.ffn_gate_inp.weight']
                if 6 <= l < 18: small += [f'blk.{l}.loop_{s}_ctrl_{p}.weight' for s in ('q', 'k', 'v', 'ffn') for p in ('down', 'forget', 'gain')]
                for n in small:
                    x = load(T[n]); store(f, T[n], nvfp4(x, gs_of(float(np.abs(x).max()))))
            if l >= 20:
                for n in ('gate_up', 'down'):
                    t = T[f'blk.{l}.ffn_{n}_exps.weight']; nb = t.data[0].nbytes
                    if t.tensor_type.name != 'BF16': continue  # mix D: already low-bit
                    for e in range(128):
                        u = np.ascontiguousarray(t.data[e]).view(np.uint16); x = (u.astype(np.uint32) << 16).view(np.float32)
                        store(f, t, nvfp4(x, gs_of(float(np.abs(x).max()))), e * nb)
            print('layer', l, flush=True)
        t = T['output.weight']; nr = t.data.shape[0]; nb = t.data[0].nbytes
        rows = lambda a, b: (np.ascontiguousarray(t.data[a:b]).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        gs = gs_of(max(float(np.abs(rows(a, min(a + 8192, 200058))).max()) for a in range(0, 200058, 8192)))
        for a in range(0, 200058, 8192):  # only the usable rows, as squid does
            x = rows(a, min(a + 8192, 200058)); store(f, t, fp8(x) if mix == 'B2' else nvfp4(x, gs), a * nb)

def check(ref):
    R = GGUFReader(ref); T = {t.name: t for t in R.tensors}
    for v in (0.0, 1.0, 448.0, 2.0 ** -9): assert e4m3(np.float32(v)) == np.float32(v)
    b = np.array([[6, -4, 3, 2, 1.5, -1, .5, 0] * 2], np.float32); assert np.allclose(nvfp4(b, 1.0), b)  # representable block is exact
    for n, i in (('blk.20.ffn_gate_up_exps.weight', 3), ('blk.20.ffn_down_exps.weight', 3), ('blk.6.attn_q.weight', None), ('blk.6.attn_output.weight', None)):
        t = T[n]; d = t.data if i is None else t.data[i]
        x = (np.ascontiguousarray(d).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        am = float(np.abs(x).max()); rel = lambda y: float(np.linalg.norm(x - y) / np.linalg.norm(x))
        print(f'{n:32s} amax={am:.3f} squid nvfp4 {rel(nvfp4(x, am / 2688)):.4f} | no global scale {rel(nvfp4(x, 1.0)):.4f} | fp8 {rel(fp8(x)):.4f}')

if __name__ == '__main__':
    check(sys.argv[1]) if sys.argv[2] == '--check' else main(*sys.argv[1:4])
