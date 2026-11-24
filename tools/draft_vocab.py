"""Build draft_vocab_32k.bin (little-endian int32 token IDs) from weighted token counts.
Usage: draft_vocab.py OUT.bin own_train.ids text.ids code.ids   (each file: whitespace/comma separated IDs)
Weights: 60% the model's own output, 20% text, 20% code; special tokens 199998..200057 always kept."""
import collections, re, sys
from array import array
N, VOCAB, SPECIAL = 32768, 200058, range(199998, 200058)

def counts(p):
    return collections.Counter(int(x) for x in re.findall(r'\d+', open(p).read()))

def build(cs, w=(.6, .2, .2)):
    score = collections.Counter()
    for c, k in zip(cs, w):
        tot = sum(c.values()) or 1
        for t, n in c.items(): score[t] += k * n / tot
    for t in SPECIAL: score[t] = 2.0
    ids = [t for t, _ in sorted(score.items(), key=lambda x: (-x[1], x[0])) if t < VOCAB][:N]
    have = set(ids)
    ids += [t for t in range(VOCAB) if t not in have][:N - len(ids)]
    assert len(set(ids)) == N and all(0 <= t < VOCAB for t in ids)
    return sorted(ids)

if __name__ == '__main__':
    out, *files = sys.argv[1:]
    cs = [counts(f) for f in files]; ids = build(cs)
    a = array('i', ids); assert a.itemsize == 4
    if sys.byteorder != 'little': a.byteswap()
    open(out, 'wb').write(a.tobytes())
    s = set(ids)
    print('wrote', len(ids), 'ids; distinct per source', [len(c) for c in cs], 'coverage', [round(sum(n for t, n in c.items() if t in s) / max(1, sum(c.values())), 4) for c in cs])
