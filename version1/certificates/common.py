"""Exact data conventions and an independent Golay/Leech enumeration.

All distributed coordinates have squared norm 128. All indices are zero-based.
No pickle, network, search program, or floating approximation is needed.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
I = [0, 2, 4, 5, 6, 10, 11, 23]
J = [i for i in range(24) if i not in I]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lexsort(x):
    return x[np.lexsort(x.T[::-1])]


def golay_code():
    exponents = [0, 2, 4, 5, 6, 10, 11]
    generators = np.zeros((12, 24), dtype=np.int64)
    for s in range(12):
        generators[s, [(s+k) % 23 for k in exponents]] = 1
        generators[s, 23] = generators[s, :23].sum() % 2
    messages = (np.arange(4096)[:, None] >> np.arange(12)) & 1
    code = (messages @ generators) % 2
    weights, counts = np.unique(code.sum(1), return_counts=True)
    require(weights.tolist() == [0, 8, 12, 16, 24], 'Golay weights')
    require(counts.tolist() == [1, 759, 2576, 759, 1], 'Golay enumerator')
    require(len(np.unique(code, axis=0)) == 4096, 'Golay rank')
    return code


def leech_shell():
    """Numerically lexicographically sorted norm-128 Leech shell in Z^24."""
    code = golay_code()
    pieces = []
    for a in range(24):
        for b in range(a+1, 24):
            for sa in (-8, 8):
                for sb in (-8, 8):
                    v = np.zeros(24, dtype=np.int16)
                    v[a], v[b] = sa, sb
                    pieces.append(v[None, :])
    signs = 1 - 2*((np.arange(256)[:, None] >> np.arange(8)) & 1)
    signs = signs[(signs < 0).sum(1) % 2 == 0]
    for word in code[code.sum(1) == 8]:
        a = np.zeros((128, 24), dtype=np.int16)
        a[:, np.flatnonzero(word)] = 4*signs
        pieces.append(a)
    for word in code:
        a = np.tile(np.where(word, 2, -2), (24, 1)).astype(np.int16)
        a[np.arange(24), np.arange(24)] *= -3
        pieces.append(a)
    shell = lexsort(np.concatenate(pieces))
    require(shell.shape == (196560, 24), 'Leech cardinality')
    require(np.all(np.sum(shell.astype(np.int64)**2, axis=1) == 128), 'Leech norms')
    require(not np.any(np.all(shell[1:] == shell[:-1], axis=1)), 'Leech duplicates')
    return shell


def integer_array(a, name, ndim=2):
    require(np.issubdtype(a.dtype, np.integer), name + ': noninteger dtype')
    require(a.ndim == ndim, name + ': wrong rank')
    # Guard conversion of uint64 and any future out-of-range data.
    if a.size:
        require(int(a.min()) >= -(2**63)+1 and int(a.max()) < 2**63,
                name + ': outside signed int64 range')
    return a.astype(np.int64)


def load_points(path):
    with np.load(path, allow_pickle=False) as z:
        require(int(z['squared_norm']) == 128, 'wrong coordinate scale')
        require(int(z['inner_product_limit']) == 64, 'wrong dot-product limit')
        x = integer_array(z['points'] if 'points' in z else z['integer_points'], 'integer points')
        n = integer_array(z['fraction_numerators'], 'rational numerators') if 'fraction_numerators' in z else np.empty((0, x.shape[1]), np.int64)
        d = integer_array(z['fraction_denominators'], 'rational denominators', 1) if 'fraction_denominators' in z else np.empty(0, np.int64)
    require(n.shape == (len(d), x.shape[1]), 'rational array dimensions')
    require(np.all(d > 0), 'denominators must be positive')
    return x, n, d


def canonical_rationals(n, d):
    rows = []
    for a, den in zip(n, d):
        vals = list(map(int, a)) + [int(den)]
        g = math.gcd(*vals)
        rows.append(tuple(v // g for v in vals))
    return np.array(sorted(rows), dtype='<i8').reshape((-1, n.shape[1]+1))


def canonical_hashes(x, n, d):
    require(not x.size or np.max(np.abs(x)) < 32768, 'int16 hash encoding overflow')
    return {
        'integer_rows_sha256': hashlib.sha256(lexsort(x).astype('<i2').tobytes()).hexdigest(),
        'rational_rows_sha256': hashlib.sha256(canonical_rationals(n, d).tobytes()).hexdigest(),
    }


def save_points(path, x, n, d):
    fields = dict(squared_norm=np.int64(128), inner_product_limit=np.int64(64))
    if len(d):
        fields.update(integer_points=x.astype(np.int16), fraction_numerators=n.astype(np.int64),
                      fraction_denominators=d.astype(np.int64), format=np.array('integer_core_and_row_rational_v1'))
    else:
        fields.update(points=x.astype(np.int16), coordinate_denominator=np.int64(2))
    np.savez_compressed(path, **fields)


def layer_counts(x):
    types = np.stack([np.sum(x[:, cols]**2, axis=1) for cols in
                      (I, J, list(range(24, x.shape[1])))], axis=1)
    u, counts = np.unique(types, axis=0, return_counts=True)
    return {','.join(map(str, row)): int(c) for row, c in zip(u, counts)}


def manifest():
    return json.loads((ROOT / 'manifest.json').read_text())
