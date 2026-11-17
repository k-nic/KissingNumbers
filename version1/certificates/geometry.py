"""Complete pairwise tests. Acceptance is exact, with no tolerance.

The fast kernel multiplies SMALL INTEGERS in binary64: the absolute sum bound
d*max(abs(A))*max(abs(B)) < 2**53 proves every product and partial sum exact.
--direct uses checked int64 multiplication instead. Rational comparisons use
int64 only after a bound check, and Python arbitrary-precision integers otherwise.
"""
import math
import numpy as np
from common import require, integer_array, canonical_rationals


def dot_blocks(a, b, direct=False):
    ma = max((abs(int(v)) for v in (a.min(), a.max())), default=0) if a.size else 0
    mb = max((abs(int(v)) for v in (b.min(), b.max())), default=0) if b.size else 0
    bound = a.shape[1] * ma * mb
    require(bound < (2**63 if direct else 2**53), 'dot-product exactness bound')
    dtype = np.int64 if direct else np.float64
    return a.astype(dtype) @ b.astype(dtype).T


def antipodal_partition(x):
    require(len(np.unique(x, axis=0)) == len(x), 'duplicate integer point')
    first = np.argmax(x != 0, axis=1)
    signs = np.sign(x[np.arange(len(x)), first])
    require(np.all(signs != 0), 'zero vector')
    orient = x*signs[:, None]
    _, idx, counts = np.unique(orient, axis=0, return_index=True, return_counts=True)
    require(np.all(counts <= 2), 'antipodal multiplicity')
    return orient[idx[counts == 2]], x[idx[counts == 1]]


def check_integer(x, norm=128, limit=64, block=512, direct=False):
    x = integer_array(x, 'integer geometry')
    require(block > 0 and norm > limit >= 0, 'bad geometry parameters')
    m = max(abs(int(x.min())), abs(int(x.max()))) if x.size else 0
    require(x.shape[1]*m*m < 2**63, 'norm accumulation overflow')
    require(np.all(np.sum(x*x, axis=1) == norm), 'integer norm failure')
    p, u = antipodal_partition(x)
    maximum = -norm
    for a, b, absolute, triangular in ((p,p,True,True),(p,u,True,False),(u,u,False,True)):
        if not len(a) or not len(b):
            continue
        # Convert once, to avoid repeatedly allocating the complete right operand.
        bound = x.shape[1]*m*m
        require(bound < (2**63 if direct else 2**53), 'exact kernel bound')
        dtype = np.int64 if direct else np.float64
        aa, bb = a.astype(dtype), b.astype(dtype)
        for i in range(0, len(a), block):
            start = i if triangular else 0
            dots = aa[i:i+block] @ bb[start:].T
            if absolute:
                np.abs(dots, out=dots)
            if triangular:
                rows = np.arange(min(block, len(a)-i))
                dots[rows, rows] = -norm
            local = int(np.max(dots))
            require(local <= limit, 'integer pair conflict')
            maximum = max(maximum, local)
    pairs = 4*math.comb(len(p),2)+len(p)+2*len(p)*len(u)+math.comb(len(u),2)
    require(pairs == math.comb(len(x),2), 'pair coverage identity')
    return {'integer_points':len(x), 'antipodal_pairs':len(p), 'unpaired_points':len(u),
            'unordered_pairs_covered':pairs, 'maximum_inner_product':maximum,
            'kernel':'int64' if direct else 'binary64 exact integers',
            'absolute_sum_bound':x.shape[1]*m*m}


def check_rationals(x, numerators, denominators, norm=128, limit=64):
    n = integer_array(numerators, 'numerators')
    d = integer_array(denominators, 'denominators', 1)
    require(n.shape == (len(d), x.shape[1]) and np.all(d > 0), 'rational shape/sign')
    canon = canonical_rationals(n,d)
    require(len(np.unique(canon, axis=0)) == len(d), 'duplicate rational point')
    # A rational/integer duplicate would also violate the cross-pair bound below.
    best_num, best_den = -1, 1
    for row, den0 in zip(n, d):
        a = list(map(int,row)); den = int(den0)
        require(sum(v*v for v in a) == norm*den*den, 'rational norm failure')
        m = max(abs(int(x.min())),abs(int(x.max()))) if x.size else 0
        require(x.shape[1]*m*max(map(abs,a)) < 2**63, 'rational/core int64 overflow')
        products = x @ row
        mx = int(products.max()) if len(x) else -norm*den
        require(mx <= limit*den, 'rational/core conflict')
        if mx*best_den > best_num*norm*den:
            best_num, best_den = mx, norm*den
    for i in range(len(d)):
        a = list(map(int,n[i]))
        for j in range(i):
            product = sum(v*int(w) for v,w in zip(a,n[j]))
            denprod = int(d[i])*int(d[j])
            require(product <= limit*denprod, 'rational/rational conflict')
            if product*best_den > best_num*norm*denprod:
                best_num,best_den = product,norm*denprod
    if len(d):
        g = math.gcd(best_num,best_den);best_num//=g;best_den//=g
    return {'rational_points':len(d), 'rational_core_pairs':len(d)*len(x),
            'rational_rational_pairs':math.comb(len(d),2),
            'maximum_affected_unit_product_numerator':best_num,
            'maximum_affected_unit_product_denominator':best_den,
            'maximum_affected_unit_product_approx':best_num/best_den}
