"""Deterministic displacement generators (with legacy make_beta interface)."""
import numpy as np

def make_beta(N, r, kind='equal', seed=20261028, rng=None):
    if N < 1 or not (0 <= r < 1):
        raise ValueError('Need N>=1 and 0<=r<1')
    if kind in ('equal', 'equal_positive'):
        w = np.ones(N)
    elif kind in ('negative_equal', 'equal_negative'):
        w = -np.ones(N)
    elif kind in ('mixed_equal', 'mixed', 'mixed_sign'):
        w = np.ones(N)
        w[1::2] = -1
    elif kind in ('anisotropic', 'anisotropic_positive'):
        w = np.power(2., np.arange(N-1,-1,-1,dtype=float))
    elif kind in ('random', 'random_positive'):
        w = (rng if rng is not None else np.random.default_rng(seed)).random(N)
    elif kind in ('random_mixed',):
        w = (rng if rng is not None else np.random.default_rng(seed)).standard_normal(N)
    else:
        raise ValueError(f'Unknown displacement kind {kind!r}')
    return np.asarray(r*w/np.linalg.norm(w),dtype=float)
