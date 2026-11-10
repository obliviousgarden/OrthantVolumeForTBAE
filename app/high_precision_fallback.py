"""Bounded whole-volume high-precision fallback for severe cancellation.

The fallback evaluates the exact *raw* subset/F_A identity using mpmath,
so it avoids float64 rounding in polynomial, G2/G3, and final summation.
It is intentionally independent of the optimized Route(T) implementation.
This is a high-precision numerical fallback, not an independent mathematical
reference and not a full floating-point/roundoff error certificate.

Never compute an unconstrained high-order F_A: all nonterminating orders are
preflighted with the analytical tail majorant and strict work caps. If a
budget cannot be met, raise HighPrecisionLimit instead of returning success.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
import mpmath as mp

from fa_tail import requested_order


@dataclass
class HighPrecisionLimit(Exception):
    reason: str
    subset_size: int = 0
    requested_order: int = 0

    def __str__(self):
        return f"{self.reason}: m={self.subset_size}, order={self.requested_order}"


def _coeff_series(xs: tuple, K: int):
    """Coefficient of z^k in product sum x_i^j z^j/((2j+1)j!)."""
    coeff = [mp.mpf(1)]
    for x in xs:
        # Cheap zero/empty cases are deliberately omitted by the caller.
        seq = [mp.mpf(1)]
        for j in range(1, K + 1):
            seq.append(seq[-1] * x * (2*j-1)/(j*(2*j+1)))
        result = [mp.mpf(0)] * (K+1)
        for i, ci in enumerate(coeff):
            if ci == 0: continue
            for j in range(K-i+1):
                result[i+j] += ci * seq[j]
        coeff = result
    return coeff


def fa_mp_at_order(p: float, xs: tuple, K: int):
    if K == 0: return mp.mpf(1)
    coeff = _coeff_series(xs, K)
    a = -mp.mpf(p)
    poch = mp.mpf(1)
    contributions = [mp.mpf(1)]
    for i in range(1, K+1):
        poch *= a + i-1
        contributions.append(poch * coeff[i])
    return mp.fsum(contributions)


def evaluate_high_precision_volume(beta, *, target_rel, chi_hint=1.,
                                  dps_min=55, dps_max=100,
                                  max_support=8, max_order=384,
                                  max_subsets=256, safety=10.,
                                  initial_order=8):
    """Return (float volume, diagnostic info) or raise HighPrecisionLimit.

    Input beta is kept as the EXACT float values used by the float64 path.
    Each nonterminating F_A is bounded by a conservative equal absolute
    subset-tail budget, with one additional final-volume bound check.
    """
    beta = tuple(float(v) for v in beta)
    N = len(beta)
    support = [i for i, v in enumerate(beta) if v != 0]
    s = len(support)
    if s > max_support or 2**s > max_subsets:
        raise HighPrecisionLimit("high_precision_support_cap", s, 0)
    if not (0 < target_rel < 1):
        raise ValueError("target_rel must be between 0 and 1")
    if not max_order >= initial_order >= 1:
        raise ValueError("bad high precision order cap")
    dps = min(int(dps_max), max(int(dps_min),
          math.ceil(-math.log10(target_rel) + math.log10(max(1., chi_hint))) + 22))
    if dps < math.ceil(-math.log10(target_rel) + math.log10(max(1., chi_hint))) + 8:
        raise HighPrecisionLimit("high_precision_dps_cap", s, 0)

    with mp.workdps(dps):
        bm = tuple(mp.mpf(v) for v in beta)
        kn = [mp.pi**(mp.mpf(j)/2)/mp.gamma(1+mp.mpf(j)/2)
              for j in range(N+1)]
        centered = kn[N]/(2**N)
        # Precompute and reuse high-precision sequences when called repeatedly.
        # First form a cheap high-precision estimate using only K=32. This
        # intentionally does NOT constitute a reference for success criteria.
        def term_at(subset, p, K):
            vals = tuple(bm[i] for i in subset)
            x = tuple(v*v for v in vals)
            coeff = kn[N-len(subset)]/2**(N-len(subset))
            return coeff*mp.fprod(vals)*fa_mp_at_order(p,x,K)

        approx = [centered]
        for m in range(1, s+1):
            for I in itertools.combinations(support,m):
                p=(N-m)/2
                K= int(p) if float(p).is_integer() else 16
                approx.append(term_at(I,p,K))
        estimate = mp.fsum(approx)
        if estimate <= 0:
            # Early approximation can have a negative sign under cancellation.
            # Tighten estimate without losing precision, or report a limit.
            approx = [centered]
            for m in range(1,s+1):
                for I in itertools.combinations(support,m):
                    p=(N-m)/2
                    K=int(p) if float(p).is_integer() else min(64,max_order)
                    approx.append(term_at(I,p,K))
            estimate=mp.fsum(approx)
        if estimate <= 0:
            raise HighPrecisionLimit('high_precision_reference_scale_unresolved', s, 64)

        abs_budget = (mp.mpf(target_rel) * abs(estimate) /
                      (mp.mpf(safety) * (2**s)))
        final_terms = [centered]
        max_k = 0
        sum_tail = mp.mpf(0)
        n_nonterminating = 0
        for m in range(1,s+1):
            for I in itertools.combinations(support,m):
                p=(N-m)/2
                weight=kn[N-m]/(2**(N-m))*mp.fprod(bm[i] for i in I)
                if float(p).is_integer():
                    K=int(p)
                else:
                    n_nonterminating += 1
                    if weight == 0:
                        K=0
                    else:
                        # The float-based tail *bound* is conservative, and
                        # the chosen work order remains explicitly bounded.
                        fa_budget = abs_budget/abs(weight)
                        if fa_budget < mp.mpf('1e-290'):
                            raise HighPrecisionLimit('high_precision_tiny_budget',m,max_order)
                        rho_sq = math.fsum(beta[i]**2 for i in I)
                        K, met, bound=requested_order(p,rho_sq,float(fa_budget),
                            initial_order=initial_order,max_order=max_order)
                        if not met:
                            raise HighPrecisionLimit('high_precision_FA_order_cap',m,K)
                        sum_tail+=abs(weight)*mp.mpf(bound)
                max_k=max(max_k,K)
                final_terms.append(weight*fa_mp_at_order(p,tuple(bm[i]**2 for i in I),K))
        result=mp.fsum(final_terms)
        if result <= 0:
            raise HighPrecisionLimit('high_precision_nonpositive_volume',s,max_k)
        if sum_tail > mp.mpf(target_rel)*abs(result)/safety:
            raise HighPrecisionLimit('high_precision_final_tail_budget',s,max_k)
        return float(result),dict(
            high_precision_used=True, high_precision_dps=dps,
            high_precision_max_order=max_k,
            high_precision_FA_terms=n_nonterminating,
            high_precision_tail_absolute_bound=float(sum_tail),
            high_precision_roundoff_certified=False,
            high_precision_status='mpmath_tail_bounded_not_certified')
