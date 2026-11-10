# Eq.45 U^(N)(beta) and F_A^{(m)}(a;1/2,...,1/2;3/2,...,3/2;x_1,...,x_m)
# 第一类 Lauricella 超几何函数 F_A的半参数版本

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

# Optional high-precision backend for high-order nonterminating series.
try:
    import mpmath as mp
except ImportError:
    mp = None

from config import DEFAULT, NumericalConfig
from utils import kappa, validate_beta

# Shared numerical backend settings.  The public simulation ceiling is 2^14;
# adaptive doubling normally stops much earlier.
FA_HIGH_PRECISION_SWITCH_ORDER = 128
FA_MP_DPS = 50

def lauricella_FA_half_parameters(
    a: float,
    x: Sequence[float],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    initial_order: int = 8,
    max_order: int = 16384,
    return_info: bool = False,
):
    # 半参型F_A(a;b;c;x)函数，b={1/2,...,1/2},c={3/2,...,3/2}, F_A in Eq.45
    r"""
    Evaluate the special Lauricella function

      F_A^{(m)}(
          a;
          1/2,...,1/2;
          3/2,...,3/2;
          x_1,...,x_m
      )

    using total-degree summation.

    General definition:
      F_A^{(m)}
      = sum_{k_1,...,k_m>=0}
          (a)_{K}
          prod_i [(1/2)_{k_i}/(3/2)_{k_i}]
          prod_i x_i^{k_i}/k_i!,
        K = sum_i k_i.

    For the present parameters:
      (1/2)_k/(3/2)_k = 1/(2k+1),

    hence
      F_A = sum_K (a)_K [z^K] prod_i
            sum_{k>=0} x_i^k / ((2k+1)k!) z^k.

    Convergence region:
      sum_i |x_i| < 1.

    In the orthant problem x_i=beta_i^2, so every subset satisfies
      sum_{i in I} beta_i^2 < 1.

    If a is a non-positive integer, the series terminates exactly.
    """
    x = np.asarray(x, dtype=float)
    m = x.size
    # F_A^0 = 1
    if m == 0:
        out = 1.0
        info = {
            "order": 0,
            "converged": True,
            "terminating": True,
            "sum_x": 0.0,
            "backend": "numpy",
            "max_order_allowed": int(max_order),
        }
        return (out, info) if return_info else out
    # x=beta^2>=0
    if np.any(x < 0.0):
        raise ValueError("This specialized evaluator expects x_i >= 0.")
    # ||beta||_2<1 (Origin inside the Ball)
    sum_x = float(np.sum(x))
    if not sum_x < 1.0:
        raise ValueError(
            f"Lauricella F_A series requires sum x_i < 1; got {sum_x:.16g}."
        )

    nearest_integer = round(a)
    # If a is a non-positive integer, F_A terminates
    terminating = (a <= 0.0) and abs(a - nearest_integer) < 1e-14
    termination_order = int(-nearest_integer) if terminating else None

    def evaluate_to_order(Kmax: int) -> float:
        # Fast low-order path.  On Windows, np.longdouble may be identical to float64;
        # high orders are therefore handled by the mpmath fallback below.
        ld = np.longdouble

        # Coefficients of prod_i q_i(z), where
        # q_i(k) = x_i^k / ((2k+1) k!).
        coeff = np.array([1.0], dtype=ld)

        for xi in x:
            seq = np.empty(Kmax + 1, dtype=ld)
            seq[0] = 1.0
            for k in range(1, Kmax + 1):
                # Stable recurrence:
                # q_k/q_{k-1}
                # = x * (2k-1) / [k(2k+1)].
                seq[k] = (
                    seq[k - 1]
                    * ld(xi)
                    * ld(2 * k - 1)
                    / (ld(k) * ld(2 * k + 1))
                )
            coeff = np.convolve(coeff, seq)[: Kmax + 1]

        poch = ld(1.0)
        total = coeff[0]

        for K in range(1, Kmax + 1):
            poch *= ld(a + K - 1)
            total += poch * coeff[K]

        return float(total)
    # If terminate: calculate to order
    def evaluate_high_order(Kmax: int) -> float:
        """High-precision total-degree sum without float64 overflow/underflow.

        Rebuilds all coefficients in mpmath: merely upgrading (a)_K is
        insufficient because the original q_i coefficients underflow too.
        This is an intentionally conservative fallback, not a fast algorithm.
        """
        if mp is None:
            raise ImportError(
                "High-order Lauricella evaluation requires mpmath. "
                "Install it with: pip install mpmath"
            )
        # Precision of intermediate products and cancellation is controlled
        # independently of the final float64 output.
        with mp.workdps(FA_MP_DPS):
            xs = [mp.mpf(float(v)) for v in x if v != 0.0]
            aa = mp.mpf(float(a))
            coeff = [mp.mpf(1)]
            for xi in xs:
                seq = [mp.mpf(1)]
                for k in range(1, Kmax + 1):
                    seq.append(seq[-1] * xi * (2*k-1) / (k*(2*k+1)))
                nxt = [mp.mpf(0)] * (Kmax+1)
                # Coefficients of each q_i are positive; high-precision
                # accumulation avoids cancellation and float64 underflow.
                for j, cj in enumerate(coeff):
                    if not cj:
                        continue
                    for k in range(Kmax-j+1):
                        nxt[j+k] += cj * seq[k]
                coeff = nxt
            poch = mp.mpf(1)
            total = coeff[0]
            for k in range(1, Kmax+1):
                poch *= aa + k - 1
                total += poch * coeff[k]
            return float(total)

    def safe_evaluate(Kmax: int) -> float:
        # Preserve the original numerical path for all previously successful
        # low-order evaluations. Windows np.longdouble often equals float64.
        if Kmax <= FA_HIGH_PRECISION_SWITCH_ORDER:
            return evaluate_to_order(Kmax)
        return evaluate_high_order(Kmax)

    if terminating:
        value = safe_evaluate(termination_order)
        info = {
            "order": termination_order,
            "converged": True,
            "terminating": True,
            "sum_x": sum_x,
            "backend": (
                "mpmath"
                if termination_order > FA_HIGH_PRECISION_SWITCH_ORDER
                else "numpy"
            ),
            "max_order_allowed": int(max_order),
        }
        return (value, info) if return_info else value

    if initial_order < 1 or max_order < 1:
        raise ValueError("initial_order and max_order must be positive")
    K = min(initial_order, max_order)
    old = safe_evaluate(K)
    # if NOT terminate; calculate to max_order
    while K < max_order:
        K2 = min(2 * K, max_order)
        new = safe_evaluate(K2)

        if not math.isfinite(new):
            raise FloatingPointError(f"Non-finite Lauricella result at K={K2}")

        if abs(new - old) <= atol + rtol * abs(new):
            info = {
                "order": K2,
                "converged": True,
                "terminating": False,
                "sum_x": sum_x,
                "backend": (
                    "mpmath"
                    if K2 > FA_HIGH_PRECISION_SWITCH_ORDER
                    else "numpy"
                ),
                "max_order_allowed": int(max_order),
            }
            return (new, info) if return_info else new

        old = new
        K = K2

    info = {
        "order": K,
        "converged": False,
        "terminating": False,
        "sum_x": sum_x,
        "backend": (
            "mpmath"
            if K > FA_HIGH_PRECISION_SWITCH_ORDER
            else "numpy"
        ),
        "max_order_allowed": int(max_order),
    }
    return (old, info) if return_info else old


def unit_ball_orthant_lauricella(
    beta: Sequence[float],
    *,
    config: NumericalConfig = DEFAULT,
    return_info: bool = False,
):
    # Eq.45
    r"""
    Exact normalized orthant volume of the unit N-ball:

      U_N(beta)
      = sum_{I subset [N]}
          kappa_{N-|I|}/2^{N-|I|}
          (prod_{i in I} beta_i)
          F_A^{(|I|)}(
              -(N-|I|)/2;
              {1/2};
              {3/2};
              {beta_i^2}
          ).

    Important adjustable parameters
    -------------------------------
    N                = len(beta)
    r                = ||beta||_2
    support s        = # nonzero beta_i
    beta direction   = equal / sparse / anisotropic / random / mixed signs
    fa_rtol, fa_atol
    fa_max_order

    Terms containing beta_i=0 vanish and are skipped automatically.
    """
    beta = validate_beta(beta)
    N = beta.size

    active = [
        i for i, b in enumerate(beta)
        if abs(b) > config.zero_tol
    ]

    terms = [kappa(N) / (2.0 ** N)]
    fa_info = []

    for m in range(1, len(active) + 1):
        for subset in itertools.combinations(active, m):
            idx = np.asarray(subset, dtype=int)
            product_beta = float(np.prod(beta[idx]))
            d = N - m

            FA, info = lauricella_FA_half_parameters(
                -d / 2.0,
                beta[idx] ** 2,
                rtol=config.fa_rtol,
                atol=config.fa_atol,
                initial_order=config.fa_initial_order,
                max_order=config.fa_max_order,
                return_info=True,
            )

            term = (
                kappa(d)
                / (2.0 ** d)
                * product_beta
                * FA
            )
            terms.append(term)
            fa_info.append(info)

    # Accurate accumulation is important for mixed-sign beta.
    U = math.fsum(terms)

    if not return_info:
        return U

    abs_sum = math.fsum(abs(t) for t in terms)
    cancellation = abs_sum / abs(U) if U != 0.0 else math.inf

    info = {
        "N": N,
        "r": float(np.linalg.norm(beta)),
        "support": len(active),
        "subset_terms_evaluated": len(terms),
        "all_FA_converged": all(x["converged"] for x in fa_info),
        "max_FA_order": max((x["order"] for x in fa_info), default=0),
        "FA_max_order_allowed": int(config.fa_max_order),
        "high_precision_FA_terms": sum(
            int(x.get("backend") == "mpmath")
            for x in fa_info
        ),
        "high_precision_switch_order": FA_HIGH_PRECISION_SWITCH_ORDER,
        "cancellation_indicator": cancellation,
        "terms": terms,
        "FA_info": fa_info,
    }
    return U, info

