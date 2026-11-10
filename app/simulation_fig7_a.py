from __future__ import annotations

"""
Fig. 7(a) — final uniform accuracy-matched low-support benchmark
=============================================================

Revision
--------
This version uses a *common relative-error target* for both methods,

    epsilon_rel,target = 1e-10,

instead of tuning Aono alone against the default Hybrid result.

For every test case a high-accuracy common reference V_ref is first constructed.
Then

    epsilon_Hybrid,rel
        = |V_Hybrid - V_ref| / |V_ref|,

    epsilon_Aono,rel
        = |V_Aono - V_ref| / |V_ref|.

The Hybrid numerical tolerances and the Aono Fourier truncation K_Aono are
selected independently so that each method satisfies the same target.

Structured reference
--------------------
s=1: 70-digit mpmath incomplete beta;
s=2: 70-digit mpmath independent one-dimensional probability quadrature;
s=3,4: 60-digit original unreduced subset-Lauricella representation,
       with an analytic absolute tail bound summed over all subsets.
If reference certification fails, the case is labelled, NOT timed or counted.

Grid: every N=4,...,20; every s=1,...,4; both displacement patterns;
17 radii r=0.25,...,0.97. 2312 configurations.

Cache discipline
----------------
Each independent subset solve (N>=5,s>=3) clears cached G1/G2/G3 values
ONCE on entry, preserving reuse within that solve but never across timing
repetitions. Analytic support=1/2 and the N4-special route do not touch
these caches, so they avoid unnecessary cache clearing.

Resume
------
Append-only JSONL checkpoint with a fingerprint of this script and the
numerical kernels; does not mix timings from different implementations.
On modification of code/dependencies, start with --fresh.

Speedup
-------
The speedup convention remains

    S_Hybrid/Aono = t_Aono / t_Hybrid.

Thus

    S_Hybrid/Aono > 1  -> Hybrid faster,
    S_Hybrid/Aono < 1  -> Aono faster.

Outputs
-------
results/
    fig7a_raw.csv
    fig7a_plot.csv
    fig7a_summary.csv
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

from memory_probe import memory_once, memory_fields
from aono_fourier import aono_fourier_orthant
from config import NumericalConfig
from proposed_hybrid import (
    analytic_structure_counts,
    is_fully_structured_exact,
    unit_ball_orthant_hybrid,
)
from utils import (
    make_sparse_beta,
    timed_warm,
    valid_orthant_volume,
    write_csv,
)


RESULTS = Path(__file__).resolve().parent / "results"
PROTOCOL = "Fig7a_V21_5_optimized_nodiag_elementary_20261104"
CHECKPOINT_NAME = "fig7a_checkpoint.jsonl"



# =============================================================================
# V21.2 Route(U/T/G) integrity / per-case checks
# =============================================================================

def verify_v21_route_evaluator() -> None:
    """Fail early when the installed proposed_hybrid.py is the pre-Route version.

    This checks actual subset execution, not just function names or the imported
    theoretical count helper.  Both Figure 7(a) and 7(b) require V21.2 code.
    """
    import inspect

    checks = (
        # N, active beta coordinates, expected (Psi, J, retained generic FA).
        (5, (+0.16, -0.12, +0.08), (3, 0, 0)),
        (6, (+0.16, -0.12, +0.08), (0, 1, 0)),
        (6, (+0.16, -0.12, +0.08, +0.06), (0, 4, 0)),
        (7, (+0.16, -0.12, +0.08, +0.06), (6, 0, 1)),
    )
    for N, active, expected in checks:
        beta = np.zeros(N, dtype=float)
        beta[:len(active)] = active
        value, info = unit_ball_orthant_hybrid(beta, return_info=True)
        try:
            actual = (
                int(info['evaluated_Psi_terms']),
                int(info['evaluated_J_terms']),
                int(info['evaluated_generic_FA_terms']),
            )
            rho = float(info['rho_FA'])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                'Figure 7 requires the V21.5 Route(U/T/G) proposed_hybrid.py. '
                'The imported implementation lacks a required Route diagnostic.'
            ) from exc
        if (actual != expected or info['route'] != 'hybrid_subset'
                or not info['all_FA_converged'] or not info['valid_volume']
                or not math.isfinite(float(value)) or not math.isfinite(rho)):
            raise RuntimeError(
                f'Wrong/inconsistent Hybrid evaluator: N={N}, s={len(active)}, '
                f'expected (Psi,J,FA)={expected}, got {actual}, '
                f'route={info.get("route")}; source: '
                f'{inspect.getfile(unit_ball_orthant_hybrid)}. '
                'Install the V21.5 Route(U/T/G) core before running Figure 7.'
            )
    print('Verified V21.5 Route(U/T/G) Hybrid evaluator: '
          f'{inspect.getfile(unit_ball_orthant_hybrid)}')


def verify_hybrid_case_route(beta, info: dict) -> None:
    """Validate the *executed* branch and FA/Psi/J call counts for each case."""
    N = len(beta)
    s = int(np.count_nonzero(np.asarray(beta) != 0.0))
    route = str(info.get('route', ''))
    if s == 0:
        expected_route='centered_exact'
    elif N <= 2:
        expected_route=f'N{N}_elementary'
    elif N == 3:
        expected_route=('N3_elementary_s1' if s==1 else
                        'N3_elementary_s2' if s==2 else 'N3_elementary')
    elif N == 4:
        expected_route=(f'N4_elementary_s{s}' if s<=2 else
                        'N4_1D_reduction')
    elif s == 1:
        expected_route='s1_incomplete_beta'
    elif s == 2:
        expected_route=info.get('route')
        if expected_route not in ('s2_elementary_iterative',
                                  's2_stable_1D_probability'):
            raise RuntimeError(f'Unexpected s2 route: {expected_route}')
    else:
        expected_route='hybrid_subset'
    if route != expected_route:
        raise RuntimeError(
            f'Unexpected Hybrid branch: N={N}, s={s}, '
            f'expected {expected_route}, got {route}'
        )
    if N >= 5 and s >= 3:
        expected_fa = sum(math.comb(s, m) for m in range(4, s + 1)
                          if (N - m) % 2 == 1)
        expected_psi = math.comb(s, 2) if N % 2 else 0
        expected_j = math.comb(s, 3) if not N % 2 else 0
        tests = (
            ('evaluated_generic_FA_terms', expected_fa),
            ('evaluated_Psi_terms', expected_psi),
            ('evaluated_J_terms', expected_j),
            ('evaluated_subset_terms', 2**s),
        )
        for key, expected in tests:
            if int(info.get(key, -1)) != expected:
                raise RuntimeError(
                    f'V21.2 Route count mismatch: N={N}, s={s}, '
                    f'{key}: got {info.get(key)}, expected {expected}'
                )

# =============================================================================
# Common accuracy target
# =============================================================================

EPSILON_REL_TARGET = 1e-10

# Reference arithmetic.  70 decimal digits is far beyond the 1e-10 benchmark
# target and is still practical for the one-dimensional structured references.
MP_DPS = 70

# Aono remains subject to the same practical ceiling used in the earlier
# benchmark.  A target not reached before this ceiling is explicitly censored.
K_AONO_MAX = 4194304

TIMING_REPEATS = 9
TIMING_WARMUP = 1


# =============================================================================
# Hybrid accuracy scan
# =============================================================================

# Scan from deliberately loose to strict settings.  The first configuration
# that actually satisfies the COMMON volume relative-error target is timed.
HYBRID_TOL_GRID = (
    1e-5,
    3e-6,
    1e-6,
    3e-7,
    1e-7,
    3e-8,
    1e-8,
    3e-9,
    1e-9,
    3e-10,
    1e-10,
    3e-11,
    1e-11,
    3e-12,
    1e-12,
    3e-13,
    1e-13,
)

HYBRID_FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384
HYBRID_QUAD_LIMIT = 600


# =============================================================================
# Parameter grids
# =============================================================================

# Complete, parity-unrestricted grid (user-specified).
SUPPORTS = (1, 2, 3, 4)
DIMENSIONS = tuple(range(4, 21))
SIGN_CASES = ("positive", "negative")  # s=1
MULTI_SIGN_CASES = ("same_sign", "mixed_sign")  # s>=2

# Independent original-series reference for s>=3.
REF_DPS = 60
REF_REL_TAIL_TARGET = 1e-13
REF_K_INITIAL = 24
REF_K_MAX = 2048
REF_VERSION = "mp_unreduced_FA_analytic_tail_V1"

# Dense displacement coverage while retaining the original 0.25--0.97 range.
R_VALUES = tuple(
    np.round(
        np.concatenate([
            np.arange(0.25, 0.9000001, 0.05),
            np.array([0.925, 0.95, 0.97], dtype=float),
        ]),
        8,
    )
)


# =============================================================================
# Generic helpers
# =============================================================================

def relative_error(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference)):
        return math.inf
    if reference == 0.0:
        return abs(value - reference)
    return abs(value - reference) / abs(reference)


def make_hybrid_config(tol: float) -> NumericalConfig:
    """
    One scalar scan parameter controls the relevant numerical tolerances.

    The final acceptance criterion is NOT the internal tolerance itself; it is
    always the actual volume relative error with respect to V_ref.
    """
    return NumericalConfig(
        fa_rtol=float(tol),
        fa_atol=max(float(tol) * 1e-2, 1e-16),
        fa_initial_order=HYBRID_FA_INITIAL_ORDER,
        fa_max_order=FA_MAX_ORDER,
        quad_epsabs=max(float(tol) * 1e-2, 1e-15),
        quad_epsrel=max(float(tol), 2e-14),
        quad_limit=HYBRID_QUAD_LIMIT,
    )


def aono_k_grid(max_k: int) -> list[int]:
    if max_k < 1:
        raise ValueError("max_k must be positive.")

    if max_k < 16:
        return [max_k]

    values = []
    K = 16

    while K <= max_k:
        values.append(K)
        K *= 2

    if values[-1] != max_k and max_k > values[-1]:
        values.append(max_k)

    return values


# =============================================================================
# High-precision structured reference
# =============================================================================

def mp_kappa(N: int):
    return mp.pi ** (mp.mpf(N) / 2) / mp.gamma(1 + mp.mpf(N) / 2)


def mp_halfspace_probability(beta, dimension: int):
    """
    High-precision probability

        P(X_1 >= -beta)

    for X uniform in the unit ``dimension``-ball.
    """
    b = mp.mpf(beta)

    if b >= 1:
        return mp.mpf(1)
    if b <= -1:
        return mp.mpf(0)
    if b == 0:
        return mp.mpf("0.5")

    x = b * b
    a = mp.mpf("0.5")
    c = (mp.mpf(dimension) + 1) / 2

    if b > 0:
        I = mp.betainc(a, c, 0, x, regularized=True)
        return mp.mpf("0.5") * (1 + I)

    # Stable complementary representation for beta < 0.
    I = mp.betainc(c, a, 0, 1 - x, regularized=True)
    return mp.mpf("0.5") * I


def reference_s1_mp(N: int, beta1: float) -> float:
    with mp.workdps(MP_DPS):
        p = mp_halfspace_probability(mp.mpf(beta1), N)
        value = mp_kappa(N) * p / (mp.mpf(2) ** (N - 1))
        return float(value)


def reference_s2_mp(N: int, beta1: float, beta2: float) -> float:
    """
    High-precision complete orthant volume for exactly two active thresholds.

    The same mathematical 1D reduction as the stable probability argument is
    used, but evaluated independently with mpmath high-precision arithmetic.
    """
    with mp.workdps(MP_DPS):
        b1 = mp.mpf(beta1)
        b2 = mp.mpf(beta2)

        marginal_constant = mp_kappa(N - 1) / mp_kappa(N)
        power = (mp.mpf(N) - 1) / 2
        lower = -b1

        def integrand(x):
            q = 1 - x * x
            if q <= 0:
                return mp.mpf(0)

            radius = mp.sqrt(q)
            conditional = mp_halfspace_probability(
                b2 / radius,
                N - 1,
            )

            return (
                marginal_constant
                * q ** power
                * conditional
            )

        points = [lower]

        if abs(b2) < 1:
            critical = mp.sqrt(max(mp.mpf(0), 1 - b2 * b2))
            for x in (-critical, critical):
                if lower < x < 1:
                    points.append(x)

        points.append(mp.mpf(1))
        points = sorted(set(points))

        probability = mp.quad(integrand, points)

        value = (
            mp_kappa(N)
            * probability
            / (mp.mpf(2) ** (N - 2))
        )

        return float(value)


def mp_fa_value_and_tail(p, x: tuple, K: int):
    """Return raw defining-series F_A and a rigorous analytic tail bound."""
    if not x:
        return mp.mpf(1), mp.mpf(0)
    p = mp.mpf(p)
    if p == int(p):
        K = min(K,int(p))
    coeff = [mp.mpf(1)] + [mp.mpf(0)]*K
    for xi in x:
        xi = mp.mpf(xi)
        seq = [mp.mpf(1)]
        for j in range(1,K+1):
            seq.append(seq[-1]*xi*(2*j-1)/(j*(2*j+1)))
        updated = [mp.mpf(0)]*(K+1)
        for i,ci in enumerate(coeff):
            if ci == 0: continue
            for j in range(K-i+1):
                if seq[j] != 0:
                    updated[i+j] += ci*seq[j]
        coeff = updated
    poch = mp.mpf(1)
    value = mp.mpf(1)
    for k in range(1,K+1):
        poch *= -p+(k-1)
        value += poch*coeff[k]
    if p == int(p):
        return value, mp.mpf(0)
    if K < int(mp.ceil(p)):
        return value, mp.inf
    rho2 = mp.fsum(x)
    bound = abs(mp.binomial(p,K+1))*rho2**(K+1)/(1-rho2)
    return value, bound


def mp_unreduced_volume(beta: np.ndarray, K: int):
    """Compute a high-precision original subset sum and TOTAL tail bound."""
    N = len(beta)
    active = [i for i,bi in enumerate(beta) if bi != 0.0]
    betas = [mp.mpf(float(b)) for b in beta]
    pieces = [mp_kappa(N)/2**N]
    total_bound = mp.mpf(0)
    for m in range(1,len(active)+1):
        d = N-m
        factor = mp_kappa(d)/2**d
        for subset in itertools.combinations(active,m):
            xx = tuple(betas[i]**2 for i in subset)
            pref = factor*mp.fprod(betas[i] for i in subset)
            FA, bound = mp_fa_value_and_tail(mp.mpf(d)/2,xx,K)
            pieces.append(pref*FA)
            total_bound += abs(pref)*bound
    return mp.fsum(pieces),total_bound


def construct_reference(beta: np.ndarray) -> dict:
    with mp.workdps(REF_DPS):
        K=REF_K_INITIAL
        while True:
            value, bound=mp_unreduced_volume(beta,K)
            if value > 0 and bound <= REF_REL_TAIL_TARGET*value:
                result=float(value)
                return {
                    'reference':result,
                    'reference_name':REF_VERSION,
                    'reference_ok':valid_orthant_volume(result,len(beta)),
                    'reference_series_order':K,
                    'reference_tail_rel_bound':float(bound/value),
                    'reference_dps':REF_DPS,
                }
            if K>=REF_K_MAX:
                # Preserve diagnostics, do not silently use an unverified ref.
                return {
                    'reference':float(value),
                    'reference_name':REF_VERSION,
                    'reference_ok':False,
                    'reference_series_order':K,
                    'reference_tail_rel_bound':float(bound/abs(value)) if value else math.inf,
                    'reference_dps':REF_DPS,
                }
            K=min(2*K,REF_K_MAX)



def structured_reference(beta: np.ndarray, support: int) -> dict:
    active = np.flatnonzero(beta != 0.0)
    N = len(beta)
    if support == 1:
        ref = reference_s1_mp(N, float(beta[active[0]]))
        return {"reference": ref,
                "reference_name": "mpmath_s1_incomplete_beta",
                "reference_ok": valid_orthant_volume(ref,N),
                "reference_series_order": 0,
                "reference_tail_rel_bound": 0.0,
                "reference_dps": MP_DPS}
    if support == 2:
        ref = reference_s2_mp(N, float(beta[active[0]]), float(beta[active[1]]))
        return {"reference": ref,
                "reference_name": "mpmath_s2_1D_probability",
                "reference_ok": valid_orthant_volume(ref,N),
                "reference_series_order": 0,
                "reference_tail_rel_bound": 0.0,
                "reference_dps": MP_DPS}
    if support in (3,4):
        return construct_reference(beta)
    raise ValueError("Unsupported support")


# =============================================================================
# Accuracy-matched Hybrid scan
# =============================================================================

def scan_hybrid_to_target(beta, reference: float) -> dict:
    selected = None
    last = None

    for tol in HYBRID_TOL_GRID:
        config = make_hybrid_config(tol)

        value, info = unit_ball_orthant_hybrid(
            beta,
            config=config,
            return_info=True,
        )

        verify_hybrid_case_route(beta, info)
        value = float(value)
        err = relative_error(value, reference)

        valid = bool(info["valid_volume"])
        converged = bool(info["all_FA_converged"])

        last = (tol, config, value, info, err, valid, converged)

        if (
            valid
            and converged
            and math.isfinite(err)
            and err <= EPSILON_REL_TARGET
        ):
            selected = last
            break

    reached = selected is not None
    chosen = selected if reached else last

    tol, config, _, _, _, _, _ = chosen

    # Time only the solver's production-value path; all diagnostics/route
    # checks, convergence checks and relative-error evaluation are untimed.
    t_h, timed_value = timed_warm(
        unit_ball_orthant_hybrid, beta, config=config, return_info=False,
        repeats=TIMING_REPEATS, warmup=TIMING_WARMUP,
    )
    timed_value=float(timed_value)
    # Route/integrity validation occurs outside timed_warm on the same beta.
    diagnostic_value,timed_info=unit_ball_orthant_hybrid(
        beta,config=config,return_info=True,
    )
    verify_hybrid_case_route(beta,timed_info)
    timed_err=relative_error(timed_value,reference)
    consistent=math.isclose(timed_value,float(diagnostic_value),rel_tol=1e-13,abs_tol=0.0)
    reached_after_timing=bool(
        consistent and timed_info['valid_volume'] and
        timed_info['all_FA_converged'] and math.isfinite(timed_err) and
        timed_err<=EPSILON_REL_TARGET
    )

    # Independent un-timed single evaluation at the selected tolerance.
    # The evaluator itself is unchanged: same Route and cache reset.
    _, memory = memory_once(unit_ball_orthant_hybrid, beta,
                            config=config, return_info=False)

    return {
        **memory_fields("Hybrid", memory),
        "Hybrid_target_reached": reached_after_timing,
        "Hybrid_value": timed_value,
        "Hybrid_rel_error": timed_err,
        "Hybrid_time_s": float(t_h),
        "Hybrid_route": str(timed_info["route"]),
        "Hybrid_selected_tol": float(tol),
        "Hybrid_fa_rtol": float(config.fa_rtol),
        "Hybrid_fa_atol": float(config.fa_atol),
        "Hybrid_quad_epsrel": float(config.quad_epsrel),
        "Hybrid_quad_epsabs": float(config.quad_epsabs),
        "Hybrid_FA_order": int(timed_info["max_FA_order"]),
        "Hybrid_rho_FA": float(timed_info["rho_FA"]),
        "Hybrid_generic_FA_evaluated": int(timed_info["evaluated_generic_FA_terms"]),
        "Hybrid_Psi_evaluated": int(timed_info["evaluated_Psi_terms"]),
        "Hybrid_J_evaluated": int(timed_info["evaluated_J_terms"]),
    }


# =============================================================================
# Accuracy-matched Aono scan
# =============================================================================

def scan_aono_to_target(beta, reference: float) -> dict:
    selected = None
    last = None
    memory_at_Kmax = None

    for K_Aono in aono_k_grid(K_AONO_MAX):
        if K_Aono == K_AONO_MAX:
            # The last permitted K is actually evaluated, with memory
            # recorded in THIS un-timed accuracy scan (no duplicate Kmax).
            raw_value, memory_at_Kmax = memory_once(
                aono_fourier_orthant, beta, n_terms=K_Aono)
            value = float(raw_value)
        else:
            value = float(aono_fourier_orthant(beta, n_terms=K_Aono))

        valid = valid_orthant_volume(
            value,
            len(beta),
        )

        err = relative_error(
            value,
            reference,
        )

        last = (
            K_Aono,
            value,
            valid,
            err,
        )

        if (
            valid
            and math.isfinite(err)
            and err <= EPSILON_REL_TARGET
        ):
            selected = last
            break

    reached = selected is not None
    K_Aono = int((selected if reached else last)[0])
    if not reached:
        # Do NOT time an invalid comparison 9x at K=4194304.
        return {
            "Aono_target_reached": False,
            "Aono_value": float(last[1]),
            "Aono_rel_error": float(last[3]),
            "Aono_time_s": math.nan,
            "K_Aono": K_Aono,
            "K_Aono_max_allowed": K_AONO_MAX,
            "Aono_censored": True,
            **memory_fields("Aono", memory_at_Kmax),
        }
    t_a, timed_value = timed_warm(
        aono_fourier_orthant, beta, n_terms=K_Aono,
        repeats=TIMING_REPEATS, warmup=TIMING_WARMUP,
    )
    timed_value = float(timed_value)
    timed_err = relative_error(timed_value, reference)
    reached_after_timing = bool(
        valid_orthant_volume(timed_value, len(beta)) and
        math.isfinite(timed_err) and timed_err <= EPSILON_REL_TARGET
    )
    if memory_at_Kmax is None or K_Aono != K_AONO_MAX:
        _, measured_memory = memory_once(aono_fourier_orthant, beta,
                                          n_terms=K_Aono)
    else:
        measured_memory = memory_at_Kmax

    return {
        **memory_fields("Aono", measured_memory),
        "Aono_target_reached": reached_after_timing,
        "Aono_value": timed_value,
        "Aono_rel_error": timed_err,
        "Aono_time_s": float(t_a),
        "K_Aono": K_Aono,
        "K_Aono_max_allowed": K_AONO_MAX,
        "Aono_censored": not reached_after_timing,
    }


# =============================================================================
# Geometry
# =============================================================================

def make_s1_beta(N: int, r: float, sign: float) -> np.ndarray:
    beta = np.zeros(N, dtype=float)
    beta[0] = float(sign) * float(r)
    return beta


def make_s2_beta(N: int, r: float, mixed: bool) -> np.ndarray:
    return make_sparse_beta(
        N,
        r,
        2,
        mixed=mixed,
    )


# =============================================================================
# Single case
# =============================================================================

def _missing_algorithm_fields():
    nan = math.nan
    return {
        **memory_fields("Hybrid", None),
        **memory_fields("Aono", None),
        "Hybrid_target_reached": False, "Hybrid_value": nan,
        "Hybrid_rel_error": nan, "Hybrid_time_s": nan,
        "Hybrid_route": "reference_failed", "Hybrid_selected_tol": nan,
        "Hybrid_fa_rtol": nan, "Hybrid_fa_atol": nan,
        "Hybrid_quad_epsrel": nan, "Hybrid_quad_epsabs": nan,
        "Hybrid_FA_order": nan, "Hybrid_rho_FA": nan,
        "Hybrid_generic_FA_evaluated": nan,
        "Hybrid_Psi_evaluated": nan, "Hybrid_J_evaluated": nan,
        "Aono_target_reached": False, "Aono_value": nan,
        "Aono_rel_error": nan, "Aono_time_s": nan,
        "K_Aono": nan, "K_Aono_max_allowed": K_AONO_MAX,
        "Aono_censored": True,
    }


def evaluate_case(*, N: int, r: float, support: int,
                  sign_case: str, beta: np.ndarray) -> dict:
    reference_info = structured_reference(beta, support)
    ref = reference_info["reference"]
    if reference_info["reference_ok"]:
        hybrid = scan_hybrid_to_target(beta, ref)
        aono = scan_aono_to_target(beta, ref)
    else:
        missing = _missing_algorithm_fields()
        hybrid = {k:v for k,v in missing.items() if k.startswith("Hybrid_")}
        aono = {k:v for k,v in missing.items() if k.startswith("Aono_") or k == "K_Aono" or k == "K_Aono_max_allowed"}
    both_reached = bool(hybrid["Hybrid_target_reached"] and aono["Aono_target_reached"])
    speedup = (aono["Aono_time_s"] / hybrid["Hybrid_time_s"]
               if both_reached and hybrid["Hybrid_time_s"] > 0.0 else math.nan)
    structure = analytic_structure_counts(N, support)
    return {
        "protocol": PROTOCOL,
        "group": f"s{support}_all_N",
        "N": N, "r": float(r), "support": support,
        "sign_case": sign_case, "beta": beta.tolist(),
        **reference_info,
        "epsilon_rel_target": EPSILON_REL_TARGET,
        "fully_structured_exact": bool(is_fully_structured_exact(N, support)),
        "Hybrid_generic_FA_terms": int(structure["generic_FA_terms"]),
        "Hybrid_terminating_terms": int(structure["terminating_polynomial_terms"]),
        "Hybrid_singleton_terms": int(structure["singleton_elementary_terms"]),
        **hybrid, **aono,
        "both_target_reached": both_reached,
        "S_Hybrid_Aono": speedup,
        "speedup_Aono_over_Hybrid": speedup,
    }


# =============================================================================
# Main simulation
# =============================================================================

def _json_numpy(item):
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, np.ndarray):
        return item.tolist()
    raise TypeError(f'Unexpected JSON type: {type(item).__name__}')


def _code_fingerprint() -> str:
    # Old runs MUST not silently survive an optimization/core edit.
    names = ("simulation_fig7_a.py", "proposed_hybrid.py", "hybrid_reductions.py",
             "lauricella.py", "n4.py", "utils.py", "config.py", "elementary.py",
             "aono_fourier.py", "memory_probe.py")
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update((Path(__file__).resolve().parent/name).read_bytes())
    return h.hexdigest()


def _all_cases():
    for support in SUPPORTS:
        for N in DIMENSIONS:
            for r in R_VALUES:
                if support == 1:
                    variants = (("positive",make_s1_beta(N,r,+1.0)),
                                ("negative",make_s1_beta(N,r,-1.0)))
                else:
                    variants = (("same_sign",make_sparse_beta(N,r,support,mixed=False)),
                                ("mixed_sign",make_sparse_beta(N,r,support,mixed=True)))
                for sign_case,beta in variants:
                    yield support,N,float(r),sign_case,beta


def _case_key(support,N,r,sign_case):
    return f"{support}|{N}|{r:.8f}|{sign_case}"


def _read_or_init_checkpoint(folder:Path,fresh:bool):
    file = folder/CHECKPOINT_NAME
    digest = _code_fingerprint()
    if fresh or not file.exists():
        with file.open('w',encoding='utf-8') as f:
            f.write(json.dumps({"kind":"meta","protocol":PROTOCOL,
                                 "code_sha256":digest})+'\n')
        return {}, file
    result = {}
    with file.open(encoding='utf-8') as f:
        head=json.loads(next(f))
        if (head.get('kind')!='meta' or head.get('protocol')!=PROTOCOL
                or head.get('code_sha256')!=digest):
            raise RuntimeError('Checkpoint code/protocol differs from current version; '
                               'use --fresh or a new --output-dir.')
        for line in f:
            if not line.strip():continue
            try: data=json.loads(line)
            except json.JSONDecodeError:break  # interrupted last append
            if data.get('kind')=='case':
                row=data['row']
                result[_case_key(row['support'],row['N'],row['r'],row['sign_case'])]=row
    return result,file


def _write_outputs(rows,folder:Path):
    if not rows:return
    write_csv(rows,folder/'fig7a_raw.csv')
    write_csv(build_plot_table(rows),folder/'fig7a_plot.csv')
    write_csv(build_summary(rows),folder/'fig7a_summary.csv')


def run_fig7a(*, supports=SUPPORTS, output_dir:Path|None=None,
              fresh=False, max_cases:int|None=None, checkpoint_every:int=10):
    verify_v21_route_evaluator()
    folder = Path(output_dir) if output_dir is not None else RESULTS
    folder.mkdir(parents=True,exist_ok=True)
    previous,file = _read_or_init_checkpoint(folder,fresh)
    to_add = 0
    total = len(SUPPORTS)*len(DIMENSIONS)*len(R_VALUES)*2
    print(f'Fig.7(a) full grid: {total} cases; resume: {len(previous)} cases',flush=True)
    for support,N,r,sign_case,beta in _all_cases():
        if support not in supports: continue
        key = _case_key(support,N,r,sign_case)
        if key in previous: continue
        row=evaluate_case(N=N,r=r,support=support,sign_case=sign_case,beta=beta)
        previous[key]=row
        with file.open('a',encoding='utf-8') as f:
            f.write(json.dumps({"kind":"case","row":row},allow_nan=True,default=_json_numpy)+'\n')
        to_add+=1
        report_case(row)
        if to_add%checkpoint_every==0:
            _write_outputs(list(previous.values()),folder)
        if max_cases is not None and to_add>=max_cases:
            break
    rows=sorted(previous.values(),key=lambda q:(q['support'],q['N'],q['r'],q['sign_case']))
    _write_outputs(rows,folder)
    return rows, build_summary(rows)


# =============================================================================
# Reporting / output tables
# =============================================================================

def report_case(row):
    S = row["S_Hybrid_Aono"]
    S_text = (
        f"{S:.4g}"
        if math.isfinite(S)
        else "NA"
    )

    print(
        f"N={row['N']:3d}, "
        f"r={row['r']:.2f}, "
        f"s={row['support']}, "
        f"{row['sign_case']:10s}, "
        f"eH={row['Hybrid_rel_error']:.2e}, "
        f"eA={row['Aono_rel_error']:.2e}, "
        f"tolH={row['Hybrid_selected_tol']:.1e}, "
        f"K_Aono={row['K_Aono']:5d}, "
        f"S={S_text:>9s}, "
        f"matched={row['both_target_reached']}"
    )


def build_plot_table(rows):
    out = []

    for row in rows:
        out.append(
            {
                "group": row["group"],
                "N": row["N"],
                "r": row["r"],
                "support": row["support"],
                "sign_case": row["sign_case"],

                "epsilon_rel_target": row[
                    "epsilon_rel_target"
                ],

                "Hybrid_target_reached": row[
                    "Hybrid_target_reached"
                ],
                "Hybrid_rel_error": row[
                    "Hybrid_rel_error"
                ],
                "Hybrid_time_s": row[
                    "Hybrid_time_s"
                ],
                "Hybrid_route": row[
                    "Hybrid_route"
                ],
                "Hybrid_selected_tol": row[
                    "Hybrid_selected_tol"
                ],
                "Hybrid_generic_FA_evaluated": row["Hybrid_generic_FA_evaluated"],
                "Hybrid_Psi_evaluated": row["Hybrid_Psi_evaluated"],
                "Hybrid_J_evaluated": row["Hybrid_J_evaluated"],
                "Hybrid_rho_FA": row["Hybrid_rho_FA"],
                "Hybrid_FA_order": row["Hybrid_FA_order"],
                "FA_max_order_allowed": FA_MAX_ORDER,

                "Aono_target_reached": row[
                    "Aono_target_reached"
                ],
                "Aono_rel_error": row[
                    "Aono_rel_error"
                ],
                "Aono_time_s": row[
                    "Aono_time_s"
                ],
                "K_Aono": row[
                    "K_Aono"
                ],

                "both_target_reached": row[
                    "both_target_reached"
                ],

                "S_Hybrid_Aono": row[
                    "S_Hybrid_Aono"
                ],

                "plot_role": "main_speedup_and_SI_diagnostics",
                "Hybrid_error_over_target": row["Hybrid_rel_error"] / EPSILON_REL_TARGET,
                "Aono_error_over_target": row["Aono_rel_error"] / EPSILON_REL_TARGET,
                "K_Aono_max_allowed": K_AONO_MAX,
                "speedup_valid": bool(row["both_target_reached"] and math.isfinite(row["S_Hybrid_Aono"]) and row["S_Hybrid_Aono"] > 0),
                "speedup_plot": row["S_Hybrid_Aono"] if row["both_target_reached"] and math.isfinite(row["S_Hybrid_Aono"]) and row["S_Hybrid_Aono"] > 0 else math.nan,


                "Hybrid_faster": bool(
                    row["both_target_reached"]
                    and math.isfinite(
                        row["S_Hybrid_Aono"]
                    )
                    and row["S_Hybrid_Aono"] > 1.0
                ),

                "Aono_faster": bool(
                    row["both_target_reached"]
                    and math.isfinite(
                        row["S_Hybrid_Aono"]
                    )
                    and row["S_Hybrid_Aono"] < 1.0
                ),

                "reference_name": row["reference_name"],
                "reference_ok":row["reference_ok"],
                "reference_series_order":row["reference_series_order"],
                "reference_tail_rel_bound":row["reference_tail_rel_bound"],
                "Aono_censored":row["Aono_censored"],
                "Hybrid_memory_peak_rss_delta_MiB":row["Hybrid_memory_peak_rss_delta_MiB"],
                "Aono_memory_peak_rss_delta_MiB":row["Aono_memory_peak_rss_delta_MiB"],
                "Hybrid_memory_peak_traced_MiB":row["Hybrid_memory_peak_traced_MiB"],
                "Aono_memory_peak_traced_MiB":row["Aono_memory_peak_traced_MiB"],
            }
        )

    return out


def build_summary(rows):
    summary = []

    keys = sorted(
        {
            (
                row["group"],
                row["support"],
                row["N"],
            )
            for row in rows
        }
    )

    for group, support, N in keys:
        subset = [
            row
            for row in rows
            if (
                row["group"] == group
                and row["support"] == support
                and row["N"] == N
            )
        ]

        matched = [
            row
            for row in subset
            if (
                row["both_target_reached"]
                and math.isfinite(
                    row["S_Hybrid_Aono"]
                )
            )
        ]

        speeds = [
            row["S_Hybrid_Aono"]
            for row in matched
        ]
        matched_K = [
            row["K_Aono"]
            for row in matched
            if math.isfinite(row["K_Aono"])
        ]
        matched_FA_orders = [
            row["Hybrid_FA_order"]
            for row in matched
            if math.isfinite(row["Hybrid_FA_order"])
        ]
        hybrid_times = [
            row["Hybrid_time_s"]
            for row in matched
            if math.isfinite(row["Hybrid_time_s"])
        ]
        aono_times = [
            row["Aono_time_s"]
            for row in matched
            if math.isfinite(row["Aono_time_s"])
        ]

        summary.append(
            {
                "group": group,
                "support": support,
                "N": N,
                "epsilon_rel_target": (
                    EPSILON_REL_TARGET
                ),

                "n_cases": len(subset),
                "n_reference_ok":sum(bool(row["reference_ok"]) for row in subset),
                "n_Aono_censored":sum(bool(row["Aono_censored"]) for row in subset),

                "n_Hybrid_target_reached": sum(
                    row["Hybrid_target_reached"]
                    for row in subset
                ),

                "n_Aono_target_reached": sum(
                    row["Aono_target_reached"]
                    for row in subset
                ),

                "n_both_target_reached": len(
                    matched
                ),
                "Hybrid_target_reached_fraction": (
                    sum(row["Hybrid_target_reached"] for row in subset) / len(subset)
                    if subset else math.nan
                ),
                "Aono_target_reached_fraction": (
                    sum(row["Aono_target_reached"] for row in subset) / len(subset)
                    if subset else math.nan
                ),

                "both_target_reached_fraction": (
                    len(matched) / len(subset)
                    if subset
                    else math.nan
                ),

                "median_S_Hybrid_Aono": (
                    float(np.median(speeds))
                    if speeds
                    else math.nan
                ),

                "min_S_Hybrid_Aono": (
                    min(speeds)
                    if speeds
                    else math.nan
                ),

                "max_S_Hybrid_Aono": (
                    max(speeds)
                    if speeds
                    else math.nan
                ),

                "median_K_Aono": (
                    float(np.median(matched_K))
                    if matched_K else math.nan
                ),
                "min_K_Aono": (min(matched_K) if matched_K else math.nan),
                "max_K_Aono": (max(matched_K) if matched_K else math.nan),
                "median_Hybrid_FA_order": (
                    float(np.median(matched_FA_orders))
                    if matched_FA_orders else math.nan
                ),
                "max_Hybrid_FA_order": (
                    max(matched_FA_orders) if matched_FA_orders else math.nan
                ),
                "median_Hybrid_time_s": (
                    float(np.median(hybrid_times)) if hybrid_times else math.nan
                ),
                "median_Aono_time_s": (
                    float(np.median(aono_times)) if aono_times else math.nan
                ),
                "FA_max_order_allowed": FA_MAX_ORDER,

                # Memory diagnostics are descriptive, not speedup criteria.
                "median_Hybrid_memory_peak_rss_delta_MiB": float(np.median([
                    row["Hybrid_memory_peak_rss_delta_MiB"] for row in subset
                    if math.isfinite(row["Hybrid_memory_peak_rss_delta_MiB"])
                ])) if any(math.isfinite(row["Hybrid_memory_peak_rss_delta_MiB"])
                           for row in subset) else math.nan,
                "median_Aono_memory_peak_rss_delta_MiB": float(np.median([
                    row["Aono_memory_peak_rss_delta_MiB"] for row in subset
                    if math.isfinite(row["Aono_memory_peak_rss_delta_MiB"])
                ])) if any(math.isfinite(row["Aono_memory_peak_rss_delta_MiB"])
                           for row in subset) else math.nan,
                "median_Hybrid_memory_peak_traced_MiB": float(np.median([
                    row["Hybrid_memory_peak_traced_MiB"] for row in subset
                    if math.isfinite(row["Hybrid_memory_peak_traced_MiB"])
                ])) if any(math.isfinite(row["Hybrid_memory_peak_traced_MiB"])
                           for row in subset) else math.nan,
                "median_Aono_memory_peak_traced_MiB": float(np.median([
                    row["Aono_memory_peak_traced_MiB"] for row in subset
                    if math.isfinite(row["Aono_memory_peak_traced_MiB"])
                ])) if any(math.isfinite(row["Aono_memory_peak_traced_MiB"])
                           for row in subset) else math.nan,

                "median_Hybrid_selected_tol": (
                    float(
                        np.median(
                            [
                                row[
                                    "Hybrid_selected_tol"
                                ]
                                for row in matched
                            ]
                        )
                    )
                    if matched
                    else math.nan
                ),

                "max_Hybrid_rel_error": (
                    max(
                        row["Hybrid_rel_error"]
                        for row in subset
                    )
                    if subset
                    else math.nan
                ),

                "max_Aono_rel_error_reached": (
                    max(
                        (
                            row["Aono_rel_error"]
                            for row in subset
                            if row[
                                "Aono_target_reached"
                            ]
                        ),
                        default=math.nan,
                    )
                ),

                "K_Aono_max_allowed": (
                    K_AONO_MAX
                ),
            }
        )

    return summary


# =============================================================================
# IDE main
# =============================================================================

def main():
    parser=argparse.ArgumentParser(description='Figure 7(a): all supports s=1..4, N=4..20')
    parser.add_argument('--supports',type=int,nargs='+',default=list(SUPPORTS),
                        choices=SUPPORTS,help='Optional subset; cached results are retained')
    parser.add_argument('--output-dir',type=Path,default=RESULTS)
    parser.add_argument('--fresh',action='store_true',help='Discard existing checkpoint (new timings)')
    parser.add_argument('--max-cases',type=int,default=None,
                        help='Optional quick smoke test; omitted in production')
    args=parser.parse_args()
    rows, summary=run_fig7a(supports=set(args.supports),output_dir=args.output_dir,
                             fresh=args.fresh,max_cases=args.max_cases)
    print('Fig7a completed:',len(rows),'/',len(SUPPORTS)*len(DIMENSIONS)*len(R_VALUES)*2)
    print('Hybrid target reached:',sum(r['Hybrid_target_reached'] for r in rows))
    print('Aono target reached:',sum(r['Aono_target_reached'] for r in rows))
    print('Both reached:',sum(r['both_target_reached'] for r in rows))
    print('Output:',args.output_dir)


if __name__=='__main__':
    main()
