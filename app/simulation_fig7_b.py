from __future__ import annotations

"""
Fig. 7(b) — final accuracy-matched dense benchmark
=============================================

Revision
--------
The final benchmark uses the COMMON
relative-error target for both methods:

    epsilon_rel,target = 1e-10.

For every dense test case,

    epsilon_Hybrid,rel
        = |V_Hybrid - V_ref| / |V_ref|,

    epsilon_Aono,rel
        = |V_Aono - V_ref| / |V_ref|.

The Hybrid numerical tolerance and the Aono Fourier truncation K_Aono are
selected independently so that both satisfy the same target before runtime is
compared.

Cache discipline
----------------
The included proposed_hybrid.py clears G1/G2/G3 at the start of each N>=5,
s>=3 subset evaluation, not for N<=4 or s<=2. This preserves independent
per-evaluation timings and avoids unnecessary overhead for simple routes.

Common reference
----------------
N = 1,2,3:
    high-precision mpmath elementary formulas.

N = 4:
    direct three-dimensional quadrature, evaluated at a strict reference
    tolerance.  This is independent of the Hybrid N=4 one-dimensional route.

N >= 5:
    a stricter Hybrid evaluation is used as the benchmark reference because no
    independent closed-form/dimension-reduced reference exists for the dense
    cases.  Reference stability is checked by comparing two stricter Hybrid
    configurations.  The reference is accepted only when the two agree well
    below the 1e-10 benchmark target.

This N>=5 reference is therefore a high-accuracy benchmarking reference rather
than a new independent validation; the correctness/stability validation is
handled in the preceding figures.

Speedup
-------
The convention remains

    S_Hybrid/Aono = t_Aono / t_Hybrid.

Thus

    S_Hybrid/Aono > 1  -> Hybrid faster,
    S_Hybrid/Aono < 1  -> Aono faster.

Outputs
-------
results/
    fig7b_raw.csv
    fig7b_plot.csv
    fig7b_summary.csv
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

from memory_probe import memory_once, memory_fields
from aono_fourier import aono_fourier_orthant
from config import NumericalConfig
from n4 import unit_ball_orthant_N4_3D
from patterns import make_beta
from proposed_hybrid import unit_ball_orthant_hybrid
from utils import (
    timed_warm,
    valid_orthant_volume,
    write_csv,
)


RESULTS = Path(__file__).resolve().parent / "results"
PROTOCOL = "Fig7b_V21_5_optimized_nodiag_elementary_20261104"
CHECKPOINT_NAME = "fig7b_checkpoint.jsonl"



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
# Common benchmark target
# =============================================================================

EPSILON_REL_TARGET = 1e-10

TIMING_REPEATS = 9
TIMING_WARMUP = 1

K_AONO_MAX = 4194304


# =============================================================================
# Hybrid accuracy scan
# =============================================================================

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
# High-accuracy reference settings
# =============================================================================

MP_DPS = 80

# N=4 direct-3D reference.
N4_REF_QUAD_EPSREL = 2e-12
N4_REF_QUAD_EPSABS = 2e-13
N4_REF_QUAD_LIMIT = 800

# N>=5 high-accuracy Hybrid reference.
REF_FA_RTOL_STRICT = 1e-13
REF_FA_ATOL_STRICT = 1e-15

REF_FA_RTOL_CHECK = 3e-13
REF_FA_ATOL_CHECK = 3e-15

# The same common FA_MAX_ORDER ceiling is used for both benchmark and
# high-accuracy reference evaluations.

# The two high-accuracy Hybrid references should agree at least one order of
# magnitude better than the final 1e-10 benchmarking target.
REFERENCE_STABILITY_LIMIT = 1e-11


# =============================================================================
# Parameter grid
# =============================================================================

DIMENSIONS = tuple(range(1, 13))

# Dense r coverage; N is already sampled at every integer from 1 through 12.
R_VALUES = tuple(np.round(np.arange(0.50, 0.9000001, 0.05), 8))

KINDS = (
    "equal",
    "mixed_equal",
    "anisotropic",
)


def kinds_for_dimension(N: int):
    # In N=1 all three nominal dense trajectories collapse to beta=(r).
    if N == 1:
        return ("equal",)
    return KINDS


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
# High-precision low-dimensional references
# =============================================================================

def mp_clip_unit(x):
    return min(mp.mpf(1), max(mp.mpf(-1), x))


def reference_N1_mp(beta) -> float:
    with mp.workdps(MP_DPS):
        b = mp.mpf(float(beta[0]))
        return float(1 + b)


def reference_N2_mp(beta) -> float:
    with mp.workdps(MP_DPS):
        b1 = mp.mpf(float(beta[0]))
        b2 = mp.mpf(float(beta[1]))

        def one_coord(x):
            return mp.mpf("0.5") * (
                x * mp.sqrt(1 - x * x)
                + mp.asin(x)
            )

        value = (
            mp.pi / 4
            + one_coord(b1)
            + one_coord(b2)
            + b1 * b2
        )
        return float(value)


def mp_Phi3(s):
    return mp.pi / 4 * (s - s**3 / 3)


def mp_Psi3(s, t):
    q = 1 - s * s - t * t
    root = mp.sqrt(max(mp.mpf(0), q))

    den_s = mp.sqrt(max(mp.mpf(0), 1 - t * t))
    den_t = mp.sqrt(max(mp.mpf(0), 1 - s * s))

    return (
        s * t / 3 * root
        + t * (3 - t * t) / 6
        * mp.asin(mp_clip_unit(s / den_s))
        + s * (3 - s * s) / 6
        * mp.asin(mp_clip_unit(t / den_t))
        - mp.mpf(1) / 3
        * mp.atan2(s * t, root)
    )


def reference_N3_mp(beta) -> float:
    with mp.workdps(MP_DPS):
        b = [
            mp.mpf(float(x))
            for x in beta
        ]

        value = mp.pi / 6

        for x in b:
            value += mp_Phi3(x)

        for i in range(3):
            for j in range(i + 1, 3):
                value += mp_Psi3(
                    b[i],
                    b[j],
                )

        value += b[0] * b[1] * b[2]
        return float(value)


# =============================================================================
# Common reference construction
# =============================================================================

def n4_reference(beta) -> dict:
    config = NumericalConfig(
        fa_rtol=1e-13,
        fa_atol=1e-15,
        fa_initial_order=8,
        fa_max_order=FA_MAX_ORDER,
        quad_epsabs=N4_REF_QUAD_EPSABS,
        quad_epsrel=N4_REF_QUAD_EPSREL,
        quad_limit=N4_REF_QUAD_LIMIT,
    )

    value = float(
        unit_ball_orthant_N4_3D(
            beta,
            config=config,
        )
    )

    return {
        "reference": value,
        "reference_name": "N4_direct_3D_strict",
        "reference_ok": math.isfinite(value),
        "reference_stability_rel_error": 0.0,
    }


def high_accuracy_hybrid_reference(beta) -> dict:
    """
    N>=5 benchmarking reference.

    Two strict Hybrid configurations are compared.  The stricter value is
    retained only when the pair is stable well below the benchmark target.
    """
    strict = NumericalConfig(
        fa_rtol=REF_FA_RTOL_STRICT,
        fa_atol=REF_FA_ATOL_STRICT,
        fa_initial_order=8,
        fa_max_order=FA_MAX_ORDER,
        quad_epsabs=2e-15,
        quad_epsrel=2e-13,
        quad_limit=800,
    )

    check = NumericalConfig(
        fa_rtol=REF_FA_RTOL_CHECK,
        fa_atol=REF_FA_ATOL_CHECK,
        fa_initial_order=8,
        fa_max_order=FA_MAX_ORDER,
        quad_epsabs=5e-15,
        quad_epsrel=5e-13,
        quad_limit=800,
    )

    value_strict, info_strict = unit_ball_orthant_hybrid(
        beta,
        config=strict,
        return_info=True,
    )

    value_check, info_check = unit_ball_orthant_hybrid(
        beta,
        config=check,
        return_info=True,
    )

    verify_hybrid_case_route(beta, info_strict)
    verify_hybrid_case_route(beta, info_check)
    value_strict = float(value_strict)
    value_check = float(value_check)

    stability = relative_error(
        value_check,
        value_strict,
    )

    ok = bool(
        info_strict["valid_volume"]
        and info_check["valid_volume"]
        and info_strict["all_FA_converged"]
        and info_check["all_FA_converged"]
        and math.isfinite(stability)
        and stability <= REFERENCE_STABILITY_LIMIT
    )

    return {
        "reference": value_strict,
        "reference_name": (
            "Hybrid_strict_stability_checked"
        ),
        "reference_ok": ok,
        "reference_stability_rel_error": stability,
        "reference_strict_FA_order": int(
            info_strict["max_FA_order"]
        ),
        "reference_check_FA_order": int(
            info_check["max_FA_order"]
        ),
    }


def construct_reference(beta) -> dict:
    N = len(beta)

    if N == 1:
        return {
            "reference": reference_N1_mp(beta),
            "reference_name": "mpmath_N1_elementary",
            "reference_ok": True,
            "reference_stability_rel_error": 0.0,
        }

    if N == 2:
        return {
            "reference": reference_N2_mp(beta),
            "reference_name": "mpmath_N2_elementary",
            "reference_ok": True,
            "reference_stability_rel_error": 0.0,
        }

    if N == 3:
        return {
            "reference": reference_N3_mp(beta),
            "reference_name": "mpmath_N3_elementary",
            "reference_ok": True,
            "reference_stability_rel_error": 0.0,
        }

    if N == 4:
        return n4_reference(beta)

    return high_accuracy_hybrid_reference(beta)


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
        err = relative_error(
            value,
            reference,
        )

        valid = bool(info["valid_volume"])
        converged = bool(info["all_FA_converged"])

        last = (
            tol,
            config,
            value,
            info,
            err,
            valid,
            converged,
        )

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
        "Hybrid_FA_order": int(
            timed_info["max_FA_order"]
        ),
        "Hybrid_rho_FA": float(
            timed_info["rho_FA"]
        ),
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
        # Aono has already been scanned independently up to K_AONO_MAX.
        # No dimension-level shortcut; do not benchmark an invalid S 9 times.
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
# Single dense case
# =============================================================================

def evaluate_dense_case(
    *,
    N: int,
    r: float,
    kind: str,
) -> dict:
    beta = make_beta(
        N,
        r,
        kind=kind,
    )

    support = int(
        np.count_nonzero(
            np.abs(beta) > 0.0
        )
    )

    if support != N:
        raise RuntimeError(
            f"Dense Fig. 7(b) requires s=N; got s={support}, N={N}."
        )

    ref_info = construct_reference(beta)

    common = {
        "N": N,
        "r": float(r),
        "kind": kind,
        "support": support,
        "beta": beta.tolist(),
        "beta_norm": float(np.linalg.norm(beta)),
        "subset_count_structural": 2**support,
        "epsilon_rel_target": EPSILON_REL_TARGET,
        **ref_info,
    }

    if not ref_info["reference_ok"]:
        return {
            **common,
            **memory_fields("Hybrid", None),
            **memory_fields("Aono", None),
            "Hybrid_target_reached": False,
            "Hybrid_value": math.nan,
            "Hybrid_rel_error": math.nan,
            "Hybrid_time_s": math.nan,
            "Hybrid_route": "reference_failed",
            "Hybrid_selected_tol": math.nan,
            "Hybrid_fa_rtol": math.nan,
            "Hybrid_fa_atol": math.nan,
            "Hybrid_quad_epsrel": math.nan,
            "Hybrid_quad_epsabs": math.nan,
            "Hybrid_FA_order": math.nan,
            "Hybrid_rho_FA": math.nan,
            "Hybrid_generic_FA_evaluated": math.nan,
            "Hybrid_Psi_evaluated": math.nan,
            "Hybrid_J_evaluated": math.nan,

            "Aono_target_reached": False,
            "Aono_value": math.nan,
            "Aono_rel_error": math.nan,
            "Aono_time_s": math.nan,
            "K_Aono": math.nan,
            "K_Aono_max_allowed": K_AONO_MAX,
            "Aono_censored": True,

            "both_target_reached": False,
            "S_Hybrid_Aono": math.nan,
        }

    reference = float(
        ref_info["reference"]
    )

    hybrid = scan_hybrid_to_target(
        beta,
        reference,
    )

    aono = scan_aono_to_target(
        beta,
        reference,
    )

    both_reached = bool(
        hybrid["Hybrid_target_reached"]
        and aono["Aono_target_reached"]
    )

    speedup = (
        aono["Aono_time_s"] / hybrid["Hybrid_time_s"]
        if (
            both_reached
            and hybrid["Hybrid_time_s"] > 0.0
        )
        else math.nan
    )

    return {
        **common,
        **hybrid,
        **aono,
        "both_target_reached": both_reached,
        "S_Hybrid_Aono": speedup,
    }


# =============================================================================
# Main simulation
# =============================================================================

def _code_fingerprint() -> str:
    names = ("simulation_fig7_b.py", "proposed_hybrid.py", "hybrid_reductions.py",
             "lauricella.py", "n4.py", "utils.py", "config.py", "elementary.py",
             "aono_fourier.py", "patterns.py", "memory_probe.py")
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update((Path(__file__).resolve().parent/name).read_bytes())
    return h.hexdigest()


def _case_key(N, r, kind):
    return f"{N}|{r:.8f}|{kind}"


def _read_or_init_checkpoint(folder:Path, fresh:bool):
    filename=folder/CHECKPOINT_NAME
    digest=_code_fingerprint()
    if fresh or not filename.exists():
        filename.write_text(json.dumps({"kind":"meta", "protocol":PROTOCOL,
                                       "code_sha256":digest})+'\n',encoding='utf-8')
        return {},filename
    with filename.open(encoding='utf-8') as stream:
        meta=json.loads(next(stream))
        if meta.get("kind")!="meta" or meta.get("protocol")!=PROTOCOL or meta.get("code_sha256")!=digest:
            raise RuntimeError("Old timing checkpoint/code mismatch. Use --fresh or a new --output-dir.")
        previous={}
        for line in stream:
            if not line.strip(): continue
            try: entry=json.loads(line)
            except json.JSONDecodeError: break
            if entry.get('kind')=='case':
                row=entry['row']
                previous[_case_key(row['N'],row['r'],row['kind'])]=row
    return previous,filename


def _write_outputs(rows,folder:Path):
    if not rows: return
    write_csv(rows,folder/'fig7b_raw.csv')
    write_csv(build_plot_table(rows),folder/'fig7b_plot.csv')
    write_csv(build_summary(rows),folder/'fig7b_summary.csv')


def run_fig7b(*,output_dir:Path|None=None,fresh=False,max_cases:int|None=None,
               checkpoint_every:int=10):
    verify_v21_route_evaluator()
    folder=Path(output_dir) if output_dir is not None else RESULTS
    folder.mkdir(parents=True,exist_ok=True)
    previous,checkpoint=_read_or_init_checkpoint(folder,fresh)
    n_new=0
    for N in DIMENSIONS:
        for r in R_VALUES:
            for kind in kinds_for_dimension(N):
                key=_case_key(N,r,kind)
                if key in previous: continue
                row=evaluate_dense_case(N=N,r=r,kind=kind)
                previous[key]=row
                with checkpoint.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({"kind":"case","row":row},allow_nan=True)+'\n')
                n_new+=1
                report_case(row)
                if n_new%checkpoint_every==0:
                    _write_outputs(list(previous.values()),folder)
                if max_cases is not None and n_new>=max_cases:
                    rows=sorted(previous.values(),key=lambda q:(q['N'],q['r'],q['kind']))
                    _write_outputs(rows,folder)
                    return rows,build_summary(rows)
    rows=sorted(previous.values(),key=lambda q:(q['N'],q['r'],q['kind']))
    _write_outputs(rows,folder)
    return rows,build_summary(rows)


# =============================================================================
# Reporting / plot-ready table
# =============================================================================

def report_case(row):
    S = row["S_Hybrid_Aono"]
    S_text = (
        f"{S:.4g}"
        if math.isfinite(S)
        else "NA"
    )

    K = row["K_Aono"]
    K_text = (
        str(int(K))
        if math.isfinite(K)
        else "NA"
    )

    eH = row["Hybrid_rel_error"]
    eA = row["Aono_rel_error"]

    eH_text = (
        f"{eH:.2e}"
        if math.isfinite(eH)
        else "NA"
    )

    eA_text = (
        f"{eA:.2e}"
        if math.isfinite(eA)
        else "NA"
    )

    print(
        f"N={row['N']:2d}, "
        f"r={row['r']:.2f}, "
        f"{row['kind']:11s}, "
        f"refOK={row['reference_ok']}, "
        f"eH={eH_text:>9s}, "
        f"eA={eA_text:>9s}, "
        f"K_Aono={K_text:>5s}, "
        f"S={S_text:>9s}, "
        f"matched={row['both_target_reached']}"
    )


def build_plot_table(rows):
    out = []

    for row in rows:
        speedup = row["S_Hybrid_Aono"]

        out.append(
            {
                "N": row["N"],
                "r": row["r"],
                "kind": row["kind"],
                "support": row["support"],

                "epsilon_rel_target": row[
                    "epsilon_rel_target"
                ],

                "reference_name": row[
                    "reference_name"
                ],
                "reference_ok": row[
                    "reference_ok"
                ],
                "reference_stability_rel_error": row[
                    "reference_stability_rel_error"
                ],
                "reference_strict_FA_order": row.get("reference_strict_FA_order", math.nan),
                "reference_check_FA_order": row.get("reference_check_FA_order", math.nan),
                "FA_max_order_allowed": FA_MAX_ORDER,

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
                "Hybrid_FA_order": row[
                    "Hybrid_FA_order"
                ],

                "Aono_target_reached": row[
                    "Aono_target_reached"
                ],
                "Aono_censored": row["Aono_censored"],
                "Hybrid_memory_peak_rss_delta_MiB":row["Hybrid_memory_peak_rss_delta_MiB"],
                "Aono_memory_peak_rss_delta_MiB":row["Aono_memory_peak_rss_delta_MiB"],
                "Hybrid_memory_peak_traced_MiB":row["Hybrid_memory_peak_traced_MiB"],
                "Aono_memory_peak_traced_MiB":row["Aono_memory_peak_traced_MiB"],
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

                "S_Hybrid_Aono": speedup,

                "plot_role": "main_speedup_and_SI_diagnostics",
                "Hybrid_error_over_target": row["Hybrid_rel_error"] / EPSILON_REL_TARGET,
                "Aono_error_over_target": row["Aono_rel_error"] / EPSILON_REL_TARGET,
                "K_Aono_max_allowed": K_AONO_MAX,
                "speedup_valid": bool(row["both_target_reached"] and math.isfinite(speedup) and speedup > 0),
                "speedup_plot": speedup if row["both_target_reached"] and math.isfinite(speedup) and speedup > 0 else math.nan,


                "Hybrid_faster": bool(
                    row["both_target_reached"]
                    and math.isfinite(speedup)
                    and speedup > 1.0
                ),

                "Aono_faster": bool(
                    row["both_target_reached"]
                    and math.isfinite(speedup)
                    and speedup < 1.0
                ),

                "subset_count_structural": row[
                    "subset_count_structural"
                ],
            }
        )

    return out


# =============================================================================
# Summary
# =============================================================================

def _summarize_group(
    subset,
    *,
    aggregation: str,
    N: int,
    kind: str,
):
    reference_ok_rows = [
        row
        for row in subset
        if row["reference_ok"]
    ]

    matched = [
        row
        for row in reference_ok_rows
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
    reference_FA_orders = [
        order
        for row in subset
        for order in (
            row.get("reference_strict_FA_order", math.nan),
            row.get("reference_check_FA_order", math.nan),
        )
        if math.isfinite(order)
    ]

    return {
        "aggregation": aggregation,
        "N": N,
        "kind": kind,
        "support": N,
        "epsilon_rel_target": EPSILON_REL_TARGET,

        "n_cases": len(subset),
        "n_reference_ok": len(
            reference_ok_rows
        ),

        "n_Hybrid_target_reached": sum(
            row["Hybrid_target_reached"]
            for row in reference_ok_rows
        ),

        "n_Aono_target_reached": sum(
            row["Aono_target_reached"]
            for row in reference_ok_rows
        ),

        "n_both_target_reached": len(
            matched
        ),
        "Hybrid_target_reached_fraction": (
            sum(row["Hybrid_target_reached"] for row in reference_ok_rows)
            / len(reference_ok_rows)
            if reference_ok_rows else math.nan
        ),
        "Aono_target_reached_fraction": (
            sum(row["Aono_target_reached"] for row in reference_ok_rows)
            / len(reference_ok_rows)
            if reference_ok_rows else math.nan
        ),

        "both_target_reached_fraction": (
            len(matched) / len(reference_ok_rows)
            if reference_ok_rows
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
            float(np.median(matched_K)) if matched_K else math.nan
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
        "max_reference_FA_order": (
            max(reference_FA_orders) if reference_FA_orders else math.nan
        ),
        "FA_max_order_allowed": FA_MAX_ORDER,
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

        "median_Hybrid_time_s": (
            float(
                np.median(
                    [
                        row["Hybrid_time_s"]
                        for row in matched
                    ]
                )
            )
            if matched
            else math.nan
        ),

        "median_Aono_time_s": (
            float(
                np.median(
                    [
                        row["Aono_time_s"]
                        for row in matched
                    ]
                )
            )
            if matched
            else math.nan
        ),

        "max_reference_stability_rel_error": (
            max(
                (
                    row[
                        "reference_stability_rel_error"
                    ]
                    for row in subset
                    if math.isfinite(
                        row[
                            "reference_stability_rel_error"
                        ]
                    )
                ),
                default=math.nan,
            )
        ),

        "K_Aono_max_allowed": (
            K_AONO_MAX
        ),
    }


def build_summary(rows):
    summary = []

    for N in DIMENSIONS:
        group_N = [
            row
            for row in rows
            if row["N"] == N
        ]

        for kind in kinds_for_dimension(N):
            subset = [
                row
                for row in group_N
                if row["kind"] == kind
            ]

            summary.append(
                _summarize_group(
                    subset,
                    aggregation="by_kind",
                    N=N,
                    kind=kind,
                )
            )

        summary.append(
            _summarize_group(
                group_N,
                aggregation="all_kinds",
                N=N,
                kind="all",
            )
        )

    return summary


# =============================================================================
# IDE main
# =============================================================================

def main():
    parser=argparse.ArgumentParser(description='Figure 7(b): dense support N=1..12')
    parser.add_argument('--output-dir',type=Path,default=RESULTS)
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--max-cases',type=int,default=None,
                        help='Quick smoke test only')
    args=parser.parse_args()
    rows,summary=run_fig7b(output_dir=args.output_dir,
                             fresh=args.fresh,max_cases=args.max_cases)
    print('Figure 7(b) cases:',len(rows),'/306')
    print('Hybrid target reached:',sum(r['Hybrid_target_reached'] for r in rows))
    print('Aono target reached:',sum(r['Aono_target_reached'] for r in rows))
    print('Both reached:',sum(r['both_target_reached'] for r in rows))
    print('Output:',args.output_dir)


if __name__=='__main__':
    main()
