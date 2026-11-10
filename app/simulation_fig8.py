from __future__ import annotations

"""
Fig. 8 — accuracy-dependent Hybrid/Aono method-selection map
============================================================

Purpose
-------
Construct a practical method-selection map for the Hybrid and Aono
representations over the two structural variables that most directly control
cost in the preceding figures:

    ambient dimension N
    effective support s = #{i : beta_i != 0}

Three application-oriented relative-error targets are compared:

    epsilon_rel,target = 1e-6   (routine accuracy)
    epsilon_rel,target = 1e-8   (high accuracy)
    epsilon_rel,target = 1e-10  (stringent accuracy)

For every geometry and every target, both methods are independently tuned to
the same target before their runtimes are compared:

    S_Hybrid/Aono = t_Aono / t_Hybrid.

Thus

    log10(S_Hybrid/Aono) > 0  -> Hybrid faster
    log10(S_Hybrid/Aono) < 0  -> Aono faster
    log10(S_Hybrid/Aono) = 0  -> equal runtime

The intended final Fig. 8 contains three N-s heatmaps (one per accuracy target)
with a common color scale:

    x = N
    y = s
    color = log10(median S_Hybrid/Aono)

The median is taken across representative displacement magnitudes and geometry
families at fixed (target, N, s).  Cells are additionally classified by the
fraction of geometries for which BOTH methods reach the requested accuracy.
This allows partial/unavailable cells to be hatched or grayed instead of being
misrepresented as a fully accuracy-matched speed comparison.

Geometry sampling
-----------------
Dimensions:

    N = 2, 3, ..., 18

Supports:

    s = 1, 2, ..., N

Displacement magnitudes:

    r = 0.50, 0.55, 0.60, ..., 0.90

Representative active-component patterns:

    equal
        all active components are positive and equal in magnitude;

    negative_equal
        all active components are negative and equal in magnitude;

    mixed_equal
        equal magnitudes with alternating signs + - + - ...;

    anisotropic
        positive active components proportional to
        2^(s-1), 2^(s-2), ..., 2, 1.

All four geometry families are included for every support s>=2. For s=1,
``mixed_equal`` and ``anisotropic`` would duplicate ``equal`` exactly, so only
the two distinct signs (``equal`` and ``negative_equal``) are included. This
avoids double-counting an identical geometric configuration in the medians.
All vectors are normalized so that ||beta||_2 = r.

This is a NEW full four-trajectory experiment, not a resume of the previous
three-family Fig. 8 run. All geometries are computed from scratch, with a new
output directory and a new checkpoint protocol.

Reference hierarchy
-------------------
The reference is constructed once per geometry and reused for all three
accuracy targets.

    s = 1:
        independent high-precision incomplete-beta half-space formula;

    s = 2:
        independent high-precision 1D conditional-probability integral;

    N = 3 and s = 3:
        independent high-precision elementary N=3 formula;

    N = 4 and s >= 3:
        independent strict direct 3D quadrature;

    N >= 5 and s >= 3:
        strict Hybrid benchmarking reference, accepted only when two stricter
        Hybrid configurations agree to <= 1e-11 relative difference.

The N>=5, s>=3 reference is therefore a high-accuracy benchmarking reference,
not an independent validation.  Independent validation is supplied by the
preceding figures.

Efficiency of the three-target scan
------------------------------------
The three accuracy targets are NOT simulated as three unrelated experiments.
For each geometry:

  1. the reference is built once;
  2. the Hybrid tolerance grid is scanned once from loose to strict and the
     first configuration reaching each target is recorded;
  3. the Aono K grid is scanned once from low to high order and the first K
     reaching each target is recorded;
  4. repeated timing is cached when two targets select the same Hybrid
     tolerance or the same Aono K. Only value-only solver calls are timed.

This preserves strict accuracy matching while keeping the final dense N-s scan
computationally practical.

Outputs
-------
results_fig8_optimized/
    fig8_raw.csv
        one row per (N, s, r, kind, accuracy target);

    fig8_plot.csv
        one heatmap-ready row per (accuracy target, N, s), aggregated over all
        representative geometries;

    fig8_summary.csv
        all-geometries and by-kind summaries, including target-reaching
        fractions, speedup quantiles, selected orders/tolerances and reference
        diagnostics.

Recommended heatmap fields
--------------------------
From fig8_plot.csv, the FIRST THREE columns are directly plottable as:

    x                         = N
    y                         = support
    z / color                 = heatmap_log10_speedup
    panel                     = epsilon_rel_target
    hatch / mask              = selection_status

The script also writes one panel-specific CSV for each accuracy target:
    fig8_plot_1e-6.csv
    fig8_plot_1e-8.csv
    fig8_plot_1e-10.csv
so each can be imported into Origin directly as X, Y, Z.

where the backward-compatible ``match_status`` is retained, while the more
explicit ``selection_status`` distinguishes

    full_speedup       both methods reach target for every geometry;
    hybrid_only        only Hybrid reaches target for every geometry;
    aono_only          only Aono reaches target for every geometry;
    mixed_with_speedup / mixed_accuracy
                       geometry-dependent mixed states;
    neither            neither method reaches target;
    reference_partial / reference_failed
                       incomplete reference validation.

For publication, ``full_speedup`` cells receive solid speed color and
``mixed_with_speedup`` cells receive matched-case median color PLUS hatching.
Never interpret a partial-cell median as an all-geometries speed comparison.
Hybrid-only and reference-partial cells do not receive speed colors.
The script also keeps the previous coarse status and matched-case diagnostics.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import mpmath as mp
import numpy as np

from aono_fourier import aono_fourier_orthant
from config import NumericalConfig
from n4 import unit_ball_orthant_N4_3D
from proposed_hybrid import unit_ball_orthant_hybrid
from utils import timed_warm, valid_orthant_volume, write_csv


BASE_DIR = Path(__file__).resolve().parent

# New four-trajectory run, with V21.5 numerical core and timing unchanged.
# Never import or merge previous Figure 8 raw or summary results.
RESULTS = BASE_DIR / "results_fig8_V20261108"
MAX_SUPPORT: int | None = None
FRESH = False
PROTOCOL = "Fig8_V21_6_four_trajectories_full_recompute_20261106"
PREVIOUS_PLOT = BASE_DIR / "results" / "fig8_plot.csv"
PREVIOUS_SUMMARY = BASE_DIR / "results" / "fig8_summary.csv"


# =============================================================================
# Accuracy targets and benchmark settings
# =============================================================================

ACCURACY_TARGETS = (
    1e-6,
    1e-8,
    1e-10,
)

# Labels are useful when importing the plot CSV directly into Origin.
ACCURACY_LABELS = {
    1e-6: "routine_1e-6",
    1e-8: "high_1e-8",
    1e-10: "stringent_1e-10",
}

TIMING_REPEATS = 9
TIMING_WARMUP = 1

# Keep the final Fig. 7 ceilings so Fig. 8 is directly comparable.
K_AONO_MAX = 4194304
HYBRID_FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384
HYBRID_QUAD_LIMIT = 600

# Scan from deliberately loose settings.  The first configuration whose ACTUAL
# error versus the common reference reaches each target is selected.
HYBRID_TOL_GRID = (
    1e-3,
    3e-4,
    1e-4,
    3e-5,
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


# =============================================================================
# Reference settings
# =============================================================================

MP_DPS = 80

# N=4 independent direct-3D reference.
N4_REF_QUAD_EPSREL = 2e-12
N4_REF_QUAD_EPSABS = 2e-13
N4_REF_QUAD_LIMIT = 800

# N>=5, s>=3 strict Hybrid benchmarking reference.
REF_FA_RTOL_STRICT = 1e-13
REF_FA_ATOL_STRICT = 1e-15
REF_FA_RTOL_CHECK = 3e-13
REF_FA_ATOL_CHECK = 3e-15
REFERENCE_STABILITY_LIMIT = 1e-11


# =============================================================================
# Dense N-s parameter grid
# =============================================================================

DIMENSIONS = tuple(range(2, 19))

# Dense r coverage, matching the final Fig. 7(b) benchmark granularity.
# For s>=2 each (N,s) cell contains 9 r values x 4 distinct geometry
# families = 36 cases. At s=1, only two distinct signs exist (18 cases).
R_VALUES = tuple(
    np.round(
        np.arange(0.50, 0.9000001, 0.05),
        8,
    )
)

KINDS = (
    "equal",
    "negative_equal",
    "mixed_equal",
    "anisotropic",
)


def supports_for_dimension(N: int) -> tuple[int, ...]:
    return tuple(range(1, min(N, MAX_SUPPORT or N) + 1))


def kinds_for_support(support: int) -> tuple[str, ...]:
    # In one dimension, mixed_equal and anisotropic are identical to equal.
    # Include the two distinct positive/negative cases exactly once each.
    # For every s >= 2, sample ALL FOUR distinct families, including negative_equal.
    if support == 1:
        return ("equal", "negative_equal")
    return KINDS


# =============================================================================
# Generic numerical helpers
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
    if max_k < 16:
        return [int(max_k)]

    values = []
    K = 16
    while K <= max_k:
        values.append(K)
        K *= 2

    if values[-1] != max_k and max_k > values[-1]:
        values.append(int(max_k))

    return values


def target_key(target: float) -> str:
    return f"{target:.0e}"


def safe_log10(x: float) -> float:
    if math.isfinite(x) and x > 0.0:
        return math.log10(x)
    return math.nan


# =============================================================================
# Geometry construction
# =============================================================================

def make_supported_beta(
    N: int,
    r: float,
    support: int,
    kind: str,
) -> np.ndarray:
    """
    Construct a vector with exactly ``support`` active components and norm r.

    The first ``support`` coordinates are active.  Coordinate permutation is
    immaterial for the spherical geometry, so this gives a deterministic and
    reproducible support convention.
    """
    if not (2 <= N <= 18):
        raise ValueError("Fig. 8 is configured for 2 <= N <= 18.")
    if not (1 <= support <= N):
        raise ValueError("Need 1 <= support <= N.")
    if not (0.0 <= r < 1.0):
        raise ValueError("Need 0 <= r < 1.")

    if support == 1 and kind not in ("equal", "negative_equal"):
        raise ValueError(
            "At support=1 use kind='equal' or kind='negative_equal'."
        )

    if kind == "equal":
        w = np.ones(support, dtype=float)

    elif kind == "negative_equal":
        w = -np.ones(support, dtype=float)

    elif kind == "mixed_equal":
        w = np.ones(support, dtype=float)
        w[1::2] *= -1.0

    elif kind == "anisotropic":
        # Same geometric idea used in the earlier anisotropic stress tests:
        # exponentially graded positive active components.
        exponents = np.arange(
            support - 1,
            -1,
            -1,
            dtype=float,
        )
        w = np.power(2.0, exponents)

    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    norm = float(np.linalg.norm(w))
    if norm == 0.0:
        raise RuntimeError("Degenerate active-direction vector.")

    beta = np.zeros(N, dtype=float)
    beta[:support] = float(r) * w / norm

    return beta


# =============================================================================
# Independent / high-accuracy reference formulas
# =============================================================================

def mp_kappa(N: int):
    return mp.pi ** (mp.mpf(N) / 2) / mp.gamma(1 + mp.mpf(N) / 2)


def mp_halfspace_probability(beta, dimension: int):
    """P(X_1 >= -beta) for X uniform in the unit ``dimension``-ball."""
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

    # Complementary representation is numerically stable for beta < 0.
    I = mp.betainc(c, a, 0, 1 - x, regularized=True)
    return mp.mpf("0.5") * I


def reference_s1_mp(N: int, beta1: float) -> float:
    with mp.workdps(MP_DPS):
        p = mp_halfspace_probability(mp.mpf(beta1), N)
        value = mp_kappa(N) * p / (mp.mpf(2) ** (N - 1))
        return float(value)


def reference_s2_mp(N: int, beta1: float, beta2: float) -> float:
    """
    Independent high-precision 1D reduction for exactly two active thresholds.
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
                * q**power
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


def mp_clip_unit(x):
    return min(mp.mpf(1), max(mp.mpf(-1), x))


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


def reference_N3_mp(beta: np.ndarray) -> float:
    with mp.workdps(MP_DPS):
        b = [mp.mpf(float(x)) for x in beta]

        value = mp.pi / 6
        for x in b:
            value += mp_Phi3(x)

        for i in range(3):
            for j in range(i + 1, 3):
                value += mp_Psi3(b[i], b[j])

        value += b[0] * b[1] * b[2]
        return float(value)


def n4_reference(beta: np.ndarray) -> dict:
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
        "reference_strict_FA_order": math.nan,
        "reference_check_FA_order": math.nan,
    }


def high_accuracy_hybrid_reference(beta: np.ndarray) -> dict:
    """
    N>=5 and support>=3 benchmarking reference.

    Two strict Hybrid evaluations must agree at least one order of magnitude
    below the strictest Fig. 8 target (1e-10).
    """
    strict = NumericalConfig(
        fa_rtol=REF_FA_RTOL_STRICT,
        fa_atol=REF_FA_ATOL_STRICT,
        fa_initial_order=HYBRID_FA_INITIAL_ORDER,
        fa_max_order=FA_MAX_ORDER,
        quad_epsabs=2e-15,
        quad_epsrel=2e-13,
        quad_limit=800,
    )

    check = NumericalConfig(
        fa_rtol=REF_FA_RTOL_CHECK,
        fa_atol=REF_FA_ATOL_CHECK,
        fa_initial_order=HYBRID_FA_INITIAL_ORDER,
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

    value_strict = float(value_strict)
    value_check = float(value_check)

    stability = relative_error(value_check, value_strict)

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
        "reference_name": "Hybrid_strict_stability_checked",
        "reference_ok": ok,
        "reference_stability_rel_error": stability,
        "reference_strict_FA_order": int(info_strict["max_FA_order"]),
        "reference_check_FA_order": int(info_check["max_FA_order"]),
    }


def construct_reference(
    beta: np.ndarray,
    support: int,
) -> dict:
    N = len(beta)
    active = np.flatnonzero(np.abs(beta) > 0.0)

    # Use the independent structured references whenever they are available,
    # even in high ambient dimension.
    if support == 1:
        value = reference_s1_mp(
            N,
            float(beta[active[0]]),
        )
        return {
            "reference": value,
            "reference_name": "mpmath_s1_incomplete_beta",
            "reference_ok": math.isfinite(value),
            "reference_stability_rel_error": 0.0,
            "reference_strict_FA_order": math.nan,
            "reference_check_FA_order": math.nan,
        }

    if support == 2:
        value = reference_s2_mp(
            N,
            float(beta[active[0]]),
            float(beta[active[1]]),
        )
        return {
            "reference": value,
            "reference_name": "mpmath_s2_1D_probability",
            "reference_ok": math.isfinite(value),
            "reference_stability_rel_error": 0.0,
            "reference_strict_FA_order": math.nan,
            "reference_check_FA_order": math.nan,
        }

    if N == 3:
        value = reference_N3_mp(beta)
        return {
            "reference": value,
            "reference_name": "mpmath_N3_elementary",
            "reference_ok": math.isfinite(value),
            "reference_stability_rel_error": 0.0,
            "reference_strict_FA_order": math.nan,
            "reference_check_FA_order": math.nan,
        }

    if N == 4:
        return n4_reference(beta)

    return high_accuracy_hybrid_reference(beta)


# =============================================================================
# Multi-target Hybrid scan and cached timing
# =============================================================================

def scan_hybrid_all_targets(
    beta: np.ndarray,
    reference: float,
) -> dict[float, dict]:
    """
    Scan the Hybrid tolerance grid once and select the first configuration that
    reaches each requested target. Repeated timing is cached by selected tol.

    Important optimization: a target that is NOT reached is never repeatedly
    timed.  Its last scan value/error is retained for diagnostics, but runtime
    is recorded as NaN because no accuracy-matched timing exists.
    """
    selected: dict[float, tuple] = {}
    last = None

    for tol in HYBRID_TOL_GRID:
        config = make_hybrid_config(tol)
        value, info = unit_ball_orthant_hybrid(
            beta,
            config=config,
            return_info=True,
        )
        verify_case_route(beta, info)

        value = float(value)
        err = relative_error(value, reference)
        valid = bool(info["valid_volume"])
        converged = bool(info["all_FA_converged"])

        last = (tol, config, value, info, err, valid, converged)

        if valid and converged and math.isfinite(err):
            for target in ACCURACY_TARGETS:
                if target not in selected and err <= target:
                    selected[target] = last

        if len(selected) == len(ACCURACY_TARGETS):
            break

    timing_cache: dict[float, tuple] = {}
    out: dict[float, dict] = {}

    for target in ACCURACY_TARGETS:
        chosen = selected.get(target)

        # Accuracy target not reached: preserve the terminal scan diagnostics,
        # but do not waste 9 repeated timings on a configuration that cannot be
        # used for the accuracy-matched speed comparison.
        if chosen is None:
            if last is None:
                out[target] = {
                    "Hybrid_target_reached": False,
                    "Hybrid_value": math.nan,
                    "Hybrid_rel_error": math.nan,
                    "Hybrid_time_s": math.nan,
                    "Hybrid_route": "scan_failed",
                    "Hybrid_selected_tol": math.nan,
                    "Hybrid_fa_rtol": math.nan,
                    "Hybrid_fa_atol": math.nan,
                    "Hybrid_quad_epsrel": math.nan,
                    "Hybrid_quad_epsabs": math.nan,
                    "Hybrid_FA_order": math.nan,
                    "Hybrid_rho_NT": math.nan,
                    "Hybrid_cancellation": math.nan,
                }
                continue

            tol, config, value, info, err, _, _ = last
            out[target] = {
                "Hybrid_target_reached": False,
                "Hybrid_value": float(value),
                "Hybrid_rel_error": float(err),
                "Hybrid_time_s": math.nan,
                "Hybrid_route": str(info["route"]),
                "Hybrid_selected_tol": float(tol),
                "Hybrid_fa_rtol": float(config.fa_rtol),
                "Hybrid_fa_atol": float(config.fa_atol),
                "Hybrid_quad_epsrel": float(config.quad_epsrel),
                "Hybrid_quad_epsabs": float(config.quad_epsabs),
                "Hybrid_FA_order": int(info["max_FA_order"]),
                "Hybrid_rho_NT": float(info["rho_NT"]),
                "Hybrid_cancellation": float(info["cancellation_indicator"]),
            }
            continue

        tol, config, _, _, _, _, _ = chosen

        if tol not in timing_cache:
            # V21.5 timing protocol: ONLY the solver's value-only path is
            # measured. Diagnostics, reference, and accuracy checks are outside.
            t_h, timed_value = timed_warm(
                unit_ball_orthant_hybrid,
                beta,
                config=config,
                return_info=False,
                repeats=TIMING_REPEATS,
                warmup=TIMING_WARMUP,
            )
            timed_value = float(timed_value)
            # Independent untimed diagnostic evaluation at the same tolerance.
            diagnostic_value, timed_info = unit_ball_orthant_hybrid(
                beta, config=config, return_info=True,
            )
            verify_case_route(beta, timed_info)
            if not math.isclose(timed_value, float(diagnostic_value),
                                rel_tol=1e-13, abs_tol=0.0):
                raise RuntimeError("Value-only and diagnostic Hybrid outputs differ")
            timing_cache[tol] = (float(t_h), timed_value, timed_info)

        t_h, timed_value, timed_info = timing_cache[tol]
        timed_err = relative_error(timed_value, reference)
        reached = bool(
            timed_info["valid_volume"]
            and timed_info["all_FA_converged"]
            and math.isfinite(timed_err)
            and timed_err <= target
        )

        out[target] = {
            "Hybrid_target_reached": reached,
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
            "Hybrid_rho_NT": float(timed_info["rho_NT"]),
            "Hybrid_cancellation": float(
                timed_info["cancellation_indicator"]
            ),
        }

    return out


# =============================================================================
# Multi-target Aono scan and cached timing
# =============================================================================

def scan_aono_all_targets(
    beta: np.ndarray,
    reference: float,
) -> dict[float, dict]:
    """
    Scan K_Aono once and select the first K reaching each target. Timing is
    cached when multiple targets select the same K.

    A target that is not reached by K_AONO_MAX retains the terminal K/value/error
    but is NOT repeatedly timed, because no accuracy-matched runtime exists.
    """
    selected: dict[float, tuple] = {}
    last = None

    for K_Aono in aono_k_grid(K_AONO_MAX):
        value = float(
            aono_fourier_orthant(
                beta,
                n_terms=K_Aono,
            )
        )

        valid = valid_orthant_volume(value, len(beta))
        err = relative_error(value, reference)
        last = (K_Aono, value, valid, err)

        if valid and math.isfinite(err):
            for target in ACCURACY_TARGETS:
                if target not in selected and err <= target:
                    selected[target] = last

        if len(selected) == len(ACCURACY_TARGETS):
            break

    timing_cache: dict[int, tuple] = {}
    out: dict[float, dict] = {}

    for target in ACCURACY_TARGETS:
        chosen = selected.get(target)

        if chosen is None:
            if last is None:
                out[target] = {
                    "Aono_target_reached": False,
                    "Aono_value": math.nan,
                    "Aono_rel_error": math.nan,
                    "Aono_time_s": math.nan,
                    "K_Aono": math.nan,
                    "K_Aono_max_allowed": K_AONO_MAX,
                }
                continue

            K_Aono, value, _, err = last
            out[target] = {
                "Aono_target_reached": False,
                "Aono_value": float(value),
                "Aono_rel_error": float(err),
                "Aono_time_s": math.nan,
                "K_Aono": int(K_Aono),
                "K_Aono_max_allowed": K_AONO_MAX,
            }
            continue

        K_Aono = int(chosen[0])

        if K_Aono not in timing_cache:
            t_a, timed_value = timed_warm(
                aono_fourier_orthant,
                beta,
                n_terms=K_Aono,
                repeats=TIMING_REPEATS,
                warmup=TIMING_WARMUP,
            )
            timing_cache[K_Aono] = (
                float(t_a),
                float(timed_value),
            )

        t_a, timed_value = timing_cache[K_Aono]
        timed_valid = valid_orthant_volume(
            timed_value,
            len(beta),
        )
        timed_err = relative_error(
            timed_value,
            reference,
        )

        reached = bool(
            timed_valid
            and math.isfinite(timed_err)
            and timed_err <= target
        )

        out[target] = {
            "Aono_target_reached": reached,
            "Aono_value": timed_value,
            "Aono_rel_error": timed_err,
            "Aono_time_s": float(t_a),
            "K_Aono": K_Aono,
            "K_Aono_max_allowed": K_AONO_MAX,
        }

    return out


# =============================================================================
# Evaluate one geometry at all three targets
# =============================================================================

def evaluate_geometry_all_targets(
    *,
    N: int,
    support: int,
    r: float,
    kind: str,
) -> list[dict]:
    beta = make_supported_beta(
        N=N,
        r=r,
        support=support,
        kind=kind,
    )

    actual_support = int(
        np.count_nonzero(np.abs(beta) > 0.0)
    )
    if actual_support != support:
        raise RuntimeError(
            f"Support mismatch: expected {support}, got {actual_support}."
        )

    ref_info = construct_reference(beta, support)

    common = {
        "protocol": PROTOCOL,
        "N": N,
        "support": support,
        "support_fraction": support / N,
        "r": float(r),
        "kind": kind,
        "beta": beta.tolist(),
        "beta_norm": float(np.linalg.norm(beta)),
        "subset_count_structural": 2**support,
        **ref_info,
    }

    rows = []

    if not ref_info["reference_ok"]:
        for target in ACCURACY_TARGETS:
            rows.append(
                {
                    **common,
                    "epsilon_rel_target": target,
                    "accuracy_label": ACCURACY_LABELS[target],
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
                    "Hybrid_rho_NT": math.nan,
                    "Hybrid_cancellation": math.nan,
                    "Aono_target_reached": False,
                    "Aono_value": math.nan,
                    "Aono_rel_error": math.nan,
                    "Aono_time_s": math.nan,
                    "K_Aono": math.nan,
                    "K_Aono_max_allowed": K_AONO_MAX,
                    "both_target_reached": False,
                    "S_Hybrid_Aono": math.nan,
                    "log10_S_Hybrid_Aono": math.nan,
                }
            )
        return rows

    reference = float(ref_info["reference"])

    hybrid_results = scan_hybrid_all_targets(beta, reference)
    aono_results = scan_aono_all_targets(beta, reference)

    for target in ACCURACY_TARGETS:
        hybrid = hybrid_results[target]
        aono = aono_results[target]

        both_reached = bool(
            hybrid["Hybrid_target_reached"]
            and aono["Aono_target_reached"]
        )

        speedup = (
            aono["Aono_time_s"] / hybrid["Hybrid_time_s"]
            if (
                both_reached
                and math.isfinite(aono["Aono_time_s"])
                and math.isfinite(hybrid["Hybrid_time_s"])
                and hybrid["Hybrid_time_s"] > 0.0
            )
            else math.nan
        )

        rows.append(
            {
                **common,
                "epsilon_rel_target": target,
                "accuracy_label": ACCURACY_LABELS[target],
                **hybrid,
                **aono,
                "both_target_reached": both_reached,
                "S_Hybrid_Aono": speedup,
                "log10_S_Hybrid_Aono": safe_log10(speedup),
            }
        )

    return rows


# =============================================================================
# Reporting
# =============================================================================

def report_geometry(rows: list[dict]):
    first = rows[0]
    chunks = []

    for row in rows:
        target = row["epsilon_rel_target"]
        S = row["S_Hybrid_Aono"]
        if math.isfinite(S):
            text = f"{target:.0e}:S={S:.3g}"
        else:
            h = int(bool(row["Hybrid_target_reached"]))
            a = int(bool(row["Aono_target_reached"]))
            text = f"{target:.0e}:H{h}A{a}"
        chunks.append(text)

    print(
        f"N={first['N']:2d}, s={first['support']:2d}, "
        f"r={first['r']:.2f}, {first['kind']:11s}, "
        f"refOK={first['reference_ok']}, "
        + " | ".join(chunks)
    )


# =============================================================================
# Heatmap summaries
# =============================================================================

def _summarize_subset(
    subset: list[dict],
    *,
    aggregation: str,
    target: float,
    N: int,
    support: int,
    kind: str,
) -> dict:
    reference_ok_rows = [
        row for row in subset
        if row["reference_ok"]
    ]

    n_total = len(subset)
    n_ref = len(reference_ok_rows)

    n_H1A1 = sum(
        bool(row["Hybrid_target_reached"])
        and bool(row["Aono_target_reached"])
        for row in reference_ok_rows
    )
    n_H1A0 = sum(
        bool(row["Hybrid_target_reached"])
        and not bool(row["Aono_target_reached"])
        for row in reference_ok_rows
    )
    n_H0A1 = sum(
        not bool(row["Hybrid_target_reached"])
        and bool(row["Aono_target_reached"])
        for row in reference_ok_rows
    )
    n_H0A0 = n_ref - n_H1A1 - n_H1A0 - n_H0A1

    matched = [
        row for row in reference_ok_rows
        if (
            row["both_target_reached"]
            and math.isfinite(row["S_Hybrid_Aono"])
            and row["S_Hybrid_Aono"] > 0.0
        )
    ]

    speeds = [row["S_Hybrid_Aono"] for row in matched]

    hybrid_reached = n_H1A1 + n_H1A0
    aono_reached = n_H1A1 + n_H0A1
    n_matched = len(matched)

    hybrid_fraction = (
        hybrid_reached / n_ref if n_ref else math.nan
    )
    aono_fraction = (
        aono_reached / n_ref if n_ref else math.nan
    )
    both_fraction = (
        n_matched / n_ref if n_ref else math.nan
    )

    # Keep the old coarse status for backward compatibility with the previous
    # fig8_plot.csv, and add a more informative selection_status for publication.
    if n_ref == 0:
        match_status = "reference_failed"
    elif n_matched == n_ref:
        match_status = "full"
    elif n_matched == 0:
        match_status = "unavailable"
    else:
        match_status = "partial"

    if n_ref == 0:
        selection_status = "reference_failed"
    elif n_ref < n_total:
        selection_status = "reference_partial"
    elif n_matched == n_ref:
        selection_status = "full_speedup"
    elif n_H1A0 == n_ref:
        selection_status = "hybrid_only"
    elif n_H0A1 == n_ref:
        selection_status = "aono_only"
    elif n_H0A0 == n_ref:
        selection_status = "neither"
    elif n_H1A1 > 0:
        selection_status = "mixed_with_speedup"
    else:
        selection_status = "mixed_accuracy"

    median_speedup = (
        float(np.median(speeds))
        if speeds else math.nan
    )

    # The figure definition is log10(median S), not median(log10 S).
    median_log_speedup = safe_log10(median_speedup)

    # Publication heatmap: matched cases are colored, including partial cells.
    # Partial cells MUST be hatched (selection_status=mixed_with_speedup);
    # the colored median refers only to geometries where both methods met
    # the target. Hybrid-only and reference-partial cells have NO speed color.
    heatmap_log10_speedup = (
        median_log_speedup
        if selection_status in ("full_speedup", "mixed_with_speedup")
        else math.nan
    )

    if selection_status == "full_speedup" and math.isfinite(median_speedup):
        if median_speedup > 1.0:
            recommended_method = "Hybrid"
        elif median_speedup < 1.0:
            recommended_method = "Aono"
        else:
            recommended_method = "Tie"
    elif n_ref > 0 and hybrid_fraction == 1.0 and aono_fraction < 1.0:
        recommended_method = "Hybrid_accuracy_advantage"
    elif n_ref > 0 and aono_fraction == 1.0 and hybrid_fraction < 1.0:
        recommended_method = "Aono_accuracy_advantage"
    else:
        recommended_method = "No_universal_recommendation"

    # A compact discrete code is also written for an optional selection-state
    # overlay/map.  +/-1 are speed winners, +/-2 are exclusive accuracy winners.
    if selection_status == "hybrid_only":
        selection_code = 2.0
    elif selection_status == "aono_only":
        selection_code = -2.0
    elif selection_status == "full_speedup" and math.isfinite(median_speedup):
        selection_code = 1.0 if median_speedup > 1.0 else (-1.0 if median_speedup < 1.0 else 0.0)
    else:
        selection_code = math.nan

    matched_K = [
        row["K_Aono"]
        for row in matched
        if math.isfinite(row["K_Aono"])
    ]
    matched_FA = [
        row["Hybrid_FA_order"]
        for row in matched
        if math.isfinite(row["Hybrid_FA_order"])
    ]
    matched_tol = [
        row["Hybrid_selected_tol"]
        for row in matched
        if math.isfinite(row["Hybrid_selected_tol"])
    ]

    reference_stabilities = [
        row["reference_stability_rel_error"]
        for row in subset
        if math.isfinite(row["reference_stability_rel_error"])
    ]

    return {
        "aggregation": aggregation,
        "epsilon_rel_target": target,
        "accuracy_label": ACCURACY_LABELS[target],
        "N": N,
        "support": support,
        "support_fraction": support / N,
        "kind": kind,

        "n_cases": n_total,
        "n_reference_ok": n_ref,
        "n_Hybrid_target_reached": hybrid_reached,
        "n_Aono_target_reached": aono_reached,
        "n_both_target_reached": n_matched,
        "n_H1A1": n_H1A1,
        "n_H1A0": n_H1A0,
        "n_H0A1": n_H0A1,
        "n_H0A0": n_H0A0,

        "Hybrid_target_reached_fraction": hybrid_fraction,
        "Aono_target_reached_fraction": aono_fraction,
        "both_target_reached_fraction": both_fraction,
        "match_status": match_status,
        "selection_status": selection_status,
        "selection_code": selection_code,

        "median_S_Hybrid_Aono": median_speedup,
        "median_log10_S_Hybrid_Aono": median_log_speedup,
        "heatmap_log10_speedup": heatmap_log10_speedup,
        "min_S_Hybrid_Aono": (
            min(speeds) if speeds else math.nan
        ),
        "max_S_Hybrid_Aono": (
            max(speeds) if speeds else math.nan
        ),
        "q25_S_Hybrid_Aono": (
            float(np.quantile(speeds, 0.25))
            if speeds else math.nan
        ),
        "q75_S_Hybrid_Aono": (
            float(np.quantile(speeds, 0.75))
            if speeds else math.nan
        ),

        "recommended_method": recommended_method,

        "median_K_Aono": (
            float(np.median(matched_K))
            if matched_K else math.nan
        ),
        "max_K_Aono": (
            max(matched_K) if matched_K else math.nan
        ),
        "K_Aono_max_allowed": K_AONO_MAX,

        "median_Hybrid_FA_order": (
            float(np.median(matched_FA))
            if matched_FA else math.nan
        ),
        "max_Hybrid_FA_order": (
            max(matched_FA) if matched_FA else math.nan
        ),
        "FA_max_order_allowed": FA_MAX_ORDER,

        "median_Hybrid_selected_tol": (
            float(np.median(matched_tol))
            if matched_tol else math.nan
        ),

        "max_reference_stability_rel_error": (
            max(reference_stabilities)
            if reference_stabilities else math.nan
        ),
    }


def build_summary(rows: list[dict]) -> list[dict]:
    summary = []

    for target in ACCURACY_TARGETS:
        rows_t = [
            row for row in rows
            if row["epsilon_rel_target"] == target
        ]

        dimensions_present = sorted({row["N"] for row in rows_t})
        for N in dimensions_present:
            rows_N = [row for row in rows_t if row["N"] == N]

            for support in supports_for_dimension(N):
                cell = [
                    row for row in rows_N
                    if row["support"] == support
                ]

                summary.append(
                    _summarize_subset(
                        cell,
                        aggregation="all_geometries",
                        target=target,
                        N=N,
                        support=support,
                        kind="all",
                    )
                )

                for kind in kinds_for_support(support):
                    cell_kind = [
                        row for row in cell
                        if row["kind"] == kind
                    ]
                    summary.append(
                        _summarize_subset(
                            cell_kind,
                            aggregation="by_kind",
                            target=target,
                            N=N,
                            support=support,
                            kind=kind,
                        )
                    )

    return summary


def build_plot_table(summary: list[dict]) -> list[dict]:
    """Return only all-geometries cells for direct three-panel heatmap plotting."""
    out = []

    for row in summary:
        if row["aggregation"] != "all_geometries":
            continue

        out.append(
            {
                # First three columns are deliberately x, y, z for direct
                # Origin heatmap import after filtering by accuracy target.
                "N": row["N"],
                "support": row["support"],
                "heatmap_log10_speedup": row[
                    "heatmap_log10_speedup"
                ],

                # Panel selector / metadata follow after the plotting triplet.
                "epsilon_rel_target": row["epsilon_rel_target"],
                "accuracy_label": row["accuracy_label"],
                "protocol": PROTOCOL,
                "support_fraction": row["support_fraction"],

                "match_status": row["match_status"],
                "selection_status": row["selection_status"],
                "selection_code": row["selection_code"],
                "hatch_flag": int(row["selection_status"] == "mixed_with_speedup"),
                "n_H1A1": row["n_H1A1"],
                "n_H1A0": row["n_H1A0"],
                "n_H0A1": row["n_H0A1"],
                "n_H0A0": row["n_H0A0"],
                "both_target_reached_fraction": row[
                    "both_target_reached_fraction"
                ],
                "Hybrid_target_reached_fraction": row[
                    "Hybrid_target_reached_fraction"
                ],
                "Aono_target_reached_fraction": row[
                    "Aono_target_reached_fraction"
                ],

                # Diagnostic color value if the author chooses to show the
                # matched-case median beneath a hatch in partial cells.
                "matched_median_log10_speedup": row[
                    "median_log10_S_Hybrid_Aono"
                ],
                "median_S_Hybrid_Aono": row[
                    "median_S_Hybrid_Aono"
                ],
                "q25_S_Hybrid_Aono": row[
                    "q25_S_Hybrid_Aono"
                ],
                "q75_S_Hybrid_Aono": row[
                    "q75_S_Hybrid_Aono"
                ],

                "recommended_method": row[
                    "recommended_method"
                ],

                "n_cases": row["n_cases"],
                "n_reference_ok": row["n_reference_ok"],
                "n_both_target_reached": row[
                    "n_both_target_reached"
                ],

                "median_K_Aono": row["median_K_Aono"],
                "median_Hybrid_FA_order": row[
                    "median_Hybrid_FA_order"
                ],
                "max_reference_stability_rel_error": row[
                    "max_reference_stability_rel_error"
                ],
            }
        )

    return out


# =============================================================================
# V21.5 route integrity and checkpoint protocol
# =============================================================================


def verify_case_route(beta: np.ndarray, info: dict) -> None:
    """Assert the executed V21.5 route, including absence of G recursion
    in the elementary/direct low-dimensional and low-support branches.
    This check is outside all timing windows.
    """
    N = len(beta)
    s = int(np.count_nonzero(np.asarray(beta) != 0.0))
    route = str(info.get("route", ""))
    if N <= 2:
        allowed = {f"N{N}_elementary"}
    elif N == 3:
        allowed = {"N3_elementary", "N3_elementary_s1", "N3_elementary_s2"}
    elif N == 4 and s <= 2:
        allowed = {f"N4_elementary_s{s}"}
    elif N == 4:
        allowed = {"N4_1D_reduction"}
    elif s == 1:
        allowed = {"s1_incomplete_beta"}
    elif s == 2:
        allowed = {"s2_elementary_iterative", "s2_stable_1D_probability"}
    else:
        allowed = {"hybrid_subset"}
    if route not in allowed:
        raise RuntimeError(f"Wrong V21.5 Hybrid route N={N} s={s}: {route}; expected {allowed}")
    if N <= 3 or s <= 2:
        for key in ("evaluated_generic_FA_terms", "evaluated_Psi_terms", "evaluated_J_terms"):
            if int(info.get(key, -1)) != 0:
                raise RuntimeError(f"Low-dimensional/support calculation used recursion: {key}={info.get(key)}")
    if N == 4 and s >= 3:
        if int(info.get("evaluated_J_terms", -1)) != math.comb(s,3):
            raise RuntimeError("N4 one-dimensional J count mismatch")
        if int(info.get("evaluated_generic_FA_terms", -1)) != 0:
            raise RuntimeError("N4 accidentally used general FA")
    if N >= 5 and s >= 3:
        expected = {
            "evaluated_generic_FA_terms": sum(math.comb(s,m) for m in range(4,s+1) if (N-m)%2==1),
            "evaluated_Psi_terms": math.comb(s,2) if N%2 else 0,
            "evaluated_J_terms": math.comb(s,3) if N%2==0 else 0,
            "evaluated_subset_terms": 2**s,
        }
        for key, count in expected.items():
            if int(info.get(key, -1)) != count:
                raise RuntimeError(f"V21.5 Route count mismatch N={N} s={s}: {key}, expected {count}, got {info.get(key)}")


def verify_v21_5_evaluator() -> None:
    for N, active in ((2,[0.21,-0.18]), (3,[0.20]), (3,[0.20,-0.15]),
                      (3,[0.20,-0.15,0.13]), (4,[0.20]),(4,[0.20,-0.15]),
                      (4,[0.20,-0.15,0.13]), (5,[0.20]),(5,[0.20,-0.15]),
                      (5,[0.20,-0.15,0.13]),(6,[0.20,-0.15,0.13]),
                      (7,[0.20,-0.15,0.13,0.09])):
        beta = np.zeros(N)
        beta[:len(active)] = active
        value, info = unit_ball_orthant_hybrid(beta, return_info=True)
        verify_case_route(beta, info)
        only = unit_ball_orthant_hybrid(beta, return_info=False)
        if not math.isclose(float(value),float(only),rel_tol=1e-13,abs_tol=0.0):
            raise RuntimeError("Inconsistent value-only/diagnostic Hybrid evaluator")
    print("V21.5 fast analytic routes and Route(U/T/G) verified.")



def verify_four_trajectory_sampling() -> None:
    """Validate full negative branch and s=1 duplicate suppression (untimed)."""
    assert kinds_for_support(1) == ("equal", "negative_equal")
    assert kinds_for_support(2) == KINDS
    for support in (1, 2, 3, 4, 10):
        N = max(5, support)
        patterns = []
        for kind in kinds_for_support(support):
            beta = make_supported_beta(N=N, r=0.7, support=support, kind=kind)
            if not np.isclose(np.linalg.norm(beta), 0.7, rtol=1e-13, atol=1e-14):
                raise RuntimeError(f"Wrong displacement norm for s={support}, kind={kind}")
            if np.count_nonzero(beta) != support:
                raise RuntimeError(f"Wrong support for s={support}, kind={kind}")
            if kind == "negative_equal" and not np.all(beta[:support] < 0):
                raise RuntimeError("negative_equal is not entirely negative")
            patterns.append(tuple(beta))
        if len(set(patterns)) != len(patterns):
            raise RuntimeError(f"Duplicate geometry patterns for s={support}")
    # Default full-grid count, independent of runtime overrides.
    counts = sum(2 if s == 1 else 4 for N in range(2, 19) for s in range(1, N+1))
    if counts * 9 != 5814:
        raise RuntimeError("Unexpected full four-trajectory geometry count")
    print("Four-trajectory sampling verified (s=1: 2, s>=2: 4; full grid: 5814 geometries).")

_CHECKPOINT_BOOLS = {"reference_ok", "Hybrid_target_reached", "Aono_target_reached", "both_target_reached"}
_CHECKPOINT_INTS = {"N", "support", "subset_count_structural", "Hybrid_FA_order", "K_Aono", "K_Aono_max_allowed", "reference_strict_FA_order", "reference_check_FA_order"}
_CHECKPOINT_TEXTS = {"protocol", "kind", "beta", "accuracy_label", "reference_name", "Hybrid_route"}

def _checkpoint_load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        values = list(csv.DictReader(fh))
    rows = []
    for row in values:
        restored = {}
        for k,v in row.items():
            if k in _CHECKPOINT_BOOLS:
                if v not in ("True","False"):
                    raise RuntimeError(f"Bad checkpoint boolean {path}: {k}={v}")
                restored[k] = v == "True"
            elif k in _CHECKPOINT_TEXTS:
                restored[k] = v
            elif v in (None, ""):
                restored[k] = math.nan
            else:
                restored[k] = float(v)
                if k in _CHECKPOINT_INTS and math.isfinite(restored[k]):
                    restored[k] = int(restored[k])
        rows.append(restored)
    return rows


def _checkpoint_write(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    write_csv(rows, tmp)
    tmp.replace(path)


def _manifest_payload() -> dict:
    # Any edit to the Figure 8 code or numerical core invalidates the cache.
    sources = ("simulation_fig8.py", "proposed_hybrid.py", "lauricella.py", "elementary.py", "hybrid_reductions.py", "n4.py", "aono_fourier.py", "config.py", "utils.py")
    return {
        "protocol": PROTOCOL,
        "source_sha256": {name: hashlib.sha256((BASE_DIR / name).read_bytes()).hexdigest() for name in sources},
        "dimensions": list(DIMENSIONS), "max_support": MAX_SUPPORT,
        "r_values": list(R_VALUES), "kinds": list(KINDS),
        "targets": list(ACCURACY_TARGETS), "hybrid_tol_grid": list(HYBRID_TOL_GRID),
        "timing_repeats": TIMING_REPEATS, "timing_warmup": TIMING_WARMUP,
        "K_AONO_MAX": K_AONO_MAX, "FA_MAX_ORDER": FA_MAX_ORDER,
    }


def _prepare_checkpoints(checkpoint_dir: Path) -> None:
    if FRESH:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        for p in RESULTS.glob("fig8_*.csv"):
            p.unlink()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_dir / "protocol.json"
    payload = _manifest_payload()
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != payload:
            raise RuntimeError("Checkpoint code/protocol differs from the V21.6 four-trajectory protocol; use --fresh or a new --output-dir.")
    else:
        if any(checkpoint_dir.glob("*.csv")):
            raise RuntimeError("Checkpoint files found without protocol manifest; use --fresh or another --output-dir.")
        manifest.write_text(json.dumps(payload,indent=2),encoding="utf-8")


# =============================================================================
# Main simulation
# =============================================================================

def run_fig8():
    rows: list[dict] = []
    n_geometries = sum(len(R_VALUES)*len(kinds_for_support(s))
                       for N in DIMENSIONS for s in supports_for_dimension(N))
    print(f"Fig. 8 V21.6 four-trajectory method-selection, {DIMENSIONS[0]}..{DIMENSIONS[-1]}; geometries={n_geometries}")
    print(f"Targets: {ACCURACY_TARGETS}; output: {RESULTS}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = RESULTS / "checkpoints"
    _prepare_checkpoints(checkpoint_dir)
    completed = 0

    for N in DIMENSIONS:
        rows_N: list[dict] = []
        for support in supports_for_dimension(N):
            path = checkpoint_dir / f"fig8_raw_N{N:02d}_s{support:02d}.csv"
            expected_geometries = len(R_VALUES)*len(kinds_for_support(support))
            if path.exists():
                cell_rows = _checkpoint_load(path)
                expected = {(N,support,float(r),kind,float(t)) for r in R_VALUES
                            for kind in kinds_for_support(support) for t in ACCURACY_TARGETS}
                actual = {(int(x["N"]),int(x["support"]),float(x["r"]),x["kind"],float(x["epsilon_rel_target"])) for x in cell_rows}
                if (len(cell_rows)!=len(expected) or actual!=expected
                        or any(x.get("protocol")!=PROTOCOL for x in cell_rows)):
                    raise RuntimeError(f"Incomplete/corrupt checkpoint: {path}; restart with --fresh")
                print(f"Resuming N={N} s={support} ({expected_geometries} geometries)")
            else:
                # Save after EACH geometry as well as each completed support.
                # Important for the expensive high-N, high-s cells.
                partial_path = checkpoint_dir / f"fig8_partial_N{N:02d}_s{support:02d}.csv"
                cell_rows = _checkpoint_load(partial_path) if partial_path.exists() else []
                valid_keys = {(N,support,float(r),kind,float(t)) for r in R_VALUES
                              for kind in kinds_for_support(support) for t in ACCURACY_TARGETS}
                actual_keys = {(int(x["N"]),int(x["support"]),float(x["r"]),x["kind"],float(x["epsilon_rel_target"])) for x in cell_rows}
                if (len(actual_keys) != len(cell_rows) or not actual_keys.issubset(valid_keys)
                        or any(x.get("protocol")!=PROTOCOL for x in cell_rows)):
                    raise RuntimeError(f"Corrupt partial checkpoint: {partial_path}; use --fresh")
                for r in R_VALUES:
                    for kind in kinds_for_support(support):
                        group_keys = {(N,support,float(r),kind,float(t)) for t in ACCURACY_TARGETS}
                        if group_keys.issubset(actual_keys):
                            continue
                        if group_keys & actual_keys:
                            raise RuntimeError(f"Partial geometry in {partial_path}; use --fresh")
                        geometry_rows = evaluate_geometry_all_targets(
                            N=N, support=support, r=r, kind=kind,
                        )
                        cell_rows.extend(geometry_rows)
                        actual_keys.update(group_keys)
                        _checkpoint_write(cell_rows, partial_path)
                        report_geometry(geometry_rows)
                if len(cell_rows) != len(valid_keys):
                    raise RuntimeError(f"Incomplete support checkpoint {N=}, {support=}")
                _checkpoint_write(cell_rows,path)
                if partial_path.exists():
                    partial_path.unlink()
                print(f"Checkpoint N={N} s={support} saved.")
            rows.extend(cell_rows)
            rows_N.extend(cell_rows)
            completed += expected_geometries
            print(f"Progress: {completed}/{n_geometries} geometries")

        _checkpoint_write(rows_N,checkpoint_dir / f"fig8_raw_N{N:02d}.csv")
        summary_N = build_summary(rows_N)
        plot_N = build_plot_table(summary_N)
        _checkpoint_write(summary_N,checkpoint_dir / f"fig8_summary_N{N:02d}.csv")
        _checkpoint_write(plot_N,checkpoint_dir / f"fig8_plot_N{N:02d}.csv")
        print(f"Dimension N={N} finished.\n")

    write_csv(
        rows,
        RESULTS / "fig8_raw.csv",
    )

    summary = build_summary(rows)
    write_csv(
        summary,
        RESULTS / "fig8_summary.csv",
    )

    plot_rows = build_plot_table(summary)
    write_csv(
        plot_rows,
        RESULTS / "fig8_plot.csv",
    )

    # Convenience files for Origin: each file is one heatmap panel and its
    # first three columns are exactly x=N, y=support, z=heatmap value.
    # A second file carries a discrete selection code for accuracy-only regions.
    for target in ACCURACY_TARGETS:
        panel_rows = [
            row for row in plot_rows
            if row["epsilon_rel_target"] == target
        ]
        target_tag = f"{target:.0e}".replace("e-0", "e-")
        write_csv(
            panel_rows,
            RESULTS / f"fig8_plot_{target_tag}.csv",
        )

        selection_rows = [
            {
                "N": row["N"],
                "support": row["support"],
                "selection_code": row["selection_code"],
                "selection_status": row["selection_status"],
                "hatch_flag": row["hatch_flag"],
                "recommended_method": row["recommended_method"],
                "n_H1A1": row["n_H1A1"],
                "n_H1A0": row["n_H1A0"],
                "n_H0A1": row["n_H0A1"],
                "n_H0A0": row["n_H0A0"],
            }
            for row in panel_rows
        ]
        write_csv(
            selection_rows,
            RESULTS / f"fig8_selection_{target_tag}.csv",
        )

    return rows, plot_rows, summary


# =============================================================================
# Optional overlap check against the previous Fig. 8 plot
# =============================================================================

def compare_with_previous_plot(plot_rows: list[dict]):
    """
    If results/fig8_plot.csv from the previous run exists, compare overlapping
    (target, N, s) cells with the optimized output.  This is diagnostic only;
    the new run is never altered by the previous file.
    """
    if not PREVIOUS_PLOT.exists():
        print(f"Previous plot reference not found: {PREVIOUS_PLOT}")
        return []

    old = {}
    with PREVIOUS_PLOT.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (
                    float(row["epsilon_rel_target"]),
                    int(float(row["N"])),
                    int(float(row["support"])),
                )
            except (KeyError, TypeError, ValueError):
                continue
            old[key] = row

    checks = []
    for row in plot_rows:
        key = (
            float(row["epsilon_rel_target"]),
            int(row["N"]),
            int(row["support"]),
        )
        old_row = old.get(key)
        if old_row is None:
            continue

        def parse_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return math.nan

        old_z = parse_float(old_row.get("heatmap_log10_speedup"))
        new_z = parse_float(row.get("heatmap_log10_speedup"))
        delta = (
            new_z - old_z
            if math.isfinite(old_z) and math.isfinite(new_z)
            else math.nan
        )

        checks.append({
            "epsilon_rel_target": key[0],
            "N": key[1],
            "support": key[2],
            "old_heatmap_log10_speedup": old_z,
            "new_heatmap_log10_speedup": new_z,
            "delta_log10_speedup": delta,
            "old_match_status": old_row.get("match_status", ""),
            "new_match_status": row.get("match_status", ""),
            "new_selection_status": row.get("selection_status", ""),
        })

    if checks:
        write_csv(checks, RESULTS / "fig8_overlap_check.csv")
        finite_deltas = [
            abs(x["delta_log10_speedup"])
            for x in checks
            if math.isfinite(x["delta_log10_speedup"])
        ]
        if finite_deltas:
            print(
                "Previous/new overlap check: "
                f"{len(checks)} cells, max |delta log10 S|="
                f"{max(finite_deltas):.4g}"
            )
        else:
            print(
                f"Previous/new overlap check: {len(checks)} overlapping cells; "
                "no pair had two finite heatmap values."
            )
    return checks


# =============================================================================
# IDE main
# =============================================================================

def main():
    global RESULTS, DIMENSIONS, R_VALUES, MAX_SUPPORT, FRESH, TIMING_REPEATS
    parser = argparse.ArgumentParser(description="Fig. 8 V21.6 four-trajectory accuracy-matched method-selection heatmaps")
    parser.add_argument("--output-dir",type=Path,default=RESULTS)
    parser.add_argument("--fresh",action="store_true",help="discard checkpoints/CSV in selected output directory")
    parser.add_argument("--min-n",type=int,default=2)
    parser.add_argument("--max-n",type=int,default=18)
    parser.add_argument("--max-support",type=int,default=None)
    parser.add_argument("--r-values",type=float,nargs="+",default=None,help="override r grid for testing")
    parser.add_argument("--timing-repeats",type=int,default=TIMING_REPEATS)
    parser.add_argument("--check-only",action="store_true",help="validate the V21.5 Hybrid core and four-trajectory sampling, then exit")
    args=parser.parse_args()
    if not (2<=args.min_n<=args.max_n<=18):
        parser.error("Require 2 <= min-n <= max-n <= 18")
    if args.max_support is not None and not (1<=args.max_support<=18):
        parser.error("Require 1 <= max-support <= 18")
    if args.timing_repeats<1:
        parser.error("timing-repeats must be >=1")
    if args.r_values is not None and (not args.r_values or any(not(0<r<1) for r in args.r_values)):
        parser.error("Require all r-values in (0,1)")
    RESULTS=args.output_dir.resolve()
    FRESH=args.fresh
    DIMENSIONS=tuple(range(args.min_n,args.max_n+1))
    MAX_SUPPORT=args.max_support
    if args.r_values is not None:
        R_VALUES=tuple(sorted(set(args.r_values)))
    TIMING_REPEATS=args.timing_repeats
    verify_v21_5_evaluator()
    verify_four_trajectory_sampling()
    if args.check_only:
        return
    rows, plot_rows, summary = run_fig8()
    print()
    print("Fig. 8 V21.6 four-trajectory simulation finished.")
    print(f"Raw rows: {len(rows)}")
    print(f"Heatmap cells: {len(plot_rows)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Raw CSV: {RESULTS / 'fig8_raw.csv'}")
    print(f"Plot CSV: {RESULTS / 'fig8_plot.csv'}")
    print(f"Summary CSV: {RESULTS / 'fig8_summary.csv'}")

    for target in ACCURACY_TARGETS:
        cells = [
            row for row in plot_rows
            if row["epsilon_rel_target"] == target
        ]
        n_full = sum(row["match_status"] == "full" for row in cells)
        n_partial = sum(row["match_status"] == "partial" for row in cells)
        n_unavailable = sum(
            row["match_status"] == "unavailable" for row in cells
        )
        n_ref_failed = sum(
            row["match_status"] == "reference_failed" for row in cells
        )
        n_hybrid_only = sum(
            row["selection_status"] == "hybrid_only" for row in cells
        )
        n_aono_only = sum(
            row["selection_status"] == "aono_only" for row in cells
        )

        print(
            f"target={target:.0e}: "
            f"full={n_full}, partial={n_partial}, "
            f"unavailable={n_unavailable}, reference_failed={n_ref_failed}, "
            f"hybrid_only={n_hybrid_only}, aono_only={n_aono_only}"
        )

    # Explicit closure points against Fig. 7(a).
    strict = [
        row for row in plot_rows
        if row["epsilon_rel_target"] == 1e-10
        and (row["N"], row["support"]) in ((17, 1), (18, 1), (18, 2))
    ]
    if strict:
        print()
        print("Strict-accuracy closure points (compare with Fig. 7(a)):")
        for row in strict:
            print(
                f"  N={row['N']}, s={row['support']}: "
                f"status={row['selection_status']}, "
                f"Hreach={row['Hybrid_target_reached_fraction']:.3f}, "
                f"Areach={row['Aono_target_reached_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
