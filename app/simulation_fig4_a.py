from __future__ import annotations

"""
Fig. 4(a) dedicated simulation
==============================

Purpose
-------
Generate a dense, publication-ready data set for the N=4 isolated positive
triple term

    J(a,b,c)
    = a b c F_A^(3)(
        -1/2;
        1/2,1/2,1/2;
        3/2,3/2,3/2;
        a^2,b^2,c^2
      )

and compare

    1. direct Lauricella F_A evaluation,
    2. direct three-dimensional quadrature,
    3. the Appendix-C one-dimensional S-E-L reduction.

This file is intentionally restricted to Fig. 4(a).  It does NOT run the
full N=4 volume benchmark or any Aono calculation.

Two smooth deterministic paths are used:
    - equal:              (1,1,1)
    - anisotropic_8_4_1:  (8,4,1)

The old random-positive path is deliberately excluded from this figure because
a newly generated random direction at each r does not define a smooth curve.

Output
------
Two CSV files are written under ./results_fig4a:

    fig4a_raw.csv
        Full numerical output for every r and every shape.

    fig4a_summary.csv
        Plot-ready wide-format table.  One row corresponds to one r value and
        contains the equal and anisotropic curves side by side.

Additional diagnostics
----------------------
The raw table now also records:
    rel_err_FA_vs_3D
    rel_err_1D_vs_3D
    ULP_FA_vs_3D
    ULP_1D_vs_3D

The summary table contains the corresponding anis_* and equal_* columns.

Important note for logarithmic plotting
---------------------------------------
At a few low-error points, two float64 methods may return exactly the same
binary number, so the raw absolute difference is exactly zero.  A logarithmic
axis cannot display zero.  Therefore:

    *_plot

columns replace ONLY exact zeros by one ULP of the corresponding 3D reference.

The original raw errors are retained unchanged in fig4a_raw.csv and also in
the *_raw columns of fig4a_summary.csv.  The *_zero_replaced flag identifies
every point where this plotting floor was used.

FA convergence
--------------
The defining series converges intrinsically more slowly as r approaches 1,
especially along the anisotropic path.  The shared Lauricella evaluator now
uses the original fast low-order path and a high-precision fallback above the
low-order range, so the near-boundary grid can be evaluated without the former
float64 Pochhammer overflow.

All simulations use the same safety ceiling

    FA_MAX_ORDER = 16384.

Adaptive doubling still stops as soon as the requested tolerance is met; the
ceiling is therefore not the routine evaluation order.

Aono is not used in Fig. 4(a); later Aono benchmarks retain their own Fourier
truncation ceiling, K_Aono,max, which is distinct from K_FA,max.
"""

import math
import time
from pathlib import Path

import numpy as np

from config import NumericalConfig
from lauricella import lauricella_FA_half_parameters
from n4 import (
    J4_direct_triple,
    J4_reduced_1D,
)
from utils import write_csv


RESULTS = (
    Path(__file__).resolve().parent
    / "results"
)


# =============================================================================
# Numerical settings
# =============================================================================

FA_RTOL = 1e-12
FA_ATOL = 1e-14
FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384

# Fig. 4(a) does not use Aono.  Kept here so that later figure-specific scripts
# use a consistent machine-aware upper bound.
AONO_MAX_K_FOR_LATER = 4194304

# Direct numerical quadrature settings.  These are intentionally tighter than
# the general large-scan settings because Fig. 4(a) is an accuracy figure.
QUAD_EPSABS = 2e-14
QUAD_EPSREL = 2e-13
QUAD_LIMIT = 600


# =============================================================================
# Dense r-grid
# =============================================================================

def make_r_grid() -> np.ndarray:
    """
    High-density r-grid for Fig. 4(a).

    Region I:   0.02 -- 0.80, step 0.01
    Region II:  0.805 -- 0.95, step 0.005
    Region III: 0.9525 -- 0.9900, step 0.0025
    Region IV:  0.9910 -- 0.9955, step 0.0005

    This deliberately oversamples the near-boundary region so that the plotted
    error curves are determined by computed points rather than interpolation.
    The robust high-order F_A backend removes the previous float64-overflow
    reason for keeping this grid sparse.
    """
    low = np.arange(0.02, 0.8000001, 0.01)
    middle = np.arange(0.805, 0.9500001, 0.005)
    high = np.arange(0.9525, 0.9900001, 0.0025)
    near_boundary = np.arange(0.9910, 0.9955001, 0.0005)

    grid = np.concatenate(
        [
            low,
            middle,
            high,
            near_boundary,
        ]
    )

    # Guard against accidental duplicate floating-point values.
    grid = np.unique(
        np.round(
            grid,
            8,
        )
    )

    return grid


# =============================================================================
# Shape definitions
# =============================================================================

def shape_direction(
    shape: str,
) -> np.ndarray:
    """
    Return a positive unit vector specifying a fixed direction in (a,b,c).
    """
    if shape == "equal":
        w = np.ones(
            3,
            dtype=float,
        )

    elif shape == "anisotropic_8_4_1":
        w = np.array(
            [8.0, 4.0, 1.0],
            dtype=float,
        )

    else:
        raise ValueError(
            f"Unknown shape: {shape}"
        )

    return w / np.linalg.norm(w)


# =============================================================================
# Numerical helpers
# =============================================================================

def j4_lauricella_with_info(
    a: float,
    b: float,
    c: float,
    *,
    config: NumericalConfig,
):
    """
    Same mathematical quantity as J4_lauricella, but returns the internal
    F_A convergence diagnostics needed for the raw table.
    """
    x = np.array(
        [
            a * a,
            b * b,
            c * c,
        ],
        dtype=float,
    )

    FA, info = (
        lauricella_FA_half_parameters(
            -0.5,
            x,
            rtol=config.fa_rtol,
            atol=config.fa_atol,
            initial_order=(
                config.fa_initial_order
            ),
            max_order=(
                config.fa_max_order
            ),
            return_info=True,
        )
    )

    value = (
        a
        * b
        * c
        * FA
    )

    return value, info


def time_once(
    func,
    *args,
    **kwargs,
):
    """
    Fig. 4(a) is an accuracy plot, not a timing benchmark.

    A single-call wall time is recorded only as a diagnostic so that the dense
    grid does not become unnecessarily expensive.  Dedicated timing belongs in
    Fig. 4(b).
    """
    t0 = time.perf_counter()

    result = func(
        *args,
        **kwargs,
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    return elapsed, result


def one_ulp_floor(
    reference: float,
) -> float:
    """
    One floating-point ULP around |reference|.

    Used ONLY for plotting an exact-zero difference on a logarithmic axis.
    """
    scale = abs(
        float(reference)
    )

    if scale == 0.0:
        return float(
            np.nextafter(
                0.0,
                1.0,
            )
        )

    floor = float(
        np.spacing(scale)
    )

    if not (
        math.isfinite(floor)
        and floor > 0.0
    ):
        floor = float(
            np.finfo(float).tiny
        )

    return floor


def relative_error_safe(
    value: float,
    reference: float,
) -> float:
    """
    Standard relative error used only for Fig. 4(a) diagnostics:

        |value-reference| / |reference|.

    Returns NaN if either input is non-finite or if reference is zero.
    """
    if not (
        math.isfinite(value)
        and math.isfinite(reference)
    ):
        return math.nan

    if reference == 0.0:
        return math.nan

    return abs(value - reference) / abs(reference)


def error_in_ulp(
    value: float,
    reference: float,
) -> float:
    r"""
    Difference measured in local floating-point ULPs of the 3D reference:

        n_ULP
        =
        |value-reference|
        /
        ulp(reference).

    This is useful for diagnosing the staircase structure in Fig. 4(a).
    A value near 1, 2, 3, ... means that the discrepancy is approximately
    one, two, three, ... adjacent float64 spacings at the reference magnitude.
    """
    if not (
        math.isfinite(value)
        and math.isfinite(reference)
    ):
        return math.nan

    ulp = one_ulp_floor(reference)

    if not (
        math.isfinite(ulp)
        and ulp > 0.0
    ):
        return math.nan

    return abs(value - reference) / ulp


def log_plot_value(
    raw_error: float,
    *,
    reference: float,
    valid: bool = True,
):
    """
    Return a log-axis-safe plotting value and a flag.

    - If the numerical method is invalid/non-converged, return NaN.
    - If raw_error > 0, preserve it exactly.
    - If raw_error == 0, replace it by one ULP of the reference.
    """
    if not valid:
        return (
            math.nan,
            False,
            math.nan,
        )

    if not math.isfinite(
        raw_error
    ):
        return (
            math.nan,
            False,
            math.nan,
        )

    floor = one_ulp_floor(
        reference
    )

    if raw_error > 0.0:
        return (
            raw_error,
            False,
            floor,
        )

    return (
        floor,
        True,
        floor,
    )


def positive_log_plot_value(
    raw_value: float,
    *,
    floor: float,
    valid: bool = True,
):
    """Return a positive plotting surrogate for a nonnegative diagnostic.

    Exact zeros are replaced only for log-axis visualization; the original
    raw value is always retained separately.
    """
    if not valid or not math.isfinite(raw_value):
        return math.nan, False
    if raw_value > 0.0:
        return raw_value, False
    return float(floor), True


# =============================================================================
# Main simulation
# =============================================================================

def fig4a_simulation(
    config: NumericalConfig,
    *,
    r_values=None,
):
    """
    Run the dedicated Fig. 4(a) simulation.

    Returns
    -------
    raw_rows, summary_rows
    """
    if r_values is None:
        r_values = make_r_grid()

    shapes = (
        "anisotropic_8_4_1",
        "equal",
    )

    raw_rows = []

    # -------------------------------------------------------------------------
    # Long-format raw table
    # -------------------------------------------------------------------------
    for shape in shapes:
        direction = shape_direction(
            shape
        )

        for point_index, r in enumerate(
            r_values,
            start=1,
        ):
            abc = (
                float(r)
                * direction
            )

            a, b, c = map(
                float,
                abc,
            )

            # -----------------------------
            # Raw F_A
            # -----------------------------
            t_fa, fa_result = (
                time_once(
                    j4_lauricella_with_info,
                    a,
                    b,
                    c,
                    config=config,
                )
            )

            J_fa, fa_info = (
                fa_result
            )

            fa_finite = (
                math.isfinite(J_fa)
            )

            fa_converged = bool(
                fa_info["converged"]
            )

            # -----------------------------
            # Direct 3D quadrature
            # -----------------------------
            t_3d, J_3d = time_once(
                J4_direct_triple,
                a,
                b,
                c,
                config=config,
            )

            # -----------------------------
            # Reduced 1D expression
            # -----------------------------
            t_1d, J_1d = time_once(
                J4_reduced_1D,
                a,
                b,
                c,
                config=config,
            )

            # -----------------------------
            # Raw discrepancies
            # -----------------------------
            err_fa_3d = (
                abs(
                    J_fa
                    - J_3d
                )
                if fa_finite
                else math.nan
            )

            err_1d_3d = abs(
                J_1d
                - J_3d
            )

            err_fa_1d = (
                abs(
                    J_fa
                    - J_1d
                )
                if fa_finite
                else math.nan
            )

            # -----------------------------
            # Relative-error diagnostics
            # -----------------------------
            rel_err_fa_3d = (
                relative_error_safe(
                    J_fa,
                    J_3d,
                )
                if fa_finite
                else math.nan
            )

            rel_err_1d_3d = (
                relative_error_safe(
                    J_1d,
                    J_3d,
                )
            )

            # -----------------------------
            # ULP-normalized diagnostics
            # -----------------------------
            ulp_fa_3d = (
                error_in_ulp(
                    J_fa,
                    J_3d,
                )
                if fa_finite
                else math.nan
            )

            ulp_1d_3d = (
                error_in_ulp(
                    J_1d,
                    J_3d,
                )
            )

            # Supplementary log-axis-safe relative/ULP diagnostics.
            # Raw values remain unchanged; only exact zeros receive a plotting
            # floor and are explicitly flagged.
            rel_floor = (
                one_ulp_floor(J_3d) / abs(J_3d)
                if J_3d != 0.0
                else np.finfo(float).tiny
            )
            rel_fa_plot, rel_fa_zero = positive_log_plot_value(
                rel_err_fa_3d, floor=rel_floor,
                valid=(fa_finite and fa_converged),
            )
            rel_1d_plot, rel_1d_zero = positive_log_plot_value(
                rel_err_1d_3d, floor=rel_floor,
                valid=math.isfinite(J_1d),
            )
            ulp_fa_plot, ulp_fa_zero = positive_log_plot_value(
                ulp_fa_3d, floor=1.0,
                valid=(fa_finite and fa_converged),
            )
            ulp_1d_plot, ulp_1d_zero = positive_log_plot_value(
                ulp_1d_3d, floor=1.0,
                valid=math.isfinite(J_1d),
            )

            # Plot-ready error values.
            # Non-converged F_A is NOT connected as an ordinary curve point.
            (
                err_fa_plot,
                fa_zero_replaced,
                fa_plot_floor,
            ) = log_plot_value(
                err_fa_3d,
                reference=J_3d,
                valid=(
                    fa_finite
                    and fa_converged
                ),
            )

            (
                err_1d_plot,
                one_d_zero_replaced,
                one_d_plot_floor,
            ) = log_plot_value(
                err_1d_3d,
                reference=J_3d,
                valid=math.isfinite(
                    J_1d
                ),
            )

            raw_rows.append(
                {
                    "point_index": point_index,

                    "shape": shape,
                    "r": float(r),
                    "one_minus_r": (
                        1.0 - float(r)
                    ),
                    "log10_one_minus_r": (
                        math.log10(
                            1.0 - float(r)
                        )
                    ),

                    "a": a,
                    "b": b,
                    "c": c,
                    "a2_b2_c2": (
                        a * a
                        + b * b
                        + c * c
                    ),

                    "FA": J_fa,
                    "direct_3D": J_3d,
                    "reduced_1D": J_1d,

                    "err_FA_vs_3D_raw": (
                        err_fa_3d
                    ),
                    "err_1D_vs_3D_raw": (
                        err_1d_3d
                    ),
                    "err_FA_vs_1D_raw": (
                        err_fa_1d
                    ),

                    # Requested Fig. 4(a) diagnostics
                    "rel_err_FA_vs_3D": (
                        rel_err_fa_3d
                    ),
                    "rel_err_1D_vs_3D": (
                        rel_err_1d_3d
                    ),
                    "ULP_FA_vs_3D": (
                        ulp_fa_3d
                    ),
                    "ULP_1D_vs_3D": (
                        ulp_1d_3d
                    ),

                    # Supplementary log-safe plotting values.
                    "rel_err_FA_vs_3D_plot": rel_fa_plot,
                    "rel_err_1D_vs_3D_plot": rel_1d_plot,
                    "ULP_FA_vs_3D_plot": ulp_fa_plot,
                    "ULP_1D_vs_3D_plot": ulp_1d_plot,
                    "rel_FA_zero_replaced_for_log_plot": rel_fa_zero,
                    "rel_1D_zero_replaced_for_log_plot": rel_1d_zero,
                    "ULP_FA_zero_replaced_for_log_plot": ulp_fa_zero,
                    "ULP_1D_zero_replaced_for_log_plot": ulp_1d_zero,

                    "err_FA_vs_3D_plot": (
                        err_fa_plot
                    ),
                    "err_1D_vs_3D_plot": (
                        err_1d_plot
                    ),

                    "FA_zero_replaced_for_log_plot": (
                        fa_zero_replaced
                    ),
                    "1D_zero_replaced_for_log_plot": (
                        one_d_zero_replaced
                    ),

                    "FA_plot_floor_1ULP": (
                        fa_plot_floor
                    ),
                    "1D_plot_floor_1ULP": (
                        one_d_plot_floor
                    ),

                    "FA_order": int(
                        fa_info["order"]
                    ),
                    "FA_converged": (
                        fa_converged
                    ),
                    "FA_finite": (
                        fa_finite
                    ),
                    "FA_sum_x": float(
                        fa_info["sum_x"]
                    ),

                    "time_FA_s_single": (
                        t_fa
                    ),
                    "time_3D_s_single": (
                        t_3d
                    ),
                    "time_1D_s_single": (
                        t_1d
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Wide-format plot-ready summary table
    # -------------------------------------------------------------------------
    by_key = {
        (
            row["shape"],
            row["r"],
        ): row
        for row in raw_rows
    }

    summary_rows = []

    for r in r_values:
        r = float(r)

        aniso = by_key[
            (
                "anisotropic_8_4_1",
                r,
            )
        ]

        equal = by_key[
            (
                "equal",
                r,
            )
        ]

        summary_rows.append(
            {
                "r": r,
                "one_minus_r": (
                    1.0 - r
                ),
                "log10_one_minus_r": (
                    math.log10(
                        1.0 - r
                    )
                ),
                "FA_max_order_allowed": FA_MAX_ORDER,

                # =========================================================
                # Left panel: anisotropic 8:4:1
                # =========================================================
                "anis_err_FA_vs_3D_raw": (
                    aniso[
                        "err_FA_vs_3D_raw"
                    ]
                ),
                "anis_err_1D_vs_3D_raw": (
                    aniso[
                        "err_1D_vs_3D_raw"
                    ]
                ),

                "anis_rel_err_FA_vs_3D": (
                    aniso[
                        "rel_err_FA_vs_3D"
                    ]
                ),
                "anis_rel_err_1D_vs_3D": (
                    aniso[
                        "rel_err_1D_vs_3D"
                    ]
                ),
                "anis_ULP_FA_vs_3D": (
                    aniso[
                        "ULP_FA_vs_3D"
                    ]
                ),
                "anis_ULP_1D_vs_3D": (
                    aniso[
                        "ULP_1D_vs_3D"
                    ]
                ),
                "anis_rel_err_FA_vs_3D_plot": aniso["rel_err_FA_vs_3D_plot"],
                "anis_rel_err_1D_vs_3D_plot": aniso["rel_err_1D_vs_3D_plot"],
                "anis_ULP_FA_vs_3D_plot": aniso["ULP_FA_vs_3D_plot"],
                "anis_ULP_1D_vs_3D_plot": aniso["ULP_1D_vs_3D_plot"],

                "anis_err_FA_vs_3D_plot": (
                    aniso[
                        "err_FA_vs_3D_plot"
                    ]
                ),
                "anis_err_1D_vs_3D_plot": (
                    aniso[
                        "err_1D_vs_3D_plot"
                    ]
                ),

                "anis_FA_converged": (
                    aniso[
                        "FA_converged"
                    ]
                ),
                "anis_FA_order": (
                    aniso[
                        "FA_order"
                    ]
                ),
                "anis_FA_high_order": bool(aniso["FA_order"] > 128),

                "anis_FA_zero_replaced": (
                    aniso[
                        "FA_zero_replaced_for_log_plot"
                    ]
                ),
                "anis_1D_zero_replaced": (
                    aniso[
                        "1D_zero_replaced_for_log_plot"
                    ]
                ),

                # =========================================================
                # Right panel: equal
                # =========================================================
                "equal_err_FA_vs_3D_raw": (
                    equal[
                        "err_FA_vs_3D_raw"
                    ]
                ),
                "equal_err_1D_vs_3D_raw": (
                    equal[
                        "err_1D_vs_3D_raw"
                    ]
                ),

                "equal_rel_err_FA_vs_3D": (
                    equal[
                        "rel_err_FA_vs_3D"
                    ]
                ),
                "equal_rel_err_1D_vs_3D": (
                    equal[
                        "rel_err_1D_vs_3D"
                    ]
                ),
                "equal_ULP_FA_vs_3D": (
                    equal[
                        "ULP_FA_vs_3D"
                    ]
                ),
                "equal_ULP_1D_vs_3D": (
                    equal[
                        "ULP_1D_vs_3D"
                    ]
                ),
                "equal_rel_err_FA_vs_3D_plot": equal["rel_err_FA_vs_3D_plot"],
                "equal_rel_err_1D_vs_3D_plot": equal["rel_err_1D_vs_3D_plot"],
                "equal_ULP_FA_vs_3D_plot": equal["ULP_FA_vs_3D_plot"],
                "equal_ULP_1D_vs_3D_plot": equal["ULP_1D_vs_3D_plot"],

                "equal_err_FA_vs_3D_plot": (
                    equal[
                        "err_FA_vs_3D_plot"
                    ]
                ),
                "equal_err_1D_vs_3D_plot": (
                    equal[
                        "err_1D_vs_3D_plot"
                    ]
                ),

                "equal_FA_converged": (
                    equal[
                        "FA_converged"
                    ]
                ),
                "equal_FA_order": (
                    equal[
                        "FA_order"
                    ]
                ),
                "equal_FA_high_order": bool(equal["FA_order"] > 128),

                "equal_FA_zero_replaced": (
                    equal[
                        "FA_zero_replaced_for_log_plot"
                    ]
                ),
                "equal_1D_zero_replaced": (
                    equal[
                        "1D_zero_replaced_for_log_plot"
                    ]
                ),
            }
        )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        raw_rows,
        RESULTS / "fig4a_raw.csv",
    )

    write_csv(
        summary_rows,
        RESULTS / "fig4a_summary.csv",
    )

    return raw_rows, summary_rows


# =============================================================================
# IDE main
# =============================================================================

def main():
    """
    Run this file directly in the IDE.

    All figures use the common ceiling FA_MAX_ORDER = 16384.
    Adaptive doubling still stops as soon as the convergence criterion is met,
    so ordinary cases do not pay the cost of the full ceiling.
    """
    config = NumericalConfig(
        fa_rtol=FA_RTOL,
        fa_atol=FA_ATOL,
        fa_initial_order=(
            FA_INITIAL_ORDER
        ),
        fa_max_order=(
            FA_MAX_ORDER
        ),

        quad_epsabs=(
            QUAD_EPSABS
        ),
        quad_epsrel=(
            QUAD_EPSREL
        ),
        quad_limit=(
            QUAD_LIMIT
        ),

        # Not used in Fig. 4(a), but retained for project compatibility.
        aono_fourier_terms=512,

        timing_repeats=1,

        qmc_power=18,
        qmc_scrambles=16,

        seed=20261013,
    )

    r_values = make_r_grid()

    raw_rows, summary_rows = (
        fig4a_simulation(
            config,
            r_values=r_values,
        )
    )

    n_total = len(
        raw_rows
    )

    n_fa_failed = sum(
        not row[
            "FA_converged"
        ]
        for row in raw_rows
    )

    n_zero_fa = sum(
        row[
            "FA_zero_replaced_for_log_plot"
        ]
        for row in raw_rows
    )

    n_zero_1d = sum(
        row[
            "1D_zero_replaced_for_log_plot"
        ]
        for row in raw_rows
    )

    print(
        "Fig. 4(a) simulation finished."
    )
    print(
        f"r points per shape: {len(r_values)}"
    )
    print(
        f"total raw rows: {n_total}"
    )
    print(
        f"non-converged F_A rows: {n_fa_failed}"
    )
    print(
        "exact-zero errors replaced by 1 ULP "
        f"for log plotting: FA={n_zero_fa}, 1D={n_zero_1d}"
    )
    print(
        f"raw CSV: {RESULTS / 'fig4a_raw.csv'}"
    )
    print(
        f"summary CSV: {RESULTS / 'fig4a_summary.csv'}"
    )


if __name__ == "__main__":
    main()
