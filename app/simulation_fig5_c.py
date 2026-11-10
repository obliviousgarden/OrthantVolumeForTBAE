from __future__ import annotations

"""
Fig. 5(c) dedicated simulation
==============================

Scientific purpose
------------------
Quantify numerical cancellation in the complete subset/Lauricella expression
for mixed-sign beta as the geometric boundary

    r = ||beta||_2 -> 1^-

is approached.

Fig. 5(b) isolates the intrinsic convergence cost of the single hardest
nonterminating Lauricella F_A term.  Fig. 5(c) addresses a different numerical
issue: cancellation among the signed subset contributions in the complete
orthant-volume formula.

For a complete subset expansion

    U_N(beta) = sum_q T_q,

define the cancellation indicator

    C_canc = (sum_q |T_q|) / |sum_q T_q|.

C_canc = 1 means no cancellation.  Larger C_canc means that individually
larger signed contributions cancel to produce a smaller final value.

For plotting, define

    D_canc = max(0, log10(C_canc)),

which estimates the number of decimal digits exposed to cancellation.

Important:
D_canc is NOT asserted to be the number of digits actually lost from the final
answer.  It is a conditioning / cancellation-severity indicator.  Actual
accuracy is checked independently for N=3 and N=4.

Main plotting variables
-----------------------
x:
    r = ||beta||_2

y:
    D_canc = max(0, log10(C_canc))

with separate curves for

    N = 3, 4, 5, 6.

Trajectory
----------
Only one deterministic mixed-sign direction is used:

    w = (1, -1, 1, -1, ...) / sqrt(N),

with

    beta = r w,

so that

    ||beta||_2 = r.

This deliberately removes direction-to-direction variability and isolates the
growth of mixed-sign cancellation as the boundary is approached.

Independent accuracy diagnostics
--------------------------------
For N=3:
    elementary closed form.

For N=4:
    Appendix-C one-dimensional reduction.

For N=5 and N=6:
    no independent low-dimensional reference is used in this figure-specific
    simulation; these remain cancellation / convergence diagnostics.

The main Fig. 5(c) plot should use D_canc only for evaluations in which every
nonterminating F_A term converged.  Any nonconverged point is retained in the
raw/diagnostic output but should not be interpreted as a reliable cancellation
measurement.

Output
------
results/
    fig5c_raw.csv
    fig5c_plot.csv
    fig5c_diagnostics.csv
"""

import math
import time
from pathlib import Path

import numpy as np

from config import NumericalConfig
from elementary import unit_ball_orthant_N3
from lauricella import unit_ball_orthant_lauricella
from n4 import unit_ball_orthant_N4_1D
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

DIMENSIONS = (
    3,
    4,
    5,
    6,
)


# =============================================================================
# Sampling grid
# =============================================================================

def make_r_grid() -> np.ndarray:
    """
    High-density boundary-aware r-grid.

    0.20 -- 0.80 : step 0.025
    0.81 -- 0.95 : step 0.01
    0.955 -- 0.990: step 0.005
    0.992 -- 0.999: step 0.001

    The dense high-r sampling resolves discrete changes in K_FA and the rapid
    boundary growth without relying on interpolation.
    """
    low = np.arange(0.20, 0.8000001, 0.025)
    middle = np.arange(0.81, 0.9500001, 0.01)
    high = np.arange(0.955, 0.9900001, 0.005)
    boundary = np.arange(0.992, 0.9990001, 0.001)
    return np.unique(np.round(np.concatenate([low, middle, high, boundary]), 8))


# =============================================================================
# Geometry
# =============================================================================

def mixed_equal_direction(
    N: int,
) -> np.ndarray:
    """
    Deterministic alternating-sign unit vector

        (+, -, +, -, ...)/sqrt(N).

    Hence beta = r*w satisfies ||beta||_2 = r exactly apart from floating-point
    roundoff.
    """
    w = np.ones(
        N,
        dtype=float,
    )

    w[1::2] = -1.0

    return (
        w
        / np.linalg.norm(w)
    )


def direction_label(
    N: int,
) -> str:
    """
    Human-readable sign pattern used in CSV diagnostics.
    """
    signs = [
        "+1" if i % 2 == 0 else "-1"
        for i in range(N)
    ]

    return ":".join(signs)


# =============================================================================
# Scalar diagnostics
# =============================================================================

def relative_error(
    value: float,
    reference: float,
) -> float:
    if (
        not math.isfinite(value)
        or not math.isfinite(reference)
        or reference == 0.0
    ):
        return math.nan

    return (
        abs(value - reference)
        / abs(reference)
    )


def nominal_tolerance(
    reference: float,
) -> float:
    if not math.isfinite(reference):
        return math.nan

    return (
        FA_ATOL
        + FA_RTOL * abs(reference)
    )


def error_over_tolerance(
    value: float,
    reference: float,
) -> float:
    tol = nominal_tolerance(
        reference
    )

    if (
        not math.isfinite(value)
        or not math.isfinite(reference)
        or not math.isfinite(tol)
        or tol <= 0.0
    ):
        return math.nan

    return (
        abs(value - reference)
        / tol
    )


def digits_exposed_to_cancellation(
    cancellation: float,
) -> float:
    """
    Convert the cancellation indicator

        C_canc >= 1

    to the plotted quantity

        D_canc = max(0, log10(C_canc)).

    Infinite/nonfinite cancellation is retained as +inf in the raw output.
    """
    if not math.isfinite(cancellation):
        return math.inf

    if cancellation <= 0.0:
        return math.nan

    return max(
        0.0,
        math.log10(cancellation),
    )


def reference_value(
    N: int,
    beta: np.ndarray,
    config: NumericalConfig,
):
    """
    Independent references used only as accuracy diagnostics.

    Returns
    -------
    (reference_name, reference)
    """
    if N == 3:
        return (
            "N3_elementary",
            float(
                unit_ball_orthant_N3(
                    beta
                )
            ),
        )

    if N == 4:
        return (
            "N4_1D",
            float(
                unit_ball_orthant_N4_1D(
                    beta,
                    config=config,
                )
            ),
        )

    return (
        "none",
        math.nan,
    )


# =============================================================================
# Main simulation
# =============================================================================

def run_fig5c(
    *,
    r_values=None,
):
    if r_values is None:
        r_values = make_r_grid()

    config = NumericalConfig(
        fa_rtol=FA_RTOL,
        fa_atol=FA_ATOL,
        fa_initial_order=(
            FA_INITIAL_ORDER
        ),
        fa_max_order=(
            FA_MAX_ORDER
        ),
    )

    rows = []

    for N in DIMENSIONS:
        w = mixed_equal_direction(
            N
        )

        label = direction_label(
            N
        )

        for r in r_values:
            r = float(r)

            beta = (
                r * w
            )

            beta_norm = float(
                np.linalg.norm(
                    beta
                )
            )

            t0 = time.perf_counter()

            value, info = (
                unit_ball_orthant_lauricella(
                    beta,
                    config=config,
                    return_info=True,
                )
            )

            elapsed = (
                time.perf_counter()
                - t0
            )

            value = float(
                value
            )

            converged = bool(
                info[
                    "all_FA_converged"
                ]
            )

            max_order = int(
                info[
                    "max_FA_order"
                ]
            )

            cancellation = float(
                info[
                    "cancellation_indicator"
                ]
            )

            D_canc = (
                digits_exposed_to_cancellation(
                    cancellation
                )
            )

            (
                reference_name,
                reference,
            ) = reference_value(
                N,
                beta,
                config,
            )

            abs_error = (
                abs(
                    value
                    - reference
                )
                if (
                    math.isfinite(value)
                    and math.isfinite(reference)
                )
                else math.nan
            )

            rel_error_value = (
                relative_error(
                    value,
                    reference,
                )
            )

            requested_tolerance = (
                nominal_tolerance(
                    reference
                )
            )

            error_tol = (
                error_over_tolerance(
                    value,
                    reference,
                )
            )

            cancellation_valid = bool(
                converged
                and math.isfinite(value)
                and math.isfinite(cancellation)
                and cancellation > 0.0
                and math.isfinite(D_canc)
            )

            rows.append(
                {
                    "N": N,
                    "trajectory": (
                        "mixed_equal"
                    ),
                    "direction_label": (
                        label
                    ),

                    "r": r,
                    "beta_norm": (
                        beta_norm
                    ),
                    "beta": (
                        beta.tolist()
                    ),

                    "value": value,
                    "time_s": elapsed,

                    "FA_order": (
                        max_order
                    ),
                    "FA_converged": (
                        converged
                    ),
                    "FA_max_order": (
                        FA_MAX_ORDER
                    ),

                    "cancellation": (
                        cancellation
                    ),
                    "D_canc": (
                        D_canc
                    ),
                    "cancellation_valid": (
                        cancellation_valid
                    ),

                    # Plot-ready separation:
                    # only converged, finite cancellation data are valid for
                    # the main D_canc(r) curve.
                    "D_canc_converged": (
                        D_canc
                        if cancellation_valid
                        else math.nan
                    ),

                    "reference_name": (
                        reference_name
                    ),
                    "reference": (
                        reference
                    ),
                    "abs_error": (
                        abs_error
                    ),
                    "rel_error": (
                        rel_error_value
                    ),
                    "requested_FA_tolerance": (
                        requested_tolerance
                    ),
                    "error_over_tolerance": (
                        error_tol
                    ),
                }
            )

            status = (
                "OK"
                if cancellation_valid
                else "FAIL"
            )

            if math.isfinite(
                D_canc
            ):
                d_text = (
                    f"{D_canc:.4f}"
                )
            else:
                d_text = "inf"

            if math.isfinite(
                cancellation
            ):
                c_text = (
                    f"{cancellation:.6g}"
                )
            else:
                c_text = "inf"

            print(
                f"N={N}, "
                f"r={r:.4f}, "
                f"C={c_text:>10s}, "
                f"D={d_text:>8s}, "
                f"K={max_order:4d}, "
                f"{status}, "
                f"t={elapsed:.3f} s"
            )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        rows,
        RESULTS
        / "fig5c_raw.csv",
    )

    plot_rows = (
        build_plot_table(
            rows
        )
    )

    write_csv(
        plot_rows,
        RESULTS
        / "fig5c_plot.csv",
    )

    diagnostics = (
        build_diagnostics(
            rows
        )
    )

    write_csv(
        diagnostics,
        RESULTS
        / "fig5c_diagnostics.csv",
    )

    return (
        rows,
        diagnostics,
    )


# =============================================================================
# Plot-ready table
# =============================================================================

def build_plot_table(
    rows,
):
    """
    Retain one row per simulated (N, r) point for direct plotting.

    With the default grid this table contains

        4 dimensions x 21 r values = 84 rows.

    Recommended main-figure columns
    --------------------------------
    x:
        r

    y:
        D_canc_converged

    grouping:
        N

    Additional columns are retained for filtering and numerical diagnostics.
    """
    out = []

    for row in rows:
        out.append(
            {
                "N": (
                    row["N"]
                ),
                "trajectory": (
                    row[
                        "trajectory"
                    ]
                ),
                "direction_label": (
                    row[
                        "direction_label"
                    ]
                ),

                "r": (
                    row["r"]
                ),
                "beta_norm": (
                    row[
                        "beta_norm"
                    ]
                ),

                "cancellation": (
                    row[
                        "cancellation"
                    ]
                ),
                "D_canc": (
                    row[
                        "D_canc"
                    ]
                ),
                "D_canc_converged": (
                    row[
                        "D_canc_converged"
                    ]
                ),
                "cancellation_valid": (
                    row[
                        "cancellation_valid"
                    ]
                ),

                "FA_order": (
                    row[
                        "FA_order"
                    ]
                ),
                "FA_converged": (
                    row[
                        "FA_converged"
                    ]
                ),
                "FA_max_order_allowed": FA_MAX_ORDER,
                "FA_high_order": bool(row["FA_order"] > 128),

                "value": (
                    row[
                        "value"
                    ]
                ),
                "time_s": (
                    row[
                        "time_s"
                    ]
                ),

                "reference_name": (
                    row[
                        "reference_name"
                    ]
                ),
                "reference": (
                    row[
                        "reference"
                    ]
                ),
                "abs_error": (
                    row[
                        "abs_error"
                    ]
                ),
                "rel_error": (
                    row[
                        "rel_error"
                    ]
                ),
                "error_over_tolerance": (
                    row[
                        "error_over_tolerance"
                    ]
                ),
            }
        )

    return out


# =============================================================================
# Diagnostics summary
# =============================================================================

def build_diagnostics(
    rows,
):
    """
    One summary row per N.

    The main figure does not need these summary quantities, but they make it
    easy to report:

    - the cancellation range,
    - D_canc at the largest simulated r,
    - whether any F_A evaluation failed,
    - and, for N=3,4, the maximum independently measured numerical error.
    """
    summary = []

    for N in DIMENSIONS:
        group = [
            row
            for row in rows
            if row["N"] == N
        ]

        valid_rows = [
            row
            for row in group
            if row[
                "cancellation_valid"
            ]
        ]

        failed_rows = [
            row
            for row in group
            if not row[
                "cancellation_valid"
            ]
        ]

        finite_D = [
            row["D_canc"]
            for row in valid_rows
            if math.isfinite(
                row["D_canc"]
            )
        ]

        finite_C = [
            row["cancellation"]
            for row in valid_rows
            if math.isfinite(
                row["cancellation"]
            )
        ]

        reference_rows = [
            row
            for row in group
            if math.isfinite(
                row["reference"]
            )
        ]

        finite_rel_errors = [
            row["rel_error"]
            for row in reference_rows
            if math.isfinite(
                row["rel_error"]
            )
        ]

        finite_abs_errors = [
            row["abs_error"]
            for row in reference_rows
            if math.isfinite(row["abs_error"])
        ]

        finite_error_tol = [
            row[
                "error_over_tolerance"
            ]
            for row in reference_rows
            if math.isfinite(
                row[
                    "error_over_tolerance"
                ]
            )
        ]

        if valid_rows:
            last_valid = max(
                valid_rows,
                key=lambda x: x["r"],
            )

            max_r_valid = (
                last_valid["r"]
            )
            D_at_max_r = (
                last_valid["D_canc"]
            )
            C_at_max_r = (
                last_valid[
                    "cancellation"
                ]
            )
        else:
            max_r_valid = math.nan
            D_at_max_r = math.nan
            C_at_max_r = math.nan

        summary.append(
            {
                "N": N,
                "trajectory": (
                    "mixed_equal"
                ),

                "n_total": (
                    len(group)
                ),
                "n_valid": (
                    len(valid_rows)
                ),
                "n_failed": (
                    len(failed_rows)
                ),
                "FA_max_order_allowed": FA_MAX_ORDER,
                "n_order_gt_128": sum(
                    row["FA_order"] > 128
                    for row in valid_rows
                ),
                "fraction_order_gt_128": (
                    sum(row["FA_order"] > 128 for row in valid_rows)
                    / len(valid_rows)
                    if valid_rows else math.nan
                ),

                "min_cancellation": (
                    min(finite_C)
                    if finite_C
                    else math.nan
                ),
                "max_cancellation": (
                    max(finite_C)
                    if finite_C
                    else math.nan
                ),

                "min_D_canc": (
                    min(finite_D)
                    if finite_D
                    else math.nan
                ),
                "max_D_canc": (
                    max(finite_D)
                    if finite_D
                    else math.nan
                ),

                "max_r_valid": (
                    max_r_valid
                ),
                "cancellation_at_max_r": (
                    C_at_max_r
                ),
                "D_canc_at_max_r": (
                    D_at_max_r
                ),

                "max_FA_order": (
                    max(
                        row["FA_order"]
                        for row in group
                    )
                    if group
                    else math.nan
                ),

                "reference_name": (
                    reference_rows[0][
                        "reference_name"
                    ]
                    if reference_rows
                    else "none"
                ),

                "max_abs_error": (
                    max(finite_abs_errors)
                    if finite_abs_errors
                    else math.nan
                ),

                "max_rel_error": (
                    max(
                        finite_rel_errors
                    )
                    if finite_rel_errors
                    else math.nan
                ),

                "max_error_over_tolerance": (
                    max(
                        finite_error_tol
                    )
                    if finite_error_tol
                    else math.nan
                ),
            }
        )

    return summary


# =============================================================================
# IDE main
# =============================================================================

def main():
    rows, diagnostics = (
        run_fig5c()
    )

    n_fail = sum(
        not row[
            "cancellation_valid"
        ]
        for row in rows
    )

    print()
    print(
        "Fig. 5(c) simulation finished."
    )
    print(
        f"Total cases: {len(rows)}"
    )
    print(
        f"Invalid/nonconverged cancellation cases: {n_fail}"
    )
    print(
        f"Raw CSV: {RESULTS / 'fig5c_raw.csv'}"
    )
    print(
        f"Plot CSV: {RESULTS / 'fig5c_plot.csv'}"
    )
    print(
        f"Diagnostics CSV: {RESULTS / 'fig5c_diagnostics.csv'}"
    )

    print()
    print(
        "Recommended Fig. 5(c) plot:"
    )
    print(
        "  x = r = ||beta||_2"
    )
    print(
        "  y = D_canc_converged"
    )
    print(
        "  grouping = N"
    )

    del diagnostics


if __name__ == "__main__":
    main()
