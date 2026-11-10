from __future__ import annotations

"""
Fig. 6(a) dedicated simulation
==============================

Scientific purpose
------------------
Quantify the computational cost of the general subset/Lauricella representation
as a function of the effective support

    s = #{i : beta_i != 0},

while separating this from the ambient dimension N.

This is the original "Part A: support scaling" experiment of the numerical
comparison plan.  Fig. 6(b) is reserved for the separate parity-profile test.

Why the raw subset/Lauricella evaluator is used here
----------------------------------------------------
Fig. 6(a) is intended to expose the combinatorial structure of the general
subset representation itself.  For a vector with only s active coordinates,
all subset terms containing an inactive coordinate vanish, so the number of
surviving subset contributions is

    2^s,

rather than 2^N.

Using the stable Hybrid evaluator here would mix this structural effect with
special support-specific routes (s=1 incomplete beta, s=2 1D probability
quadrature, N=4 1D reduction).  Therefore this figure deliberately evaluates
the general subset/Lauricella representation directly through

    unit_ball_orthant_lauricella(...).

The Hybrid/Aono comparison belongs to the later comparison figure.

Geometry
--------
For each (N, r, s), use the positive equal-active sparse vector

    beta_i = r / sqrt(s),    i = 1,...,s,
    beta_i = 0,              i > s,

so that

    ||beta||_2 = r.

Parameter grid
--------------
Ambient dimensions:

    N = 4, 5, 6, 7, 8, 9, 10, 11, 12.

Effective supports:

    s = 1, 2, ..., N, for every N.

Normalized displacement magnitudes:

    r = 0.50, 0.90.

The r=0.50 series is the cleanest structural scaling view because it is well
away from the defining-series convergence boundary.  The r=0.90 series is
retained as a higher-displacement comparison.

Main plotting recommendation
----------------------------
For the cleanest Fig. 6(a):

    filter: r = 0.50
    x:      effective support, s
    y:      evaluation time
    group:  ambient dimension, N
    y-scale: logarithmic

The r=0.90 rows are the supplementary robustness panel (same axes/grouping).
Use subset_terms=2^s as a structural reference, not a fit to wall time.

Interpretation
--------------
The structural subset count should satisfy exactly

    subset_terms = 2^s.

The measured wall time need not follow a perfect 2^s law because the cost of an
individual subset term depends on whether its Lauricella series terminates and,
for nonterminating terms, on the required truncation order.  Fig. 6(b) isolates
that parity-dependent structure separately.

Output
------
results/
    fig6a_raw.csv
    fig6a_plot.csv
    fig6a_summary.csv
"""

import math
from pathlib import Path

import numpy as np

from config import NumericalConfig
from lauricella import unit_ball_orthant_lauricella
from utils import (
    count_fa_term_types,
    timed_warm,
    write_csv,
)


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

TIMING_REPEATS = 11
TIMING_WARMUP = 2

# Adaptive batching makes sub-millisecond evaluations much less sensitive to
# timer/scheduler noise.
MIN_BATCH_TIME = 0.05


# =============================================================================
# Parameter grid
# =============================================================================

# Dense ambient-dimension coverage.
DIMENSIONS = tuple(range(4, 13))

R_VALUES = (
    0.50,
    0.90,
)


def support_grid(
    N: int,
):
    """Use every admissible effective support, s = 1,...,N."""
    return tuple(range(1, N + 1))


# =============================================================================
# Geometry
# =============================================================================

def make_sparse_equal_beta(
    N: int,
    r: float,
    support: int,
) -> np.ndarray:
    """
    Positive equal-active vector with exactly ``support`` nonzero components.

    beta_i = r/sqrt(s), i=1,...,s,
    beta_i = 0,         i>s,

    hence ||beta||_2 = r.
    """
    if not (
        1 <= support <= N
    ):
        raise ValueError(
            "Need 1 <= support <= N."
        )

    if not (
        0.0 <= r < 1.0
    ):
        raise ValueError(
            "Need 0 <= r < 1."
        )

    beta = np.zeros(
        N,
        dtype=float,
    )

    beta[:support] = (
        float(r)
        / math.sqrt(
            support
        )
    )

    return beta


# =============================================================================
# Single case
# =============================================================================

def evaluate_case(
    *,
    N: int,
    r: float,
    support: int,
    config: NumericalConfig,
):
    beta = make_sparse_equal_beta(
        N,
        r,
        support,
    )

    beta_norm = float(
        np.linalg.norm(
            beta
        )
    )

    counts = (
        count_fa_term_types(
            N,
            support,
        )
    )

    t_eval, result = (
        timed_warm(
            unit_ball_orthant_lauricella,
            beta,
            config=config,
            return_info=True,
            repeats=TIMING_REPEATS,
            warmup=TIMING_WARMUP,
            min_batch_time=(
                MIN_BATCH_TIME
            ),
        )
    )

    value, info = result

    value = float(
        value
    )

    subset_terms = int(
        info[
            "subset_terms_evaluated"
        ]
    )

    subset_terms_expected = int(
        counts[
            "subset_terms_expected"
        ]
    )

    FA_order = int(
        info[
            "max_FA_order"
        ]
    )

    FA_converged = bool(
        info[
            "all_FA_converged"
        ]
    )

    cancellation = float(
        info[
            "cancellation_indicator"
        ]
    )

    structural_check = bool(
        subset_terms
        == subset_terms_expected
        == 2**support
    )

    time_per_subset = (
        float(t_eval)
        / subset_terms
        if subset_terms > 0
        else math.nan
    )

    return {
        "N": N,
        "r": float(r),
        "support": support,

        "beta_norm": beta_norm,
        "beta": beta.tolist(),

        "value": value,

        "time_s": (
            float(t_eval)
        ),
        "time_ms": (
            1e3
            * float(t_eval)
        ),
        "time_per_subset_s": (
            time_per_subset
        ),

        "subset_terms": (
            subset_terms
        ),
        "subset_terms_expected": (
            subset_terms_expected
        ),
        "structural_check": (
            structural_check
        ),

        "terminating_FA_terms": (
            int(
                counts[
                    "terminating_FA_terms"
                ]
            )
        ),
        "nonterminating_FA_terms": (
            int(
                counts[
                    "nonterminating_FA_terms"
                ]
            )
        ),
        "FA_terms_total": (
            int(
                counts[
                    "FA_terms_total"
                ]
            )
        ),
        "terminating_fraction": (
            float(
                counts[
                    "terminating_fraction"
                ]
            )
            if math.isfinite(
                counts[
                    "terminating_fraction"
                ]
            )
            else math.nan
        ),

        "FA_order": (
            FA_order
        ),
        "FA_converged": (
            FA_converged
        ),

        "cancellation": (
            cancellation
        ),
    }


# =============================================================================
# Main simulation
# =============================================================================

def run_fig6a():
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

    for r in R_VALUES:
        for N in DIMENSIONS:
            for support in support_grid(
                N
            ):
                row = evaluate_case(
                    N=N,
                    r=r,
                    support=support,
                    config=config,
                )

                rows.append(
                    row
                )

                status = (
                    "OK"
                    if (
                        row[
                            "FA_converged"
                        ]
                        and row[
                            "structural_check"
                        ]
                    )
                    else "CHECK"
                )

                print(
                    f"N={N:2d}, "
                    f"r={r:.2f}, "
                    f"s={support:2d}, "
                    f"2^s={row['subset_terms']:5d}, "
                    f"K={row['FA_order']:4d}, "
                    f"t={row['time_s']:.6e} s, "
                    f"{status}"
                )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        rows,
        RESULTS
        / "fig6a_raw.csv",
    )

    plot_rows = (
        build_plot_table(
            rows
        )
    )

    write_csv(
        plot_rows,
        RESULTS
        / "fig6a_plot.csv",
    )

    summary = (
        build_summary(
            rows
        )
    )

    write_csv(
        summary,
        RESULTS
        / "fig6a_summary.csv",
    )

    return (
        rows,
        summary,
    )


# =============================================================================
# Plot-ready table
# =============================================================================

def build_plot_table(
    rows,
):
    """
    Long-format plot table.

    Recommended main Fig. 6(a):
        filter r == 0.50
        x = support
        y = time_s
        grouping = N

    The same table also contains r=0.90 for optional comparison.
    """
    out = []

    for row in rows:
        reliable = bool(
            row[
                "FA_converged"
            ]
            and row[
                "structural_check"
            ]
            and math.isfinite(
                row[
                    "time_s"
                ]
            )
        )

        is_dense = bool(
            row[
                "support"
            ]
            == row[
                "N"
            ]
        )

        out.append(
            {
                "N": (
                    row[
                        "N"
                    ]
                ),

                "r": (
                    row[
                        "r"
                    ]
                ),

                "support": (
                    row[
                        "support"
                    ]
                ),

                "plot_role": "main_r05" if row["r"] == 0.50 else "supplementary_r09",
                "structural_2pow_s": 2 ** row["support"],
                "log2_structural_terms": row["support"],
                "time_over_2pow_s": row["time_s"] / (2 ** row["support"]),

                "is_dense": (
                    is_dense
                ),

                "time_s": (
                    row[
                        "time_s"
                    ]
                ),

                "time_ms": (
                    row[
                        "time_ms"
                    ]
                ),

                "time_per_subset_s": (
                    row[
                        "time_per_subset_s"
                    ]
                ),

                "subset_terms": (
                    row[
                        "subset_terms"
                    ]
                ),

                "subset_terms_expected": (
                    row[
                        "subset_terms_expected"
                    ]
                ),

                "terminating_FA_terms": (
                    row[
                        "terminating_FA_terms"
                    ]
                ),

                "nonterminating_FA_terms": (
                    row[
                        "nonterminating_FA_terms"
                    ]
                ),

                "terminating_fraction": (
                    row[
                        "terminating_fraction"
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

                "reliable": (
                    reliable
                ),
            }
        )

    return out


# =============================================================================
# Summary
# =============================================================================

def build_summary(
    rows,
):
    """
    One summary row per (N, r).

    Useful diagnostics:
      - all structural 2^s checks passed,
      - all F_A evaluations converged,
      - runtime span across the tested support range,
      - dense/s=1 runtime ratio where both are available.
    """
    summary = []

    for r in R_VALUES:
        for N in DIMENSIONS:
            group = [
                row
                for row in rows
                if (
                    row[
                        "N"
                    ]
                    == N
                    and row[
                        "r"
                    ]
                    == r
                )
            ]

            group = sorted(
                group,
                key=lambda x: x[
                    "support"
                ],
            )

            valid = [
                row
                for row in group
                if (
                    row[
                        "FA_converged"
                    ]
                    and row[
                        "structural_check"
                    ]
                    and math.isfinite(
                        row[
                            "time_s"
                        ]
                    )
                )
            ]

            s1_rows = [
                row
                for row in valid
                if row[
                    "support"
                ]
                == 1
            ]

            dense_rows = [
                row
                for row in valid
                if row[
                    "support"
                ]
                == N
            ]

            if (
                s1_rows
                and dense_rows
                and s1_rows[0][
                    "time_s"
                ]
                > 0.0
            ):
                dense_over_s1 = (
                    dense_rows[0][
                        "time_s"
                    ]
                    / s1_rows[0][
                        "time_s"
                    ]
                )
            else:
                dense_over_s1 = (
                    math.nan
                )

            times = [
                row[
                    "time_s"
                ]
                for row in valid
            ]
            times_per_subset = [
                row["time_per_subset_s"]
                for row in valid
                if math.isfinite(row["time_per_subset_s"])
            ]

            summary.append(
                {
                    "N": N,
                    "r": r,

                    "n_supports": (
                        len(group)
                    ),

                    "all_structural_checks": (
                        all(
                            row[
                                "structural_check"
                            ]
                            for row in group
                        )
                    ),

                    "all_FA_converged": (
                        all(
                            row[
                                "FA_converged"
                            ]
                            for row in group
                        )
                    ),
                    "FA_max_order_allowed": FA_MAX_ORDER,
                    "n_order_gt_128": sum(
                        row["FA_order"] > 128
                        for row in valid
                    ),
                    "fraction_order_gt_128": (
                        sum(row["FA_order"] > 128 for row in valid) / len(valid)
                        if valid else math.nan
                    ),

                    "support_min": (
                        min(
                            row[
                                "support"
                            ]
                            for row in group
                        )
                        if group
                        else math.nan
                    ),

                    "support_max": (
                        max(
                            row[
                                "support"
                            ]
                            for row in group
                        )
                        if group
                        else math.nan
                    ),

                    "time_min_s": (
                        min(times)
                        if times
                        else math.nan
                    ),

                    "time_max_s": (
                        max(times)
                        if times
                        else math.nan
                    ),

                    "median_time_per_subset_s": (
                        float(np.median(times_per_subset))
                        if times_per_subset else math.nan
                    ),
                    "min_time_per_subset_s": (
                        min(times_per_subset) if times_per_subset else math.nan
                    ),
                    "max_time_per_subset_s": (
                        max(times_per_subset) if times_per_subset else math.nan
                    ),

                    "dense_over_s1_time_ratio": (
                        dense_over_s1
                    ),

                    "max_FA_order": (
                        max(
                            row[
                                "FA_order"
                            ]
                            for row in valid
                        )
                        if valid
                        else math.nan
                    ),
                }
            )

    return summary


# =============================================================================
# IDE main
# =============================================================================

def main():
    rows, summary = (
        run_fig6a()
    )

    n_nonconverged = sum(
        not row[
            "FA_converged"
        ]
        for row in rows
    )

    n_structure_fail = sum(
        not row[
            "structural_check"
        ]
        for row in rows
    )

    print()
    print(
        "Fig. 6(a) simulation finished."
    )

    print(
        f"Total cases: {len(rows)}"
    )

    print(
        f"Nonconverged cases: {n_nonconverged}"
    )

    print(
        f"Structural 2^s check failures: {n_structure_fail}"
    )

    print(
        f"Raw CSV: {RESULTS / 'fig6a_raw.csv'}"
    )

    print(
        f"Plot CSV: {RESULTS / 'fig6a_plot.csv'}"
    )

    print(
        f"Summary CSV: {RESULTS / 'fig6a_summary.csv'}"
    )

    print()
    print(
        "Recommended Fig. 6(a):"
    )

    print(
        "  primary filter: r == 0.50"
    )

    print(
        "  x = effective support, s"
    )

    print(
        "  y = evaluation time (log scale)"
    )

    print(
        "  grouping = ambient dimension, N"
    )

    print(
        "  retain r == 0.90 for optional comparison"
    )

    del summary


if __name__ == "__main__":
    main()
