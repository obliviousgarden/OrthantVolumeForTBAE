from __future__ import annotations

"""
Fig. 5(b) dedicated simulation
==============================

Scientific purpose
------------------
Quantify the numerical convergence cost of the *raw defining-series*
Lauricella F_A terms as the most demanding nonterminating subset approaches
its formal convergence boundary.

For a subset I of size m,

    a = -(N-m)/2,

and the F_A term is nonterminating when the residual dimension

    d = N - m

is odd.  For every beta, define the hardest nonterminating subset I_* as the
one maximizing

    rho_NT = sqrt(sum_{i in I_*} beta_i^2).

The formal defining-series boundary is therefore

    rho_NT -> 1^-.

Instead of timing the complete 2^N subset sum, this script evaluates the
single hardest nonterminating F_A term directly.  This isolates the intrinsic
series-convergence problem from cancellation and from the number of subsets.
Cancellation is a separate issue and can be treated in Fig. 5(c), if retained.

Main plotting variables
-----------------------
x:
    rho_NT

y:
    FA_order

with a logarithmic y axis (base 2 is conceptually natural, but base 10 is
also fine).

Converged and failed evaluations are stored separately:

    FA_order_converged
    FA_order_failed

A failed evaluation is now classified explicitly as either

    failure_reason = "max_order"

for a finite series evaluation that did not satisfy the stopping criterion by
FA_MAX_ORDER, or

    failure_reason = "nonfinite"

when the raw defining-series evaluator first produces NaN/Inf.  In the latter
case the routine stops immediately at the first non-finite truncation order so
that numerical breakdown is not mislabeled as slow convergence all the way to
FA_MAX_ORDER.

Convergence diagnostics
-----------------------
At every adaptive doubling step K_prev -> K,

    delta_K_abs = |F_A^(K) - F_A^(K_prev)|,

    delta_K_rel = delta_K_abs / |F_A^(K)|,

and the actual stopping-test ratio is

    convergence_ratio
      = delta_K_abs / (FA_ATOL + FA_RTOL * |F_A^(K)|).

Therefore

    convergence_ratio <= 1

is exactly equivalent to the numerical convergence criterion used here.

To keep the standard three-file output convention, the complete convergence
history is embedded in fig5b_raw.csv as list-valued columns rather than being
written to a separate history CSV.

Trajectories
------------
Two deterministic directions are used:

1. equal:
       w = (1, ..., 1) / sqrt(N)

2. anisotropic:
       w proportional to (2^(N-1), 2^(N-2), ..., 2, 1)

with beta = r w and ||beta||_2 = r.

The anisotropic family pushes the hardest (N-1)-coordinate nonterminating
subset much closer to rho_NT = 1 than the equal family and is therefore the
main near-boundary stress test.

Output
------
results/
    fig5b_raw.csv
    fig5b_summary.csv
    fig5b_plot.csv
"""

import itertools
import math
import time
from pathlib import Path

import numpy as np

from config import NumericalConfig
from utils import write_csv
from lauricella import lauricella_FA_half_parameters


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

TRAJECTORIES = (
    "equal",
    "anisotropic",
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

def direction(
    N: int,
    kind: str,
) -> np.ndarray:
    """
    Unit vector w defining beta = r*w.
    """
    if kind == "equal":
        w = np.ones(
            N,
            dtype=float,
        )

    elif kind == "anisotropic":
        w = np.asarray(
            [
                2.0 ** (
                    N - 1 - i
                )
                for i in range(N)
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            f"Unknown trajectory: {kind}"
        )

    return (
        w
        / np.linalg.norm(w)
    )


def hardest_nonterminating_subset(
    beta,
):
    """
    Find the nonterminating subset with the largest sum(beta_i^2).

    A subset of size m is nonterminating iff

        d = N-m

    is odd, because then

        a = -d/2

    is a negative half-integer rather than a non-positive integer.
    """
    beta = np.asarray(
        beta,
        dtype=float,
    )

    N = beta.size
    best = None

    active = [
        i
        for i, b in enumerate(beta)
        if b != 0.0
    ]

    for m in range(
        1,
        len(active) + 1,
    ):
        d = N - m

        # d odd -> a=-d/2 is a negative half-integer -> nonterminating.
        if d % 2 == 0:
            continue

        for subset in itertools.combinations(
            active,
            m,
        ):
            idx = np.asarray(
                subset,
                dtype=int,
            )

            x = beta[idx] ** 2
            sum_x = float(
                np.sum(x)
            )

            candidate = {
                "subset": tuple(
                    int(i)
                    for i in subset
                ),
                "m": m,
                "residual_dimension": d,
                "a": -d / 2.0,
                "sum_x": sum_x,
                "rho_NT": math.sqrt(
                    sum_x
                ),
                "x": x,
            }

            if (
                best is None
                or sum_x > best["sum_x"]
            ):
                best = candidate

    if best is None:
        raise RuntimeError(
            "No nonterminating subset exists for this beta."
        )

    if not best["sum_x"] < 1.0:
        raise ValueError(
            "Hardest nonterminating subset is outside the "
            "defining-series convergence domain."
        )

    return best


# =============================================================================
# Lauricella defining-series evaluator with embedded convergence history
# =============================================================================

def lauricella_FA_half_parameters_with_history(
    a: float,
    x,
    *,
    rtol: float,
    atol: float,
    initial_order: int,
    max_order: int,
):
    """
    Evaluate the specialized Lauricella F_A defining series while retaining
    every adaptive doubling step.

    The stopping test is

        |F_A^(K) - F_A^(K_prev)|
            <= atol + rtol * |F_A^(K)|.

    Numerical-safety change relative to the previous Fig. 5(b) diagnostic:
    if a newly evaluated truncation value is non-finite, stop immediately and
    classify the case as ``failure_reason='nonfinite'``.  This prevents NaN/Inf
    from propagating through all later doubling orders and makes the reported
    failure order physically interpretable as the first numerical breakdown.
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    m = x.size

    if m == 0:
        value = 1.0
        history = [
            {
                "K_FA": 0,
                "K_prev": math.nan,
                "FA_value_K": value,
                "FA_value_prev": math.nan,
                "delta_K_abs": math.nan,
                "delta_K_rel": math.nan,
                "delta_K": math.nan,
                "convergence_threshold": (
                    atol + rtol * abs(value)
                ),
                "convergence_ratio": math.nan,
                "step_converged": True,
                "is_finite": True,
            }
        ]
        info = {
            "order": 0,
            "converged": True,
            "terminating": True,
            "sum_x": 0.0,
            "failure_reason": "",
            "numerical_failure": False,
            "first_nonfinite_order": math.nan,
            "last_finite_order": 0,
            "last_finite_value": value,
        }
        return value, info, history

    if np.any(x < 0.0):
        raise ValueError(
            "This specialized evaluator expects x_i >= 0."
        )

    sum_x = float(
        np.sum(x)
    )

    if not sum_x < 1.0:
        raise ValueError(
            "Lauricella F_A series requires sum x_i < 1; "
            f"got {sum_x:.16g}."
        )

    nearest_integer = round(a)
    terminating = (
        a <= 0.0
        and abs(a - nearest_integer) < 1e-14
    )
    termination_order = (
        int(-nearest_integer)
        if terminating
        else None
    )

    def evaluate_to_order(Kmax: int) -> float:
        """Evaluate the exact Kmax truncation using the shared robust backend.

        initial_order=max_order=Kmax intentionally forces evaluation of this
        truncation, without invoking the backend's adaptive doubling logic.
        The outer Fig. 5(b) routine retains its own complete step history.
        """
        value, _ = lauricella_FA_half_parameters(
            a, x,
            rtol=rtol, atol=atol,
            initial_order=Kmax, max_order=Kmax,
            return_info=True,
        )
        return float(value)

    if terminating:
        value = evaluate_to_order(
            termination_order
        )
        finite = math.isfinite(value)

        history = [
            {
                "K_FA": termination_order,
                "K_prev": math.nan,
                "FA_value_K": value,
                "FA_value_prev": math.nan,
                "delta_K_abs": 0.0 if finite else math.nan,
                "delta_K_rel": 0.0 if finite else math.nan,
                "delta_K": 0.0 if finite else math.nan,
                "convergence_threshold": (
                    atol + rtol * abs(value)
                    if finite
                    else math.nan
                ),
                "convergence_ratio": 0.0 if finite else math.inf,
                "step_converged": finite,
                "is_finite": finite,
            }
        ]

        info = {
            "order": termination_order,
            "converged": finite,
            "terminating": True,
            "sum_x": sum_x,
            "failure_reason": "" if finite else "nonfinite",
            "numerical_failure": not finite,
            "first_nonfinite_order": (
                math.nan if finite else termination_order
            ),
            "last_finite_order": (
                termination_order if finite else math.nan
            ),
            "last_finite_value": (
                value if finite else math.nan
            ),
        }
        return value, info, history

    K = min(
        initial_order,
        max_order,
    )
    old = evaluate_to_order(K)
    old_finite = math.isfinite(old)

    history = [
        {
            "K_FA": K,
            "K_prev": math.nan,
            "FA_value_K": old,
            "FA_value_prev": math.nan,
            "delta_K_abs": math.nan,
            "delta_K_rel": math.nan,
            "delta_K": math.nan,
            "convergence_threshold": (
                atol + rtol * abs(old)
                if old_finite
                else math.nan
            ),
            "convergence_ratio": (
                math.nan
                if old_finite
                else math.inf
            ),
            "step_converged": False,
            "is_finite": old_finite,
        }
    ]

    if not old_finite:
        info = {
            "order": K,
            "converged": False,
            "terminating": False,
            "sum_x": sum_x,
            "failure_reason": "nonfinite",
            "numerical_failure": True,
            "first_nonfinite_order": K,
            "last_finite_order": math.nan,
            "last_finite_value": math.nan,
        }
        return math.nan, info, history

    last_finite_order = K
    last_finite_value = old

    while K < max_order:
        K2 = min(
            2 * K,
            max_order,
        )
        new = evaluate_to_order(K2)
        new_finite = math.isfinite(new)

        if not new_finite:
            history.append(
                {
                    "K_FA": K2,
                    "K_prev": K,
                    "FA_value_K": new,
                    "FA_value_prev": old,
                    "delta_K_abs": math.nan,
                    "delta_K_rel": math.nan,
                    "delta_K": math.nan,
                    "convergence_threshold": math.nan,
                    "convergence_ratio": math.inf,
                    "step_converged": False,
                    "is_finite": False,
                }
            )

            info = {
                "order": K2,
                "converged": False,
                "terminating": False,
                "sum_x": sum_x,
                "failure_reason": "nonfinite",
                "numerical_failure": True,
                "first_nonfinite_order": K2,
                "last_finite_order": last_finite_order,
                "last_finite_value": last_finite_value,
            }
            return math.nan, info, history

        delta_abs = abs(
            new - old
        )

        if new != 0.0:
            delta_rel = (
                delta_abs
                / abs(new)
            )
        elif delta_abs == 0.0:
            delta_rel = 0.0
        else:
            delta_rel = math.inf

        threshold = (
            atol
            + rtol * abs(new)
        )

        if threshold > 0.0:
            convergence_ratio = (
                delta_abs
                / threshold
            )
        elif delta_abs == 0.0:
            convergence_ratio = 0.0
        else:
            convergence_ratio = math.inf

        step_converged = (
            delta_abs
            <= threshold
        )

        history.append(
            {
                "K_FA": K2,
                "K_prev": K,
                "FA_value_K": new,
                "FA_value_prev": old,
                "delta_K_abs": delta_abs,
                "delta_K_rel": delta_rel,
                "delta_K": delta_rel,
                "convergence_threshold": threshold,
                "convergence_ratio": convergence_ratio,
                "step_converged": step_converged,
                "is_finite": True,
            }
        )

        last_finite_order = K2
        last_finite_value = new

        if step_converged:
            info = {
                "order": K2,
                "converged": True,
                "terminating": False,
                "sum_x": sum_x,
                "failure_reason": "",
                "numerical_failure": False,
                "first_nonfinite_order": math.nan,
                "last_finite_order": last_finite_order,
                "last_finite_value": last_finite_value,
            }
            return new, info, history

        old = new
        K = K2

    info = {
        "order": K,
        "converged": False,
        "terminating": False,
        "sum_x": sum_x,
        "failure_reason": "max_order",
        "numerical_failure": False,
        "first_nonfinite_order": math.nan,
        "last_finite_order": last_finite_order,
        "last_finite_value": last_finite_value,
    }

    return old, info, history


# =============================================================================
# Main simulation
# =============================================================================

def _history_list(
    history,
    key,
):
    """Return one history field as a plain Python list for CSV serialization."""
    return [
        step[key]
        for step in history
    ]


def run_fig5b(
    *,
    r_values=None,
):
    if r_values is None:
        r_values = make_r_grid()

    rows = []

    for N in DIMENSIONS:
        for kind in TRAJECTORIES:
            w = direction(
                N,
                kind,
            )

            direction_label = (
                "equal"
                if kind == "equal"
                else ":".join(
                    str(
                        2 ** (
                            N - 1 - i
                        )
                    )
                    for i in range(N)
                )
            )

            for r in r_values:
                r = float(r)
                beta = r * w

                hard = (
                    hardest_nonterminating_subset(
                        beta
                    )
                )

                t0 = time.perf_counter()

                value, info, history = (
                    lauricella_FA_half_parameters_with_history(
                        hard["a"],
                        hard["x"],
                        rtol=FA_RTOL,
                        atol=FA_ATOL,
                        initial_order=(
                            FA_INITIAL_ORDER
                        ),
                        max_order=(
                            FA_MAX_ORDER
                        ),
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                rho = float(
                    hard["rho_NT"]
                )

                converged = bool(
                    info["converged"]
                )

                order = int(
                    info["order"]
                )

                final_history = history[-1]

                rows.append(
                    {
                        # -----------------------------------------------------
                        # Original raw-column block: keep order/meaning stable.
                        # -----------------------------------------------------
                        "N": N,
                        "trajectory": kind,
                        "direction_label": direction_label,
                        "r": r,
                        "beta": beta.tolist(),
                        "hard_subset": list(
                            hard["subset"]
                        ),
                        "subset_size_m": hard["m"],
                        "residual_dimension_d": (
                            hard[
                                "residual_dimension"
                            ]
                        ),
                        "FA_a": hard["a"],
                        "sum_x": hard["sum_x"],
                        "rho_NT": rho,
                        "one_minus_rho_NT": (
                            1.0 - rho
                        ),
                        "minus_log10_1_minus_rho_NT": (
                            -math.log10(
                                1.0 - rho
                            )
                        ),
                        "FA_value": value,
                        "FA_order": order,
                        "FA_converged": converged,
                        "FA_terminating": bool(
                            info["terminating"]
                        ),
                        "FA_order_converged": (
                            order
                            if converged
                            else math.nan
                        ),
                        "FA_order_failed": (
                            order
                            if not converged
                            else math.nan
                        ),
                        "FA_max_order": FA_MAX_ORDER,
                        "time_s": elapsed,

                        # -----------------------------------------------------
                        # New numerical-status diagnostics.
                        # -----------------------------------------------------
                        "FA_failure_reason": info[
                            "failure_reason"
                        ],
                        "FA_numerical_failure": bool(
                            info[
                                "numerical_failure"
                            ]
                        ),
                        "FA_first_nonfinite_order": (
                            info[
                                "first_nonfinite_order"
                            ]
                        ),
                        "FA_last_finite_order": (
                            info[
                                "last_finite_order"
                            ]
                        ),
                        "FA_last_finite_value": (
                            info[
                                "last_finite_value"
                            ]
                        ),

                        # Final tested-step diagnostics.
                        "delta_K_abs": final_history[
                            "delta_K_abs"
                        ],
                        "delta_K_rel": final_history[
                            "delta_K_rel"
                        ],
                        "delta_K": final_history[
                            "delta_K"
                        ],
                        "convergence_threshold": (
                            final_history[
                                "convergence_threshold"
                            ]
                        ),
                        "convergence_ratio": (
                            final_history[
                                "convergence_ratio"
                            ]
                        ),

                        # -----------------------------------------------------
                        # Complete doubling history embedded in raw.csv.
                        # This replaces the previous fourth history CSV.
                        # -----------------------------------------------------
                        "K_history": _history_list(
                            history,
                            "K_FA",
                        ),
                        "FA_value_history": _history_list(
                            history,
                            "FA_value_K",
                        ),
                        "delta_K_abs_history": _history_list(
                            history,
                            "delta_K_abs",
                        ),
                        "delta_K_rel_history": _history_list(
                            history,
                            "delta_K_rel",
                        ),
                        "convergence_ratio_history": _history_list(
                            history,
                            "convergence_ratio",
                        ),
                        "step_converged_history": _history_list(
                            history,
                            "step_converged",
                        ),
                        "is_finite_history": _history_list(
                            history,
                            "is_finite",
                        ),
                    }
                )

                if converged:
                    status = "OK"
                elif info["numerical_failure"]:
                    status = "NUMFAIL"
                else:
                    status = "MAXORDER"

                print(
                    f"N={N}, "
                    f"{kind:11s}, "
                    f"r={r:.4f}, "
                    f"rho_NT={rho:.6f}, "
                    f"K={order:4d}, "
                    f"{status}, "
                    f"t={elapsed:.3f} s"
                )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Standard three-file output only.
    write_csv(
        rows,
        RESULTS
        / "fig5b_raw.csv",
    )

    summary = build_summary(
        rows
    )
    write_csv(
        summary,
        RESULTS
        / "fig5b_summary.csv",
    )

    plot_rows = build_plot_table(
        rows
    )
    write_csv(
        plot_rows,
        RESULTS
        / "fig5b_plot.csv",
    )

    return (
        rows,
        summary,
        plot_rows,
    )


# =============================================================================
# Plot-ready table
# =============================================================================

def build_plot_table(
    rows,
):
    """
    Retain one row per simulated point for direct plotting.

    The original plot-table column format is intentionally retained so existing
    plotting code remains compatible.
    """
    out = []

    for row in rows:
        out.append(
            {
                "N": row["N"],
                "trajectory": row["trajectory"],
                "direction_label": row[
                    "direction_label"
                ],
                "r": row["r"],
                "rho_NT": row["rho_NT"],
                "one_minus_rho_NT": row[
                    "one_minus_rho_NT"
                ],
                "minus_log10_1_minus_rho_NT": (
                    row[
                        "minus_log10_1_minus_rho_NT"
                    ]
                ),
                "FA_order": row["FA_order"],
                "FA_converged": row[
                    "FA_converged"
                ],
                "FA_order_converged": row[
                    "FA_order_converged"
                ],
                "FA_order_failed": row[
                    "FA_order_failed"
                ],
                "FA_value": row["FA_value"],
                "delta_K_abs": row["delta_K_abs"],
                "delta_K_rel": row["delta_K_rel"],
                "convergence_ratio": row["convergence_ratio"],
                "FA_failure_reason": row["FA_failure_reason"],
                "FA_numerical_failure": row["FA_numerical_failure"],
                "FA_max_order_allowed": FA_MAX_ORDER,
                "time_s": row["time_s"],
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
    Publication/supplement-ready convergence summary.

    In addition to the original convergence envelope, record the order
    distribution, final stopping ratios, timing, and explicit failure counts.
    The complete doubling histories remain embedded in fig5b_raw.csv.
    """
    summary = []

    for N in DIMENSIONS:
        for kind in TRAJECTORIES:
            group = [
                row
                for row in rows
                if (
                    row["N"] == N
                    and row[
                        "trajectory"
                    ] == kind
                )
            ]

            converged_rows = [
                row
                for row in group
                if row[
                    "FA_converged"
                ]
            ]

            failed_rows = [
                row
                for row in group
                if not row[
                    "FA_converged"
                ]
            ]

            if converged_rows:
                max_order_converged = max(
                    row["FA_order"]
                    for row in converged_rows
                )

                max_rho_converged = max(
                    row["rho_NT"]
                    for row in converged_rows
                )
            else:
                max_order_converged = math.nan
                max_rho_converged = math.nan

            if failed_rows:
                min_rho_failed = min(
                    row["rho_NT"]
                    for row in failed_rows
                )
            else:
                min_rho_failed = math.nan

            orders = [
                row["FA_order"]
                for row in converged_rows
            ]
            times = [
                row["time_s"]
                for row in group
                if math.isfinite(row["time_s"])
            ]
            final_ratios = [
                row["convergence_ratio"]
                for row in converged_rows
                if math.isfinite(row["convergence_ratio"])
            ]
            n_numerical_failure = sum(
                bool(row["FA_numerical_failure"])
                for row in failed_rows
            )
            n_max_order_failure = sum(
                row["FA_failure_reason"] == "max_order"
                for row in failed_rows
            )

            summary.append(
                {
                    "N": N,
                    "trajectory": kind,
                    "direction_label": (
                        group[0][
                            "direction_label"
                        ]
                        if group
                        else ""
                    ),
                    "n_total": len(group),
                    "n_converged": len(
                        converged_rows
                    ),
                    "n_failed": len(
                        failed_rows
                    ),
                    "converged_fraction": (
                        len(converged_rows) / len(group)
                        if group else math.nan
                    ),
                    "n_numerical_failure": n_numerical_failure,
                    "n_max_order_failure": n_max_order_failure,
                    "FA_max_order_allowed": FA_MAX_ORDER,
                    "median_converged_order": (
                        float(np.median(orders)) if orders else math.nan
                    ),
                    "p90_converged_order": (
                        float(np.quantile(orders, 0.90)) if orders else math.nan
                    ),
                    "n_order_gt_128": sum(order > 128 for order in orders),
                    "fraction_order_gt_128": (
                        sum(order > 128 for order in orders) / len(orders)
                        if orders else math.nan
                    ),
                    "median_time_s": (
                        float(np.median(times)) if times else math.nan
                    ),
                    "max_time_s": (max(times) if times else math.nan),
                    "median_final_convergence_ratio": (
                        float(np.median(final_ratios))
                        if final_ratios else math.nan
                    ),
                    "max_final_convergence_ratio": (
                        max(final_ratios) if final_ratios else math.nan
                    ),
                    "max_converged_order": (
                        max_order_converged
                    ),
                    "max_rho_NT_converged": (
                        max_rho_converged
                    ),
                    "min_rho_NT_failed": (
                        min_rho_failed
                    ),
                }
            )

    return summary


# =============================================================================
# IDE main
# =============================================================================

def main():
    # NumericalConfig is instantiated only to keep the same parameter naming
    # convention as the rest of the project.  The direct F_A diagnostic call
    # above uses the constants explicitly.
    _config = NumericalConfig(
        fa_rtol=FA_RTOL,
        fa_atol=FA_ATOL,
        fa_initial_order=(
            FA_INITIAL_ORDER
        ),
        fa_max_order=(
            FA_MAX_ORDER
        ),
    )

    del _config

    rows, summary, plot_rows = run_fig5b()

    n_fail = sum(
        not row["FA_converged"]
        for row in rows
    )
    n_numfail = sum(
        row["FA_numerical_failure"]
        for row in rows
    )
    n_maxorder = sum(
        row["FA_failure_reason"]
        == "max_order"
        for row in rows
    )

    print()
    print(
        "Fig. 5(b) simulation finished."
    )
    print(
        f"Total cases: {len(rows)}"
    )
    print(
        f"Converged cases: {len(rows) - n_fail}"
    )
    print(
        f"Failed cases: {n_fail}"
    )
    print(
        f"  numerical non-finite failures: {n_numfail}"
    )
    print(
        f"  finite max-order failures: {n_maxorder}"
    )
    print(
        f"Raw CSV: {RESULTS / 'fig5b_raw.csv'}"
    )
    print(
        f"Summary CSV: {RESULTS / 'fig5b_summary.csv'}"
    )
    print(
        f"Plot CSV: {RESULTS / 'fig5b_plot.csv'}"
    )


if __name__ == "__main__":
    main()
