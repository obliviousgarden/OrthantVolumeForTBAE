from __future__ import annotations

"""
Fig. 4(b) dedicated simulation -- paired-timing version
=======================================================

Purpose
-------
Benchmark the computational speedup of the Appendix-C one-dimensional
reduction for the COMPLETE normalized N=4 orthant volume U^(4)(beta):

    S_1D/3D = t_3D / t_1D.

This version is designed specifically to stabilize the high-r part of Fig. 4(b).

Key changes from the previous timing script
-------------------------------------------
1. The horizontal variable remains

       r = ||beta||_2.

2. The r-grid is substantially denser for r >= 0.90.

3. Timing is PAIRED:
   for each timing sample, both methods are measured in the same local timing
   cycle and the speedup is formed sample by sample,

       S_j = t_3D,j / t_1D,j.

   The plotted value is then

       median_j S_j,

   instead of taking the ratio of two independently estimated medians.

4. Measurement order alternates between

       1D -> 3D
       3D -> 1D

   from sample to sample to reduce systematic order / thermal bias.

5. Each method uses an independently calibrated batch size so that fast
   1D calls are timed over a sufficiently long interval.

6. The default main-paper benchmark includes only the two trajectories that
   are intended for Fig. 4(b):

       equal_positive
       anisotropic_8_4_2_1

   Set INCLUDE_MIXED_EQUAL = True if the mixed-sign trajectory is also wanted
   for supplementary diagnostics.

Output
------
Two CSV files are written under ./results_fig4b_paired:

    fig4b_raw.csv
        One row per paired timing sample.

    fig4b_summary.csv
        Plot-ready wide table with one row per r.

The summary keeps the familiar columns

    equal_speedup_1D_vs_3D
    anis_speedup_1D_vs_3D

so the plotting workflow remains simple.
"""

import gc
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from config import NumericalConfig
from n4 import (
    unit_ball_orthant_N4_1D,
    unit_ball_orthant_N4_3D,
)
from utils import write_csv


RESULTS = (
    Path(__file__).resolve().parent
    / "results"
)


# =============================================================================
# Numerical settings
# =============================================================================

QUAD_EPSABS = 2e-14
QUAD_EPSREL = 2e-13
QUAD_LIMIT = 600

# Not used directly by the two routes benchmarked here, but retained for
# project-level NumericalConfig compatibility.
FA_RTOL = 1e-12
FA_ATOL = 1e-14
FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384


# =============================================================================
# Timing settings
# =============================================================================

# More samples than the previous version because the speedup itself is now
# estimated from a distribution of paired ratios.
PAIRED_TIMING_SAMPLES = 21

# Warm up both numerical routes before calibration.
WARMUP_CALLS = 2

# Approximate minimum wall time per timing batch for the fast route.
# Direct 3D calls may already exceed this with batch_size = 1.
TARGET_BATCH_SECONDS = 0.10

MAX_BATCH_1D = 50000
MAX_BATCH_3D = 100

# Main-paper trajectories.
INCLUDE_MIXED_EQUAL = False


# =============================================================================
# Dense r-grid
# =============================================================================

def make_r_grid() -> np.ndarray:
    """
    High-density r-grid for the paired-timing Fig. 4(b) benchmark.

    Region I:   0.10 -- 0.80, step 0.01
    Region II:  0.81 -- 0.90, step 0.005
    Region III: 0.902 -- 0.990, step 0.002
    Region IV:  0.991 -- 0.999, step 0.001

    Every point is independently timed; no interpolation or smoothing is used.
    """
    low = np.arange(0.10, 0.8000001, 0.01)
    middle = np.arange(0.81, 0.9000001, 0.005)
    high = np.arange(0.902, 0.9900001, 0.002)
    near_boundary = np.arange(0.991, 0.9990001, 0.001)

    grid = np.concatenate(
        [
            low,
            middle,
            high,
            near_boundary,
        ]
    )

    return np.unique(
        np.round(
            grid,
            8,
        )
    )


# =============================================================================
# Fixed N=4 trajectories
# =============================================================================

def direction_vector(
    shape: str,
) -> np.ndarray:
    """
    Return a fixed unit vector w such that beta = r*w.
    """
    if shape == "equal_positive":
        w = np.array(
            [1.0, 1.0, 1.0, 1.0],
            dtype=float,
        )

    elif shape == "mixed_equal":
        w = np.array(
            [1.0, 1.0, -1.0, -1.0],
            dtype=float,
        )

    elif shape == "anisotropic_8_4_2_1":
        w = np.array(
            [8.0, 4.0, 2.0, 1.0],
            dtype=float,
        )

    else:
        raise ValueError(
            f"Unknown shape: {shape}"
        )

    return (
        w
        / np.linalg.norm(w)
    )


def shape_list():
    shapes = [
        "equal_positive",
        "anisotropic_8_4_2_1",
    ]

    if INCLUDE_MIXED_EQUAL:
        shapes.insert(
            1,
            "mixed_equal",
        )

    return tuple(
        shapes
    )


# =============================================================================
# General helpers
# =============================================================================

def percentile_summary(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    return {
        "median": float(
            np.median(values)
        ),
        "q25": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        "q75": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def relative_error(
    value: float,
    reference: float,
) -> float:
    if not (
        math.isfinite(value)
        and math.isfinite(reference)
    ):
        return math.nan

    if reference == 0.0:
        return math.nan

    return (
        abs(
            value
            - reference
        )
        / abs(reference)
    )


def valid_u4(
    value: float,
) -> bool:
    """
    Valid normalized N=4 orthant volume range:
        0 <= U^(4) <= kappa_4 = pi^2/2.
    """
    kappa4 = (
        math.pi**2
        / 2.0
    )

    return (
        math.isfinite(value)
        and value >= 0.0
        and value <= kappa4
    )


# =============================================================================
# Timing helpers
# =============================================================================

def call_once(
    func,
    beta,
    *,
    config: NumericalConfig,
):
    t0 = time.perf_counter_ns()

    value = func(
        beta,
        config=config,
    )

    elapsed_ns = (
        time.perf_counter_ns()
        - t0
    )

    return (
        elapsed_ns * 1e-9,
        value,
    )


def choose_batch_size(
    single_call_s: float,
    *,
    target_batch_s: float,
    max_batch: int,
) -> int:
    if not (
        math.isfinite(single_call_s)
        and single_call_s > 0.0
    ):
        return 1

    batch = int(
        math.ceil(
            target_batch_s
            / single_call_s
        )
    )

    return max(
        1,
        min(
            batch,
            int(max_batch),
        ),
    )


def run_batch(
    func,
    beta,
    *,
    config: NumericalConfig,
    batch_size: int,
):
    """
    Time one batch and return per-call time and the final numerical value.
    """
    t0 = time.perf_counter_ns()

    value = math.nan

    for _ in range(
        batch_size
    ):
        value = func(
            beta,
            config=config,
        )

    elapsed_s = (
        time.perf_counter_ns()
        - t0
    ) * 1e-9

    per_call_s = (
        elapsed_s
        / batch_size
    )

    return (
        elapsed_s,
        per_call_s,
        value,
    )


def prepare_timing(
    beta,
    *,
    config: NumericalConfig,
):
    """
    Warm up both routes, measure one calibration call for each, and determine
    independent batch sizes.
    """
    value_1d = math.nan
    value_3d = math.nan

    for _ in range(
        WARMUP_CALLS
    ):
        value_1d = (
            unit_ball_orthant_N4_1D(
                beta,
                config=config,
            )
        )

        value_3d = (
            unit_ball_orthant_N4_3D(
                beta,
                config=config,
            )
        )

    calibration_1d_s, value_1d = (
        call_once(
            unit_ball_orthant_N4_1D,
            beta,
            config=config,
        )
    )

    calibration_3d_s, value_3d = (
        call_once(
            unit_ball_orthant_N4_3D,
            beta,
            config=config,
        )
    )

    batch_1d = choose_batch_size(
        calibration_1d_s,
        target_batch_s=(
            TARGET_BATCH_SECONDS
        ),
        max_batch=MAX_BATCH_1D,
    )

    batch_3d = choose_batch_size(
        calibration_3d_s,
        target_batch_s=(
            TARGET_BATCH_SECONDS
        ),
        max_batch=MAX_BATCH_3D,
    )

    return {
        "calibration_1D_s": (
            calibration_1d_s
        ),
        "calibration_3D_s": (
            calibration_3d_s
        ),
        "batch_1D": batch_1d,
        "batch_3D": batch_3d,
        "value_1D": value_1d,
        "value_3D": value_3d,
    }


def paired_benchmark(
    beta,
    *,
    config: NumericalConfig,
    shape: str,
    r: float,
    point_index: int,
):
    """
    Measure paired 1D/3D timings.

    For odd timing samples:
        1D -> 3D

    For even timing samples:
        3D -> 1D

    The speedup is calculated within each paired sample:
        S_j = t_3D,j / t_1D,j.
    """
    prep = prepare_timing(
        beta,
        config=config,
    )

    batch_1d = int(
        prep["batch_1D"]
    )

    batch_3d = int(
        prep["batch_3D"]
    )

    raw_rows = []

    times_1d = []
    times_3d = []
    paired_speedups = []

    U_1d = float(
        prep["value_1D"]
    )

    U_3d = float(
        prep["value_3D"]
    )

    gc_was_enabled = (
        gc.isenabled()
    )

    try:
        gc.disable()

        for sample_index in range(
            1,
            PAIRED_TIMING_SAMPLES + 1,
        ):
            if sample_index % 2 == 1:
                pair_order = "1D->3D"

                (
                    elapsed_1d,
                    per_call_1d,
                    value_1d,
                ) = run_batch(
                    unit_ball_orthant_N4_1D,
                    beta,
                    config=config,
                    batch_size=batch_1d,
                )

                (
                    elapsed_3d,
                    per_call_3d,
                    value_3d,
                ) = run_batch(
                    unit_ball_orthant_N4_3D,
                    beta,
                    config=config,
                    batch_size=batch_3d,
                )

            else:
                pair_order = "3D->1D"

                (
                    elapsed_3d,
                    per_call_3d,
                    value_3d,
                ) = run_batch(
                    unit_ball_orthant_N4_3D,
                    beta,
                    config=config,
                    batch_size=batch_3d,
                )

                (
                    elapsed_1d,
                    per_call_1d,
                    value_1d,
                ) = run_batch(
                    unit_ball_orthant_N4_1D,
                    beta,
                    config=config,
                    batch_size=batch_1d,
                )

            U_1d = float(
                value_1d
            )
            U_3d = float(
                value_3d
            )

            if (
                per_call_1d > 0.0
                and math.isfinite(
                    per_call_1d
                )
                and math.isfinite(
                    per_call_3d
                )
            ):
                speedup = (
                    per_call_3d
                    / per_call_1d
                )
            else:
                speedup = math.nan

            times_1d.append(
                per_call_1d
            )
            times_3d.append(
                per_call_3d
            )
            paired_speedups.append(
                speedup
            )

            raw_rows.append(
                {
                    "point_index": (
                        point_index
                    ),
                    "shape": shape,
                    "r": r,

                    "beta_1": float(
                        beta[0]
                    ),
                    "beta_2": float(
                        beta[1]
                    ),
                    "beta_3": float(
                        beta[2]
                    ),
                    "beta_4": float(
                        beta[3]
                    ),

                    "sample_index": (
                        sample_index
                    ),
                    "pair_order": (
                        pair_order
                    ),

                    "batch_1D": (
                        batch_1d
                    ),
                    "batch_3D": (
                        batch_3d
                    ),

                    "elapsed_1D_s": (
                        elapsed_1d
                    ),
                    "elapsed_3D_s": (
                        elapsed_3d
                    ),

                    "per_call_1D_s": (
                        per_call_1d
                    ),
                    "per_call_3D_s": (
                        per_call_3d
                    ),

                    "paired_speedup_1D_vs_3D": (
                        speedup
                    ),

                    "U4_1D": U_1d,
                    "U4_3D": U_3d,

                    "abs_err_1D_vs_3D": (
                        abs(
                            U_1d
                            - U_3d
                        )
                        if (
                            math.isfinite(U_1d)
                            and math.isfinite(U_3d)
                        )
                        else math.nan
                    ),

                    "rel_err_1D_vs_3D": (
                        relative_error(
                            U_1d,
                            U_3d,
                        )
                    ),
                }
            )

    finally:
        if gc_was_enabled:
            gc.enable()

    stats_1d = percentile_summary(
        times_1d
    )

    stats_3d = percentile_summary(
        times_3d
    )

    stats_speedup = percentile_summary(
        paired_speedups
    )

    point_row = {
        "point_index": point_index,
        "shape": shape,
        "r": r,

        "beta_1": float(beta[0]),
        "beta_2": float(beta[1]),
        "beta_3": float(beta[2]),
        "beta_4": float(beta[3]),
        "beta_norm": float(
            np.linalg.norm(beta)
        ),

        "U4_1D": U_1d,
        "U4_3D": U_3d,

        "U4_1D_valid": (
            valid_u4(U_1d)
        ),
        "U4_3D_valid": (
            valid_u4(U_3d)
        ),

        "abs_err_1D_vs_3D": (
            abs(
                U_1d
                - U_3d
            )
            if (
                math.isfinite(U_1d)
                and math.isfinite(U_3d)
            )
            else math.nan
        ),

        "rel_err_1D_vs_3D": (
            relative_error(
                U_1d,
                U_3d,
            )
        ),

        "calibration_1D_s": (
            prep[
                "calibration_1D_s"
            ]
        ),
        "calibration_3D_s": (
            prep[
                "calibration_3D_s"
            ]
        ),

        "batch_1D": batch_1d,
        "batch_3D": batch_3d,

        "time_1D_median_s": (
            stats_1d["median"]
        ),
        "time_1D_q25_s": (
            stats_1d["q25"]
        ),
        "time_1D_q75_s": (
            stats_1d["q75"]
        ),
        "time_1D_min_s": stats_1d["min"],
        "time_1D_max_s": stats_1d["max"],

        "time_3D_median_s": (
            stats_3d["median"]
        ),
        "time_3D_q25_s": (
            stats_3d["q25"]
        ),
        "time_3D_q75_s": (
            stats_3d["q75"]
        ),
        "time_3D_min_s": stats_3d["min"],
        "time_3D_max_s": stats_3d["max"],

        # Primary Fig. 4(b) quantity:
        # median of paired sample-by-sample speedup ratios.
        "speedup_1D_vs_3D": (
            stats_speedup["median"]
        ),
        "speedup_q25": (
            stats_speedup["q25"]
        ),
        "speedup_q75": (
            stats_speedup["q75"]
        ),
        "speedup_min": (
            stats_speedup["min"]
        ),
        "speedup_max": (
            stats_speedup["max"]
        ),

        # Diagnostic comparison with the old estimator.
        "ratio_of_median_times": (
            stats_3d["median"]
            / stats_1d["median"]
            if (
                stats_1d["median"] > 0.0
            )
            else math.nan
        ),
    }

    return (
        point_row,
        raw_rows,
    )


# =============================================================================
# Main simulation
# =============================================================================

def fig4b_simulation(
    config: NumericalConfig,
    *,
    r_values=None,
):
    if r_values is None:
        r_values = make_r_grid()

    shapes = shape_list()

    raw_rows = []
    point_rows = []

    point_index = 0

    for shape in shapes:
        w = direction_vector(
            shape
        )

        for r in r_values:
            point_index += 1

            r = float(r)

            beta = (
                r
                * w
            )

            if not (
                np.linalg.norm(beta)
                < 1.0
            ):
                raise ValueError(
                    f"Need ||beta||_2 < 1, got r={r}."
                )

            (
                point_row,
                raw_point_rows,
            ) = paired_benchmark(
                beta,
                config=config,
                shape=shape,
                r=r,
                point_index=point_index,
            )

            point_rows.append(
                point_row
            )

            raw_rows.extend(
                raw_point_rows
            )

            print(
                f"[{shape:24s}] "
                f"r={r:0.4f}  "
                f"S={point_row['speedup_1D_vs_3D']:8.2f}  "
                f"t1D={point_row['time_1D_median_s']:.3e} s  "
                f"t3D={point_row['time_3D_median_s']:.3e} s"
            )

    # =========================================================================
    # Wide-format plot-ready summary
    # =========================================================================
    by_key = {
        (
            row["shape"],
            row["r"],
        ): row
        for row in point_rows
    }

    summary_rows = []

    for r in r_values:
        r = float(r)

        summary = {
            "r": r,
            "one_minus_r": (
                1.0 - r
            ),
            "FA_max_order_allowed": FA_MAX_ORDER,
            "paired_timing_samples": PAIRED_TIMING_SAMPLES,
        }

        shape_prefixes = [
            (
                "equal_positive",
                "equal",
            ),
            (
                "anisotropic_8_4_2_1",
                "anis",
            ),
        ]

        if INCLUDE_MIXED_EQUAL:
            shape_prefixes.insert(
                1,
                (
                    "mixed_equal",
                    "mixed",
                ),
            )

        for shape, prefix in shape_prefixes:
            source = by_key[
                (
                    shape,
                    r,
                )
            ]

            summary[
                f"{prefix}_speedup_1D_vs_3D"
            ] = source[
                "speedup_1D_vs_3D"
            ]

            summary[
                f"{prefix}_speedup_q25"
            ] = source[
                "speedup_q25"
            ]

            summary[
                f"{prefix}_speedup_q75"
            ] = source[
                "speedup_q75"
            ]
            summary[f"{prefix}_speedup_min"] = source["speedup_min"]
            summary[f"{prefix}_speedup_max"] = source["speedup_max"]

            summary[
                f"{prefix}_ratio_of_median_times"
            ] = source[
                "ratio_of_median_times"
            ]

            summary[
                f"{prefix}_time_1D_median_s"
            ] = source[
                "time_1D_median_s"
            ]

            summary[
                f"{prefix}_time_1D_q25_s"
            ] = source[
                "time_1D_q25_s"
            ]

            summary[
                f"{prefix}_time_1D_q75_s"
            ] = source[
                "time_1D_q75_s"
            ]
            summary[f"{prefix}_time_1D_min_s"] = source["time_1D_min_s"]
            summary[f"{prefix}_time_1D_max_s"] = source["time_1D_max_s"]

            summary[
                f"{prefix}_time_3D_median_s"
            ] = source[
                "time_3D_median_s"
            ]

            summary[
                f"{prefix}_time_3D_q25_s"
            ] = source[
                "time_3D_q25_s"
            ]

            summary[
                f"{prefix}_time_3D_q75_s"
            ] = source[
                "time_3D_q75_s"
            ]
            summary[f"{prefix}_time_3D_min_s"] = source["time_3D_min_s"]
            summary[f"{prefix}_time_3D_max_s"] = source["time_3D_max_s"]

            summary[
                f"{prefix}_U4_1D"
            ] = source[
                "U4_1D"
            ]

            summary[
                f"{prefix}_U4_3D"
            ] = source[
                "U4_3D"
            ]

            summary[
                f"{prefix}_abs_err_1D_vs_3D"
            ] = source[
                "abs_err_1D_vs_3D"
            ]

            summary[
                f"{prefix}_rel_err_1D_vs_3D"
            ] = source[
                "rel_err_1D_vs_3D"
            ]

            summary[
                f"{prefix}_valid_1D"
            ] = source[
                "U4_1D_valid"
            ]

            summary[
                f"{prefix}_valid_3D"
            ] = source[
                "U4_3D_valid"
            ]

            summary[
                f"{prefix}_batch_1D"
            ] = source[
                "batch_1D"
            ]

            summary[
                f"{prefix}_batch_3D"
            ] = source[
                "batch_3D"
            ]

        summary_rows.append(
            summary
        )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        raw_rows,
        RESULTS
        / "fig4b_raw.csv",
    )

    write_csv(
        summary_rows,
        RESULTS
        / "fig4b_summary.csv",
    )

    return (
        raw_rows,
        summary_rows,
        point_rows,
    )


# =============================================================================
# IDE main
# =============================================================================

def main():
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

        # Project compatibility only; not used in Fig. 4(b).
        aono_fourier_terms=512,

        timing_repeats=1,

        qmc_power=18,
        qmc_scrambles=16,

        seed=20261015,
    )

    r_values = make_r_grid()

    (
        raw_rows,
        summary_rows,
        point_rows,
    ) = fig4b_simulation(
        config,
        r_values=r_values,
    )

    invalid_points = [
        row
        for row in point_rows
        if not (
            row["U4_1D_valid"]
            and row["U4_3D_valid"]
        )
    ]

    finite_speedups = [
        row["speedup_1D_vs_3D"]
        for row in point_rows
        if math.isfinite(
            row["speedup_1D_vs_3D"]
        )
    ]

    max_rel_error = max(
        (
            row["rel_err_1D_vs_3D"]
            for row in point_rows
            if math.isfinite(
                row[
                    "rel_err_1D_vs_3D"
                ]
            )
        ),
        default=math.nan,
    )

    print()
    print(
        "Fig. 4(b) paired-timing simulation finished."
    )
    print(
        f"Python: {sys.version.split()[0]}"
    )
    print(
        f"Platform: {platform.platform()}"
    )
    print(
        f"r points per trajectory: {len(r_values)}"
    )
    print(
        f"trajectories: {len(shape_list())}"
    )
    print(
        f"paired timing samples per point: {PAIRED_TIMING_SAMPLES}"
    )
    print(
        f"full U4 parameter points: {len(point_rows)}"
    )
    print(
        f"raw paired timing rows: {len(raw_rows)}"
    )
    print(
        f"invalid full-volume points: {len(invalid_points)}"
    )
    print(
        "maximum relative discrepancy "
        f"(1D vs 3D): {max_rel_error:.6e}"
    )

    if finite_speedups:
        print(
            "paired-median speedup range: "
            f"{min(finite_speedups):.3f} -- "
            f"{max(finite_speedups):.3f}"
        )

    print(
        f"raw CSV: {RESULTS / 'fig4b_raw.csv'}"
    )
    print(
        f"summary CSV: {RESULTS / 'fig4b_summary.csv'}"
    )


if __name__ == "__main__":
    main()
