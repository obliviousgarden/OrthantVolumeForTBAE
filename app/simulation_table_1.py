from __future__ import annotations

"""
Table 1 dedicated simulation
============================

Purpose
-------
Rebuild the former ``validation_0_geometric_invariants`` calculation as a
standalone Table-1 simulation.

The calculation checks two exact geometric identities for the raw general
subset/Lauricella representation:

(A) Centered orthant identity

        U_N(0) = kappa_N / 2^N.

(B) Full orthant-partition identity

        sum_{sigma in {+1,-1}^N}
            U_N(sigma * gamma)
        = kappa_N,

    for any fixed gamma with ||gamma||_2 < 1.

The second identity is a strong global consistency check because every sign
pattern is evaluated independently and the resulting orthant volumes must sum
to the full N-ball volume.

This script intentionally uses the RAW general Lauricella evaluator rather
than Proposed_Hybrid.  Table 1 is meant to validate the general analytical
representation itself, not the later support-specific numerical routes.

Output
------
Two CSV files are written under ./results_table_1:

    table1_raw.csv
        One row for every centered test and every orthant-partition parameter
        case.

    table1_summary.csv
        Manuscript-oriented summary with one row per N.  Partition-error
        statistics are computed ONLY from successful finite/converged cases.
        Numerical failures are retained separately through success/failure
        counts and a failed-case description, so one failed random direction
        cannot turn the whole N-summary into inf.

Parameter coverage
------------------
Centered identity:
    N = 1,2,3,4,5,6,8,10,12,16,24,32

Orthant partition:
    N = 2,3,4,5,6

    r = 0.20,0.35,0.50,0.65,0.80,0.90

    directions:
        equal
        anisotropic
        random (3 independent deterministic replicates per N,r)

The partition test is intentionally limited to N<=6 because it explicitly
evaluates all 2^N orthants, and each orthant evaluation itself contains a
subset expansion.  This keeps Table 1 a correctness test rather than turning
it into a computational-scaling experiment.

Numerical settings
------------------
The FA maximum order is set uniformly to 16384 and the requested tolerance is tightened
relative to the earlier broad validation scan.  Since r<=0.90 here, these
settings are conservative without pushing the defining series unnecessarily
close to its convergence boundary.
"""

import itertools
import math
from pathlib import Path

import numpy as np

from config import NumericalConfig
from lauricella import unit_ball_orthant_lauricella
from patterns import make_beta
from utils import (
    kappa,
    relative_error,
    valid_orthant_volume,
    write_csv,
)


RESULTS = (
    Path(__file__).resolve().parent
    / "results"
)


# =============================================================================
# Table-1 parameter grid
# =============================================================================

CENTERED_DIMENSIONS = (
    1, 2, 3, 4, 5, 6,
    8, 10, 12, 16, 24, 32,
)

PARTITION_DIMENSIONS = (
    2, 3, 4, 5, 6,
)

PARTITION_R_VALUES = (
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.90,
)

PARTITION_KINDS = (
    "equal",
    "anisotropic",
    "random",
)

RANDOM_REPLICATES = 3


# =============================================================================
# Numerical settings
# =============================================================================

FA_RTOL = 1e-12
FA_ATOL = 1e-14
FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384


# =============================================================================
# Helpers
# =============================================================================

def evaluate_raw_fa(
    beta,
    *,
    config: NumericalConfig,
):
    """
    Evaluate the raw subset/Lauricella representation and return value + info.
    """
    return unit_ball_orthant_lauricella(
        beta,
        config=config,
        return_info=True,
    )


def safe_max(
    values,
    *,
    default=math.nan,
):
    finite = [
        x
        for x in values
        if math.isfinite(x)
    ]

    if not finite:
        return default

    return max(finite)


def safe_median(
    values,
    *,
    default=math.nan,
):
    finite = [
        x
        for x in values
        if math.isfinite(x)
    ]

    if not finite:
        return default

    return float(
        np.median(
            finite
        )
    )


# =============================================================================
# Main simulation
# =============================================================================

def simulation_table_1(
    config: NumericalConfig,
    *,
    centered_dimensions=CENTERED_DIMENSIONS,
    partition_dimensions=PARTITION_DIMENSIONS,
    r_values=PARTITION_R_VALUES,
    kinds=PARTITION_KINDS,
    random_replicates=RANDOM_REPLICATES,
):
    """
    Run the complete Table-1 validation.

    Returns
    -------
    raw_rows, summary_rows
    """
    raw_rows = []

    rng = np.random.default_rng(
        config.seed
    )

    # =========================================================================
    # A. Centered identity
    # =========================================================================
    centered_by_N = {}

    for N in centered_dimensions:
        beta = np.zeros(
            N,
            dtype=float,
        )

        value, info = evaluate_raw_fa(
            beta,
            config=config,
        )

        reference = math.ldexp(
            kappa(N),
            -N,
        )

        abs_error = abs(
            value
            - reference
        )

        rel_error_value = (
            relative_error(
                value,
                reference,
            )
        )

        valid = valid_orthant_volume(
            value,
            N,
        )

        row = {
            "test": "centered_orthant",
            "N": N,

            "r": 0.0,
            "kind": "zero",
            "replicate": 0,

            "beta": beta.tolist(),

            "value": value,
            "reference": reference,

            "abs_error": abs_error,
            "rel_error": rel_error_value,

            "FA_order": info[
                "max_FA_order"
            ],
            "FA_converged": info[
                "all_FA_converged"
            ],

            "cancellation": info[
                "cancellation_indicator"
            ],

            "valid_volume": valid,

            "orthants_summed": 1,
            "orthants_valid": (
                1 if valid else 0
            ),
            "orthants_invalid": (
                0 if valid else 1
            ),
        }

        raw_rows.append(row)
        centered_by_N[N] = row

    # =========================================================================
    # B. Full sign-partition identity
    # =========================================================================
    partition_rows = []

    for N in partition_dimensions:
        for r in r_values:
            for kind in kinds:

                if kind == "random":
                    replicate_ids = range(
                        1,
                        random_replicates + 1,
                    )
                else:
                    replicate_ids = (1,)

                for replicate in replicate_ids:
                    gamma = make_beta(
                        N,
                        r,
                        kind=kind,
                        rng=rng,
                    )

                    values = []

                    all_converged = True
                    max_order = 0
                    max_cancellation = 0.0

                    orthants_valid = 0
                    orthants_invalid = 0
                    orthants_nonfinite = 0
                    orthants_nonconverged = 0

                    for signs_tuple in itertools.product(
                        [-1.0, 1.0],
                        repeat=N,
                    ):
                        signs = np.asarray(
                            signs_tuple,
                            dtype=float,
                        )

                        beta = (
                            signs
                            * gamma
                        )

                        value, info = evaluate_raw_fa(
                            beta,
                            config=config,
                        )

                        values.append(
                            value
                        )

                        converged = bool(
                            info[
                                "all_FA_converged"
                            ]
                        )

                        all_converged = (
                            all_converged
                            and converged
                        )

                        if not converged:
                            orthants_nonconverged += 1

                        max_order = max(
                            max_order,
                            int(
                                info[
                                    "max_FA_order"
                                ]
                            ),
                        )

                        cancellation = float(
                            info[
                                "cancellation_indicator"
                            ]
                        )

                        if math.isfinite(
                            cancellation
                        ):
                            max_cancellation = max(
                                max_cancellation,
                                cancellation,
                            )
                        else:
                            max_cancellation = (
                                math.inf
                            )

                        if not math.isfinite(
                            value
                        ):
                            orthants_nonfinite += 1

                        if valid_orthant_volume(
                            value,
                            N,
                        ):
                            orthants_valid += 1
                        else:
                            orthants_invalid += 1

                    if all(
                        math.isfinite(x)
                        for x in values
                    ):
                        partition_sum = (
                            math.fsum(
                                values
                            )
                        )
                    else:
                        partition_sum = (
                            math.nan
                        )

                    reference = kappa(N)

                    if math.isfinite(
                        partition_sum
                    ):
                        abs_error = abs(
                            partition_sum
                            - reference
                        )

                        rel_error_value = (
                            relative_error(
                                partition_sum,
                                reference,
                            )
                        )
                    else:
                        abs_error = math.inf
                        rel_error_value = math.inf

                    partition_success = (
                        all_converged
                        and orthants_invalid == 0
                        and orthants_nonfinite == 0
                        and math.isfinite(
                            partition_sum
                        )
                        and math.isfinite(
                            abs_error
                        )
                        and math.isfinite(
                            rel_error_value
                        )
                    )

                    row = {
                        "test": (
                            "orthant_partition_sum"
                        ),
                        "N": N,

                        "r": float(r),
                        "kind": kind,
                        "replicate": replicate,

                        "beta": gamma.tolist(),

                        "value": partition_sum,
                        "reference": reference,

                        "abs_error": abs_error,
                        "rel_error": rel_error_value,

                        "FA_order": max_order,
                        "FA_converged": (
                            all_converged
                        ),

                        "cancellation": (
                            max_cancellation
                        ),

                        "valid_volume": (
                            orthants_invalid == 0
                        ),

                        "partition_success": (
                            partition_success
                        ),
                        "partition_failure_reason": (
                            ""
                            if partition_success
                            else (
                                "FA_nonconvergence"
                                if not all_converged
                                else (
                                    "nonfinite_orthant"
                                    if orthants_nonfinite > 0
                                    else (
                                        "invalid_orthant"
                                        if orthants_invalid > 0
                                        else (
                                            "nonfinite_partition_sum"
                                            if not math.isfinite(
                                                partition_sum
                                            )
                                            else "other"
                                        )
                                    )
                                )
                            )
                        ),

                        "orthants_summed": (
                            2**N
                        ),
                        "orthants_valid": (
                            orthants_valid
                        ),
                        "orthants_invalid": (
                            orthants_invalid
                        ),
                        "orthants_nonfinite": (
                            orthants_nonfinite
                        ),
                        "orthants_nonconverged": (
                            orthants_nonconverged
                        ),
                    }

                    raw_rows.append(row)
                    partition_rows.append(
                        row
                    )

    # =========================================================================
    # C. Manuscript-oriented summary
    # =========================================================================
    summary_rows = []

    all_dimensions = sorted(
        set(
            centered_dimensions
        )
        | set(
            partition_dimensions
        )
    )

    for N in all_dimensions:
        centered = centered_by_N.get(
            N
        )

        partition_N = [
            row
            for row in partition_rows
            if row["N"] == N
        ]

        if partition_N:
            partition_case_count = len(
                partition_N
            )

            successful_partition_N = [
                row
                for row in partition_N
                if row.get(
                    "partition_success",
                    False,
                )
            ]

            failed_partition_N = [
                row
                for row in partition_N
                if not row.get(
                    "partition_success",
                    False,
                )
            ]

            partition_success_count = len(
                successful_partition_N
            )

            partition_failure_count = len(
                failed_partition_N
            )

            partition_success_fraction = (
                partition_success_count
                / partition_case_count
                if partition_case_count > 0
                else math.nan
            )

            if successful_partition_N:
                worst_rel = max(
                    successful_partition_N,
                    key=lambda row: row[
                        "rel_error"
                    ],
                )

                partition_max_abs_error = (
                    safe_max(
                        [
                            row["abs_error"]
                            for row in successful_partition_N
                        ]
                    )
                )

                partition_max_rel_error = (
                    safe_max(
                        [
                            row["rel_error"]
                            for row in successful_partition_N
                        ]
                    )
                )

                partition_median_rel_error = (
                    safe_median(
                        [
                            row["rel_error"]
                            for row in successful_partition_N
                        ]
                    )
                )

                partition_worst_r = (
                    worst_rel["r"]
                )

                partition_worst_kind = (
                    worst_rel["kind"]
                )

                partition_worst_replicate = (
                    worst_rel[
                        "replicate"
                    ]
                )
            else:
                partition_max_abs_error = (
                    math.nan
                )
                partition_max_rel_error = (
                    math.nan
                )
                partition_median_rel_error = (
                    math.nan
                )
                partition_worst_r = (
                    math.nan
                )
                partition_worst_kind = ""
                partition_worst_replicate = (
                    math.nan
                )

            partition_all_converged = all(
                row["FA_converged"]
                for row in partition_N
            )

            partition_max_FA_order = max(
                row["FA_order"]
                for row in partition_N
            )

            # Keep cancellation diagnostics only from successful finite cases,
            # so a failed/nonfinite evaluation does not contaminate the
            # reported worst successful cancellation level.
            partition_max_cancellation = (
                safe_max(
                    [
                        row["cancellation"]
                        for row in successful_partition_N
                    ]
                )
            )

            partition_max_invalid_orthants = max(
                row[
                    "orthants_invalid"
                ]
                for row in partition_N
            )

            partition_max_nonfinite_orthants = max(
                row[
                    "orthants_nonfinite"
                ]
                for row in partition_N
            )

            if failed_partition_N:
                failed_cases = "; ".join(
                    (
                        f"r={row['r']},"
                        f"kind={row['kind']},"
                        f"rep={row['replicate']},"
                        f"reason={row.get('partition_failure_reason','')}"
                    )
                    for row in failed_partition_N
                )
            else:
                failed_cases = ""

        else:
            partition_case_count = 0
            partition_success_count = 0
            partition_failure_count = 0
            partition_success_fraction = (
                math.nan
            )
            failed_cases = ""

            partition_max_abs_error = (
                math.nan
            )
            partition_max_rel_error = (
                math.nan
            )
            partition_median_rel_error = (
                math.nan
            )

            partition_all_converged = (
                math.nan
            )
            partition_max_FA_order = (
                math.nan
            )
            partition_max_cancellation = (
                math.nan
            )

            partition_max_invalid_orthants = (
                math.nan
            )
            partition_max_nonfinite_orthants = (
                math.nan
            )

            partition_worst_r = (
                math.nan
            )
            partition_worst_kind = ""
            partition_worst_replicate = (
                math.nan
            )

        if centered is not None:
            centered_value = (
                centered["value"]
            )
            centered_reference = (
                centered["reference"]
            )
            centered_abs_error = (
                centered["abs_error"]
            )
            centered_rel_error = (
                centered["rel_error"]
            )
            centered_FA_order = (
                centered["FA_order"]
            )
            centered_converged = (
                centered[
                    "FA_converged"
                ]
            )
            centered_cancellation = (
                centered[
                    "cancellation"
                ]
            )
        else:
            centered_value = math.nan
            centered_reference = (
                math.nan
            )
            centered_abs_error = (
                math.nan
            )
            centered_rel_error = (
                math.nan
            )
            centered_FA_order = (
                math.nan
            )
            centered_converged = (
                math.nan
            )
            centered_cancellation = (
                math.nan
            )

        summary_rows.append(
            {
                "N": N,
                "FA_max_order_allowed": FA_MAX_ORDER,

                # ---------------------------------------------------------
                # Centered identity
                # ---------------------------------------------------------
                "centered_value": (
                    centered_value
                ),
                "centered_reference": (
                    centered_reference
                ),
                "centered_abs_error": (
                    centered_abs_error
                ),
                "centered_rel_error": (
                    centered_rel_error
                ),
                "centered_FA_order": (
                    centered_FA_order
                ),
                "centered_FA_converged": (
                    centered_converged
                ),
                "centered_cancellation": (
                    centered_cancellation
                ),

                # ---------------------------------------------------------
                # Partition identity: summary over all r/kind/replicates
                # ---------------------------------------------------------
                "partition_case_count": (
                    partition_case_count
                ),
                "partition_success_count": (
                    partition_success_count
                ),
                "partition_failure_count": (
                    partition_failure_count
                ),
                "partition_success_fraction": (
                    partition_success_fraction
                ),
                "partition_success_label": (
                    f"{partition_success_count}/{partition_case_count}"
                    if partition_case_count > 0
                    else ""
                ),
                "partition_failed_cases": (
                    failed_cases
                ),

                "partition_r_min": (
                    min(r_values)
                    if partition_case_count
                    else math.nan
                ),
                "partition_r_max": (
                    max(r_values)
                    if partition_case_count
                    else math.nan
                ),

                "partition_max_abs_error": (
                    partition_max_abs_error
                ),
                "partition_max_rel_error": (
                    partition_max_rel_error
                ),
                "partition_median_rel_error": (
                    partition_median_rel_error
                ),

                "partition_all_converged": (
                    partition_all_converged
                ),
                "partition_max_FA_order": (
                    partition_max_FA_order
                ),
                "partition_cases_order_gt_128": (
                    sum(
                        row["FA_order"] > 128
                        for row in partition_N
                    )
                    if partition_N
                    else 0
                ),
                "partition_max_cancellation": (
                    partition_max_cancellation
                ),

                "partition_max_invalid_orthants": (
                    partition_max_invalid_orthants
                ),
                "partition_max_nonfinite_orthants": (
                    partition_max_nonfinite_orthants
                ),

                "partition_worst_r": (
                    partition_worst_r
                ),
                "partition_worst_kind": (
                    partition_worst_kind
                ),
                "partition_worst_replicate": (
                    partition_worst_replicate
                ),
            }
        )

    # =========================================================================
    # Write files
    # =========================================================================
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        raw_rows,
        RESULTS / "table1_raw.csv",
    )

    write_csv(
        summary_rows,
        RESULTS / "table1_summary.csv",
    )

    return raw_rows, summary_rows


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

        quad_epsabs=1e-10,
        quad_epsrel=1e-10,
        quad_limit=400,

        # Not used in Table 1; retained for compatibility with the project
        # NumericalConfig definition.
        aono_fourier_terms=512,

        timing_repeats=1,

        qmc_power=18,
        qmc_scrambles=16,

        seed=20261014,
    )

    raw_rows, summary_rows = (
        simulation_table_1(
            config,
        )
    )

    centered_rows = [
        row
        for row in raw_rows
        if row["test"]
        == "centered_orthant"
    ]

    partition_rows = [
        row
        for row in raw_rows
        if row["test"]
        == "orthant_partition_sum"
    ]

    max_centered_rel = max(
        (
            row["rel_error"]
            for row in centered_rows
            if math.isfinite(
                row["rel_error"]
            )
        ),
        default=math.nan,
    )

    successful_partition_rows = [
        row
        for row in partition_rows
        if row.get(
            "partition_success",
            False,
        )
    ]

    max_partition_rel = max(
        (
            row["rel_error"]
            for row in successful_partition_rows
            if math.isfinite(
                row["rel_error"]
            )
        ),
        default=math.nan,
    )

    total_failures = sum(
        not row.get(
            "partition_success",
            False,
        )
        for row in partition_rows
    )

    total_invalid_orthants = sum(
        row.get(
            "orthants_invalid",
            0,
        )
        for row in partition_rows
    )

    print(
        "Table 1 simulation finished."
    )

    print(
        f"centered cases: {len(centered_rows)}"
    )

    print(
        f"partition cases: {len(partition_rows)}"
    )

    print(
        "maximum centered relative error: "
        f"{max_centered_rel:.6e}"
    )

    print(
        "maximum partition relative error "
        "(successful cases only): "
        f"{max_partition_rel:.6e}"
    )

    print(
        "successful partition cases: "
        f"{len(successful_partition_rows)}/{len(partition_rows)}"
    )

    print(
        "failed partition cases: "
        f"{total_failures}"
    )

    print(
        "total invalid individual orthant evaluations: "
        f"{total_invalid_orthants}"
    )

    print(
        f"raw CSV: {RESULTS / 'table1_raw.csv'}"
    )

    print(
        f"summary CSV: {RESULTS / 'table1_summary.csv'}"
    )


if __name__ == "__main__":
    main()
