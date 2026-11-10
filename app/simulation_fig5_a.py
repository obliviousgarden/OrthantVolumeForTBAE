from __future__ import annotations

"""
Fig. 5(a) dedicated simulation
==============================

Purpose
-------
Independently validate the affine extension from the normalized N-ball
orthant volume to an axis-aligned ellipsoid,

    V_E = (prod_i R_i) U^(N)(beta),
    beta_i = sigma_i c_i / R_i,

against randomized Sobol quasi-Monte Carlo (QMC).

The main diagnostic is the standardized QMC discrepancy

    z_QMC = (V_QMC - V_exact) / SE_QMC,

and its absolute value

    |z_QMC| = |V_QMC - V_exact| / SE_QMC.

Using the QMC standard error as the normalization is preferable to plotting
the raw absolute error because the physical ellipsoid volume changes with
dimension and semiaxis product.

Dimensions
----------
N = 2, 3, 4.

For N <= 3, ellipsoid_orthant_affine(method="auto") uses the complete
elementary ball expressions.  For N = 4 it uses the independently validated
one-dimensional reduction.

Geometry
--------
Three semiaxis families are used at each N:

    isotropic
    moderate anisotropy
    strong anisotropy

and four center / sign mappings are tested:

    equal_positive
    mixed_center
    mixed_sigma
    anisotropic_positive

The first three explicitly test the mapping beta_i = sigma_i c_i / R_i,
including sign handling.  The fourth adds a non-equal normalized displacement
direction.

Important QMC detail
--------------------
The physical semiaxes disappear after exact affine normalization.  Therefore
using the same Sobol realization for geometrically equivalent normalized
cases would produce duplicated standardized residuals.  This script assigns
a deterministic but different QMC seed to every physical test case so that
the independent numerical check is not artificially duplicated.

Output
------
Two CSV files are written under ./results_fig5a:

    fig5a_raw.csv
        One row for every (N, r, semiaxis family, mapping case).

    fig5a_summary.csv
        Plot-ready wide table with one row per r and summary diagnostics for
        N = 2, 3, 4.  In addition to mean signed z, it records signed-z SD,
        SEM, quartiles, minima/maxima, custom min/max error-bar spans, and
        explicit counts outside 3 sigma.

The raw table is retained for diagnostics.  A separate long-format boxplot
table is also written with exactly three columns, N, r, and z, where z is
the signed QMC-standardized discrepancy.
"""

import math
from pathlib import Path

import numpy as np

from config import NumericalConfig
from ellipsoid import ellipsoid_orthant_affine
from qmc import ellipsoid_orthant_qmc
from utils import write_csv


RESULTS = (
    Path(__file__).resolve().parent
    / "results"
)


# =============================================================================
# Numerical settings
# =============================================================================

# Exact N=4 route.
QUAD_EPSABS = 2e-14
QUAD_EPSREL = 2e-13
QUAD_LIMIT = 600

# Included for NumericalConfig compatibility.
FA_RTOL = 1e-12
FA_ATOL = 1e-14
FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384

# QMC:
# 2^15 points per scramble x 16 independent scrambles
# = 524,288 samples per physical test case.
QMC_POWER = 15
QMC_SCRAMBLES = 16

BASE_SEED = 20261015


# =============================================================================
# r-grid
# =============================================================================

def make_r_grid() -> np.ndarray:
    """
    High-density grid for the affine/QMC validation figure.

    The range remains 0.10 <= r <= 0.90 because Fig. 5(a) is an independent
    affine-validation test rather than the near-boundary stress test of Fig. 5(b).
    A 0.025 spacing provides substantially more independently validated points.
    """
    return np.round(np.arange(0.10, 0.9000001, 0.025), 8)


# =============================================================================
# Semiaxis families
# =============================================================================

def semiaxis_families(
    N: int,
):
    """
    Return named axis-aligned ellipsoid shapes.

    The minimum semiaxis is normalized to one.  The strong family reaches
    an aspect ratio of four in every dimension.
    """
    if N == 2:
        return (
            (
                "isotropic",
                np.array(
                    [1.0, 1.0],
                    dtype=float,
                ),
            ),
            (
                "moderate",
                np.array(
                    [1.0, 2.0],
                    dtype=float,
                ),
            ),
            (
                "strong",
                np.array(
                    [1.0, 4.0],
                    dtype=float,
                ),
            ),
        )

    if N == 3:
        return (
            (
                "isotropic",
                np.array(
                    [1.0, 1.0, 1.0],
                    dtype=float,
                ),
            ),
            (
                "moderate",
                np.array(
                    [1.0, 1.5, 2.0],
                    dtype=float,
                ),
            ),
            (
                "strong",
                np.array(
                    [1.0, 2.0, 4.0],
                    dtype=float,
                ),
            ),
        )

    if N == 4:
        return (
            (
                "isotropic",
                np.array(
                    [1.0, 1.0, 1.0, 1.0],
                    dtype=float,
                ),
            ),
            (
                "moderate",
                np.array(
                    [1.0, 1.4, 1.8, 2.4],
                    dtype=float,
                ),
            ),
            (
                "strong",
                np.array(
                    [1.0, 1.5, 2.5, 4.0],
                    dtype=float,
                ),
            ),
        )

    raise ValueError(
        f"Unsupported dimension N={N}"
    )


# =============================================================================
# Normalized displacement directions and physical mappings
# =============================================================================

def unit_direction(
    N: int,
    kind: str,
) -> np.ndarray:
    """
    Return a normalized direction w with ||w||_2 = 1.
    """
    if kind == "equal_positive":
        w = np.ones(
            N,
            dtype=float,
        )

    elif kind == "mixed_equal":
        w = np.ones(
            N,
            dtype=float,
        )
        w[1::2] *= -1.0

    elif kind == "anisotropic_positive":
        # Descending deterministic positive weights.
        w = np.array(
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
            f"Unknown direction kind: {kind}"
        )

    return (
        w
        / np.linalg.norm(w)
    )


def physical_case(
    N: int,
    r: float,
    semiaxes: np.ndarray,
    mapping_case: str,
):
    """
    Construct physical center c and orthant sign sigma.

    Returns
    -------
    center, sigma, beta_expected
    """
    sigma_positive = np.ones(
        N,
        dtype=float,
    )

    if mapping_case == "equal_positive":
        beta = (
            r
            * unit_direction(
                N,
                "equal_positive",
            )
        )

        sigma = sigma_positive

        # beta = sigma*c/R and sigma=+1
        center = (
            beta
            * semiaxes
        )

    elif mapping_case == "mixed_center":
        beta = (
            r
            * unit_direction(
                N,
                "mixed_equal",
            )
        )

        sigma = sigma_positive

        # Mixed physical center, positive orthant signs.
        center = (
            beta
            * semiaxes
        )

    elif mapping_case == "mixed_sigma":
        beta_positive = (
            r
            * unit_direction(
                N,
                "equal_positive",
            )
        )

        center = (
            beta_positive
            * semiaxes
        )

        sigma = np.ones(
            N,
            dtype=float,
        )
        sigma[1::2] = -1.0

        # This produces a mixed-sign beta through sigma*c/R.
        beta = (
            sigma
            * center
            / semiaxes
        )

    elif mapping_case == "anisotropic_positive":
        beta = (
            r
            * unit_direction(
                N,
                "anisotropic_positive",
            )
        )

        sigma = sigma_positive

        center = (
            beta
            * semiaxes
        )

    else:
        raise ValueError(
            f"Unknown mapping_case: {mapping_case}"
        )

    return (
        center,
        sigma,
        beta,
    )


MAPPING_CASES = (
    "equal_positive",
    "mixed_center",
    "mixed_sigma",
    "anisotropic_positive",
)


# =============================================================================
# Numerical helpers
# =============================================================================

def case_seed(
    N: int,
    r_index: int,
    aspect_index: int,
    mapping_index: int,
) -> int:
    """
    Deterministic independent seed for every physical QMC test case.
    """
    return int(
        BASE_SEED
        + 100000 * N
        + 1000 * r_index
        + 100 * aspect_index
        + mapping_index
    )


def standardized_discrepancy(
    exact: float,
    qmc: float,
    stderr: float,
):
    """
    Signed and absolute QMC-standardized discrepancy.
    """
    if not (
        math.isfinite(exact)
        and math.isfinite(qmc)
        and math.isfinite(stderr)
        and stderr > 0.0
    ):
        return (
            math.nan,
            math.nan,
        )

    z = (
        qmc
        - exact
    ) / stderr

    return (
        z,
        abs(z),
    )


def finite_stats(
    values,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if arr.size == 0:
        return {
            "count": 0,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "rms": math.nan,
        }

    if arr.size > 1:
        std = float(
            np.std(
                arr,
                ddof=1,
            )
        )
        sem = (
            std
            / math.sqrt(
                arr.size
            )
        )
    else:
        std = math.nan
        sem = math.nan

    return {
        "count": int(
            arr.size
        ),
        "median": float(
            np.median(arr)
        ),
        "q25": float(
            np.quantile(
                arr,
                0.25,
            )
        ),
        "q75": float(
            np.quantile(
                arr,
                0.75,
            )
        ),
        "min": float(
            np.min(arr)
        ),
        "max": float(
            np.max(arr)
        ),
        "mean": float(
            np.mean(arr)
        ),
        "std": std,
        "sem": sem,
        "rms": float(
            np.sqrt(
                np.mean(
                    arr * arr
                )
            )
        ),
    }



# =============================================================================
# Boxplot-table helper
# =============================================================================

def build_boxplot_table_long(
    raw_rows,
):
    """
    Build a long-format table for direct grouped boxplot input.

    Output columns are exactly:

        N, r, z

    where

        z = z_QMC
          = (V_QMC - V_exact) / SE_QMC.

    Each row corresponds to one physical validation case.
    With the current simulation design, every (N, r) group contains

        3 semiaxis families x 4 mapping cases = 12

    signed standardized discrepancies.

    This format is convenient for Origin, Excel, pandas, matplotlib,
    or other software that can group boxplots by N and r.
    """
    rows = []

    # Preserve the scientifically natural ordering:
    # first N=2 and all r values, then N=3, then N=4.
    for N in (
        2,
        3,
        4,
    ):
        subset_N = [
            row
            for row in raw_rows
            if row["N"] == N
        ]

        r_values_N = sorted(
            {
                float(row["r"])
                for row in subset_N
            }
        )

        for r in r_values_N:
            subset = [
                row
                for row in subset_N
                if (
                    float(row["r"]) == r
                    and math.isfinite(
                        row["signed_z_qmc"]
                    )
                )
            ]

            for row in subset:
                rows.append(
                    {
                        "N": N,
                        "r": r,
                        "z": float(
                            row[
                                "signed_z_qmc"
                            ]
                        ),
                    }
                )

    return rows


# =============================================================================
# Main simulation
# =============================================================================

def fig5a_simulation(
    config: NumericalConfig,
    *,
    r_values=None,
):
    if r_values is None:
        r_values = make_r_grid()

    raw_rows = []

    for N in (
        2,
        3,
        4,
    ):
        aspect_sets = (
            semiaxis_families(
                N
            )
        )

        for r_index, r in enumerate(
            r_values,
            start=1,
        ):
            r = float(r)

            for (
                aspect_index,
                (
                    aspect_name,
                    semiaxes,
                ),
            ) in enumerate(
                aspect_sets,
                start=1,
            ):
                aspect_ratio = float(
                    np.max(semiaxes)
                    / np.min(semiaxes)
                )

                semiaxis_product = float(
                    np.prod(semiaxes)
                )

                for (
                    mapping_index,
                    mapping_case,
                ) in enumerate(
                    MAPPING_CASES,
                    start=1,
                ):
                    (
                        center,
                        sigma,
                        beta_expected,
                    ) = physical_case(
                        N,
                        r,
                        semiaxes,
                        mapping_case,
                    )

                    beta_actual = (
                        sigma
                        * center
                        / semiaxes
                    )

                    beta_mapping_error = float(
                        np.max(
                            np.abs(
                                beta_actual
                                - beta_expected
                            )
                        )
                    )

                    beta_norm = float(
                        np.linalg.norm(
                            beta_actual
                        )
                    )

                    seed = case_seed(
                        N,
                        r_index,
                        aspect_index,
                        mapping_index,
                    )

                    exact = (
                        ellipsoid_orthant_affine(
                            center,
                            semiaxes,
                            sigma,
                            config=config,
                            method="auto",
                        )
                    )

                    (
                        qmc_value,
                        qmc_stderr,
                    ) = ellipsoid_orthant_qmc(
                        center,
                        semiaxes,
                        sigma,
                        power=QMC_POWER,
                        scrambles=(
                            QMC_SCRAMBLES
                        ),
                        seed=seed,
                    )

                    abs_error = abs(
                        qmc_value
                        - exact
                    )

                    rel_error = (
                        abs_error
                        / abs(exact)
                        if exact != 0.0
                        else math.nan
                    )

                    (
                        signed_z,
                        abs_z,
                    ) = standardized_discrepancy(
                        exact,
                        qmc_value,
                        qmc_stderr,
                    )

                    # Normalize out the affine Jacobian.  These columns are
                    # useful if a supplementary collapse plot is desired.
                    exact_normalized = (
                        exact
                        / semiaxis_product
                    )

                    qmc_normalized = (
                        qmc_value
                        / semiaxis_product
                    )

                    qmc_stderr_normalized = (
                        qmc_stderr
                        / semiaxis_product
                    )

                    raw_rows.append(
                        {
                            "N": N,
                            "r": r,

                            "aspect_name": (
                                aspect_name
                            ),
                            "aspect_ratio": (
                                aspect_ratio
                            ),
                            "semiaxes": (
                                semiaxes.tolist()
                            ),
                            "semiaxis_product": (
                                semiaxis_product
                            ),

                            "mapping_case": (
                                mapping_case
                            ),
                            "center": (
                                center.tolist()
                            ),
                            "sigma": (
                                sigma.tolist()
                            ),

                            "beta": (
                                beta_actual.tolist()
                            ),
                            "beta_norm": (
                                beta_norm
                            ),
                            "beta_mapping_max_abs_error": (
                                beta_mapping_error
                            ),

                            "qmc_power": (
                                QMC_POWER
                            ),
                            "qmc_scrambles": (
                                QMC_SCRAMBLES
                            ),
                            "qmc_seed": seed,

                            "exact": exact,
                            "qmc": qmc_value,
                            "qmc_stderr": (
                                qmc_stderr
                            ),

                            "abs_error": (
                                abs_error
                            ),
                            "rel_error": (
                                rel_error
                            ),

                            "signed_z_qmc": (
                                signed_z
                            ),
                            "abs_z_qmc": (
                                abs_z
                            ),
                            "error_over_qmc_stderr": (
                                abs_z
                            ),

                            "within_1sigma": (
                                bool(
                                    math.isfinite(
                                        abs_z
                                    )
                                    and abs_z <= 1.0
                                )
                            ),
                            "within_2sigma": (
                                bool(
                                    math.isfinite(
                                        abs_z
                                    )
                                    and abs_z <= 2.0
                                )
                            ),
                            "within_3sigma": (
                                bool(
                                    math.isfinite(
                                        abs_z
                                    )
                                    and abs_z <= 3.0
                                )
                            ),

                            "exact_normalized": (
                                exact_normalized
                            ),
                            "qmc_normalized": (
                                qmc_normalized
                            ),
                            "qmc_stderr_normalized": (
                                qmc_stderr_normalized
                            ),
                        }
                    )

            print(
                f"N={N}, r={r:.2f} finished."
            )

    # =========================================================================
    # Plot-ready summary: one row per r, separate N=2,3,4 columns.
    # =========================================================================
    summary_rows = []

    for r in r_values:
        r = float(r)

        row = {
            "r": r,
            "FA_max_order_allowed": FA_MAX_ORDER,
            "qmc_power": QMC_POWER,
            "qmc_scrambles": QMC_SCRAMBLES,
        }

        for N in (
            2,
            3,
            4,
        ):
            subset = [
                x
                for x in raw_rows
                if (
                    x["N"] == N
                    and x["r"] == r
                )
            ]

            abs_z_values = [
                x[
                    "abs_z_qmc"
                ]
                for x in subset
            ]

            signed_z_values = [
                x[
                    "signed_z_qmc"
                ]
                for x in subset
            ]

            abs_stats = finite_stats(
                abs_z_values
            )

            signed_stats = finite_stats(
                signed_z_values
            )

            finite_abs = np.asarray(
                [
                    x
                    for x in abs_z_values
                    if math.isfinite(x)
                ],
                dtype=float,
            )

            if finite_abs.size > 0:
                frac_1 = float(
                    np.mean(
                        finite_abs <= 1.0
                    )
                )
                frac_2 = float(
                    np.mean(
                        finite_abs <= 2.0
                    )
                )
                frac_3 = float(
                    np.mean(
                        finite_abs <= 3.0
                    )
                )
            else:
                frac_1 = math.nan
                frac_2 = math.nan
                frac_3 = math.nan

            prefix = f"N{N}"

            row[
                f"{prefix}_case_count"
            ] = len(subset)
            row[f"{prefix}_finite_z_count"] = int(finite_abs.size)

            row[
                f"{prefix}_median_abs_z"
            ] = abs_stats[
                "median"
            ]

            row[
                f"{prefix}_q25_abs_z"
            ] = abs_stats[
                "q25"
            ]

            row[
                f"{prefix}_q75_abs_z"
            ] = abs_stats[
                "q75"
            ]

            row[
                f"{prefix}_max_abs_z"
            ] = abs_stats[
                "max"
            ]

            row[
                f"{prefix}_mean_signed_z"
            ] = signed_stats[
                "mean"
            ]
            row[f"{prefix}_median_signed_z"] = signed_stats["median"]

            # Signed-z spread diagnostics.
            # These retain the individual-case variability that is lost when
            # only the mean signed residual is plotted.
            row[
                f"{prefix}_std_signed_z"
            ] = signed_stats[
                "std"
            ]

            row[
                f"{prefix}_sem_signed_z"
            ] = signed_stats[
                "sem"
            ]

            row[
                f"{prefix}_min_signed_z"
            ] = signed_stats[
                "min"
            ]

            row[
                f"{prefix}_max_signed_z"
            ] = signed_stats[
                "max"
            ]

            row[
                f"{prefix}_q25_signed_z"
            ] = signed_stats[
                "q25"
            ]

            row[
                f"{prefix}_q75_signed_z"
            ] = signed_stats[
                "q75"
            ]

            # Convenient custom-error-bar spans for plotting
            # mean_signed_z with whiskers reaching the actual minimum/maximum.
            if (
                math.isfinite(
                    signed_stats["mean"]
                )
                and math.isfinite(
                    signed_stats["max"]
                )
            ):
                errplus_to_max = (
                    signed_stats["max"]
                    - signed_stats["mean"]
                )
            else:
                errplus_to_max = math.nan

            if (
                math.isfinite(
                    signed_stats["mean"]
                )
                and math.isfinite(
                    signed_stats["min"]
                )
            ):
                errminus_to_min = (
                    signed_stats["mean"]
                    - signed_stats["min"]
                )
            else:
                errminus_to_min = math.nan

            row[
                f"{prefix}_errplus_to_max"
            ] = errplus_to_max

            row[
                f"{prefix}_errminus_to_min"
            ] = errminus_to_min

            row[
                f"{prefix}_rms_signed_z"
            ] = signed_stats[
                "rms"
            ]

            row[
                f"{prefix}_fraction_within_1sigma"
            ] = frac_1

            row[
                f"{prefix}_fraction_within_2sigma"
            ] = frac_2

            row[
                f"{prefix}_fraction_within_3sigma"
            ] = frac_3

            if finite_abs.size > 0:
                count_outside_3 = int(
                    np.sum(
                        finite_abs > 3.0
                    )
                )
                fraction_outside_3 = (
                    count_outside_3
                    / finite_abs.size
                )
            else:
                count_outside_3 = 0
                fraction_outside_3 = math.nan

            row[
                f"{prefix}_count_outside_3sigma"
            ] = count_outside_3

            row[
                f"{prefix}_fraction_outside_3sigma"
            ] = fraction_outside_3

            row[
                f"{prefix}_max_beta_mapping_error"
            ] = max(
                (
                    x[
                        "beta_mapping_max_abs_error"
                    ]
                    for x in subset
                ),
                default=math.nan,
            )

        summary_rows.append(
            row
        )

    # =========================================================================
    # Write output
    # =========================================================================
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        raw_rows,
        RESULTS
        / "fig5a_raw.csv",
    )

    write_csv(
        summary_rows,
        RESULTS
        / "fig5a_summary.csv",
    )

    # ---------------------------------------------------------------------
    # Boxplot-ready long table.
    # Columns are exactly: N, r, z.
    # ---------------------------------------------------------------------
    boxplot_rows = (
        build_boxplot_table_long(
            raw_rows
        )
    )

    write_csv(
        boxplot_rows,
        RESULTS
        / "fig5a_boxplot_long.csv",
    )

    return (
        raw_rows,
        summary_rows,
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

        aono_fourier_terms=512,
        timing_repeats=1,

        qmc_power=QMC_POWER,
        qmc_scrambles=(
            QMC_SCRAMBLES
        ),

        seed=BASE_SEED,
    )

    r_values = make_r_grid()

    (
        raw_rows,
        summary_rows,
    ) = fig5a_simulation(
        config,
        r_values=r_values,
    )

    finite_z = np.asarray(
        [
            row["abs_z_qmc"]
            for row in raw_rows
            if math.isfinite(
                row["abs_z_qmc"]
            )
        ],
        dtype=float,
    )

    mapping_errors = np.asarray(
        [
            row[
                "beta_mapping_max_abs_error"
            ]
            for row in raw_rows
        ],
        dtype=float,
    )

    print()
    print(
        "Fig. 5(a) simulation finished."
    )
    print(
        f"r points: {len(r_values)}"
    )
    print(
        f"physical test cases: {len(raw_rows)}"
    )
    print(
        f"QMC points per scramble: {2**QMC_POWER}"
    )
    print(
        f"QMC scrambles per case: {QMC_SCRAMBLES}"
    )

    if finite_z.size:
        print(
            "absolute standardized discrepancy:"
        )
        print(
            f"  median = {np.median(finite_z):.4f}"
        )
        print(
            f"  max    = {np.max(finite_z):.4f}"
        )
        print(
            f"  <=1 sigma: {np.mean(finite_z <= 1.0):.3f}"
        )
        print(
            f"  <=2 sigma: {np.mean(finite_z <= 2.0):.3f}"
        )
        print(
            f"  <=3 sigma: {np.mean(finite_z <= 3.0):.3f}"
        )

    print(
        "maximum beta mapping error: "
        f"{np.max(mapping_errors):.6e}"
    )

    print(
        f"raw CSV: {RESULTS / 'fig5a_raw.csv'}"
    )
    print(
        f"summary CSV: {RESULTS / 'fig5a_summary.csv'}"
    )
    print(
        f"boxplot CSV: {RESULTS / 'fig5a_boxplot_long.csv'}"
    )
    print(
        "boxplot columns: N, r, z"
    )


if __name__ == "__main__":
    main()
