# Benchmark with err, runtime, and export as CSV
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aono import aono_erf_orthant, aono_fourier_orthant
from config import DEFAULT, NumericalConfig
from elementary import unit_ball_orthant_N1, unit_ball_orthant_N2, unit_ball_orthant_N3
from lauricella import lauricella_FA_half_parameters, unit_ball_orthant_lauricella
from n4 import J4_direct_triple, J4_reduced_1D, unit_ball_orthant_N4_1D
from patterns import make_beta
from qmc import unit_ball_orthant_qmc
from utils import error_metrics, timed, validate_beta

def validate_low_dimensions(
    *,
    n_random: int = 20,
    r_max: float = 0.90,
    config: NumericalConfig = DEFAULT,
    seed: int = 12345,
) -> list[dict]:
    # Benchmark 1:
    """
    Compare:
      N=1,2,3 elementary formula
      vs
      general Lauricella formula.

    This is the first validation stage.
    """
    rng = np.random.default_rng(seed)
    rows = []

    elementary = {
        1: lambda b: unit_ball_orthant_N1(b),
        2: unit_ball_orthant_N2,
        3: unit_ball_orthant_N3,
    }

    for N in (1, 2, 3):
        for case in range(n_random):
            direction = rng.normal(size=N)
            direction /= np.linalg.norm(direction)
            r = r_max * rng.random()
            beta = r * direction

            ref = elementary[N](beta)
            val, info = unit_ball_orthant_lauricella(
                beta,
                config=config,
                return_info=True,
            )

            err = error_metrics(val, ref)

            rows.append({
                "stage": "low_dim",
                "N": N,
                "case": case,
                "r": r,
                "support": info["support"],
                "elementary": ref,
                "lauricella": val,
                **err,
                "FA_order": info["max_FA_order"],
                "FA_converged": info["all_FA_converged"],
                "cancellation": info["cancellation_indicator"],
            })

    return rows


def validate_N4_J(
    triples: Iterable[Sequence[float]],
    *,
    config: NumericalConfig = DEFAULT,
) -> list[dict]:
    """
    Compare N=4 three-coordinate J:
      direct 3D quadrature
      reduced 1D form
      Lauricella F_A^{(3)} through the general evaluator.
    """
    rows = []

    for case, triple in enumerate(triples):
        a, b, c = map(float, triple)
        if a*a + b*b + c*c >= 1.0:
            raise ValueError("Each N=4 triple must satisfy a^2+b^2+c^2<1.")

        t3, J3 = timed(
            J4_direct_triple,
            a, b, c,
            config=config,
            repeats=config.timing_repeats,
        )

        t1, J1 = timed(
            J4_reduced_1D,
            a, b, c,
            config=config,
            repeats=config.timing_repeats,
        )

        tfa, FA_pair = timed(
            lauricella_FA_half_parameters,
            -0.5,
            [a*a, b*b, c*c],
            rtol=config.fa_rtol,
            atol=config.fa_atol,
            initial_order=config.fa_initial_order,
            max_order=config.fa_max_order,
            return_info=True,
            repeats=config.timing_repeats,
        )
        FA_value, FA_info = FA_pair
        Jfa = a * b * c * FA_value

        rows.append({
            "stage": "N4_J",
            "case": case,
            "a": a,
            "b": b,
            "c": c,
            "r3": math.sqrt(a*a+b*b+c*c),
            "J_3D": J3,
            "J_1D": J1,
            "J_FA": Jfa,
            "err_1D_vs_3D": abs(J1-J3),
            "err_FA_vs_3D": abs(Jfa-J3),
            "time_3D_s": t3,
            "time_1D_s": t1,
            "time_FA_s": tfa,
            "speedup_1D_vs_3D": t3/t1 if t1 else math.inf,
            "FA_order": FA_info["order"],
            "FA_converged": FA_info["converged"],
        })

    return rows


def benchmark_case(
    beta: Sequence[float],
    *,
    config: NumericalConfig = DEFAULT,
    qmc_reference: bool = False,
) -> dict:
    """
    Compare for one beta:
      proposed general Lauricella
      Aono Fourier--Fresnel truncation
      Aono erf leading approximation
      optional QMC reference.

    For N <= 4 an elementary/reduced exact reference is also used.
    """
    beta = validate_beta(beta)
    N = beta.size

    # Preferred exact low-dimensional reference.
    if N == 1:
        reference = unit_ball_orthant_N1(beta)
        reference_name = "N1_elementary"
    elif N == 2:
        reference = unit_ball_orthant_N2(beta)
        reference_name = "N2_elementary"
    elif N == 3:
        reference = unit_ball_orthant_N3(beta)
        reference_name = "N3_elementary"
    elif N == 4:
        reference = unit_ball_orthant_N4_1D(beta, config=config)
        reference_name = "N4_1D"
    else:
        reference = math.nan
        reference_name = "none"

    t_prop, prop_pair = timed(
        unit_ball_orthant_lauricella,
        beta,
        config=config,
        return_info=True,
        repeats=config.timing_repeats,
    )
    proposed, prop_info = prop_pair

    t_aono, aono = timed(
        aono_fourier_orthant,
        beta,
        n_terms=config.aono_terms,
        repeats=config.timing_repeats,
    )

    t_erf, erf = timed(
        aono_erf_orthant,
        beta,
        repeats=config.timing_repeats,
    )

    row = {
        "N": N,
        "r": float(np.linalg.norm(beta)),
        "support": prop_info["support"],
        "beta": beta.tolist(),
        "reference_name": reference_name,
        "reference": reference,
        "proposed": proposed,
        "aono_fourier": aono,
        "aono_erf": erf,
        "time_proposed_s": t_prop,
        "time_aono_fourier_s": t_aono,
        "time_aono_erf_s": t_erf,
        "FA_order": prop_info["max_FA_order"],
        "FA_converged": prop_info["all_FA_converged"],
        "cancellation": prop_info["cancellation_indicator"],
        "aono_terms": config.aono_terms,
    }

    if not math.isnan(reference):
        row.update({
            "prop_abs_error": abs(proposed-reference),
            "aono_abs_error": abs(aono-reference),
            "erf_abs_error": abs(erf-reference),
            "prop_rel_error": abs(proposed-reference)/abs(reference)
                if reference else math.nan,
            "aono_rel_error": abs(aono-reference)/abs(reference)
                if reference else math.nan,
            "erf_rel_error": abs(erf-reference)/abs(reference)
                if reference else math.nan,
        })

    if qmc_reference:
        qmc_mean, qmc_stderr = unit_ball_orthant_qmc(
            beta,
            power=config.qmc_power,
            scrambles=config.qmc_scrambles,
            seed=config.seed,
        )
        row.update({
            "qmc": qmc_mean,
            "qmc_stderr": qmc_stderr,
            "prop_minus_qmc": proposed-qmc_mean,
            "aono_minus_qmc": aono-qmc_mean,
            "erf_minus_qmc": erf-qmc_mean,
        })

    return row


def benchmark_grid(
    N_values: Sequence[int],
    r_values: Sequence[float],
    *,
    kinds: Sequence[str] = ("equal", "mixed_equal", "anisotropic"),
    support_values: Sequence[int] | None = None,
    config: NumericalConfig = DEFAULT,
    include_qmc_for_high_N: bool = False,
) -> list[dict]:
    """
    Structured benchmark grid.

    Recommended:
      N_values = [2,3,4,5,6,8,10,12]
      r_values = [0.25,0.50,0.75,0.90,0.97]
      kinds = ["equal","mixed_equal","anisotropic","sparse"]

    Warning:
      The general subset formula scales exponentially with effective support.
      Start with N <= 8 before extending to N >= 10.
    """
    rows = []
    rng = np.random.default_rng(config.seed)

    for N in N_values:
        for r in r_values:
            for kind in kinds:
                if kind in {"sparse", "mixed_sparse"}:
                    supports = (
                        support_values
                        if support_values is not None
                        else [min(1, N), min(2, N), min(4, N), N]
                    )
                else:
                    supports = [N]

                for support in supports:
                    if support > N:
                        continue

                    beta = make_beta(
                        N,
                        r,
                        kind=kind,
                        support=support,
                        rng=rng,
                    )

                    row = benchmark_case(
                        beta,
                        config=config,
                        qmc_reference=(
                            include_qmc_for_high_N
                            and N > 4
                        ),
                    )
                    row["kind"] = kind
                    row["requested_support"] = support
                    rows.append(row)

    return rows


def write_csv(rows: list[dict], filename: str | Path) -> None:
    """
    Write a list of dictionaries to CSV.
    beta arrays are serialized as Python-list strings.
    """
    if not rows:
        return

    filename = Path(filename)

    # Union of all keys.
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

