from __future__ import annotations

"""Figure 6(b): dense-in-N validation of Route(U/T/G), V21.6.

Run in the same directory as the V21.2 core modules:
    python simulation_fig6_b.py

Produces (overwriting the previous Figure 6(b) parity-only tables):
    results/fig6b_raw.csv
    results/fig6b_plot.csv
    results/fig6b_summary.csv

Compare the unreduced subset--Lauricella evaluator and the final Route(U/T/G)
Hybrid at identical numerical settings.  A common high-precision *independent*
reference is not available for the full grid: discrepancies are agreement checks,
not certified error bounds.  The runtime ratio is reported only if both methods
converged, the volumes and structural counts are valid, and the discrepancy is
within COMPARISON_REL_TOL.

The original Figure 6(a) script is not modified.

The case N=5, s=2, r=0.50 can have S_Route<1: this is not filtered
out or repaired. The raw evaluator has only one nonterminating term
with modest K, whereas the stable s=2 Route(U) performs quadrature.
New diagnostic flags identify this case and every *measured* slowdown;
none of them changes the numerical algorithm, timing protocol, or plotted values.

Figure 6(b1) and 6(b2) plot the equal-active family at r=0.50 and r=0.90,
respectively, with all five support values. Full integer coverage N=4,...,24
resolves the parity-alternating route costs instead of connecting sparse
high-dimensional even-N points. For the main panels, N>=5 is recommended
because N=4 uses the special full-volume Route(U).
"""

import math
from pathlib import Path

import numpy as np

from config import NumericalConfig
from lauricella import unit_ball_orthant_lauricella
from proposed_hybrid import analytic_structure_counts, unit_ball_orthant_hybrid
from utils import count_fa_term_types, timed_warm, valid_orthant_volume, write_csv

RESULTS = Path(__file__).resolve().parent / "results"

# Same FA numerical settings as Figure 6(a) and the earlier Fig. 6(b).
FA_RTOL = 1e-12
FA_ATOL = 1e-14
FA_INITIAL_ORDER = 8
FA_MAX_ORDER = 16384
COMPARISON_REL_TOL = 1e-10

# Same repeated-timing protocol as the earlier Figure 6(b).
TIMING_REPEATS = 11
TIMING_WARMUP = 2
MIN_BATCH_TIME = 0.05

# Dense consecutive N makes the high-dimensional odd/even alternation visible.
# Do not add only odd N to sparse even samples: use all integers N=4,...,24.
# Figure 6(b1): r=0.50, equal-active; Figure 6(b2): r=0.90, equal-active.
# Five supports are plotted in both panels; N=4 is a Route(U) special case.
DIMENSIONS = tuple(range(4, 25))
SUPPORTS = (1, 2, 3, 4, 6)
R_VALUES = (0.50, 0.90)
KINDS = ("equal", "mixed_equal")


def make_beta(N: int, support: int, r: float, kind: str) -> np.ndarray:
    if not 1 <= support <= N:
        raise ValueError("Require 1 <= support <= N")
    if not 0 <= r < 1:
        raise ValueError("Require 0 <= r < 1")
    beta = np.zeros(N, dtype=float)
    beta[:support] = r / math.sqrt(support)
    if kind == "mixed_equal":
        beta[1:support:2] *= -1.0
    elif kind != "equal":
        raise ValueError(f"Unsupported kind: {kind}")
    return beta


def route_counts(N: int, support: int) -> dict:
    """Separate nominal subset decomposition from actually executed Route(U)."""
    raw = count_fa_term_types(N, support)
    final = analytic_structure_counts(N, support)
    direct_U = N == 4 or support <= 2
    raw_NT = int(raw["nonterminating_FA_terms"])
    retained = 0 if direct_U else int(final["nu_FA"])
    return {
        "raw_terminating_FA_terms": int(raw["terminating_FA_terms"]),
        "raw_nonterminating_FA_terms": raw_NT,
        "nominal_route_FA_terms": int(final["nu_FA"]),
        "nominal_route_Psi_terms": int(final["elementary_Psi_terms"]),
        "nominal_route_J_terms": int(final["J_terms"]),
        "expected_executed_FA_terms": retained,
        "removed_nonterminating_FA_terms": raw_NT - retained,
        "FA_removal_fraction": (1.0 - retained / raw_NT if raw_NT else math.nan),
        "route_U_direct": bool(direct_U),
    }


def rel_diff(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    if b == 0:
        return 0.0 if a == 0 else math.inf
    return abs(a - b) / abs(b)


def evaluate_case(N: int, support: int, r: float, kind: str,
                  config: NumericalConfig) -> dict:
    beta = make_beta(N, support, r, kind)
    theoretical = route_counts(N, support)
    record = {
        "N": N,
        "N_parity": "even" if N % 2 == 0 else "odd",
        "support": support, "r": r, "kind": kind,
        "beta": beta.tolist(), "beta_norm": float(np.linalg.norm(beta)),
        "expected_subset_terms": 2 ** support,
        **theoretical,
        "FA_rtol": config.fa_rtol, "FA_atol": config.fa_atol,
        "FA_initial_order": config.fa_initial_order,
        "FA_max_order_allowed": config.fa_max_order,
        "comparison_rel_tol": COMPARISON_REL_TOL,
        "timing_repeats": TIMING_REPEATS,
        "timing_warmup": TIMING_WARMUP,
        "is_N5_s2_case": bool(N == 5 and support == 2),
    }
    try:
        t_raw, (raw_value, raw_info) = timed_warm(
            unit_ball_orthant_lauricella,
            beta, config=config, return_info=True,
            repeats=TIMING_REPEATS, warmup=TIMING_WARMUP,
            min_batch_time=MIN_BATCH_TIME,
        )
        t_hybrid, (hybrid_value, hybrid_info) = timed_warm(
            unit_ball_orthant_hybrid,
            beta, config=config, return_info=True,
            repeats=TIMING_REPEATS, warmup=TIMING_WARMUP,
            min_batch_time=MIN_BATCH_TIME,
        )
        raw_value, hybrid_value = float(raw_value), float(hybrid_value)
        discrepancy = rel_diff(hybrid_value, raw_value)
        raw_converged = bool(raw_info.get("all_FA_converged", False))
        hybrid_converged = bool(hybrid_info.get("all_FA_converged", False))
        valid = (valid_orthant_volume(raw_value, N) and
                 valid_orthant_volume(hybrid_value, N))
        fa_actual = int(hybrid_info["evaluated_generic_FA_terms"])
        psi_actual = int(hybrid_info["evaluated_Psi_terms"])
        j_actual = int(hybrid_info["evaluated_J_terms"])
        raw_subset_actual = int(raw_info["subset_terms_evaluated"])
        counts_ok = (raw_subset_actual == 2 ** support and
                     fa_actual == theoretical["expected_executed_FA_terms"])
        if not theoretical["route_U_direct"]:
            counts_ok = counts_ok and (psi_actual == theoretical["nominal_route_Psi_terms"]
                                       and j_actual == theoretical["nominal_route_J_terms"])
        # N=4 executes direct Route(U). J counts may be recorded there, but are
        # deliberately not compared to the nominal subset-Route(T) counts.
        if not valid:
            status = "invalid_volume"
        elif not raw_converged:
            status = "raw_FA_not_converged"
        elif not hybrid_converged:
            status = "hybrid_FA_not_converged"
        elif not counts_ok:
            status = "route_count_mismatch"
        elif discrepancy > COMPARISON_REL_TOL:
            status = "volume_disagreement"
        else:
            status = "matched"
        ready = status == "matched"
        speedup = t_raw / t_hybrid if ready and t_hybrid > 0 else math.nan
        hybrid_over_raw = t_hybrid / t_raw if ready and t_raw > 0 else math.nan
        # A slowdown is an observed timing result, not a failed Route check.
        # In particular, DO NOT remove N=5, s=2, r=0.5 from the plot.
        slowdown = bool(math.isfinite(speedup) and speedup < 1.0)
        record.update({
            "raw_value": raw_value, "hybrid_value": hybrid_value,
            "relative_discrepancy_to_raw_FA": discrepancy,
            "raw_time_s": float(t_raw), "hybrid_time_s": float(t_hybrid),
            "raw_time_ms": 1000.0 * t_raw, "hybrid_time_ms": 1000.0 * t_hybrid,
            "speedup_raw_over_hybrid": speedup,
            "hybrid_over_raw_time_ratio": hybrid_over_raw,
            "is_measured_slowdown": slowdown,
            "raw_FA_converged": raw_converged,
            "hybrid_FA_converged": hybrid_converged,
            "raw_max_FA_order": int(raw_info.get("max_FA_order", 0)),
            "hybrid_max_FA_order": int(hybrid_info.get("max_FA_order", 0)),
            "raw_subset_terms_executed": raw_subset_actual,
            "route": str(hybrid_info["route"]),
            "executed_Psi_terms": psi_actual,
            "executed_J_terms": j_actual,
            "executed_generic_FA_terms": fa_actual,
            "executed_terminating_terms": int(hybrid_info["evaluated_terminating_terms"]),
            "structural_counts_match": counts_ok,
            "status": status, "ready_for_speedup": ready,
            "error_message": "",
        })
    except Exception as exc:
        record.update({
            "status": "exception", "ready_for_speedup": False,
            "raw_value": math.nan, "hybrid_value": math.nan,
            "relative_discrepancy_to_raw_FA": math.nan,
            "raw_time_s": math.nan, "hybrid_time_s": math.nan,
            "raw_time_ms": math.nan, "hybrid_time_ms": math.nan,
            "speedup_raw_over_hybrid": math.nan,
            "hybrid_over_raw_time_ratio": math.nan,
            "is_measured_slowdown": False,
            "error_message": f"{type(exc).__name__}: {exc}",
        })
    return record


def build_plot_table(rows: list[dict]) -> list[dict]:
    """Origin data: filter plot_role='main_r05' or 'main_r09', N>=5."""
    cols = [
        "N", "support", "r", "kind", "N_parity", "plot_role",
        "is_N5_s2_case", "is_measured_slowdown",
        "raw_nonterminating_FA_terms", "expected_executed_FA_terms",
        "removed_nonterminating_FA_terms", "FA_removal_fraction",
        "nominal_route_Psi_terms", "nominal_route_J_terms",
        "executed_generic_FA_terms", "executed_Psi_terms", "executed_J_terms",
        "raw_time_ms", "hybrid_time_ms", "speedup_raw_over_hybrid",
        "hybrid_over_raw_time_ratio",
        "relative_discrepancy_to_raw_FA", "raw_max_FA_order",
        "hybrid_max_FA_order", "route", "route_U_direct", "status",
        "ready_for_speedup",
    ]
    result = []
    for row in rows:
        if row["kind"] == "equal" and row["N"] == 4:
            role = "special_N4"
        elif row["kind"] == "equal":
            role = "main_r05" if row["r"] == 0.5 else "main_r09"
        else:
            role = "supplementary_mixed"
        enriched = {**row, "plot_role": role}
        result.append({key: enriched.get(key, math.nan) for key in cols})
    return result


def build_summary(rows: list[dict]) -> list[dict]:
    """Group by fixed r, support, parity and displacement family."""
    summary = []
    for r in R_VALUES:
        for s in SUPPORTS:
            for parity in ("even", "odd"):
                for kind in KINDS:
                    group = [row for row in rows if row["r"] == r
                             and row["support"] == s
                             and row["N_parity"] == parity
                             and row["kind"] == kind]
                    if not group:
                        continue
                    matched = [row for row in group if row["ready_for_speedup"]]
                    speeds = [row["speedup_raw_over_hybrid"] for row in matched
                              if math.isfinite(row["speedup_raw_over_hybrid"])]
                    errs = [row["relative_discrepancy_to_raw_FA"] for row in group
                            if math.isfinite(row.get("relative_discrepancy_to_raw_FA", math.nan))]
                    summary.append({
                        "r": r, "support": s, "N_parity": parity, "kind": kind,
                        "N_min": min(row["N"] for row in group),
                        "N_max": max(row["N"] for row in group),
                        "n_cases": len(group), "n_matched": len(matched),
                        "n_censored": len(group)-len(matched),
                        "n_measured_slowdown": sum(
                            bool(row.get("is_measured_slowdown", False)) for row in matched
                        ),
                        "min_speedup_raw_over_hybrid": min(speeds) if speeds else math.nan,
                        "max_speedup_raw_over_hybrid": max(speeds) if speeds else math.nan,
                        "raw_nonterminating_FA_terms": group[0]["raw_nonterminating_FA_terms"],
                        "hybrid_expected_generic_FA_terms": group[0]["expected_executed_FA_terms"],
                        "removed_nonterminating_FA_terms": group[0]["removed_nonterminating_FA_terms"],
                        "max_relative_discrepancy": max(errs) if errs else math.nan,
                        "median_speedup_raw_over_hybrid": (
                            float(np.median(speeds)) if speeds else math.nan),
                        "status": "complete" if len(matched)==len(group) else "partial",
                    })
    return summary


def main():
    config = NumericalConfig(
        fa_rtol=FA_RTOL, fa_atol=FA_ATOL,
        fa_initial_order=FA_INITIAL_ORDER, fa_max_order=FA_MAX_ORDER,
    )
    rows = []
    for r in R_VALUES:
        for N in DIMENSIONS:
            for support in SUPPORTS:
                if support > N:
                    continue
                for kind in KINDS:
                    if support == 1 and kind != "equal":
                        # With one active coordinate, alternating signs are identical.
                        continue
                    row = evaluate_case(N, support, r, kind, config)
                    rows.append(row)
                    print(
                        f"N={N:2d} s={support} r={r:.2f} {kind:11s} "
                        f"{row['status']:22s} "
                        f"err={row.get('relative_discrepancy_to_raw_FA', math.nan):.3e} "
                        f"S={row.get('speedup_raw_over_hybrid', math.nan):.3g}",
                        flush=True,
                    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(rows, RESULTS / "fig6b_raw.csv")
    write_csv(build_plot_table(rows), RESULTS / "fig6b_plot.csv")
    write_csv(build_summary(rows), RESULTS / "fig6b_summary.csv")
    n_ok = sum(bool(row["ready_for_speedup"]) for row in rows)
    print(f"\nFigure 6(b) Route comparison: {n_ok}/{len(rows)} matched. ")
    print("Main plots: r=0.50 or 0.90, kind=equal, N>=5, all supports")
    print("\nN=5, s=2 special comparison (retain slowdowns; no fallback):")
    for row in rows:
        if row["is_N5_s2_case"]:
            print(
                f"  r={row['r']:.2f}, kind={row['kind']:11s}, "
                f"K_raw={row.get('raw_max_FA_order', math.nan)}, "
                f"t_raw={row.get('raw_time_ms', math.nan):.4f} ms, "
                f"t_Hybrid={row.get('hybrid_time_ms', math.nan):.4f} ms, "
                f"S={row.get('speedup_raw_over_hybrid', math.nan):.4f}, "
                f"slowdown={row.get('is_measured_slowdown', False)}, "
                f"status={row['status']}"
            )
    print("Outputs:")
    for name in ("fig6b_raw.csv", "fig6b_plot.csv", "fig6b_summary.csv"):
        print(" ", RESULTS / name)


if __name__ == "__main__":
    main()
