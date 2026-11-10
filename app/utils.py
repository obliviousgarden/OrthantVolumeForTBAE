# 工具集：公用函数，误差统计，计时器
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import special


# =============================================================================
# Geometry / numerical basics
# =============================================================================

def kappa(n: int) -> float:
    # N维单位球体积公式，Eq.35
    """#Volume of the n-dimensional unit ball."""
    return math.pi ** (n / 2.0) / special.gamma(1.0 + n / 2.0)


def validate_beta(
        beta: Sequence[float],
        strict: bool = True,
) -> np.ndarray:
    """
    Convert beta to a 1D NumPy array and check ||beta||_2 < 1.
    """
    # beta的输入前验证
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1:
        raise ValueError("beta must be one-dimensional.")
    # ||beta||_2<1. Eq.28
    norm = float(np.linalg.norm(beta))
    if strict and not norm < 1.0:
        raise ValueError(f"Need ||beta||_2 < 1, got {norm:.16g}.")
    return beta


def clip_unit(x: float) -> float:
    # 截断浮点数据结构导致的ASIN的溢出
    """Clip only roundoff-level excursions before asin."""
    return float(np.clip(x, -1.0, 1.0))


def normalized_probability(U: float, n: int) -> float:
    # 归一化
    """
    P = U / kappa_n, so 0 <= P <= 1.
    This is useful when comparing different dimensions.
    """
    return U / kappa(n)


def centered_orthant_scale(n: int) -> float:
    """
    Natural volume scale of one centered orthant:

        kappa_n / 2^n.
    """
    return math.ldexp(kappa(n), -n)


def sparse_partition_reference(
        n: int,
        support: int,
) -> float:
    r"""
    Exact reference for the active-sign partition identity.

    Fix the magnitudes of ``support`` active thresholds and sum the orthant
    volumes over all 2^support sign combinations of those active thresholds.
    The active directions are then completely partitioned, while the remaining
    n-support zero-threshold coordinates still contribute one fixed orthant.

    Therefore

        sum_{epsilon in {+1,-1}^support}
            U_N(epsilon_1 a_1, ..., epsilon_s a_s, 0, ..., 0)

        = kappa_n / 2^(n-support).

    This identity is independent of the actual active magnitudes as long as
    ||beta||_2 < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    if not 0 <= support <= n:
        raise ValueError("Need 0 <= support <= n.")

    return math.ldexp(
        kappa(n),
        -(n - support),
    )


def valid_orthant_volume(
        value: float,
        n: int,
        *,
        upper_rtol: float = 1e-12,
) -> bool:
    """
    Sanity check for a strict-interior orthant volume.

    Under ||beta||_2 < 1, the intersection has strictly positive volume and
    cannot exceed the full N-ball volume.
    """
    if not math.isfinite(value):
        return False

    if not value > 0.0:
        return False

    upper = kappa(n)

    return value <= upper * (1.0 + upper_rtol)


# =============================================================================
# Error metrics
# =============================================================================

def relative_error(
        value: float,
        reference: float,
) -> float:
    """
    Relative error robust to NaN/inf and zero reference values.

    Returns
    -------
    float
        |value-reference|/|reference|.

        If either input is non-finite, returns inf.
        If reference == 0, returns 0 only when value == 0; otherwise inf.
    """
    if not (
            math.isfinite(value)
            and math.isfinite(reference)
    ):
        return math.inf

    if reference == 0.0:
        return 0.0 if value == 0.0 else math.inf

    return abs(value - reference) / abs(reference)


def orthant_scaled_error(
        value: float,
        reference: float,
        n: int,
) -> float:
    r"""
    Error normalized by the larger of the actual reference volume and the
    natural centered-orthant scale:

        eps_orth
        =
        |value-reference|
        /
        max(
            |reference|,
            kappa_n / 2^n
        ).

    This is useful in very high dimension when a tail orthant volume is much
    smaller than the already tiny centered-orthant volume.
    """
    if not (
            math.isfinite(value)
            and math.isfinite(reference)
    ):
        return math.inf

    scale = max(
        abs(reference),
        centered_orthant_scale(n),
    )

    if scale == 0.0:
        return 0.0 if value == reference else math.inf

    return abs(value - reference) / scale


def error_metrics(value: float, reference: float) -> dict:
    # 绝对误差abs和相对误差rel
    """
    Return absolute and relative errors.
    """
    abs_error = abs(value - reference)

    return {
        "abs_error": abs_error,
        "rel_error": relative_error(
            value,
            reference,
        ),
    }


def nominal_tolerance(
        reference: float,
        config,
) -> float:
    """
    Nominal absolute tolerance implied by the Lauricella settings:

        atol + rtol*|reference|.
    """
    return (
            float(config.fa_atol)
            + float(config.fa_rtol) * abs(reference)
    )


def error_over_tolerance(
        value: float,
        reference: float,
        config,
) -> float:
    """
    Absolute error divided by the nominal Lauricella tolerance.
    """
    tol = nominal_tolerance(
        reference,
        config,
    )

    if tol <= 0.0:
        return math.nan

    return abs(value - reference) / tol


def count_fa_term_types(
        N: int,
        support: int,
) -> dict:
    r"""
    Count subset terms by residual-dimension parity.

    For a subset of size m,

        d = N-m.

    If d is even, the first Lauricella parameter -d/2 is a non-positive
    integer and the F_A series terminates.  If d is odd, it is a negative
    half-integer and the defining series is nonterminating.

    The centered empty-subset term is reported separately through
    ``subset_terms_expected = 2**support``.
    """
    if not 0 <= support <= N:
        raise ValueError("Need 0 <= support <= N.")

    terminating = 0
    nonterminating = 0

    for m in range(1, support + 1):
        number = math.comb(
            support,
            m,
        )

        d = N - m

        if d % 2 == 0:
            terminating += number
        else:
            nonterminating += number

    total = (
            terminating
            + nonterminating
    )

    return {
        "terminating_FA_terms": terminating,
        "nonterminating_FA_terms": nonterminating,
        "FA_terms_total": total,
        "subset_terms_expected": 2 ** support,
        "terminating_fraction": (
            terminating / total
            if total > 0
            else math.nan
        ),
    }


def estimated_digits_exposed(
        cancellation: float,
) -> float:
    r"""
    Estimate the number of decimal digits exposed to subtractive cancellation:

        max(0, log10(chi)),

    where chi is the cancellation indicator.
    """
    if cancellation <= 0.0:
        return math.inf

    if not math.isfinite(cancellation):
        return math.inf

    return max(
        0.0,
        math.log10(cancellation),
    )


# =============================================================================
# Timing
# =============================================================================

def timed(
        func: Callable,
        *args,
        repeats: int = 5,
        **kwargs,
):
    # 计时器
    """
    Evaluate func several times and return

        (median_wall_time, final_result).

    No warm-up call is made.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    times = []
    result = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return float(np.median(times)), result


def timed_warm(
        func: Callable,
        *args,
        repeats: int = 9,
        warmup: int = 1,
        min_batch_time: float = 0.02,
        max_batch_size: int = 100000,
        **kwargs,
):
    """
    Warm-up + adaptive batching + median timing.

    Very fast Hybrid evaluations can take only a few tens of microseconds.
    Timing a single call is then strongly affected by scheduler/timer noise.
    This routine automatically repeats the function enough times that each
    timing batch lasts approximately ``min_batch_time`` seconds, and reports
    the median per-call time.

    The return signature remains

        (median_time_per_call, final_result)

    so it is compatible with the previous runner.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    if warmup < 0:
        raise ValueError("warmup must be >= 0.")

    for _ in range(warmup):
        func(*args, **kwargs)

    # Pilot call.
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    pilot = time.perf_counter() - t0

    if min_batch_time <= 0.0:
        batch_size = 1
    else:
        safe_pilot = max(pilot, 1e-9)
        batch_size = int(
            math.ceil(min_batch_time / safe_pilot)
        )
        batch_size = max(
            1,
            min(batch_size, max_batch_size),
        )

    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()

        for _ in range(batch_size):
            result = func(*args, **kwargs)

        elapsed = time.perf_counter() - t0
        times.append(elapsed / batch_size)

    return float(np.median(times)), result


# =============================================================================
# Beta-pattern helper shared by final simulations
# =============================================================================
def make_sparse_beta(
        N: int,
        r: float,
        support: int,
        *,
        mixed: bool = False,
) -> np.ndarray:
    """
    Construct an equal-amplitude sparse beta vector with prescribed norm r.

    Parameters
    ----------
    N:
        Ambient dimension.

    r:
        Desired Euclidean norm ||beta||_2. Need 0 <= r < 1.

    support:
        Number of nonzero components.

    mixed:
        If True, alternate signs among active coordinates:
            + - + - ...

    Returns
    -------
    numpy.ndarray
        beta with exactly ``support`` active coordinates and ||beta||_2 = r.
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    if not 0.0 <= r < 1.0:
        raise ValueError("Need 0 <= r < 1.")

    if not 1 <= support <= N:
        raise ValueError("Need 1 <= support <= N.")

    beta = np.zeros(N, dtype=float)
    beta[:support] = r / math.sqrt(support)

    if mixed:
        beta[1:support:2] *= -1.0

    return beta


# =============================================================================
# CSV output
# =============================================================================

def write_csv(
        rows: list[dict],
        filename: str | Path,
) -> None:
    # 结果写入CSV
    """
    Write a list of dictionaries to a CSV file.

    The union of all dictionary keys is used as the CSV header, so rows from
    different validation/comparison branches may contain different fields.

    Parameters
    ----------
    rows:
        List of dictionaries containing numerical results.

    filename:
        Output CSV path. Parent directories are created automatically.
    """
    if not rows:
        return

    filename = Path(filename)
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve the order in which keys first appear.
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with filename.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
