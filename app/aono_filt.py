# Aono's exact Laplace-erf transform + FILT inversion
# FIXME: 暂时不使用这个方法,
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special

from utils import validate_beta


def aono_laplace_pdf_transform(
    s: complex,
    lower: Sequence[float],
    upper: Sequence[float],
) -> complex:
    r"""
    Aono--Nguyen Eq. (9):

      L[rho_{sum X_i^2}](s)
      = (pi/s)^(N/2)
        prod_i
        [erf(b_i sqrt(s))-erf(a_i sqrt(s))]
        / [2(b_i-a_i)].

    This is the exact transformed function. It is not yet the volume.
    """
    lower = np.asarray(lower,float)
    upper = np.asarray(upper,float)
    widths = upper-lower

    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower/upper must be same-length 1D arrays.")
    if np.any(widths <= 0):
        raise ValueError("Need upper_i>lower_i.")

    N = lower.size
    s = complex(s)
    sq = np.sqrt(s)

    ratio = (
        special.erf(upper*sq)
        - special.erf(lower*sq)
    ) / (2*widths)

    return (math.pi/s)**(N/2) * np.prod(ratio)


def aono_laplace_cdf_transform(
    s: complex,
    lower: Sequence[float],
    upper: Sequence[float],
) -> complex:
    r"""
    Transform whose inverse at t=1 gives

      Pr(sum X_i^2 <= 1).

    Since a CDF contributes an additional 1/s:
      F_CDF(s) = L[rho_sum](s)/s.
    """
    return aono_laplace_pdf_transform(s, lower, upper) / s


def aono_filt_probability(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    k: int = 30,
    J: int = 30,
    gamma1: float = 20.0,
    t: float = 1.0,
) -> float:
    r"""
    Hosono/FILT approximation corresponding to Aono--Nguyen Eq. (11),
    with the Euler/van-Wijngaarden acceleration described immediately after it.

    The exact inverse is the Bromwich integral in Eq. (10). Numerically,
    the paper evaluates it through FILT rather than direct contour quadrature.

    Parameters
    ----------
    k:
        Number of explicit alternating-series terms kept before Euler acceleration.
    J:
        Number of Euler-transformed tail levels.
    gamma1:
        Shift parameter in Hosono's inversion.
        For float64, values around 15--25 are usually numerically safer.
        Aono--Nguyen used much higher precision and the heuristic
        gamma1=max(50,30+3*sqrt(N)) in their large-dimensional experiments.
    """
    lower = np.asarray(lower,float)
    upper = np.asarray(upper,float)

    if k < 1 or J < 1:
        raise ValueError("k and J must be positive.")

    # Need F_m through m=k+J.
    m = np.arange(1,k+J+1,dtype=float)
    s = (
        gamma1
        + (m-0.5)*math.pi*1j
    ) / t

    Fm = np.array([
        np.imag(
            aono_laplace_cdf_transform(
                z, lower, upper
            )
        )
        for z in s
    ],dtype=float)

    # First k terms of sum (-1)^m F_m.
    initial = float(np.sum(
        ((-1.0)**m[:k]) * Fm[:k]
    ))

    # Euler transform of the tail:
    # sum_{j=1}^J (-1)^j Delta^{j-1}F_{k+1}/2^j.
    work = Fm[k:].copy()
    tail = 0.0

    for j in range(1,J+1):
        tail += ((-1.0)**j) * work[0] / (2.0**j)
        if len(work) == 1:
            break
        work = np.diff(work)

    alternating_sum = (
        initial
        + ((-1.0)**k)*tail
    )

    return math.exp(gamma1)/t * alternating_sum


def aono_filt_volume(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    k: int = 30,
    J: int = 30,
    gamma1: float = 20.0,
) -> float:
    """
    Ball-box intersection volume from FILT:
      volume = box_volume * Pr(sum X_i^2<=1).
    """
    lower = np.asarray(lower,float)
    upper = np.asarray(upper,float)

    p = aono_filt_probability(
        lower,upper,k=k,J=J,gamma1=gamma1,t=1.0
    )
    return float(np.prod(upper-lower))*p


def aono_filt_orthant(
    beta,
    *,
    k: int = 30,
    J: int = 30,
    gamma1: float = 20.0,
) -> float:
    beta = validate_beta(beta)
    return aono_filt_volume(
        -beta,
        np.ones_like(beta),
        k=k,
        J=J,
        gamma1=gamma1,
    )
