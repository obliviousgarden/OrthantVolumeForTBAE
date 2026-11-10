# Aono's Fourier/fresnel function with an erf leading approximation
# TODO: Eq.9 对应ERF的精确方程不用么？
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special

from utils import kappa, validate_beta

def _aono_fresnel_CS(x: np.ndarray | float):
    # Aono's Fresnel Function C and S
    r"""
    Aono convention:
      C_A(x) = int_0^x cos(t^2) dt,
      S_A(x) = int_0^x sin(t^2) dt.

    SciPy convention:
      C_S(z) = int_0^z cos(pi t^2/2) dt,
      S_S(z) = int_0^z sin(pi t^2/2) dt.

    Conversion:
      C_A(x) = sqrt(pi/2) C_S(x sqrt(2/pi)),
      S_A(x) = sqrt(pi/2) S_S(x sqrt(2/pi)).
    """
    z = np.asarray(x) * math.sqrt(2.0 / math.pi)
    S_scipy, C_scipy = special.fresnel(z)
    scale = math.sqrt(math.pi / 2.0)
    return scale * C_scipy, scale * S_scipy


def aono_fourier_fresnel(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    n_terms: int = 256,
) -> float:
    # Eq.7
    r"""
    Aono--Nguyen Fourier-series / Fresnel representation for

      Vol(Ball_N(1) intersection prod_j [a_j,b_j]).

    Let
      ell = sum_j max(a_j^2,b_j^2).

    If ell <= 1, the entire box is inside the unit ball:
      V = prod_j (b_j-a_j).

    If ell > 1:
      V = prod_j(b_j-a_j) *
          [
            1/2
            - sum_j(a_j^2+b_j^2+a_j b_j)/(3 ell)
            + 1/ell
            + (1/pi) Im sum_{k>=1}
                Phi(-2 pi k/ell)/k * exp(2 pi i k/ell)
          ],

    where
      Phi(omega)
      = prod_j
          {
            [C(b_j sqrt|omega|)-C(a_j sqrt|omega|)]
            + i sgn(omega)
              [S(b_j sqrt|omega|)-S(a_j sqrt|omega|)]
          }
          / [(b_j-a_j)sqrt|omega|].

    The exact formula is an infinite series.
    n_terms controls the numerical truncation.

    For the orthant problem:
      a_i = -beta_i,
      b_i = 1,
      ell = N  (because |beta_i|<1).
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower and upper must be same-length 1D arrays.")
    if np.any(upper <= lower):
        raise ValueError("Need upper_i > lower_i.")
    if n_terms < 1:
        raise ValueError("n_terms must be positive.")

    widths = upper - lower
    box_volume = float(np.prod(widths))

    ell = float(np.sum(np.maximum(lower**2, upper**2)))

    if ell <= 1.0 + 1e-15:
        return box_volume

    base = (
        0.5
        - float(np.sum(lower**2 + upper**2 + lower * upper))
        / (3.0 * ell)
        + 1.0 / ell
    )

    # Vectorized k = 1,...,n_terms.
    k = np.arange(1, n_terms + 1, dtype=float)
    omega = -2.0 * math.pi * k / ell
    root = np.sqrt(np.abs(omega))

    # Shape: (K, N)
    arg_upper = root[:, None] * upper[None, :]
    arg_lower = root[:, None] * lower[None, :]

    Cb, Sb = _aono_fresnel_CS(arg_upper)
    Ca, Sa = _aono_fresnel_CS(arg_lower)

    numerator = (
        (Cb - Ca)
        + 1j * np.sign(omega)[:, None] * (Sb - Sa)
    )
    denominator = widths[None, :] * root[:, None]
    # Function Phi in Aono's paper, Function F as Eq.8
    Phi = np.prod(numerator / denominator, axis=1)

    series = np.sum(
        Phi / k
        * np.exp(2j * math.pi * k / ell)
    )

    return box_volume * (
        base
        + series.imag / math.pi
    )


def aono_fourier_orthant(
    beta: Sequence[float],
    *,
    n_terms: int = 256,
) -> float:
    # Orthant Volume using Eq.7
    """
    Aono Fourier benchmark for the same orthant geometry.

    Mapping:
      lower_i = -beta_i,
      upper_i = 1.
    """
    beta = validate_beta(beta)
    return aono_fourier_fresnel(
        -beta,
        np.ones_like(beta),
        n_terms=n_terms,
    )


def aono_erf_leading(
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    # Eq.10
    r"""
    First (fast) term of the Aono--Nguyen Laguerre/erf representation:

      V_erf^(0)
      = kappa_N / 2^N
        prod_j [
          erf(b_j sqrt((N+2)/2))
          - erf(a_j sqrt((N+2)/2))
        ].

    This is an approximation, not the full exact infinite series.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower and upper must be same-length 1D arrays.")
    if np.any(upper <= lower):
        raise ValueError("Need upper_i > lower_i.")

    N = lower.size
    scale = math.sqrt((N + 2.0) / 2.0)

    return (
        kappa(N)
        / (2.0 ** N)
        * float(
            np.prod(
                special.erf(upper * scale)
                - special.erf(lower * scale)
            )
        )
    )


def aono_erf_orthant(beta: Sequence[float]) -> float:
    # Orthant Volume using Eq.10
    r"""
    Orthant specialization:

      U_erf^(0)(beta)
      = kappa_N / 2^N
        prod_i [
          erf(s_N) + erf(s_N beta_i)
        ],
      s_N = sqrt((N+2)/2).
    """
    beta = validate_beta(beta)
    return aono_erf_leading(-beta, np.ones_like(beta))

