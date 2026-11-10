# Aono's Constales/Fourier-Fresnel exact series, Eq.7
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special

from utils import validate_beta


def _fresnel_aono(x):
    """
    Aono convention:
      C(x)=int_0^x cos(t^2)dt,
      S(x)=int_0^x sin(t^2)dt.

    scipy.special.fresnel uses pi*t^2/2, so a scaling is required.
    """
    z = np.asarray(x) * math.sqrt(2/math.pi)
    S_scipy, C_scipy = special.fresnel(z)
    scale = math.sqrt(math.pi/2)
    return scale*C_scipy, scale*S_scipy


def aono_fourier_fresnel(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    n_terms: int = 256,
) -> float:
    """
    Aono--Nguyen / Constales exact Fourier-series representation,
    numerically truncated after n_terms.
    """
    lower = np.asarray(lower,float)
    upper = np.asarray(upper,float)

    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower/upper must be same-length 1D arrays.")
    if np.any(upper <= lower):
        raise ValueError("Need upper_i>lower_i.")

    widths = upper-lower
    box_volume = float(np.prod(widths))
    ell = float(np.sum(np.maximum(lower**2,upper**2)))

    if ell <= 1+1e-15:
        return box_volume

    base = (
        0.5
        - float(np.sum(lower**2+upper**2+lower*upper))/(3*ell)
        + 1/ell
    )

    k = np.arange(1,n_terms+1,dtype=float)
    omega = -2*math.pi*k/ell
    root = np.sqrt(np.abs(omega))

    Cb,Sb = _fresnel_aono(root[:,None]*upper[None,:])
    Ca,Sa = _fresnel_aono(root[:,None]*lower[None,:])

    Phi = np.prod(
        (
            (Cb-Ca)
            + 1j*np.sign(omega)[:,None]*(Sb-Sa)
        )
        /
        (widths[None,:]*root[:,None]),
        axis=1,
    )

    series = np.sum(Phi/k*np.exp(2j*math.pi*k/ell))
    return box_volume*(base+series.imag/math.pi)


def aono_fourier_orthant(beta, *, n_terms=256):
    beta = validate_beta(beta)
    return aono_fourier_fresnel(
        -beta,
        np.ones_like(beta),
        n_terms=n_terms,
    )
