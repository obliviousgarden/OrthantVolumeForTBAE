# Aono's Tibken/Laguerre-erf's leading erf approximation, Eq.12
from __future__ import annotations

import math
import numpy as np
from scipy import special

from utils import kappa, validate_beta


def aono_tibken_erf_leading(lower, upper) -> float:
    """
    First term of the exact Tibken/Laguerre-erf infinite series.

      V0 = kappa_N/2^N *
           prod_i [
             erf(b_i sqrt((N+2)/2))
             - erf(a_i sqrt((N+2)/2))
           ].

    This is a fast approximation, not the full exact series.
    """
    lower = np.asarray(lower,float)
    upper = np.asarray(upper,float)

    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower/upper must be same-length 1D arrays.")

    N = lower.size
    s = math.sqrt((N+2)/2)

    return (
        kappa(N)/(2**N)
        * float(np.prod(
            special.erf(upper*s)-special.erf(lower*s)
        ))
    )


def aono_tibken_erf_orthant(beta) -> float:
    beta = validate_beta(beta)
    return aono_tibken_erf_leading(
        -beta,
        np.ones_like(beta),
    )
