# 轴上椭球向单位球的仿射变换和QMC验证
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special
from scipy.stats import qmc

from app import unit_ball_orthant_N4_1D
from config import DEFAULT, NumericalConfig
from elementary import unit_ball_orthant_N1, unit_ball_orthant_N2, unit_ball_orthant_N3
from lauricella import unit_ball_orthant_lauricella
from utils import kappa, validate_beta

def ellipsoid_orthant_affine(
    center: Sequence[float],
    semiaxes: Sequence[float],
    sigma: Sequence[float] | None = None,
    *,
    config: NumericalConfig = DEFAULT,
    method: str = "auto",
) -> float:
    # 仿射变换(Eq.73)后应用单位球对应的体积公式
    r"""
    Exact affine-normalization formula:

      beta_i = sigma_i c_i / R_i,

      V_E
      = (prod_i R_i) U_N(beta).

    Adjustable parameters:
      center c_i
      semiaxes R_i > 0
      orthant signs sigma_i
      aspect ratios R_i/R_j

    Large-scale scanning over R_i is not necessary for the ball theory itself;
    ellipsoid tests mainly verify the affine Jacobian and threshold mapping.
    """
    center = np.asarray(center, dtype=float)
    semiaxes = np.asarray(semiaxes, dtype=float)

    if center.ndim != 1 or semiaxes.shape != center.shape:
        raise ValueError("center and semiaxes must be same-length 1D arrays.")
    if np.any(semiaxes <= 0.0):
        raise ValueError("All semiaxes must be positive.")

    N = center.size

    if sigma is None:
        sigma = np.ones(N)
    sigma = np.asarray(sigma, dtype=float)

    if sigma.shape != center.shape or not np.all(np.isin(sigma, [-1.0, 1.0])):
        raise ValueError("sigma_i must be +/-1.")

    beta = sigma * center / semiaxes
    validate_beta(beta)
    # "auto": use special form when N=1~4
    if method == "auto":
        if N == 1:
            U = unit_ball_orthant_N1(beta)
        elif N == 2:
            U = unit_ball_orthant_N2(beta)
        elif N == 3:
            U = unit_ball_orthant_N3(beta)
        elif N == 4:
            U = unit_ball_orthant_N4_1D(beta, config=config)
        else:
            U = unit_ball_orthant_lauricella(beta, config=config)
    elif method == "lauricella":
        U = unit_ball_orthant_lauricella(beta, config=config)
    else:
        raise ValueError("method must be 'auto' or 'lauricella'.")

    return float(np.prod(semiaxes)) * U




