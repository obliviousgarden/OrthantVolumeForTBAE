# QMC Monte Carlo Check for Unit Ball
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special
from scipy.stats import qmc

from utils import kappa, validate_beta

def unit_ball_orthant_qmc(
    beta: Sequence[float],
    *,
    power: int = 16,
    scrambles: int = 8,
    seed: int = 12345,
) -> tuple[float, float]:
    # QMC Check
    """
    Independent QMC estimate of U_N(beta) for higher dimensions.
    """
    beta = validate_beta(beta)
    N = beta.size
    estimates = []

    for r in range(scrambles):
        sampler = qmc.Sobol(
            d=N + 1,
            scramble=True,
            seed=seed + r,
        )
        U = sampler.random_base2(power)

        eps = np.finfo(float).eps
        G = special.ndtri(
            np.clip(U[:, :N], eps, 1.0 - eps)
        )
        G /= np.linalg.norm(G, axis=1, keepdims=True)

        radial = U[:, N] ** (1.0 / N)
        Z = G * radial[:, None]

        fraction = float(
            np.mean(
                np.all(Z >= -beta[None, :], axis=1)
            )
        )
        estimates.append(kappa(N) * fraction)

    estimates = np.asarray(estimates)

    mean = float(np.mean(estimates))
    stderr = (
        float(np.std(estimates, ddof=1) / math.sqrt(scrambles))
        if scrambles > 1
        else math.nan
    )
    return mean, stderr

def ellipsoid_orthant_qmc(
    center: Sequence[float],
    semiaxes: Sequence[float],
    sigma: Sequence[float] | None = None,
    *,
    power: int = 16,
    scrambles: int = 8,
    seed: int = 12345,
) -> tuple[float, float]:
    # Monte-Carlo Check
    r"""
    Independent randomized Sobol-QMC check.

    Uniform unit-ball sampling:
      g_i = Phi^{-1}(u_i),
      direction = g / ||g||,
      radius = u_{N+1}^{1/N},
      z = radius * direction.

    Map to ellipsoid:
      x_i = c_i + R_i z_i.

    The estimator is
      V_hat = kappa_N prod_i R_i *
              mean[ sigma_i x_i >= 0 for all i ].

    Returns:
      mean estimate over scrambles,
      standard error across scrambles.

    Main adjustable parameters:
      power       number of points per scramble = 2**power
      scrambles   randomized replicate count
      aspect ratio / center / sigma
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

    total_volume = kappa(N) * float(np.prod(semiaxes))
    estimates = []

    for r in range(scrambles):
        sampler = qmc.Sobol(
            d=N + 1,
            scramble=True,
            seed=seed + r,
        )
        U = sampler.random_base2(power)

        # Avoid infinities in inverse normal transform.
        eps = np.finfo(float).eps
        G = special.ndtri(
            np.clip(U[:, :N], eps, 1.0 - eps)
        )
        G_norm = np.linalg.norm(G, axis=1, keepdims=True)
        directions = G / G_norm

        radii = U[:, N] ** (1.0 / N)
        Z = directions * radii[:, None]

        X = center[None, :] + semiaxes[None, :] * Z

        inside_orthant = np.all(
            sigma[None, :] * X >= 0.0,
            axis=1,
        )

        estimates.append(
            total_volume * float(np.mean(inside_orthant))
        )

    estimates = np.asarray(estimates, dtype=float)

    mean = float(np.mean(estimates))
    if scrambles > 1:
        stderr = float(
            np.std(estimates, ddof=1)
            / math.sqrt(scrambles)
        )
    else:
        stderr = math.nan

    return mean, stderr