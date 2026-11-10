"""Numerical settings for the V21.2 Route(G/T/U)-only Hybrid simulations.

This file preserves constructor compatibility with the original Figure/Table
scripts.  The four legacy settings below are accepted for compatibility;
Route(G/T/U) and the FA evaluator do not use them to change their algorithm.

No chi-adaptive precision controller, FA tail-bound order selection, or
high-precision whole-volume fallback is used by the production Hybrid.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalConfig:
    # Original Lauricella adaptive total-degree stopping settings.
    fa_rtol: float = 1e-12
    fa_atol: float = 1e-14
    fa_initial_order: int = 8
    fa_max_order: int = 16384

    # Numerical quadrature.
    quad_epsabs: float = 1e-12
    quad_epsrel: float = 1e-11
    quad_limit: int = 300

    # Geometry and random sampling.
    zero_tol: float = 0.0
    seed: int = 20261028

    # Backward-compatible fields passed by the original Fig. 4(a/b),
    # Fig. 5(a), and Table 1 scripts. The active simulation functions retain
    # their own explicit timing, QMC, and Aono parameters where applicable.
    aono_fourier_terms: int = 512
    timing_repeats: int = 1
    qmc_power: int = 18
    qmc_scrambles: int = 16


DEFAULT = NumericalConfig()
