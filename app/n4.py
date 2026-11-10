# N=4,4维超球坐标体积公式,Eq.62
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
from scipy import integrate

from lauricella import lauricella_FA_half_parameters,unit_ball_orthant_lauricella
from config import DEFAULT, NumericalConfig
from elementary import Psi3
from utils import clip_unit, validate_beta

def A4(a: float) -> float:
    # Function A, Eq.63
    r"""
    N=4, |I|=1:
      A(a)
      = pi/6 int_0^a (1-u^2)^(3/2) du
      = pi/48 [3 asin(a) + a sqrt(1-a^2)(5-2a^2)].
    """
    return (
        math.pi / 48.0
        * (
            3.0 * math.asin(a)
            + a * math.sqrt(max(0.0, 1.0 - a * a))
            * (5.0 - 2.0 * a * a)
        )
    )


def B4(a: float, b: float) -> float:
    # Function B, Eq.64
    r"""
    N=4, |I|=2:
      B(a,b)
      = pi ab/4 [1-(a^2+b^2)/3].
    """
    return (
        math.pi * a * b / 4.0
        * (1.0 - (a * a + b * b) / 3.0)
    )

def J4_lauricella(
    a: float,
    b: float,
    c: float,
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    # Function J(F_A), Eq.66
    """
    J = abc F_A^(3)(-1/2; 1/2,1/2,1/2; 3/2,3/2,3/2; a^2,b^2,c^2).
    """
    if a*a+b*b+c*c >= 1:
        raise ValueError("Need a^2+b^2+c^2<1.")

    FA = lauricella_FA_half_parameters(
        -0.5,
        [a*a, b*b, c*c],
        rtol=config.fa_rtol,
        atol=config.fa_atol,
        initial_order=config.fa_initial_order,
        max_order=config.fa_max_order,
    )
    return a*b*c*FA
def J4_direct_triple(
    a: float,
    b: float,
    c: float,
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    # Function J(triple), Eq.65
    r"""
    Direct reference form for the N=4 three-coordinate contribution:

      J(a,b,c)
      = int_0^a int_0^b int_0^c
          sqrt(1-u^2-v^2-w^2) dw dv du.

    It is evaluated after mapping each oriented interval to [0,1]^3:

      J = abc int_[0,1]^3
            sqrt(1-a^2 x^2-b^2 y^2-c^2 z^2) dz dy dx.

    Main adjustable parameters:
      quad_epsabs, quad_epsrel.

    This method is intended primarily as an independent N=4 reference.
    """
    if a * a + b * b + c * c >= 1.0:
        raise ValueError("Need a^2+b^2+c^2 < 1.")

    prefactor = a * b * c
    if prefactor == 0.0:
        return 0.0

    def integrand(z, y, x):
        q = 1.0 - (a * x) ** 2 - (b * y) ** 2 - (c * z) ** 2
        return math.sqrt(max(0.0, q))

    value, _ = integrate.nquad(
        integrand,
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        opts={
            "epsabs": config.quad_epsabs,
            "epsrel": config.quad_epsrel,
        },
    )
    return prefactor * value


def _J4_reduced_positive(
    a: float,
    b: float,
    c: float,
    *,
    config: NumericalConfig,
) -> float:
    # J = S/4+E/4-L/8, Eq.79
    """Positive-argument core of the Appendix-C S-E-L reduction."""

    def S4(a: float, b: float, c: float) -> float:
        # Eq.73
        """Elementary boundary term S(a,b,c) in the N=4 reduction."""
        return (
                a * (1.0 - a * a) ** 1.5
                * Psi3(
                b / math.sqrt(1.0 - a * a),
                c / math.sqrt(1.0 - a * a),
            )
                    + b * (1.0 - b * b) ** 1.5
                    * Psi3(
                a / math.sqrt(1.0 - b * b),
                c / math.sqrt(1.0 - b * b),
            )
                    + c * (1.0 - c * c) ** 1.5
                    * Psi3(
                a / math.sqrt(1.0 - c * c),
                b / math.sqrt(1.0 - c * c),
            )
        )

    def E4(a: float, b: float, c: float) -> float:
        # Eq.77
        """Elementary term E(a,b,c) in the N=4 reduction."""
        D = math.sqrt(max(0.0, 1.0 - a * a - b * b - c * c))

        def asin_ratio(num: float, den: float) -> float:
            return math.asin(clip_unit(num / den))

        return (
                a * c
                * asin_ratio(
                b,
                math.sqrt(1.0 - a * a - c * c),
            )
                    + b * c
                    * asin_ratio(
                a,
                math.sqrt(1.0 - b * b - c * c),
            )
                    - a * math.sqrt(1.0 - a * a) / 2.0
                    * math.atan2(
                b * c,
                math.sqrt(1.0 - a * a) * D,
            )
                    - b * math.sqrt(1.0 - b * b) / 2.0
                    * math.atan2(
                a * c,
                math.sqrt(1.0 - b * b) * D,
            )
                    + a * b
                    * asin_ratio(
                c,
                math.sqrt(1.0 - a * a - b * b),
            )
                    - c * math.sqrt(1.0 - c * c) / 2.0
                    * math.atan2(
                a * b,
                math.sqrt(1.0 - c * c) * D,
            )
        )

    def L4(a: float, b: float, c: float, *, config: NumericalConfig = DEFAULT,
    ) -> float:
        # Eq.80
        r"""
        Final one-dimensional residual integral:

          L(a,b,c)
          = int_0^{asin(c)}
              atan[
                ab /
                (cos(theta)
                 sqrt(1-a^2-b^2-sin^2(theta)))
              ] dtheta.

        This form removes the factor 1/sqrt(1-w^2) from the w-form.
        """
        if a * a + b * b + c * c >= 1.0:
            raise ValueError("Need a^2+b^2+c^2 < 1.")
        if a == 0.0 or b == 0.0 or c == 0.0:
            return 0.0

        upper = math.asin(c)

        def integrand(theta: float) -> float:
            cos_theta = math.cos(theta)
            q = (
                    1.0
                    - a * a
                    - b * b
                    - math.sin(theta) ** 2
            )
            return math.atan2(
                a * b,
                cos_theta * math.sqrt(max(0.0, q)),
            )

        value, _ = integrate.quad(
            integrand,
            0.0,
            upper,
            epsabs=config.quad_epsabs,
            epsrel=config.quad_epsrel,
            limit=config.quad_limit,
        )
        return value

    S = S4(a, b, c)
    E = E4(a, b, c)
    L = L4(a, b, c)

    return 0.25 * S + 0.25 * E - 0.125 * L

def J4_reduced_1D(
    a: float,
    b: float,
    c: float,
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    """
    Appendix-C one-dimensional reduction:
      J = S/4 + E/4 - L/8.

    The derivation is written for positive arguments. For signed oriented
    limits, J(a,b,c)=sgn(abc) J(|a|,|b|,|c|).
    """
    if a * a + b * b + c * c >= 1:
        raise ValueError("Need a^2+b^2+c^2<1.")

    if a == 0 or b == 0 or c == 0:
        return 0.0

    sign = 1.0 if a * b * c > 0 else -1.0
    return sign * _J4_reduced_positive(
        abs(a), abs(b), abs(c), config=config
    )


def _assemble_U4(beta, J_function, config):
    b = validate_beta(beta)
    if b.size != 4:
        raise ValueError("N=4 requires four beta components.")

    return (
        math.pi**2/32
        + math.fsum(A4(float(x)) for x in b)
        + math.fsum(
            B4(float(b[i]),float(b[j]))
            for i,j in itertools.combinations(range(4),2)
        )
        + math.fsum(
            J_function(
                float(b[i]),float(b[j]),float(b[k]),
                config=config,
            )
            for i,j,k in itertools.combinations(range(4),3)
        )
        + float(np.prod(b))
    )


def unit_ball_orthant_N4_1D(
    beta: Sequence[float],
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    return _assemble_U4(beta, J4_reduced_1D, config)


def unit_ball_orthant_N4_3D(
    beta: Sequence[float],
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    """Full N=4 formula with every J evaluated by direct triple quadrature."""
    return _assemble_U4(beta, J4_direct_triple, config)


def unit_ball_orthant_N4_FA(
    beta: Sequence[float],
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    """Full N=4 volume from the general Lauricella subset formula."""
    b = validate_beta(beta)
    if b.size != 4:
        raise ValueError("N=4 requires four beta components.")
    return unit_ball_orthant_lauricella(b, config=config)


