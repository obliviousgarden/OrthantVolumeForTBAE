# N=1,2,3的初等闭式公式 Eq.54,Eq.55,Eq.57
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from utils import clip_unit, validate_beta

def unit_ball_orthant_N1(beta: Sequence[float] | float) -> float:
    # N=1,1维球（线段）对应的坐标体积公式，Eq.54
    r"""
    N=1:
      U_1(beta) = 1 + beta.
    """
    b = float(np.asarray(beta).reshape(-1)[0])
    if abs(b) >= 1.0:
        raise ValueError("Need |beta| < 1.")
    return 1.0 + b


def unit_ball_orthant_N2(beta: Sequence[float]) -> float:
    # N=2,2维球（圆）对应的坐标体积公式，Eq.55
    r"""
    N=2:
      U_2(b1,b2)
      = pi/4
        + 1/2 sum_i [b_i sqrt(1-b_i^2) + asin(b_i)]
        + b1 b2.
    """
    b = validate_beta(beta)
    if b.size != 2:
        raise ValueError("N=2 requires two beta components.")

    def one_coord(x: float) -> float:
        return 0.5 * (
            x * math.sqrt(max(0.0, 1.0 - x * x))
            + math.asin(x)
        )

    return (
        math.pi / 4.0
        + one_coord(float(b[0]))
        + one_coord(float(b[1]))
        + float(b[0] * b[1])
    )


def Phi3(s: float) -> float:
    # N=3, function Phi, Eq.58
    r"""
    N=3 one-coordinate contribution:
      Phi(s) = pi/4 * (s - s^3/3).
    """
    return math.pi / 4.0 * (s - s**3 / 3.0)


def Psi3(s: float, t: float) -> float:
    # N=3, function Psi, Eq.60
    r"""
    N=3 two-coordinate contribution:
      Psi(s,t) = int_0^s int_0^t sqrt(1-u^2-v^2) dv du

    Elementary expression:
      st/3 sqrt(1-s^2-t^2)
      + t(3-t^2)/6 asin[s/sqrt(1-t^2)]
      + s(3-s^2)/6 asin[t/sqrt(1-s^2)]
      - 1/3 atan[st/sqrt(1-s^2-t^2)].
    """
    q = 1.0 - s * s - t * t
    if q < -1e-13:
        raise ValueError("Psi3 requires s^2+t^2 < 1.")
    root = math.sqrt(max(0.0, q))

    den_s = math.sqrt(max(0.0, 1.0 - t * t))
    den_t = math.sqrt(max(0.0, 1.0 - s * s))

    return (
        s * t / 3.0 * root
        + t * (3.0 - t * t) / 6.0
        * math.asin(clip_unit(s / den_s))
        + s * (3.0 - s * s) / 6.0
        * math.asin(clip_unit(t / den_t))
        - 1.0 / 3.0 * math.atan2(s * t, root)
    )


def unit_ball_orthant_N3(beta: Sequence[float]) -> float:
    # N=3,3维球对应的坐标体积公式，Eq.57
    r"""
    N=3:
      U_3(beta)
      = pi/6
        + sum_i Phi(beta_i)
        + sum_{i<j} Psi(beta_i,beta_j)
        + beta_1 beta_2 beta_3.
    """
    b = validate_beta(beta)
    if b.size != 3:
        raise ValueError("N=3 requires three beta components.")

    U = math.pi / 6.0
    U += math.fsum(Phi3(float(x)) for x in b)
    U += math.fsum(
        Psi3(float(b[i]), float(b[j]))
        for i, j in itertools.combinations(range(3), 2)
    )
    U += float(np.prod(b))
    return U

