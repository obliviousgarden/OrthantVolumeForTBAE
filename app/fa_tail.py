"""Enhanced absolute tail majorant for F_A(-p;1/2,...;3/2,...;x).

The formula is valid for p>0, M=K+1>p and 0<=rho_sq<1.
It bounds the *exact-arithmetic series tail*, not coefficient roundoff,
quadrature error, or final subset-summation rounding.
"""
from __future__ import annotations
import math


def coefficient_b(M: int, p: float) -> float:
    if p <= 0 or M <= p:
        raise ValueError('Requires p>0 and M>p')
    # |(-p)_M|/M! = |Gamma(M-p)/Gamma(-p)| / Gamma(M+1).
    return math.exp(math.lgamma(M-p)-math.lgamma(-p)-math.lgamma(M+1))


def tail_bound(K: int, p: float, rho_sq: float, *, improved: bool=True) -> float:
    if not (0 <= rho_sq < 1) or p <= 0 or K+1 <= p:
        return math.inf
    if rho_sq == 0:
        return 0.0
    M = K + 1
    log_b = math.lgamma(M-p)-math.lgamma(-p)-math.lgamma(M+1)
    geometry = min(1/(1-rho_sq), M/p) if improved else 1/(1-rho_sq)
    log_bound = log_b + M*math.log(rho_sq) + math.log(geometry)
    return math.exp(log_bound) if log_bound > -745 else 0.0


def requested_order(p: float, rho_sq: float, absolute_budget: float,
                    *, initial_order: int=8, max_order: int=16384):
    """Return (chosen K, met, bound). Choose first admissible doubling order.

    A requested order >max_order is *not* accepted as converged.
    The K grid follows 8,16,32,...,max_order (last cap included).
    """
    if absolute_budget <= 0 or not math.isfinite(absolute_budget):
        raise ValueError('absolute_budget must be positive and finite')
    if max_order < 1 or initial_order < 1:
        raise ValueError('Orders must be positive')
    K=min(initial_order,max_order)
    while True:
        bound=tail_bound(K,p,rho_sq)
        if bound<=absolute_budget:
            return K,True,bound
        if K>=max_order:
            return K,False,bound
        K=min(2*K,max_order)
