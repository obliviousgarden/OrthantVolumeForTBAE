from __future__ import annotations

"""
Hybrid evaluator for the translated-ball orthant volume.

This module implements the analytic hierarchy developed in the manuscript:

1. N <= 3:
   complete elementary formulas.

2. N = 4:
   Appendix-C one-dimensional reduction.

3. N >= 5:
    support s=1:
        stable incomplete-beta half-space formula;

    support s=2:
        direct finite elementary recurrence without cached G1/G2/G3,
        with stable one-dimensional probability quadrature only when the
        elementary sum would suffer significant cancellation;

    support s>=3:
        subset decomposition with
          |I|=1               -> elementary one-coordinate integral,
          N-|I| even          -> terminating polynomial,
          |I|=2, N-|I| odd    -> elementary Psi reduction,
          |I|=3, N-|I| odd    -> J plus elementary E_k remainder,
          |I|>=4, N-|I| odd  -> generic nonterminating F_A series.

The direct s=1 and conditioned s=2 routes avoid G-function recursion and
unneeded diagnostics on the timed value-only path.  For rare ill-conditioned
s=2 threshold configurations, the stable probability integral is retained.
"""

import itertools
import math
from typing import Sequence

import numpy as np
from scipy import integrate, special

from config import DEFAULT, NumericalConfig
from elementary import (
    Phi3, Psi3,
    unit_ball_orthant_N1,
    unit_ball_orthant_N2,
    unit_ball_orthant_N3,
)
from lauricella import lauricella_FA_half_parameters
from n4 import A4, B4, unit_ball_orthant_N4_1D
from utils import (
    kappa,
    validate_beta,
    valid_orthant_volume,
)


def _zero_tol(config: NumericalConfig) -> float:
    return float(
        getattr(
            config,
            "zero_tol",
            0.0,
        )
    )


# =============================================================================
# Stable support-specific formulas
# =============================================================================

def _halfspace_probability(
    beta: float,
    dimension: int,
) -> float:
    r"""
    Probability that the first coordinate X of a point uniformly distributed
    in the unit ``dimension``-ball satisfies

        X >= -beta.

    Stable incomplete-beta forms are used on the two sides of beta=0.
    """
    b = float(beta)

    if dimension < 1:
        raise ValueError("dimension must be >= 1.")

    if b >= 1.0:
        return 1.0

    if b <= -1.0:
        return 0.0

    if b == 0.0:
        return 0.5

    x = b * b
    a = 0.5
    c = (dimension + 1.0) / 2.0

    if b > 0.0:
        return 0.5 * (
            1.0
            + special.betainc(
                a,
                c,
                x,
            )
        )

    # For b<0 use the complementary argument directly, avoiding
    # 1 - I_x when the desired tail probability is very small.
    return 0.5 * special.betainc(
        c,
        a,
        1.0 - x,
    )


def stable_single_support_volume(
    beta: float,
    N: int,
) -> float:
    r"""
    Stable complete orthant volume for exactly one active threshold.

    The inactive N-1 coordinates contribute a sign-symmetry factor
    2^{-(N-1)}.  Hence

        U
        =
        kappa_N / 2^{N-1}
        * Pr(X_1 >= -beta),

    where X is uniform in the N-ball.
    """
    probability = _halfspace_probability(
        beta,
        N,
    )

    return (
        math.ldexp(
            kappa(N),
            -(N - 1),
        )
        * probability
    )


def stable_two_support_volume(
    beta1: float,
    beta2: float,
    N: int,
    *,
    config: NumericalConfig = DEFAULT,
) -> float:
    r"""
    Stable complete orthant volume for exactly two active thresholds.

    Instead of summing centered + singleton + pair corrections, compute the
    full two-coordinate probability directly:

        P_2
        =
        int_{x >= -beta1}
            f_{X_1}(x)
            Pr(
                X_2 >= -beta2 | X_1=x
            )
        dx.

    Conditional on X_1=x, the remaining coordinates are uniform in an
    (N-1)-ball of radius sqrt(1-x^2), so the inner probability is another
    stable incomplete-beta half-space probability.

    Finally,

        U = kappa_N / 2^{N-2} * P_2.

    This route is a one-dimensional semi-analytic quadrature and avoids the
    high-dimensional cancellation observed when the mathematically
    terminating subset polynomial is accumulated directly in float64.
    """
    b1 = float(beta1)
    b2 = float(beta2)

    if b1 * b1 + b2 * b2 >= 1.0:
        raise ValueError(
            "Need beta1^2 + beta2^2 < 1."
        )

    marginal_constant = (
        kappa(N - 1)
        / kappa(N)
    )

    power = (N - 1.0) / 2.0
    lower = -b1

    def integrand(x: float) -> float:
        q = 1.0 - x * x

        if q <= 0.0:
            return 0.0

        radius = math.sqrt(q)

        conditional = _halfspace_probability(
            b2 / radius,
            N - 1,
        )

        return (
            marginal_constant
            * q ** power
            * conditional
        )

    # The conditional probability changes analytic branch where
    # sqrt(1-x^2)=|beta2|.  Supplying these points helps adaptive quadrature.
    points = []

    if abs(b2) < 1.0:
        critical = math.sqrt(
            max(
                0.0,
                1.0 - b2 * b2,
            )
        )

        for x in (-critical, critical):
            if lower < x < 1.0:
                points.append(x)

    epsrel = max(
        float(config.quad_epsrel),
        2e-14,
    )

    probability, _ = integrate.quad(
        integrand,
        lower,
        1.0,
        epsabs=0.0,
        epsrel=epsrel,
        limit=getattr(
            config,
            "quad_limit",
            300,
        ),
        points=points if points else None,
    )

    return (
        math.ldexp(
            kappa(N),
            -(N - 2),
        )
        * probability
    )


# =============================================================================
# Fast s=2 elementary route (no G1/G2/G3 recursion)
# =============================================================================

def _G2_elementary_iterative(b1: float, b2: float, power: float) -> float:
    """Oriented double integral of (1-u²-v²)^power for integer/half-integer
    power. Finite forward recurrences only; never enters the cached G functions.

    For integer power, G2(p=0)=b1*b2. For half-integer power,
    G2(p=1/2)=Psi3(b1,b2). The two G1 boundary integrals and G2 are
    advanced together, with no recursive calls or FA series.
    """
    x=float(b1); y=float(b2)
    ax=1.0-x*x; ay=1.0-y*y
    q=1.0-x*x-y*y
    if q <= 0.0:
        raise ValueError("Need b1² + b2² < 1")
    is_integer=abs(power-round(power))<1e-12
    if is_integer:
        start=0.0
        pair=x*y
        # G1(A-x², y, 0) and G1(A-y², x, 0)
        edge_y=y; edge_x=x
    else:
        start=0.5
        pair=Psi3(x,y)
        qroot=math.sqrt(q)
        edge_y=0.5*(y*qroot+ax*math.asin(y/math.sqrt(ax)))
        edge_x=0.5*(x*qroot+ay*math.asin(x/math.sqrt(ay)))
    p=start+1.0
    while p <= power+1e-10:
        qp=q**p
        edge_y_next=(2*p*ax*edge_y + y*qp)/(2*p+1.0)
        edge_x_next=(2*p*ay*edge_x + x*qp)/(2*p+1.0)
        pair=(2*p*pair+x*edge_y_next+y*edge_x_next)/(2*p+2.0)
        edge_y,edge_x=edge_y_next,edge_x_next
        p+=1.0
    return pair


def fast_two_support_volume(
    beta1:float, beta2:float, N:int, *,
    config:NumericalConfig=DEFAULT,
    return_route:bool=False,
):
    """Fast exact-form s=2 subset result with stability-aware fallback.

    The absolute-term condition number limits cancellation in the sum of four
    elementary contributions. If it exceeds the conservative limit, use the
    original stable 1D probability quadrature (needed for rare small tails).
    Route selection is part of the numerical algorithm, not its diagnostics.
    """
    a=float(beta1); b=float(beta2)
    t0=math.ldexp(kappa(N),-N)
    t1=one_coordinate_subset_term(a,N)
    t2=one_coordinate_subset_term(b,N)
    d=N-2
    t12=math.ldexp(kappa(d),-d)*_G2_elementary_iterative(a,b,d/2.0)
    value=math.fsum((t0,t1,t2,t12))
    magnitude=abs(t0)+abs(t1)+abs(t2)+abs(t12)
    # <=~2e3 gives a modest floating-point cancellation budget relative to
    # epsilon_rel=1e-10. A small volume always uses the stable tail formula.
    if not math.isfinite(value) or value<=0 or magnitude > 2000.0*value:
        value=stable_two_support_volume(a,b,N,config=config)
        route='s2_stable_1D_probability'
    else:
        route='s2_elementary_iterative'
    return (value,route) if return_route else value


def _low_support_evaluate(b, N:int, active, s:int,
                          config:NumericalConfig):
    """Direct low-dimensional/low-support exact formulas, no G recursion.

    Returns (volume, actual_route) and is used by BOTH timing and diagnostics.
    """
    if s==0:
        return math.ldexp(kappa(N),-N), 'centered_exact'
    if N==1:
        return unit_ball_orthant_N1(b),'N1_elementary'
    if N==2:
        return unit_ball_orthant_N2(b),'N2_elementary'
    if N==3 and s==1:
        return (math.pi/6.0+Phi3(float(b[active[0]]))), 'N3_elementary_s1'
    if N==3 and s==2:
        a=float(b[active[0]]); c=float(b[active[1]])
        return (math.pi/6.0+Phi3(a)+Phi3(c)+Psi3(a,c)), 'N3_elementary_s2'
    if N==3:
        return unit_ball_orthant_N3(b),'N3_elementary'
    if N==4 and s==1:
        return (math.pi**2/32.0+A4(float(b[active[0]]))), 'N4_elementary_s1'
    if N==4 and s==2:
        a=float(b[active[0]]); c=float(b[active[1]])
        return (math.pi**2/32.0+A4(a)+A4(c)+B4(a,c)), 'N4_elementary_s2'
    if N==4:
        return unit_ball_orthant_N4_1D(b,config=config), 'N4_1D_reduction'
    if s==1:
        return stable_single_support_volume(float(b[active[0]]),N), 's1_incomplete_beta'
    if s==2:
        return fast_two_support_volume(float(b[active[0]]),float(b[active[1]]),N,
                                       config=config,return_route=True)
    raise AssertionError('Only s<=2 or N<=4 accepted')


# =============================================================================
# Existing analytic subset components
# =============================================================================

def one_coordinate_integral(
    beta: float,
    N: int,
) -> float:
    r"""
    I_N(beta)
    =
    int_0^beta (1-u^2)^((N-1)/2) du.

    This helper is still useful inside the generic s>=3 Hybrid decomposition.
    """
    x = float(beta)

    if not abs(x) < 1.0:
        raise ValueError("Need |beta| < 1.")

    if N < 1:
        raise ValueError("N must be positive.")

    if N % 2 == 1:
        q = (N - 1) // 2

        return math.fsum(
            ((-1.0) ** k)
            * math.comb(q, k)
            * x ** (2 * k + 1)
            / (2 * k + 1)
            for k in range(q + 1)
        )

    root = math.sqrt(
        max(
            0.0,
            1.0 - x * x,
        )
    )

    value = 0.5 * (
        x * root
        + math.asin(x)
    )

    p = 0.5
    target = (N - 1) / 2.0

    while p < target - 1e-15:
        p_next = p + 1.0

        value = (
            x
            * (1.0 - x * x) ** p_next
            / (2.0 * p_next + 1.0)
            + (2.0 * p_next)
            / (2.0 * p_next + 1.0)
            * value
        )

        p = p_next

    return value


def one_coordinate_subset_term(
    beta: float,
    N: int,
) -> float:
    return (
        math.ldexp(
            kappa(N - 1),
            -(N - 1),
        )
        * one_coordinate_integral(
            beta,
            N,
        )
    )





# =============================================================================
# Route(G/T/U) structural counts and evaluator
# =============================================================================

def analytic_structure_counts(N: int, support: int) -> dict:
    """Analytic terms in the subset hierarchy, prior to Route(U) shortcuts."""
    if not 0 <= support <= N:
        raise ValueError("Need 0<=support<=N")
    term = sum(math.comb(support,m) for m in range(2,support+1)
               if (N-m)%2==0)
    jcount = math.comb(support,3) if (support>=3 and N%2==0) else 0
    fa = sum(math.comb(support,m) for m in range(4,support+1)
             if (N-m)%2==1)
    psi = math.comb(support,2) if (support>=2 and N%2==1) else 0
    return {
        "empty_terms":1,
        "singleton_elementary_terms":support,
        "terminating_polynomial_terms":term,
        "elementary_Psi_terms":psi,
        "J_terms":jcount,
        "generic_FA_terms":fa,
        "nu_FA":fa,
        "nu_J":jcount,
        "total_surviving_terms":2**support,
    }


def rho_FA(beta:Sequence[float],N:int,zero_tol:float=0.0)->float:
    """Maximum radius among retained generic F_A terms, m>=4 and N-m odd."""
    squared=sorted((float(x)**2 for x in beta if abs(x)>zero_tol),reverse=True)
    s=len(squared)
    choices=[m for m in range(4,s+1) if (N-m)%2==1]
    return math.sqrt(sum(squared[:max(choices)])) if choices else 0.0


def rho_nonterminating(beta,N,zero_tol=0.0):
    """Historical field alias; now refers to final Route(T) FA subsets."""
    return rho_FA(beta,N,zero_tol)


def is_fully_structured_exact(N:int,support:int)->bool:
    return analytic_structure_counts(N,support)['generic_FA_terms']==0


def unit_ball_orthant_hybrid(beta:Sequence[float],*,
                             config:NumericalConfig=DEFAULT,
                             return_info:bool=False):
    """Route(U)->Route(T)->Route(G) evaluator with bounded F_A doubling.

    No chi-adaptive controller, tail-bound stopping, or high-precision fallback.
    Nonconverged raw F_A values are flagged in return_info.  Those are NOT
    accuracy-validated results; benchmark callers must reject them.
    """
    b=validate_beta(beta)
    N=len(b)
    active=[i for i,v in enumerate(b) if abs(v)>_zero_tol(config)]
    s=len(active)
    fa_details=[] if return_info else None
    count_psi=count_j=count_poly=count_fa=count_single=0
    total_terms=0
    if N<=4 or s<=2:
        val,route=_low_support_evaluate(b,N,active,s,config)
        if not return_info:
            # The benchmark measures only the numerical solve. No diagnostics,
            # structure counting, route checks, or dictionaries are constructed.
            return val
        if N==4 and s>=3:
            count_j=math.comb(s,3)
    else:
        from hybrid_reductions import G2,G3,clear_reduction_caches
        # Cached G recurrences are cleared for every *independent* timed solve.
        clear_reduction_caches()
        route='hybrid_subset'
        terms=[math.ldexp(kappa(N),-N)]
        for m in range(1,s+1):
            for I in itertools.combinations(active,m):
                q=[float(b[i]) for i in I]
                d=N-m
                p=d/2.0
                if m==1:
                    term=one_coordinate_subset_term(q[0],N)
                    if return_info: count_single+=1
                elif d==0:
                    term=math.prod(q)  # G_{m,0}
                    if return_info: count_poly+=1
                elif d%2==0:  # Exact terminating polynomial from F_A
                    fa_result=lauricella_FA_half_parameters(
                        -p,[z*z for z in q],rtol=config.fa_rtol,
                        atol=config.fa_atol,initial_order=config.fa_initial_order,
                        max_order=config.fa_max_order,return_info=return_info)
                    fa=fa_result[0] if return_info else fa_result
                    term=math.ldexp(kappa(d),-d)*math.prod(q)*fa
                    if return_info: count_poly+=1
                elif m==2:
                    term=math.ldexp(kappa(d),-d)*G2(1.0,q[0],q[1],p)
                    if return_info: count_psi+=1
                elif m==3:
                    term=math.ldexp(kappa(d),-d)*G3(
                        1.0,*q,p,config.quad_epsrel,config.quad_epsabs)
                    if return_info: count_j+=1
                else:
                    fa_result=lauricella_FA_half_parameters(
                        -p,[z*z for z in q],rtol=config.fa_rtol,
                        atol=config.fa_atol,initial_order=config.fa_initial_order,
                        max_order=config.fa_max_order,return_info=return_info)
                    fa=fa_result[0] if return_info else fa_result
                    term=math.ldexp(kappa(d),-d)*math.prod(q)*fa
                    if return_info:
                        count_fa+=1
                        fa_details.append(fa_result[1])
                terms.append(term)
        val=math.fsum(terms)
        if return_info: total_terms=len(terms)
    if not return_info:
        return val
    counts=analytic_structure_counts(N,s)
    all_conv=all(x['converged'] for x in fa_details)
    order=max((x['order'] for x in fa_details),default=0)
    # Compatibility placeholders: dedicated diagnostics script handles chi.
    info={
        'N':N,'support':s,'route':route,
        'all_FA_converged':all_conv,
        'max_FA_order':order,'FA_max_order_allowed':int(config.fa_max_order),
        'evaluated_singleton_terms':count_single,
        'evaluated_terminating_terms':count_poly,
        'evaluated_Psi_terms':count_psi,
        'evaluated_J_terms':count_j,
        'evaluated_generic_FA_terms':count_fa,
        'evaluated_FA':count_fa,'evaluated_J':count_j,
        'evaluated_subset_terms':total_terms,
        'rho_FA':rho_FA(b,N,_zero_tol(config)),
        'rho_NT':rho_FA(b,N,_zero_tol(config)),
        'cancellation_indicator':math.nan,
        'valid_volume':valid_orthant_volume(val,N),
        'numerically_usable':all_conv and valid_orthant_volume(val,N),
        **counts,
    }
    return val,info
