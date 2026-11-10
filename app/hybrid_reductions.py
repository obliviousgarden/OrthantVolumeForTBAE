"""Constructive low-variable G_{m,p} routes from Sections 6.2 and 6.3.

Only integer and half-integer nonnegative p are used. Return oriented signed
integrals. The low-variable half-integer branches require *no* general F_A.
"""
from __future__ import annotations
import math
from functools import lru_cache
from elementary import Psi3
from n4 import J4_reduced_1D


def _is_integer(p):
    return abs(p-round(p))<1e-11

@lru_cache(maxsize=4096)
def G1(A:float,b:float,p:float)->float:
    if not A > b*b or p<0:
        raise ValueError('Need A>b^2 and p>=0')
    if _is_integer(p):
        n=int(round(p))
        return math.fsum(math.comb(n,j)*(-1)**j*A**(n-j)*b**(2*j+1)/(2*j+1)
                         for j in range(n+1))
    if p==0.5:
        return 0.5*(b*math.sqrt(A-b*b)+A*math.asin(b/math.sqrt(A)))
    prev=G1(A,b,p-1)
    return (2*p*A*prev+b*(A-b*b)**p)/(1+2*p)

@lru_cache(maxsize=4096)
def G2(A:float,b:float,c:float,p:float)->float:
    if not A>b*b+c*c or p<0:
        raise ValueError('Need A>b^2+c^2 and p>=0')
    if _is_integer(p):
        n=int(round(p))
        # finite polynomial via stable low-order terminating FA
        from lauricella import lauricella_FA_half_parameters
        f=lauricella_FA_half_parameters(-n,[b*b/A,c*c/A])
        return A**n*b*c*f
    if p==0.5:
        return A**1.5*Psi3(b/math.sqrt(A),c/math.sqrt(A))
    return (2*p*A*G2(A,b,c,p-1) + b*G1(A-b*b,c,p)
            + c*G1(A-c*c,b,p))/(2+2*p)

@lru_cache(maxsize=4096)
def G3(A:float,a:float,b:float,c:float,p:float,quad_epsrel:float=1e-11,quad_epsabs:float=1e-12):
    if not A>a*a+b*b+c*c or p<0:
        raise ValueError('Need A>sum b_i^2 and p>=0')
    if _is_integer(p):
        n=int(round(p))
        from lauricella import lauricella_FA_half_parameters
        return A**n*a*b*c*lauricella_FA_half_parameters(-n,[a*a/A,b*b/A,c*c/A])
    if p==0.5:
        from config import NumericalConfig
        return A**2*J4_reduced_1D(a/math.sqrt(A),b/math.sqrt(A),c/math.sqrt(A),
              config=NumericalConfig(quad_epsrel=quad_epsrel,quad_epsabs=quad_epsabs))
    return (2*p*A*G3(A,a,b,c,p-1,quad_epsrel,quad_epsabs) +
        a*G2(A-a*a,b,c,p)+b*G2(A-b*b,a,c,p)+c*G2(A-c*c,a,b,p))/(3+2*p)


def C_k(k:int)->float:
    if k<0: raise ValueError('k must be >=0')
    out=1.0
    for i in range(1,k+1):out *= (2*i+1)/(2*i+4)
    return out


def E_k(k:int,a:float,b:float,c:float)->float:
    """Elementary remainder of the N=4+2k three-coordinate hierarchy."""
    if k<0:raise ValueError('k must be >=0')
    result=0.0
    for i in range(1,k+1):
        p=i+0.5
        boundary=(a*G2(1-a*a,b,c,p)+b*G2(1-b*b,a,c,p)+c*G2(1-c*c,a,b,p))
        result=(2*i+1)/(2*i+4)*result +boundary/(2*i+4)
    return result


def clear_reduction_caches() -> None:
    """Reset cross-evaluation caches while preserving reuse within one solve.

    Call once at the start of each subset-based Hybrid evaluation. This
    prevents timed_warm from benchmarking memoized J/Psi values rather than
    the actual numerical integrations for each independent solve.
    """
    G1.cache_clear()
    G2.cache_clear()
    G3.cache_clear()
