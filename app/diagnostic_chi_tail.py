"""Optional research-only diagnostics. Not part of the production Hybrid.

Compute the cancellation indicator of the original full subset expansion and
an exact-arithmetic F_A tail majorant.  This program is NOT a certified error
analysis, and does not control the Hybrid truncation or its running time.
"""
from __future__ import annotations
import argparse,itertools,math,csv
from pathlib import Path
import numpy as np
from config import NumericalConfig
from lauricella import lauricella_FA_half_parameters
from utils import kappa
from patterns import make_beta

def fa_tail_majorant(p:float,r2:float,K:int)->float:
    """Bound for a nonterminating F_A with -p and x_i>=0; M=K+1>p."""
    M=K+1
    if p<=0 or abs(p-round(p))<1e-12 or M<=p or r2<0 or r2>=1:
        return math.nan
    if r2==0: return 0.0
    log_b=math.lgamma(M-p)-math.lgamma(-p)-math.lgamma(M+1)
    log_t=log_b+M*math.log(r2)
    return math.exp(log_t)*min(1/(1-r2),M/p) if log_t>-745 else 0.0

def evaluate_diagnostics(beta, *,rtol=1e-12,atol=1e-14,max_order=16384):
    b=np.asarray(beta,float); n=len(b); active=np.flatnonzero(b!=0)
    terms=[math.ldexp(kappa(n),-n)]; bound_sum=0.; records=[]
    for m in range(1,len(active)+1):
        for I in itertools.combinations(active,m):
            d=n-m;p=d/2.; q=b[list(I)]
            fa,info=lauricella_FA_half_parameters(-p,q*q,rtol=rtol,atol=atol,
                initial_order=8,max_order=max_order,return_info=True)
            scale=math.ldexp(kappa(d),-d)*float(np.prod(q))
            terms.append(scale*fa)
            if (d%2==1 and m>=4):
                bound=abs(scale)*fa_tail_majorant(p,float(np.dot(q,q)),info['order'])
                bound_sum+=bound if math.isfinite(bound) else 0.
                records.append(dict(m=m,p=p,r_I=float(np.linalg.norm(q)),
                    K=info['order'],converged=info['converged'],
                    FA_tail_bound=bound))
    u=math.fsum(terms)
    chi=(math.fsum(abs(z) for z in terms)/abs(u)) if u!=0 else math.inf
    return dict(U_raw=u,chi=chi,log10_chi=math.log10(chi) if chi>0 else math.nan,
       generic_tail_bound_sum=bound_sum,generic_FA_count=len(records),all_generic_FA_converged=all(z['converged'] for z in records),details=records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--N',type=int,default=6)
    ap.add_argument('--r',type=float,default=0.9)
    ap.add_argument('--kind',default='mixed_equal',choices=['equal','mixed_equal','negative_equal','anisotropic'])
    ap.add_argument('--max-order',type=int,default=16384)
    ap.add_argument('--output',default='results/optional_chi_tail.csv')
    a=ap.parse_args()
    beta=make_beta(a.N,a.r,kind=a.kind)
    out=evaluate_diagnostics(beta,max_order=a.max_order)
    row={k:v for k,v in out.items() if k!='details'}
    row.update(N=a.N,r=a.r,kind=a.kind)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    print(row,'\nFA subsets:',out['details'])
if __name__=='__main__':main()
