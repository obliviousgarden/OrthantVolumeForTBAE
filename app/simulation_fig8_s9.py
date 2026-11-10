#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 8 Supplementary S9: complete, independent stability audit of V21.6.

One invocation audits ALL 5,814 geometries / 17,442 stored accuracy trials.
It never modifies Hybrid, Aono, or the original Fig. 8 timings. There are
NO case-family/scope switches and NO plotting operations.

Output (always exactly three public CSVs):
    fig8_s9_raw.csv       17,442 geometry/target records
    fig8_s9_summary.csv     510 (N, support, target) summaries
    fig8_s9_plot.csv        510 Origin-ready (N, support, target) records

Internal resumability files start with '.s9_' in the output directory.
Independent reference: inverse Laplace by de Hoog and Talbot, twice at
separate mpmath decimal precisions, subject to agreement & physical bounds.
The reference is numerically cross-checked, NOT rigorously error-certified.

An archived strict float64 Hybrid result is used where the previous Fig. 8
reference was itself calculated by strict Hybrid (N>=5, support>=3), even if
that old reference was invalid/negative. For low-support cases, the saved
Hybrid trial at the strictest target is used as an explicitly labeled proxy.
Crucially, we never substitute the old Hybrid reference for the independent
reference; this is a diagnostic, NOT a new accuracy-matched timing benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import mpmath as mp

VERSION = 'Fig8_S9_AllGeometries_V2_20261109'  # Numerical protocol unchanged in V2.1; only paths updated
TARGETS = (1e-6, 1e-8, 1e-10)
TARGET_LABELS = {1e-6: '1e-6', 1e-8: '1e-8', 1e-10: '1e-10'}
KINDS = ('equal', 'negative_equal', 'mixed_equal', 'anisotropic')
EPS64 = 2.220446049250313e-16

RAW_FIELDS = [
    'protocol','N','support','support_fraction','r','kind','beta',
    'epsilon_rel_target','accuracy_label',
    'original_reference','original_reference_name','original_reference_ok',
    'independent_reference','independent_reference_status',
    'ref_internal_max_rel','ref_dehoog_low_high_rel',
    'ref_talbot_low_high_rel','ref_dehoog_talbot_high_rel',
    'ref_dehoog_high','ref_talbot_high',
    'strict_Hybrid_value','strict_Hybrid_source','strict_Hybrid_rel_error_HP',
    'original_reference_rel_error_HP',
    'positive_displacement_volume_HP','log10_C_sum','log10_eps64_C_sum',
    'original_Hybrid_value','original_Hybrid_FA_rtol','original_Hybrid_FA_order',
    'original_Hybrid_target_reached','Hybrid_rel_error_HP','Hybrid_reached_HP',
    'original_Aono_value','original_K_Aono',
    'original_Aono_target_reached','Aono_rel_error_HP','Aono_reached_HP',
    'original_both_target_reached','both_reached_HP',
    'n_reference_dps_low','n_reference_dps_high','message',
]
SUMMARY_FIELDS = [
    'protocol','N','support','support_fraction','epsilon_rel_target',
    'n_geometries','n_HP_verified','n_HP_unverified',
    'n_Hybrid_evaluated','n_Hybrid_reached_HP','n_Hybrid_failed_HP',
    'n_Hybrid_unavailable','n_Aono_evaluated','n_Aono_reached_HP',
    'n_both_reached_HP','n_old_Hybrid_reached','n_old_Aono_reached',
    'n_old_Hybrid_false_positives','n_old_Aono_false_positives',
    'n_strict_evaluated','n_strict_failed_HP','max_strict_rel_error_HP',
    'max_old_reference_rel_error_HP','median_log10_C_all',
    'max_log10_C_all','n_equal_failed_strict','n_negative_equal_failed_strict',
    'n_mixed_equal_failed_strict','n_anisotropic_failed_strict',
    'n_negative_equal','n_negative_equal_HP_verified',
    'first_failed_r_negative_equal','last_passed_r_negative_equal',
    'max_log10_C_negative_equal','median_log10_C_negative_equal',
    'max_strict_rel_error_negative_equal','n_negative_unverified',
    'completed_cell',
]
PLOT_FIELDS = [
    'N','support','epsilon_rel_target','accuracy_label','protocol',
    'first_failed_r_negative_equal','last_passed_r_negative_equal',
    'failure_fraction_negative_equal','n_negative_equal_failed_strict',
    'n_negative_equal_HP_verified','n_negative_unverified',
    'max_log10_C_negative_equal','median_log10_C_negative_equal',
    'max_log10_strict_error_negative_equal',
    'max_log10_strict_error_all',
    'n_strict_failed_HP','n_strict_evaluated',
    'n_Hybrid_failed_HP','n_Hybrid_unavailable',
    'n_old_Hybrid_false_positives','n_HP_unverified',
    'n_equal_failed_strict','n_mixed_equal_failed_strict',
    'n_anisotropic_failed_strict','completed_cell',
]


def truth(x):
    return str(x).strip().lower() in ('true', '1', 'yes')


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (ValueError, TypeError):
        return math.nan


def mpstr(x, digits=48):
    return mp.nstr(x, digits, min_fixed=-6, max_fixed=12)


def mp_ratio(v, reference):
    if not mp.isfinite(v) or not reference or not mp.isfinite(reference):
        return mp.nan
    return abs(v - reference) / abs(reference)


def raw_key(n, s, r, kind):
    return (int(n), int(s), round(float(r), 12), str(kind))


def stable_id(key):
    return f'{key[0]}:{key[1]}:{key[2]:.12f}:{key[3]}'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_input(path, complete_grid=True):
    cases = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'N','support','r','kind','beta','reference','reference_name',
                    'reference_ok','epsilon_rel_target','Hybrid_value','Aono_value'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing Fig. 8 columns: ' + ', '.join(sorted(missing)))
        for row in reader:
            n, s, r, kind = int(row['N']), int(row['support']), float(row['r']), row['kind']
            key = raw_key(n, s, r, kind)
            beta = json.loads(row['beta'])
            if not (2 <= n <= 18 and 1 <= s <= n and 0.5 - 1e-12 <= r <= 0.9 + 1e-12):
                raise ValueError('Unexpected geometry: ' + str(key))
            if kind not in KINDS or len(beta) != n:
                raise ValueError('Unexpected kind or beta length: ' + str(key))
            if sum(abs(v) > 0 for v in beta) != s:
                raise ValueError('Incorrect support: ' + str(key))
            if abs(math.sqrt(sum(float(v)**2 for v in beta)) - r) > 1e-12:
                raise ValueError('Incorrect displacement norm: ' + str(key))
            if s == 1 and kind not in ('equal','negative_equal'):
                raise ValueError('Duplicate single-support trajectory: ' + str(key))
            target = float(row['epsilon_rel_target'])
            if target not in TARGETS:
                raise ValueError('Unexpected target: ' + str((key,target)))
            if key not in cases:
                cases[key] = {'N':n,'s':s,'r':r,'kind':kind,'beta':beta,
                              'reference':row['reference'],
                              'reference_name':row['reference_name'],
                              'reference_ok':row['reference_ok'],
                              'trials':{}}
            c = cases[key]
            if (row['reference'] != c['reference'] or
                row['reference_name'] != c['reference_name'] or
                row['reference_ok'] != c['reference_ok'] or
                beta != c['beta']):
                raise ValueError('Inconsistent case across accuracy targets: ' + str(key))
            if target in c['trials']:
                raise ValueError('Duplicate trial: ' + str((key,target)))
            c['trials'][target] = row
    for key, c in cases.items():
        if set(c['trials']) != set(TARGETS):
            raise ValueError('Missing accuracy targets: ' + str(key))
    if complete_grid:
        expected = set()
        for n in range(2,19):
            for s in range(1,n+1):
                for j in range(9):
                    r = round(.5 + j*.05,12)
                    kinds = ('equal','negative_equal') if s==1 else KINDS
                    for k in kinds:
                        expected.add(raw_key(n,s,r,k))
        if set(cases) != expected:
            miss = list(expected-set(cases))[:5]
            extra = list(set(cases)-expected)[:5]
            raise ValueError(f'Fig8 V21.6 grid mismatch: expected {len(expected)}, '
                             f'got {len(cases)}, missing examples {miss}, extra examples {extra}')
    # Equal first => positive-displacement reference can be reused in the
    # other two equal-magnitude families at identical N,s,r.
    kind_order = {x:i for i,x in enumerate(KINDS)}
    return dict(sorted(cases.items(), key=lambda pair:
                       (pair[0][0],pair[0][1],pair[0][2],kind_order[pair[0][3]])))


def high_precision_beta(c):
    n, s, kind = c['N'], c['s'], c['kind']
    r = mp.mpf(str(c['r']))
    if kind == 'anisotropic':
        weights = [mp.power(2, s-1-i) for i in range(s)]
        norm = mp.sqrt(sum(w*w for w in weights))
        active = [r*w/norm for w in weights]
    else:
        amp = r/mp.sqrt(s)
        active = [(-amp if kind=='negative_equal' or
                   (kind=='mixed_equal' and i%2) else amp) for i in range(s)]
    return active+[mp.mpf(0)]*(n-s)


def laplace_transform(p, n, beta):
    value = mp.pi**(mp.mpf(n)/2) / (mp.power(2,n)*p**(mp.mpf(n)/2+1))
    root = mp.sqrt(p)
    for b in beta:
        if b:
            value *= mp.erfc(-b*root)
    return value


def inverse(c, digits, method):
    with mp.workdps(digits):
        beta = high_precision_beta(c)
        v = mp.invertlaplace(lambda p: laplace_transform(p,c['N'],beta),
                             mp.mpf(1),method=method)
        if not mp.isfinite(v):
            raise ArithmeticError('Nonfinite inverse Laplace result')
        return +v


def independent_reference(c, low, high, tolerance):
    dl = inverse(c,low,'dehoog')
    tl = inverse(c,low,'talbot')
    dh = inverse(c,high,'dehoog')
    th = inverse(c,high,'talbot')
    with mp.workdps(high+12):
        if dh == 0 or th == 0:
            differences = (mp.inf,mp.inf,mp.inf)
        else:
            differences = (abs(dl-dh)/abs(dh),abs(tl-th)/abs(th),
                           abs(dh-th)/abs(dh))
        full_ball = mp.pi**(mp.mpf(c['N'])/2)/mp.gamma(mp.mpf(c['N'])/2+1)
        is_physical = dh>0 and th>0 and dh<=full_ball and th<=full_ball
        status = 'PASS' if is_physical and max(differences)<tolerance else 'UNVERIFIED'
        return {
            'status': status,'hp':mpstr(dh), 'dehoog_high':mpstr(dh),
            'talbot_high':mpstr(th),
            'max_rel':mpstr(max(differences),24),
            'dehoog_precision_rel':mpstr(differences[0],24),
            'talbot_precision_rel':mpstr(differences[1],24),
            'cross_method_rel':mpstr(differences[2],24),
            'message':'' if status=='PASS' else 'Independent inversion mismatch or geometric bounds failure'
        }


def compute_geometry(c, args, positive_reference):
    rec={'N':c['N'],'support':c['s'],'r':c['r'],'kind':c['kind'],
         'reference_status':'ERROR','message':'','hp':'','max_rel':'',
         'dehoog_precision_rel':'','talbot_precision_rel':'','cross_method_rel':'',
         'dehoog_high':'','talbot_high':'','positive_volume':'',
         'log10_C_sum':'','log10_eps64_C_sum':'',
         'strict_Hybrid_value':'','strict_Hybrid_source':'','strict_Hybrid_rel_error_HP':'',
         'original_reference_rel_error_HP':'',
         'dps_low':args.dps_low,'dps_high':args.dps_high}
    try:
        ref = independent_reference(c,args.dps_low,args.dps_high,
                                    mp.mpf(args.agreement_tol))
        rec.update(ref)
        rec['reference_status']=ref['status']
        if ref['status']!='PASS':
            return rec
        with mp.workdps(args.dps_high+12):
            v = mp.mpf(ref['hp'])
            if c['kind'] in ('equal','anisotropic'):
                upos=v
            else:
                # The positive-center equal-magnitude case is guaranteed to
                # be present in the COMPLETE Fig. 8 input grid and processed
                # earlier. It has identical |beta|, not merely similar norm.
                poskey=raw_key(c['N'],c['s'],c['r'],'equal')
                pos=positive_reference.get(poskey)
                if pos is None or pos.get('reference_status')!='PASS':
                    rec['reference_status']='UNVERIFIED'
                    rec['message']='Missing/invalid equal-magnitude positive reference for C_sum'
                    return rec
                upos=mp.mpf(pos['hp'])
            rec['positive_volume']=mpstr(upos)
            cond=upos/abs(v)
            lg=mp.log10(cond)
            rec['log10_C_sum']=mpstr(lg,22)
            rec['log10_eps64_C_sum']=mpstr(lg+mp.log10(mp.mpf(str(EPS64))),22)
            old=safe_float(c['reference'])
            if math.isfinite(old):
                rec['original_reference_rel_error_HP']=mpstr(mp_ratio(mp.mpf(str(old)),v),23)
            if c['reference_name']=='Hybrid_strict_stability_checked':
                hstrict=old
                rec['strict_Hybrid_source']='Fig8_stored_strict_float64_Hybrid'
            else:
                # Low-support/low-dimensional archived reference is a
                # different analytic route, so never label it as Hybrid.
                hstrict=safe_float(c['trials'][1e-10]['Hybrid_value'])
                rec['strict_Hybrid_source']='Fig8_stored_Hybrid_target_1e-10_proxy'
            if math.isfinite(hstrict):
                rec['strict_Hybrid_value']=repr(hstrict)
                rec['strict_Hybrid_rel_error_HP']=mpstr(mp_ratio(mp.mpf(str(hstrict)),v),23)
        return rec
    except Exception as e:
        rec['reference_status']='ERROR'
        rec['message']=f'{type(e).__name__}: {e}'
        return rec


def build_raw_row(c, result, target):
    orig=c['trials'][target]
    row=dict.fromkeys(RAW_FIELDS,'')
    row.update(protocol=VERSION, N=c['N'], support=c['s'],
               support_fraction=c['s']/c['N'],r=c['r'],kind=c['kind'],
               beta=json.dumps(c['beta']),epsilon_rel_target=target,
               accuracy_label=TARGET_LABELS[target],
               original_reference=c['reference'],
               original_reference_name=c['reference_name'],
               original_reference_ok=c['reference_ok'],
               independent_reference=result.get('hp',''),
               independent_reference_status=result.get('reference_status',''),
               ref_internal_max_rel=result.get('max_rel',''),
               ref_dehoog_low_high_rel=result.get('dehoog_precision_rel',''),
               ref_talbot_low_high_rel=result.get('talbot_precision_rel',''),
               ref_dehoog_talbot_high_rel=result.get('cross_method_rel',''),
               ref_dehoog_high=result.get('dehoog_high',''),
               ref_talbot_high=result.get('talbot_high',''),
               strict_Hybrid_value=result.get('strict_Hybrid_value',''),
               strict_Hybrid_source=result.get('strict_Hybrid_source',''),
               strict_Hybrid_rel_error_HP=result.get('strict_Hybrid_rel_error_HP',''),
               original_reference_rel_error_HP=result.get('original_reference_rel_error_HP',''),
               positive_displacement_volume_HP=result.get('positive_volume',''),
               log10_C_sum=result.get('log10_C_sum',''),
               log10_eps64_C_sum=result.get('log10_eps64_C_sum',''),
               original_Hybrid_value=orig.get('Hybrid_value',''),
               original_Hybrid_FA_rtol=orig.get('Hybrid_fa_rtol',''),
               original_Hybrid_FA_order=orig.get('Hybrid_FA_order',''),
               original_Hybrid_target_reached=orig.get('Hybrid_target_reached',''),
               original_Aono_value=orig.get('Aono_value',''),
               original_K_Aono=orig.get('K_Aono',''),
               original_Aono_target_reached=orig.get('Aono_target_reached',''),
               original_both_target_reached=orig.get('both_target_reached',''),
               n_reference_dps_low=result.get('dps_low',''),
               n_reference_dps_high=result.get('dps_high',''),
               message=result.get('message',''))
    if result.get('reference_status')=='PASS':
        with mp.workdps(86):
            ref=mp.mpf(result['hp'])
            for label in ('Hybrid','Aono'):
                val=safe_float(orig.get(label+'_value'))
                if math.isfinite(val):
                    e=mp_ratio(mp.mpf(str(val)),ref)
                    row[label+'_rel_error_HP']=mpstr(e,24)
                    row[label+'_reached_HP']=str(bool(e<=mp.mpf(str(target))))
            row['both_reached_HP']=(
                str(row['Hybrid_reached_HP']=='True' and row['Aono_reached_HP']=='True')
                if row['Hybrid_reached_HP'] and row['Aono_reached_HP'] else '')
    return row


def finite_values(rows,col,filter_kind=None):
    return [safe_float(r.get(col)) for r in rows
            if (filter_kind is None or r['kind']==filter_kind)
            and math.isfinite(safe_float(r.get(col)))]


def maximum(vals):
    return max(vals) if vals else ''


def median(vals):
    if not vals:return ''
    vals=sorted(vals)
    m=len(vals)//2
    return vals[m] if len(vals)%2 else (vals[m-1]+vals[m])/2


def log10_or_blank(v):
    return math.log10(v) if isinstance(v,(float,int)) and math.isfinite(v) and v>0 else ''


def summarize(cases, saved, complete_grid):
    groups=defaultdict(list)
    for key,c in cases.items():
        if key not in saved:continue
        result=saved[key]
        for target in TARGETS:
            groups[(c['N'],c['s'],target)].append(build_raw_row(c,result,target))
    summary=[];plots=[]
    for (n,s,t),rr in sorted(groups.items()):
        valid=[r for r in rr if r['independent_reference_status']=='PASS']
        bad=[r for r in rr if r['independent_reference_status']!='PASS']
        testable=lambda r: r['independent_reference_status']=='PASS'
        hs=[r for r in valid if r['Hybrid_reached_HP'] in ('True','False')]
        aa=[r for r in valid if r['Aono_reached_HP'] in ('True','False')]
        st=[r for r in valid if math.isfinite(safe_float(r['strict_Hybrid_rel_error_HP']))]
        neg=[r for r in valid if r['kind']=='negative_equal']
        neg_all=[r for r in rr if r['kind']=='negative_equal']
        neg_strict=[r for r in neg if math.isfinite(safe_float(r['strict_Hybrid_rel_error_HP']))]
        fail=lambda r: safe_float(r['strict_Hybrid_rel_error_HP'])>t
        fail_rows=[r for r in st if fail(r)]
        neg_bad=[r for r in neg_strict if fail(r)]
        neg_good=[r for r in neg_strict if not fail(r)]
        equal_count=len([r for r in fail_rows if r['kind']=='equal'])
        mixed_count=len([r for r in fail_rows if r['kind']=='mixed_equal'])
        aniso_count=len([r for r in fail_rows if r['kind']=='anisotropic'])
        expected = (18 if s==1 else 36)
        is_complete=(len(rr)==expected) if complete_grid else (len(rr)==len(rr))
        n_old_false=sum(truth(r['original_Hybrid_target_reached']) and r['Hybrid_reached_HP']=='False' for r in valid)
        n_old_aono_false=sum(truth(r['original_Aono_target_reached']) and r['Aono_reached_HP']=='False' for r in valid)
        lgs=finite_values(valid,'log10_C_sum')
        neg_lgs=finite_values(valid,'log10_C_sum','negative_equal')
        strict_e=finite_values(st,'strict_Hybrid_rel_error_HP')
        neg_e=finite_values(neg_strict,'strict_Hybrid_rel_error_HP')
        old_e=finite_values(valid,'original_reference_rel_error_HP')
        row=dict(protocol=VERSION,N=n,support=s,support_fraction=s/n,epsilon_rel_target=t,
            n_geometries=len(rr),n_HP_verified=len(valid),n_HP_unverified=len(bad),
            n_Hybrid_evaluated=len(hs),n_Hybrid_reached_HP=sum(r['Hybrid_reached_HP']=='True' for r in hs),
            n_Hybrid_failed_HP=sum(r['Hybrid_reached_HP']=='False' for r in hs),
            n_Hybrid_unavailable=len(valid)-len(hs),
            n_Aono_evaluated=len(aa),n_Aono_reached_HP=sum(r['Aono_reached_HP']=='True' for r in aa),
            n_both_reached_HP=sum(r['both_reached_HP']=='True' for r in valid),
            n_old_Hybrid_reached=sum(truth(r['original_Hybrid_target_reached']) for r in rr),
            n_old_Aono_reached=sum(truth(r['original_Aono_target_reached']) for r in rr),
            n_old_Hybrid_false_positives=n_old_false,n_old_Aono_false_positives=n_old_aono_false,
            n_strict_evaluated=len(st),n_strict_failed_HP=len(fail_rows),
            max_strict_rel_error_HP=maximum(strict_e),max_old_reference_rel_error_HP=maximum(old_e),
            median_log10_C_all=median(lgs),max_log10_C_all=maximum(lgs),
            n_equal_failed_strict=equal_count,
            n_negative_equal_failed_strict=len(neg_bad),
            n_mixed_equal_failed_strict=mixed_count,
            n_anisotropic_failed_strict=aniso_count,
            n_negative_equal=len(neg_all),n_negative_equal_HP_verified=len(neg),
            first_failed_r_negative_equal=min((float(r['r']) for r in neg_bad),default=''),
            last_passed_r_negative_equal=max((float(r['r']) for r in neg_good),default=''),
            max_log10_C_negative_equal=maximum(neg_lgs),
            median_log10_C_negative_equal=median(neg_lgs),
            max_strict_rel_error_negative_equal=maximum(neg_e),
            n_negative_unverified=len(neg_all)-len(neg),
            completed_cell=str(is_complete))
        summary.append(row)
        plots.append(dict(
            N=n,support=s,epsilon_rel_target=t,
            accuracy_label=TARGET_LABELS[t],protocol=VERSION,
            first_failed_r_negative_equal=row['first_failed_r_negative_equal'],
            last_passed_r_negative_equal=row['last_passed_r_negative_equal'],
            failure_fraction_negative_equal=(len(neg_bad)/len(neg_strict) if neg_strict else ''),
            n_negative_equal_failed_strict=row['n_negative_equal_failed_strict'],
            n_negative_equal_HP_verified=row['n_negative_equal_HP_verified'],
            n_negative_unverified=row['n_negative_unverified'],
            max_log10_C_negative_equal=row['max_log10_C_negative_equal'],
            median_log10_C_negative_equal=row['median_log10_C_negative_equal'],
            max_log10_strict_error_negative_equal=log10_or_blank(maximum(neg_e)),
            max_log10_strict_error_all=log10_or_blank(maximum(strict_e)),
            n_strict_failed_HP=row['n_strict_failed_HP'],
            n_strict_evaluated=row['n_strict_evaluated'],
            n_Hybrid_failed_HP=row['n_Hybrid_failed_HP'],
            n_Hybrid_unavailable=row['n_Hybrid_unavailable'],
            n_old_Hybrid_false_positives=n_old_false,
            n_HP_unverified=row['n_HP_unverified'],
            n_equal_failed_strict=equal_count,
            n_mixed_equal_failed_strict=mixed_count,
            n_anisotropic_failed_strict=aniso_count,
            completed_cell=str(is_complete)))
    raw=[]
    for key,c in cases.items():
        if key not in saved:continue
        for t in TARGETS:raw.append(build_raw_row(c,saved[key],t))
    return raw,summary,plots


def save_csv_atomic(path,columns,records):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(records)
    tmp.replace(path)


def validate_outputs(raw,summary,plot,cases,saved,complete_grid):
    if len(raw)!=3*len(saved):raise AssertionError('Incorrect raw count')
    if len(summary)!=len(plot):raise AssertionError('Incorrect summary/plot count')
    if complete_grid and len(saved)==len(cases):
        if len(raw)!=17442 or len(summary)!=510 or len(plot)!=510:
            raise AssertionError('Incorrect full-grid output sizes')
    rkeys={(x['N'],x['support'],x['r'],x['kind'],x['epsilon_rel_target']) for x in raw}
    if len(rkeys)!=len(raw):raise AssertionError('Duplicate raw output')
    for x,y in zip(summary,plot):
        if (x['N'],x['support'],x['epsilon_rel_target'])!=(y['N'],y['support'],y['epsilon_rel_target']):
            raise AssertionError('Summary and plot have different ordering')
        if x['n_HP_verified']+x['n_HP_unverified']!=x['n_geometries']:
            raise AssertionError('Broken reference counts')
        if x['n_strict_failed_HP']>x['n_strict_evaluated']:
            raise AssertionError('Broken strict-Hybrid counts')
    return {'raw_rows':len(raw),'summary_rows':len(summary),'plot_rows':len(plot),
            'completed_geometry':len(saved),'expected_geometry':len(cases),
            'hp_verified':sum(c['reference_status']=='PASS' for c in saved.values()),
            'hp_unverified':sum(c['reference_status']!='PASS' for c in saved.values())}


def validate_written_csvs(output, expected):
    """Verify the three on-disk CSVs (not just in-memory DataFrames)."""
    specs=(('fig8_s9_raw.csv',RAW_FIELDS,'raw_rows'),
           ('fig8_s9_summary.csv',SUMMARY_FIELDS,'summary_rows'),
           ('fig8_s9_plot.csv',PLOT_FIELDS,'plot_rows'))
    tables={}
    for filename,fields,count_name in specs:
        with open(output/filename,encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=fields:
                raise AssertionError('CSV header mismatch: '+filename)
            rows=list(reader)
        if len(rows)!=expected[count_name]:
            raise AssertionError('CSV row count mismatch: '+filename)
        tables[filename]=rows
    raw=tables['fig8_s9_raw.csv']
    if len({(x['N'],x['support'],x['r'],x['kind'],x['epsilon_rel_target']) for x in raw})!=len(raw):
        raise AssertionError('Duplicate key in saved raw CSV')
    by_cell={(x['N'],x['support'],x['epsilon_rel_target']):x
             for x in tables['fig8_s9_summary.csv']}
    plot_keys={(x['N'],x['support'],x['epsilon_rel_target'])
               for x in tables['fig8_s9_plot.csv']}
    if len(by_cell)!=len(tables['fig8_s9_summary.csv']) or set(by_cell)!=plot_keys:
        raise AssertionError('Saved summary/plot (N,s,epsilon) keys disagree')
    for x in tables['fig8_s9_summary.csv']:
        if int(x['n_HP_verified'])+int(x['n_HP_unverified'])!=int(x['n_geometries']):
            raise AssertionError('Inconsistent saved reference-count arithmetic')
    return True


def write_results(output,cases,saved,complete_grid):
    raw,summary,plots=summarize(cases,saved,complete_grid)
    report=validate_outputs(raw,summary,plots,cases,saved,complete_grid)
    save_csv_atomic(output/'fig8_s9_raw.csv',RAW_FIELDS,raw)
    save_csv_atomic(output/'fig8_s9_summary.csv',SUMMARY_FIELDS,summary)
    save_csv_atomic(output/'fig8_s9_plot.csv',PLOT_FIELDS,plots)
    validate_written_csvs(output,report)
    return report


def read_saved(path):
    saved={}
    if not path.exists():return saved
    with open(path,encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            try:
                rec=json.loads(line)
            except json.JSONDecodeError:
                # A torn final write after forced shutdown is safe to ignore,
                # provided the truncated record is the last record.
                if not list(f):break
                raise RuntimeError(f'Checkpoint corruption at line {line_no}')
            key=raw_key(rec['N'],rec['support'],rec['r'],rec['kind'])
            saved[key]=rec
    return saved


def append_checkpoint(file,record):
    with open(file,'a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(record,ensure_ascii=True,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def resolve_fig8_raw(explicit=None):
    """Find full per-geometry Fig.8 results; plot/summary are insufficient.

    Works when launched from a project root, from app/, or from the
    S9 script's own directory. --raw overrides automatic discovery.
    """
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if path.name.lower().startswith(('fig8_plot', 'fig8_summary')):
            raise FileNotFoundError(
                f'S9 requires fig8_raw.csv (per-geometry data), not {path.name}. '
                f'Check {path.parent / "fig8_raw.csv"}.')
        if not path.is_file():
            raise FileNotFoundError(f'Fig8 raw CSV not found: {path}')
        return path

    # The user's directory layout: app/results_fig8_V20261108/fig8_plot.csv.
    # We specifically require its companion fig8_raw.csv.
    bases=[Path.cwd(), Path(__file__).resolve().parent]
    bases += list(Path(__file__).resolve().parents)[:3]
    seen=set()
    candidates=[]
    for base in bases:
        for rel in (
            'app/results_fig8_V20261108/fig8_raw.csv',
            'results_fig8_V20261108/fig8_raw.csv',
            'fig8_raw.csv',
            'fig8_raw(2).csv',
        ):
            path=(base/rel).resolve()
            if path not in seen:
                seen.add(path)
                candidates.append(path)
                if path.is_file():
                    return path
    raise FileNotFoundError(
        'Fig8 per-geometry input not found. Expected '
        'app/results_fig8_V20261108/fig8_raw.csv '
        '(beside fig8_plot.csv). Use --raw PATH to specify another location. '
        'S9 cannot reconstruct individual geometry data from fig8_plot.csv. '
        'Searched: '+', '.join(str(x) for x in candidates[:8])
    )


def resolve_output_dir(explicit,source):
    if explicit:
        return Path(explicit).expanduser().resolve()
    # Keep S9 data alongside the original Fig8 results, never in that
    # original directory, to avoid overwriting fig8_plot/summary/raw.csv.
    return (source.parent.parent/'results_fig8_S9').resolve()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw',default=None,help='Optional Fig8 raw CSV; automatically detects app/results_fig8_V20261108/fig8_raw.csv')
    p.add_argument('--output-dir',default=None,help='Output directory; default app/results_fig8_S9/ beside the Fig8 source')
    p.add_argument('--dps-low',type=int,default=50)
    p.add_argument('--dps-high',type=int,default=75)
    p.add_argument('--agreement-tol',default='1e-22')
    p.add_argument('--save-every',type=int,default=100,help='Refresh all three CSVs every K geometries')
    p.add_argument('--fresh',action='store_true',help='Start from zero; refuse to overwrite an existing checkpoint')
    p.add_argument('--verify-csv-only',action='store_true',help='Validate/rebuild three CSVs from checkpoint without computing')
    # For internal small-grid smoke tests only. Do not use in final Fig.8 S9.
    p.add_argument('--allow-incomplete-input',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.dps_low<35 or a.dps_high<a.dps_low+15:p.error('dps-high must exceed dps-low by >=15; dps-low>=35')
    if a.save_every<1:p.error('--save-every must be positive')
    if not 0 < float(a.agreement_tol) < 1:p.error('Invalid agreement tolerance')
    if a.fresh and a.verify_csv_only:p.error('Do not combine --fresh and --verify-csv-only')
    return a


def main(argv=None):
    a=parse_args(argv)
    source=resolve_fig8_raw(a.raw)
    cases=load_input(source,not a.allow_incomplete_input)
    dest=resolve_output_dir(a.output_dir, source)
    dest.mkdir(parents=True,exist_ok=True)
    manifest=dest/'.s9_protocol.json'
    checkpoint=dest/'.s9_checkpoint.jsonl'
    spec=dict(script=VERSION,source_hash=sha256_file(source),
              dps_low=a.dps_low,dps_high=a.dps_high,
              agreement_tol=str(mp.mpf(a.agreement_tol)),geometry_count=len(cases))
    if a.fresh and (manifest.exists() or checkpoint.exists()):
        raise RuntimeError('--fresh refuses to overwrite an existing S9 run. Use a new --output-dir.')
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old!=spec:
            raise RuntimeError('Input/precision/protocol mismatch with checkpoint; use new --output-dir.')
    elif checkpoint.exists():
        raise RuntimeError('Checkpoint found without protocol metadata; use new --output-dir.')
    else:
        manifest.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    saved=read_saved(checkpoint)
    if set(saved)-set(cases):
        raise RuntimeError('Checkpoint contains configurations absent from current input')
    # Resume references, reusing the equal-configuration high-precision result.
    todo=[] if a.verify_csv_only else [k for k in cases if k not in saved]
    print(f'S9 complete-grid stability audit: {len(cases)} geometries / {len(cases)*3} trials',flush=True)
    print('Already completed:',len(saved),'remaining:',len(todo),' output:',dest,flush=True)
    if a.verify_csv_only and not saved:
        raise RuntimeError('No checkpoint records available to validate/rebuild')
    for count,key in enumerate(todo,1):
        c=cases[key]
        data=compute_geometry(c,a,saved)
        append_checkpoint(checkpoint,data)
        saved[key]=data
        if count<=3 or count%20==0 or data['reference_status']!='PASS':
            print(f'[{count}/{len(todo)}] N={c["N"]} s={c["s"]} r={c["r"]:.2f} '
                  f'{c["kind"]}: {data["reference_status"]}, '
                  f'log10(C)={data.get("log10_C_sum","")}, '
                  f'strict_err={data.get("strict_Hybrid_rel_error_HP","")}',flush=True)
        if count%a.save_every==0:
            counts=write_results(dest,cases,saved,not a.allow_incomplete_input)
            print('  CSV checkpoint:',counts,flush=True)
    counts=write_results(dest,cases,saved,not a.allow_incomplete_input)
    print('DONE:',counts,flush=True)
    print('CSV outputs: fig8_s9_raw.csv, fig8_s9_summary.csv, fig8_s9_plot.csv',flush=True)
    return 0 if len(saved)==len(cases) else 3


if __name__=='__main__':
    main()
