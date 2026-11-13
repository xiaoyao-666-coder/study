"""Predeclared macro economic comparisons; no candidate selection."""
import pandas as pd
import numpy as np
import runtime as rt
from kernel import METHODS
FIELDS=['gain','regret','vs_raw','vs_static','harmed_raw','harmed_static','negative_loss','negative_rate','repairs_raw','restores_raw','spoils_correct_raw']
def macro(f):return f.groupby('site').mean(numeric_only=True)['effect'].mean()
def summarize(frame):
    # Each target has 4 donors x 3 seeds; they are correlated model versions.
    needcols=['site','date','mature','method']
    cycles=frame.groupby(needcols,as_index=False)[FIELDS].mean()
    tables=[];contrasts=[];site_effects=[];seed_effects=[]
    for scope in ['mature','all']:
        src=frame[frame.mature==1] if scope=='mature' else frame
        cy=cycles[cycles.mature==1] if scope=='mature' else cycles
        site=cy.groupby(['site','method'])[FIELDS].mean().reset_index()
        for method in METHODS:
            row=site[site.method==method][FIELDS].mean().to_dict()
            row.update(scope=scope,method=method,cycles=len(cy[cy.method==method]),worst_regret=float(src[src.method==method].regret.max()))
            tables.append(row)
        wide=cy.pivot(index=['site','date','mature'],columns='method',values='gain')
        for ref in ['static','raw','reverse']:
            effect=(wide['on']-wide[ref]).rename('effect').reset_index();ss=effect.groupby('site').effect.mean()
            best=effect.effect.idxmax();drop=effect.drop(index=best)
            seeds=[]
            for seed,group in src.groupby('seed'):
                z=group.groupby(['site','date','method']).gain.mean().unstack('method')
                sv=float((z['on']-z[ref]).groupby('site').mean().mean());seeds.append(sv)
                seed_effects.append(dict(scope=scope,reference=ref,seed=int(seed),effect=sv))
            for name,v in ss.items():site_effects.append(dict(scope=scope,reference=ref,site=name,effect=float(v)))
            contrasts.append(dict(scope=scope,reference=ref,effect=float(ss.mean()),positive_sites=int((ss>1e-10).sum()),
                seeds_min=float(min(seeds)),remove_best=float(macro(drop)),
                min_leave_site=float(min(ss.drop(s).mean() for s in ss.index)),
                best_site=str(effect.loc[best,'site']),best_date=str(effect.loc[best,'date'])))
    table=pd.DataFrame(tables);cc=pd.DataFrame(contrasts);gates={}
    for ref in ['static','raw','reverse']:
        r=cc[(cc.scope=='mature')&(cc.reference==ref)].iloc[0];gates['mature_positive_'+ref]=bool(r.effect>1e-10)
    r=cc[(cc.scope=='mature')&(cc.reference=='static')].iloc[0]
    gates.update(three_sites=bool(r.positive_sites>=3),each_seed=bool(r.seeds_min>1e-10),remove_best=bool(r.remove_best>1e-10),leave_each_site=bool(r.min_leave_site>1e-10))
    for scope in ['mature','all']:
        tab=table[table.scope==scope].set_index('method')
        for ref in ['static','raw']:
            for field in ['negative_loss','negative_rate','worst_regret']:
                gates[scope+'_'+field+'_vs_'+ref]=bool(tab.loc['on',field]<=tab.loc[ref,field]+1e-10)
            if scope=='all':gates['all_positive_'+ref]=bool(float(cc[(cc.scope=='all')&(cc.reference==ref)].iloc[0].effect)>1e-10)
    return dict(cycles=cycles,summary=table,contrasts=cc,sites=pd.DataFrame(site_effects),seeds=pd.DataFrame(seed_effects),gates=gates)
def run():
    rt.require_server();tasks=rt.protocol()['tasks'];frames=[]
    freeze=rt.read(rt.OUT/'score_freeze.json');rt.need(len(freeze['tasks'])==60,'Score completeness')
    for t in tasks:
        frames.append(pd.read_csv(rt.prior_file('score',t['id'],'scores.csv')))
    f=pd.concat(frames,ignore_index=True)
    rt.need(f.groupby(['site','date','method']).size().eq(12).all(),'Twelve balanced donor-seed versions')
    rt.need(f.groupby(['task','date']).method.nunique().eq(4).all(),'Four complete methods')
    d=rt.OUT/'aggregate';d.mkdir();f.to_csv(d/'versions.csv',index=False)
    result=summarize(f)
    for n in ['cycles','summary','contrasts','sites','seeds']:result[n].to_csv(d/(n+'.csv'),index=False)
    gates=result['gates'];rt.put(d/'screening.json',dict(engineering_passed=True,science_passed=all(gates.values()),gates=gates,automatic_promotion=False))
    text=['# Model-matched feedback','', 'Fixed development screen: '+('PASS' if all(gates.values()) else 'FAIL'),'',
          'Positive on-reference is higher true net gain. Water error is not an adoption criterion.','',result['contrasts'].to_string(index=False),
          '', 'All 2016 years are reused development data. This is fixed-logger replay, not a closed-loop policy result.']
    (d/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    rt.put(d/'seal.json',dict(files=rt.filemap(d)))
