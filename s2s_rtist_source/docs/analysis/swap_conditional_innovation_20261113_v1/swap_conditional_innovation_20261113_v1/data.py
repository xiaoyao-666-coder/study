"""Fold-local preprocessing. Only callers define the permitted site/year scope."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
from schema import KEYS,CYCLE,STATES,INPUTS,LABELS,GRID,PHYSICAL,RAIN,INITIAL,COUNTS
from rootstate import attach,fit_normalizer,apply_normalizer,RAW
from soil_features import augment
from bspline_basis import CubicBSplineBasis

GROUPS=[list(range(2)),list(range(2,7)),[7,50,51]]+[list(range(8+7*i,15+7*i)) for i in range(6)]

def need(ok,msg):
    if not ok:raise ValueError(msg)

def weights(sites):
    s=np.asarray(sites);unique=np.unique(s)
    return np.array([1/(len(unique)*np.sum(s==x)) for x in s])

def load(sites,years,truth=False):
    import runtime as rt
    frames=[]
    for site in sorted(sites):
        for year in sorted(years):
            name=f'years/{year}/{site}/'+('labels.csv' if truth else 'inputs.csv')
            f=pd.read_csv(rt.checked(name),float_precision='round_trip',keep_default_na=False,dtype={k:str for k in KEYS[:4]})
            need(f.columns.tolist()==(LABELS if truth else INPUTS),'Column whitelist '+name)
            need(set(f.site_id)=={site} and set(f.target_year)=={str(year)},'Partition identity')
            need(year in range(2014,2017) and f.sample_id.is_unique,'Allowed unique keys')
            need(len(f)==8*COUNTS[site][year-2014],'Exact partition cycle count '+name)
            if not truth:
                soil=pd.read_csv(rt.checked(f'soil/{site}/soil.csv'),float_precision='round_trip')
                f,_=augment(f,soil)
                root=pd.read_csv(rt.checked(f'years/{year}/{site}/rootstate.csv'),float_precision='round_trip',dtype={'target_year':str})
                f=attach(f,root)
            frames.append(f)
    f=pd.concat(frames,ignore_index=True).sort_values(CYCLE+['irrigation_mm'],kind='stable').reset_index(drop=True)
    validate(f,truth)
    return f

def validate(f,truth=False):
    need(not f.empty and f.sample_id.is_unique and len(f)%8==0,'Nonempty complete cycles')
    need(f.irrigation_mm.tolist()==GRID*(len(f)//8),'Ordered eight-dose grid')
    need(f.groupby(CYCLE,sort=False).size().eq(8).all(),'Unique eight-dose cycles')
    need(np.isfinite(f[LABELS[4:] if truth else STATES+RAW+[INITIAL]].to_numpy(float)).all(),'Finite data')
    if truth:
        a=f[PHYSICAL].to_numpy(float)
        need(np.all(f.target_net_gain_7d.to_numpy()[::8]==0),'True zero anchor')
        need(np.all(a[:,0]>=0) and np.all((a[:,1:]>=0)&(a[:,1:]<=1)),'Physical labels domain')
    else:
        need(not f.groupby(CYCLE)[STATES+RAW+[INITIAL]].nunique(dropna=False).gt(1).any().any(),'Cycle context invariant')
    for row in f.iloc[::8].itertuples():
        d=date.fromisoformat(row.decision_date)
        need(str(d.year)==str(row.target_year) and d+timedelta(days=7)<date(d.year+1,1,1),'Label maturity before next training cutoff')

def fit_preprocessing(f,y,sites):
    need(f[KEYS].equals(y[KEYS]),'Input/label alignment')
    ctx=f.iloc[::8];raw=ctx[STATES].to_numpy(float);w=weights(ctx.site_id)
    mean=w@raw;std=np.sqrt(w@((raw-mean)**2));active=std>1e-6
    scale=np.where(active,std,1.);fw=np.zeros(52)
    groups=[[i for i in g if active[i]] for g in GROUPS];groups=[g for g in groups if g]
    for g in groups:fw[g]=1/(len(groups)*len(g))
    norm=fit_normalizer(f,sites)
    rw=np.repeat(w/8,8);gy=y.target_net_gain_7d.to_numpy(float);mu=rw@gy
    gs=max(float(np.sqrt(rw@((gy-mu)**2))),1.)
    ph=y[PHYSICAL].to_numpy(float);rain=f[RAIN].sum(axis=1).to_numpy(float)+f.irrigation_mm.to_numpy(float)
    def stdof(a):
        if a.ndim==1:return float(np.sqrt(rw@((a-rw@a)**2)))
        mu=float(np.sum(rw[:,None]*a)/a.shape[1]);return float(np.sqrt(np.sum(rw[:,None]*(a-mu)**2)/a.shape[1]))
    scales=np.array([max(stdof(ph[:,0]),1.),max(stdof(ph[:,1:]),.01),max(stdof(rain),1.)])
    av=max(float(rw@ph[:,0]),1e-3);v=np.clip(rw@ph[:,1:],1e-4,1-1e-4)
    inv=lambda x:x+np.log(-np.expm1(-x))
    offsets=np.r_[inv(av),np.log(v/(1-v)),inv(1.)]
    prep=dict(mean=mean,scale=scale,weights=fw,state_mean=norm['state_mean'],state_scale=norm['state_scale'],
              gain_scale=np.array([gs]),physical_scales=scales,latent=np.r_[scales[0],np.ones(7),scales[2]],offsets=offsets)
    evidence=dict(raw=raw,population_std=std,cycle_weights=w,normalization_sample_ids=ctx.sample_id.to_numpy(dtype=str),
                  root_population_std=norm['state_population_std'],root_raw=norm['normalization_raw'])
    return prep,evidence

def tensors(f,prep,y=None):
    raw=f.iloc[::8][STATES].to_numpy(float);n=len(raw)
    projected=raw.copy();weather=projected[:,8:50].reshape(n,6,7);weather[:]=weather.mean(2,keepdims=True)
    z=(projected-prep['mean'])/prep['scale']*np.sqrt(prep['weights'])
    _,u=apply_normalizer(f.iloc[::8],prep['state_mean'],prep['state_scale'])
    dose=f.irrigation_mm.to_numpy(float).reshape(n,8)
    B=CubicBSplineBasis(lower=0,upper=60,basis_count=6).evaluate(dose.ravel()).reshape(n,8,6)
    ev=dict(z=z,u=u[:,:3]/np.sqrt(3),B=B,T=B*dose[:,:,None]/60,dose=dose,w=weights(f.site_id.iloc[::8]),
            rain=f[RAIN].sum(axis=1).to_numpy(float).reshape(n,8),storage=f[INITIAL].to_numpy(float).reshape(n,8),
            sample_ids=f.sample_id.to_numpy(dtype=str).reshape(n,8),sites=f.site_id.to_numpy(dtype=str)[::8],
            dates=f.decision_date.to_numpy(dtype=str)[::8],years=f.target_year.to_numpy(dtype=str)[::8],
            **{k:prep[k] for k in ('physical_scales','latent','offsets')})
    if y is not None:
        need(f[KEYS].equals(y[KEYS]),'Aligned tensor labels')
        ev.update(y=y.target_net_gain_7d.to_numpy(float).reshape(n,8)/prep['gain_scale'][0],
                  physical=y[PHYSICAL].to_numpy(float).reshape(n,8,8))
    return ev
