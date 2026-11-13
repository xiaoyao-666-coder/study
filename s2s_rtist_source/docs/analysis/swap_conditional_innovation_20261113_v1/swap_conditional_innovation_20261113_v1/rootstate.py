"""Audited predecision state join and source-only cycle normalization."""
from datetime import date,timedelta
import numpy as np
from inference_model import numeric
CYCLE_KEYS=('target_year','site_id','decision_date')
GRID=(0,10,15,20,25,30,40,60)

COLS=['sample_id','target_year','site_id','decision_date','state_timestamp','crop_timestamp','root_depth_cm','prior_start','prior_end','rootzone_phead_cm','prior3_tact_mm','prior3_tpot_mm']
RAW=['rootzone_phead_cm','prior3_tact_mm','prior3_tpot_mm']
ATTACHED=[c for c in COLS if c not in ['sample_id',*CYCLE_KEYS]]
def need(v,m):
    if not v:raise ValueError(m)

def attach(frame,state):
    need(state.columns.tolist()==COLS,'Root-state whitelist schema')
    need(not any(c in frame for c in ATTACHED),'Already attached root state')
    need(not frame.sample_id.duplicated().any() and not state.sample_id.duplicated().any(),'Duplicate sample identity')
    keys=list(CYCLE_KEYS)
    need(not state.duplicated(keys).any(),'Duplicate state cycle')
    need(set(frame.site_id.astype(str))==set(state.site_id.astype(str)),'State site mismatch')
    f=frame.copy();r=state.copy()
    for k in keys:f[k]=f[k].astype(str);r[k]=r[k].astype(str)
    zero=f[f.irrigation_mm==0]
    need(len(zero)==len(r) and len(f)==8*len(r),'State cycle coverage')
    need(set(map(tuple,zero[keys+['sample_id']].to_numpy()))==set(map(tuple,r[keys+['sample_id']].to_numpy())),'State zero-dose identity')
    for row in r.itertuples(index=False):
        d=date.fromisoformat(row.decision_date)
        need(str(d.year)==str(row.target_year) and 2014<=d.year<=2018,'State year')
        need(row.state_timestamp==str(d)+'T06:00:00','State must be D06')
        need(row.crop_timestamp==row.prior_end==str(d-timedelta(days=1)) and row.prior_start==str(d-timedelta(days=3)),'Antecedent interval')
    v=numeric(r,RAW+['root_depth_cm'])
    need(np.isfinite(v).all() and np.all(v[:,1:3]>=0) and np.all((v[:,3]>0)&(v[:,3]<=100)),'Root-state numerical domain')
    need(np.all(v[v[:,2]==0,1]==0),'Nonzero Tact with zero Tpot')
    need(np.all(v[:,1]<=v[:,2]+1e-9),'Tact exceeds Tpot')
    out=f.merge(r.drop(columns='sample_id'),on=keys,how='left',validate='many_to_one',sort=False)
    need(out.sample_id.tolist()==f.sample_id.tolist() and not out[ATTACHED].isna().any().any(),'State join order/coverage')
    for _,g in out.groupby(keys,sort=False):need(g.irrigation_mm.tolist()==list(GRID),'Root-state dose grid')
    return out

def raw_coordinates(frame):
    v=numeric(frame,RAW);need(np.isfinite(v).all(),'Finite root state')
    h,tact,tpot=v.T
    need(np.all(tact>=0) and np.all(tpot>=0) and np.all(tact[tpot==0]==0) and np.all(tact<=tpot+1e-9),'Transpiration domain')
    ratio=np.ones(len(v));np.divide(tact,tpot,out=ratio,where=tpot>0)
    return np.column_stack([np.sign(h)*np.log1p(abs(h)),ratio,(tpot==0).astype(float)])

def fit_normalizer(frame,source_sites):
    need(len(source_sites) in (3,4,5) and len(set(source_sites))==len(source_sites) and set(frame.site_id)==set(source_sites),'Only declared three to five training sites may fit normalization')
    need(set(map(int,frame.target_year))<=set(range(2014,2019)),'Normalization source years')
    need(not frame.groupby(list(CYCLE_KEYS))[RAW].nunique(dropna=False).gt(1).any().any(),'State varies within a cycle')
    ix=[]
    for _,indices in frame.reset_index(drop=True).groupby(list(CYCLE_KEYS),sort=False).indices.items():
        need(frame.irrigation_mm.iloc[indices].tolist()==list(GRID),'Normalization complete ordered grid')
        ix.append(indices[0])
    raw=raw_coordinates(frame);unique=raw[ix,:2]
    mean=unique.mean(axis=0);std=unique.std(axis=0,ddof=0);scale=np.where(std==0,1.,std)
    need(np.isfinite(mean).all() and np.isfinite(scale).all(),'Finite source normalization')
    return dict(state_mean=mean,state_scale=scale,state_population_std=std,
        normalization_sample_ids=frame.sample_id.to_numpy(dtype=str)[ix],normalization_source_sites=np.asarray(source_sites,dtype=str),
        normalization_raw=unique)

def apply_normalizer(frame,mean,scale):
    mean=np.asarray(mean);scale=np.asarray(scale)
    need(mean.shape==scale.shape==(2,) and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale>0),'Frozen normalizer schema')
    raw=raw_coordinates(frame);u=np.column_stack([(raw[:,:2]-mean)/scale,raw[:,2],np.ones(len(raw))])
    need(np.isfinite(u).all(),'Finite standardized state')
    return raw,u
