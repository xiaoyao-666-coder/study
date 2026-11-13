"""Pure arithmetic kernels. Real inputs are used only by guarded workers."""
from datetime import date,timedelta
import numpy as np
from scipy.special import expit
GRID=np.array([0,10,15,20,25,30,40,60],float)
METHODS=['raw','static','on','reverse']
def need(v,msg):
    if not v:raise ValueError(msg)
def forward(pre,B,m):
    hf=np.c_[np.ones(len(pre)),np.tanh(pre)];T=B*GRID[None,:,None]/60
    gain=np.einsum('nh,hk,nak->na',hf,m['heads'][0],T)*float(m['gain_scale'][0])
    water=expit(np.einsum('nh,hkd,nak->nad',hf,m['C'].reshape(9,6,7),B)+m['offsets'][1:8])
    need(np.isfinite(gain).all() and np.isfinite(water).all() and not gain[:,0].any(),'Finite anchored forward')
    return gain,water,T
def supports(sites,years,dates):
    dates=[date.fromisoformat(str(s)) for s in dates];out=[]
    for i,di in enumerate(dates):
        legal=[j for j in range(i) if sites[j]==sites[i] and years[j]==years[i] and dates[j]+timedelta(days=7)<=di]
        out.append(max(legal,key=lambda j:dates[j]) if legal else -1)
    return np.asarray(out,int)
def feature(r,rms):
    u=np.asarray(r)/rms;return u/np.hypot(1.,np.linalg.norm(u,axis=-1))[...,None]
def year_weights(years):
    years=np.asarray(years);unique=np.unique(years)
    need(len(unique)==2,'Both source years must have mature episodes')
    return np.array([1/(len(unique)*np.sum(years==y)) for y in years])
def decode(raw,T,c,W,f,gain_scale,mature):
    shift=np.einsum('nak,kj->na',T,c);dynamic=np.einsum('nak,kj,nj->na',T,W,f)
    need(np.max(np.abs(shift))<=.125+1e-12 and np.max(np.abs(dynamic))<=.125+1e-12,'Fixed output budget')
    curves=np.repeat(raw[:,None,:],4,axis=1)
    for k,delta in ((1,shift),(2,shift+dynamic),(3,shift-dynamic)):curves[mature,k]+=gain_scale*delta[mature]
    need(np.isfinite(curves).all() and not curves[:,:,0].any(),'Anchored finite decoder')
    return curves
def score(curves,truth):
    actions=curves.argmax(-1);selected=np.take_along_axis(truth[:,None,:],actions[:,:,None],axis=2)[:,:,0]
    return actions,selected,truth.max(1)[:,None]-selected
