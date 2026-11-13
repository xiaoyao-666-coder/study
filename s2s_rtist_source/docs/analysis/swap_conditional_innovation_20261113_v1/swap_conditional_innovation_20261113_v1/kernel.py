"""Fixed conditional-innovation algebra; production callers are server guarded."""
from datetime import date,timedelta
import numpy as np
from scipy.special import expit
GRID=np.array([0,10,15,20,25,30,40,60],float)
METHODS=['raw','static','on','reverse','uncentered']
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
def context(z,u,dates,support,actions):
    v=np.c_[z,u];need(v.shape[1]==55,'Original encoder coordinates')
    ii=np.flatnonzero(support>=0);jj=support[ii]
    lag=np.array([(date.fromisoformat(str(dates[i]))-date.fromisoformat(str(dates[j]))).days for i,j in zip(ii,jj)])
    need(np.all(lag>=7),'Mature context')
    C=np.c_[v[ii],v[jj],GRID[actions[jj]]/60,(lag-7)/(lag+7)]
    need(C.shape==(len(ii),112) and np.isfinite(C).all(),'Context shape')
    return C,ii,jj
def transform(C,mu,sd):
    need(mu.shape==sd.shape==(112,) and np.all(sd>0),'Context transform')
    return np.c_[np.ones(len(C)),np.tanh((C-mu)/sd)/np.sqrt(112)]/np.sqrt(2)
def nuisance(C,r):
    need(len(C)>=2 and r.shape==(len(C),7),'Nuisance episodes')
    mu=C.mean(0);sd=C.std(0);sd=np.where(sd<1e-6,1.,sd);x=transform(C,mu,sd)
    penalty=np.diag(np.r_[0.,np.ones(112)])*.01
    G=x.T@x/len(x)+penalty;rhs=x.T@r/len(x)
    np.linalg.cholesky(G);A=np.linalg.solve(G,rhs)
    error=G@A-rhs;relative=float(np.max(abs(error))/max(1.,np.max(abs(rhs))))
    need(np.isfinite(A).all() and relative<=1e-10,'Nuisance normal equation')
    rms=max(.001,float(np.sqrt(np.mean(np.sum(r*r,axis=1)))))
    return dict(mu=mu,sd=sd,A=A,rms=np.asarray(rms)),dict(x=x,G=G,rhs=rhs,penalty=penalty,
        normal_residual=error,relative_residual=np.asarray(relative),rhat=x@A)
def feature(r,rms):
    u=np.asarray(r)/rms;return u/np.hypot(1.,np.linalg.norm(u,axis=-1))[...,None]
def decode(raw,T,c,W,f,gain_scale,mature,x,uncentered_W,uncentered_f):
    shift=np.einsum('nak,kj,nj->na',T,c,x);dynamic=np.einsum('nak,kj,nj->na',T,W,f)
    control=np.einsum('nak,kj,nj->na',T,uncentered_W,uncentered_f)
    need(all(np.max(abs(a))<=.125+1e-12 for a in (shift,dynamic,control)),'Fixed output budget')
    curves=np.repeat(raw[:,None,:],len(METHODS),axis=1)
    for k,delta in ((1,shift),(2,shift+dynamic),(3,shift-dynamic),(4,shift+control)):curves[mature,k]+=gain_scale*delta[mature]
    need(np.isfinite(curves).all() and not curves[:,:,0].any(),'Anchored finite decoder')
    return curves
def score(curves,truth):
    actions=curves.argmax(-1);selected=np.take_along_axis(truth[:,None,:],actions[:,:,None],axis=2)[:,:,0]
    return actions,selected,truth.max(1)[:,None]-selected
