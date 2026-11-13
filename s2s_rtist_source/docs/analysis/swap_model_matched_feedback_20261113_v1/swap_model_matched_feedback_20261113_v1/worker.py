"""Separate source supervision, causal issuance, and globally frozen scoring."""
import csv,json,sys
from pathlib import Path
import numpy as np
import data
import runtime as rt
from kernel import GRID,METHODS,forward,supports,feature,year_weights,decode,score
from optimizer import Objective,optimize
PREP=('mean','scale','weights','state_mean','state_scale','gain_scale','physical_scales','latent','offsets')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,**v):
    with Path(p).open('wb') as f:np.savez_compressed(f,**v)
def source(t,d):
    m=load(rt.checked(f'models/{t["base_id"]}.npz'))
    rt.need(list(m['source_sites'])==t['source_sites'] and t['donor'] not in m['source_sites'] and t['target'] not in m['source_sites'],'Model excludes both sites')
    x=data.load([t['donor']],[2014,2015]);y=data.load([t['donor']],[2014,2015],truth=True)
    ev=data.tensors(x,{k:m[k] for k in PREP},y);p=m['parameters']
    pre=ev['z']@p[:416].reshape(52,8)+p[416:424]+ev['u']@p[478:].reshape(3,8)
    raw,water,T=forward(pre,ev['B'],m);action=raw.argmax(1);s=supports(ev['sites'],ev['years'],ev['dates']);ii=np.flatnonzero(s>=0)
    jj=s[ii];aa=action[jj];observed=ev['physical'][jj,aa,1:8]
    sw=float(m['physical_scales'][1]);sg=float(m['gain_scale'][0]);r=(observed-water[jj,aa])/sw
    w=year_weights(ev['years'][ii]);rms=max(.001,float(np.sqrt(np.sum(w*np.sum(r*r,axis=1)))));f=feature(r,rms)
    base=raw[ii]/sg;target=ev['y'][ii];T=T[ii]
    binding=rt.digest(rt.BASE/'manifest.json')+'|'+t['id']
    def fit(kind,features,b):
        obj=Objective(T,features,b,target,w,kind,.125)
        theta,report,state=optimize(obj,binding+'|'+kind,checkpoint=lambda st:save(d/('checkpoint_'+kind+'.npz'),**st))
        rt.need(report['accepted'] and report['final_objective']<=report['initial_objective']+1e-10,'Unaccepted '+kind)
        rt.put(d/(kind+'_fit.json'),report);return theta
    c=fit('static',np.ones((len(ii),1)),base)
    W=fit('feedback',f,base+np.einsum('nak,kj->na',T,c))
    save(d/'fit.npz',c=c,W=W,rms=np.asarray(rms),gain_scale=np.asarray(sg),vwc_scale=np.asarray(sw),
         base_id=np.asarray(t['base_id']),donor=np.asarray(t['donor']),target=np.asarray(t['target']),parent_sha256=m['parent_sha256'])
    save(d/'episodes.npz',sample_ids=ev['sample_ids'][ii],support_ids=ev['sample_ids'][jj,aa],
         dates=ev['dates'][ii],support_dates=ev['dates'][jj],support_actions=aa,raw=raw[ii],target=target,
         observed=observed,predicted_water=water[jj,aa],r=r,f=f,weights=w,T=T)
    rt.put(d/'identity.json',dict(model_sha256=rt.digest(rt.checked(f'models/{t["base_id"]}.npz')),task=t,episodes=len(ii)))
def predict(t,d,releases):
    tid=t['id'];c=load(rt.checked(f'inputs/{tid}/context.npz'));m=load(rt.checked(f'models/{t["base_id"]}.npz'))
    fit=load(rt.prior_file('source',tid,'fit.npz'))
    for k in ('base_id','donor','target'):rt.need(str(fit[k])==t[k],'Fitted identity '+k)
    rt.need(str(fit['parent_sha256'])==str(m['parent_sha256']),'Fitted parent identity')
    rt.need(float(fit['gain_scale'])==float(m['gain_scale'][0]) and float(fit['vwc_scale'])==float(m['physical_scales'][1]),'Frozen scales')
    rt.need(set(c['sites'])=={t['target']} and set(c['years'].astype(int))=={2016},'Target partition')
    s=supports(c['sites'],c['years'],c['dates']);np.testing.assert_array_equal(s,c['support'])
    raw,water,T=forward(c['pre'],c['B'],m)
    np.testing.assert_allclose(raw,c['curves'],atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(water,c['water0'],atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(raw.argmax(1),c['actions'])
    # Saved expert curves are the exact reference after the reconstruction check.
    raw=c['curves'];f=np.zeros((len(s),7));byh={r['history']:r for r in releases};used=[]
    for i,h in enumerate(s):
        rt.advance(i)
        if h<0:continue
        r=byh[int(h)];rt.need(r['action']==int(c['actions'][h]),'Logged action')
        fb=load(rt.checked(r['file']));rt.need(str(fb['sample_id'])==c['sample_ids'][h,r['action']],'Released identity')
        obs=fb['observed'];rt.need(obs.shape==(7,) and np.isfinite(obs).all() and np.all((obs>=0)&(obs<=1)),'Full seven-day feedback')
        f[i]=feature((obs-c['water0'][h,r['action']])/float(fit['vwc_scale']),float(fit['rms']))
        used.append(dict(index=i,history=int(h),sample_id=str(fb['sample_id']),file=r['file']))
    curves=decode(raw,T,fit['c'],fit['W'],f,float(fit['gain_scale']),s>=0)
    np.testing.assert_array_equal(curves[s<0],np.repeat(raw[s<0,None,:],4,axis=1))
    save(d/'predictions.npz',curves=curves,actions=curves.argmax(2),f=f,support=s,
         sample_ids=c['sample_ids'],sites=c['sites'],dates=c['dates'],methods=np.array(METHODS))
    rt.put(d/'feedback_used.json',used)
def scoring(t,d):
    p=load(rt.prior_file('predict',t['id'],'predictions.npz'));g=load(rt.checked(f'scores/{t["id"]}.npz'))
    np.testing.assert_array_equal(p['sample_ids'],g['sample_ids']);truth=g['gain']
    rt.need(np.isfinite(truth).all() and not truth[:,0].any(),'Finite anchored target gains')
    actions,selected,regret=score(p['curves'],truth);np.testing.assert_array_equal(actions,p['actions'])
    left,right=np.triu_indices(8,1);rows=[];pairs=[]
    for i in range(len(truth)):
        for k,method in enumerate(METHODS):
            delta=float(selected[i,k]-selected[i,0]);vs_static=float(selected[i,k]-selected[i,1])
            rows.append(dict(task=t['id'],base=t['base_id'],donor=t['donor'],site=t['target'],seed=t['seed'],date=str(p['dates'][i]),
                mature=int(p['support'][i]>=0),method=method,action=float(GRID[actions[i,k]]),gain=float(selected[i,k]),regret=float(regret[i,k]),
                vs_raw=delta,vs_static=vs_static,harmed_raw=int(delta < -1e-10),harmed_static=int(vs_static < -1e-10),
                negative_loss=float(max(-selected[i,k],0)),negative_rate=int(selected[i,k]<-1e-10),
                repairs_raw=int(regret[i,0]>1e-10 and delta>1e-10),restores_raw=int(k>=2 and actions[i,k]==actions[i,0] and actions[i,1]!=actions[i,0]),
                spoils_correct_raw=int(regret[i,0]<=1e-10 and regret[i,k]>1e-10)))
        tg=truth[i,left]-truth[i,right]
        # Explicit per-pair values avoid an ambiguous advanced-index axis order.
        for a,b,y in zip(left,right,tg):
            ds=p['curves'][i,:,a]-p['curves'][i,:,b]
            pairs.append(dict(site=t['target'],date=str(p['dates'][i]),task=t['id'],a=float(GRID[a]),b=float(GRID[b]),truth=float(y),
                **{method:float(ds[k]) for k,method in enumerate(METHODS)}))
    for n,records in [('scores.csv',rows),('pairs.csv',pairs)]:
        with (d/n).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def main():
    rt.require_server();role,tid=sys.argv[1:];rt.need(role in rt.ROLES,'Role');t=rt.task(tid);d=rt.directory(role,tid);d.mkdir(parents=True)
    releases=[];release_index={}
    if role=='predict':
        releases=rt.read(rt.checked(f'inputs/{tid}/releases.json'));c=load(rt.checked(f'inputs/{tid}/context.npz'))
        for r in releases:
            hits=np.flatnonzero(c['support']==r['history']);rt.need(len(hits)>0,'Release unused');release_index[r['file']]=int(hits.min())
    events=rt.restrict(role,t,release_index)
    if role=='source':source(t,d)
    elif role=='predict':predict(t,d,releases)
    else:scoring(t,d)
    rt.put(d/'reads.json',events);rt.seal(role,tid);print('ACCEPTED '+role+' '+tid,flush=True)
if __name__=='__main__':main()
