"""Artificial fixtures only. Never import any real model or numeric payload."""
from pathlib import Path
from datetime import date,timedelta
from itertools import combinations
import ast,hashlib,json,os,shutil,subprocess,sys,tempfile
H=Path(__file__).resolve().parent;P=H/H.name
sys.path.insert(0,str(P))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import kernel as k
import runtime as rt
from optimizer import Objective,optimize,project
from bspline_basis import CubicBSplineBasis
from schema import INPUTS,LABELS,KEYS,PHYSICAL,STATES
from rootstate import COLS
from soil_features import COLUMNS,SITE_KEYS,DEPTHS
from aggregate import summarize
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def save(p,**v):p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**v)
def numeric_tests():
    rng=np.random.default_rng(402);n=12
    B=np.repeat(CubicBSplineBasis(lower=0,upper=60,basis_count=6).evaluate(k.GRID)[None,:,:],n,axis=0);T=B*k.GRID[None,:,None]/60
    f=k.feature(rng.normal(size=(n,7)),2.);w=np.ones(n)/n;raw=rng.normal(size=(n,8))*.1;raw[:,0]=0
    target=raw+rng.normal(size=(n,8))*.04;target[:,0]=0
    obj=Objective(T,f,raw,target,w,'feedback',.125);theta=rng.normal(size=(6,7))*.005
    value,g=obj(theta);eps=1e-6;fd=np.zeros_like(theta)
    for ix in np.ndindex(theta.shape):
        a=theta.copy();b=theta.copy();a[ix]+=eps;b[ix]-=eps;fd[ix]=(obj(a)[0]-obj(b)[0])/(2*eps)
    assert np.max(abs(fd-g))<2e-8
    solution,report,state=optimize(obj,'synthetic')
    assert report['accepted'] and report['projected_gradient_inf']<=1e-6
    ref=minimize(lambda x:obj(x.reshape(6,7))[0],np.zeros(42),jac=lambda x:obj(x.reshape(6,7))[1].ravel(),method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda x:.125**2-np.sum(x.reshape(6,7)**2,axis=1))],options=dict(ftol=1e-12,maxiter=3000))
    assert ref.success and abs(obj(solution)[0]-ref.fun)<1e-8
    _,_,paused=optimize(obj,'synthetic',max_steps=2);resumed,rr,_=optimize(obj,'synthetic',resume=paused)
    np.testing.assert_array_equal(solution,resumed)
    try:optimize(obj,'wrong',resume=paused);assert False
    except ValueError:pass
    c=np.ones((6,1))*.01;zero=np.zeros_like(f);mature=np.arange(n)>0
    curves=k.decode(raw,T,c,solution,f,2.,mature);cz=k.decode(raw,T,c,solution,zero,2.,mature)
    np.testing.assert_array_equal(cz[:,1],cz[:,2]);np.testing.assert_array_equal(cz[:,1],cz[:,3]);np.testing.assert_array_equal(curves[0],np.repeat(raw[0,None,:],4,axis=0))
    np.testing.assert_allclose(curves[:,2]+curves[:,3],2*curves[:,1],atol=1e-15)
    aa,gg,reg=k.score(curves,target)
    for i in range(n):
        for j in range(4):assert gg[i,j]==target[i,aa[i,j]] and reg[i,j]==target[i].max()-gg[i,j]
    dates=['2014-06-01','2014-06-07','2014-06-08','2015-06-01','2015-06-08']
    np.testing.assert_array_equal(k.supports(['P1']*5,['2014']*3+['2015']*2,dates),[-1,-1,0,-1,3])
    return dict(gradient_max_abs=float(np.max(abs(fd-g))),independent_solver_objective_difference=float(abs(obj(solution)[0]-ref.fun)),
        checkpoint_exact_resume=True,zero_reverse_anchor_and_tie=True)
def make_fixture(root):
    base=root/'package';base.mkdir();out=root/'output';out.mkdir();rng=np.random.default_rng(727)
    for p in P.glob('*.py'):shutil.copyfile(p,base/p.name)
    sites=['P1','P15','P2','P3','P4'];basis=CubicBSplineBasis(lower=0,upper=60,basis_count=6).evaluate(k.GRID)
    for site in sites:
        soil=[]
        for depth in DEPTHS:soil.append([site,SITE_KEYS[site],'embedded_confirmed_profiles',depth,.05,.45,.02,1.5,4.,.02])
        dest=base/'soil'/site;dest.mkdir(parents=True);pd.DataFrame(soil,columns=COLUMNS).to_csv(dest/'soil.csv',index=False)
        for year in [2014,2015]:
            inputs=[];labels=[];roots=[]
            for i in range(3):
                day=date(year,6,1)+timedelta(days=i*7);sid=site+'_'+str(day)
                roots.append([sid+'_0',str(year),site,str(day),str(day)+'T06:00:00',str(day-timedelta(days=1)),30.,
                    str(day-timedelta(days=3)),str(day-timedelta(days=1)),-10.,2.,3.])
                for a in k.GRID:
                    row={c:.2 for c in INPUTS};row.update(sample_id=sid+'_'+str(int(a)),target_year=str(year),site_id=site,decision_date=str(day),irrigation_mm=a,
                        predecision_dvs=.5,predecision_lai=2.,predecision_crop_root_depth_cm=30.,predecision_cwdm_kg_ha=100.,predecision_cwso_kg_ha=0.,
                        predecision_soil_vwc_0_100cm=.25,physics_initial_soil_storage_0_100cm_mm=250.)
                    inputs.append(row);lab={c:row[c] for c in KEYS};lab.update(target_net_gain_7d=float(.05*a-.002*a*a),target_aet_7d_mm=10.)
                    lab.update({c:.28+.0001*a+.001*i for c in PHYSICAL[1:]});labels.append(lab)
            dest=base/'years'/str(year)/site;dest.mkdir(parents=True)
            pd.DataFrame(inputs,columns=INPUTS).to_csv(dest/'inputs.csv',index=False);pd.DataFrame(labels,columns=LABELS).to_csv(dest/'labels.csv',index=False)
            pd.DataFrame(roots,columns=COLS).to_csv(dest/'rootstate.csv',index=False)
    tasks=[]
    for trio in combinations(sites,3):
        excluded=[s for s in sites if s not in trio]
        for seed in [1729,1730,1731]:
            bid='_'.join(trio)+'_'+str(seed)
            m=dict(mean=np.zeros(52),scale=np.ones(52),weights=np.ones(52)/52,state_mean=np.zeros(2),state_scale=np.ones(2),gain_scale=np.array([10.]),
                physical_scales=np.array([1.,.1,1.]),latent=np.ones(9),offsets=np.r_[0.,np.ones(7)*-1.,0.],parameters=rng.normal(size=502)*.0001,
                heads=rng.normal(size=(4,9,6))*.01,C=rng.normal(size=(54,7))*.01,source_sites=np.array(trio),parent_sha256=np.array('synthetic'))
            save(base/f'models/{bid}.npz',**m)
            for target in excluded:
                donor=next(s for s in excluded if s!=target);tid=bid+'_'+donor+'_'+target;t=dict(id=tid,base_id=bid,donor=donor,target=target,source_sites=list(trio),seed=seed);tasks.append(t)
                pre=rng.normal(size=(3,8))*.01;B=np.repeat(basis[None,:,:],3,axis=0);raw,water,_=k.forward(pre,B,m);actions=raw.argmax(1)
                dates=np.array(['2016-06-01','2016-06-08','2016-06-15']);ids=np.array([[target+'_'+d+'_'+str(int(a)) for a in k.GRID] for d in dates]);support=np.array([-1,0,1])
                save(base/f'inputs/{tid}/context.npz',pre=pre,B=B,curves=raw,water0=water,actions=actions,dates=dates,sample_ids=ids,
                    sites=np.array([target]*3),years=np.array(['2016']*3),support=support)
                save(base/f'scores/{tid}.npz',sample_ids=ids,gain=np.repeat((.05*k.GRID-.002*k.GRID**2)[None,:],3,axis=0))
                rel=[]
                for h in [0,1]:
                    name=f'feedback/{tid}/{h:04d}.npz';save(base/name,observed=water[h,actions[h]]+.001,sample_id=np.array(ids[h,actions[h]]));rel.append(dict(history=h,action=int(actions[h]),file=name))
                put(base/f'inputs/{tid}/releases.json',rel)
    put(base/'protocol.json',dict(tasks=tasks,worker_timeout_seconds=1800,total_timeout_seconds=10800))
    put(base/'manifest.json',dict(files={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()}))
    return base,out,tasks
def pipeline_tests(root):
    base,out,tasks=make_fixture(root)
    # Override execution location only in an external test harness; production guard stays intact.
    harness=root/'harness.py'
    harness.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,sys.argv[1])\nimport runtime as rt, worker, aggregate, data\nrt.BASE=Path(sys.argv[1]);rt.OUT=Path(sys.argv[2]);rt.SERVER=rt.OUT.parent;rt.require_server=lambda:None\ndata.COUNTS={s:[3,3] for s in ["P1","P15","P2","P3","P4"]}\nmode=sys.argv[3]\nif mode=="aggregate":aggregate.run()\nelse:\n sys.argv=["worker",mode,sys.argv[4]];worker.main()\n',encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Full stages: all 60 direction identities with 3 synthetic cycles per year/site.
    from concurrent.futures import ThreadPoolExecutor
    def run(role,t):
        p=subprocess.run([sys.executable,'-B',str(harness),str(base),str(out),role,t['id']],env=env,capture_output=True,text=True)
        if p.returncode:raise AssertionError(role+' '+t['id']+'\n'+p.stdout+p.stderr)
    for role in ['source','predict','score']:
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda t:run(role,t),tasks))
        put(out/(role+'_freeze.json'),dict(tasks={t['id']:sha(out/'tasks'/t['id']/role/'seal.json') for t in tasks}))
    p=subprocess.run([sys.executable,'-B',str(harness),str(base),str(out),'aggregate'],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    report=json.loads((out/'aggregate/screening.json').read_text());assert report['engineering_passed']
    # A failed science screen is allowed in invented data; its boolean is not a result.
    t=tasks[0];c=np.load(base/f'inputs/{t["id"]}/context.npz');rel=json.loads((base/f'inputs/{t["id"]}/releases.json').read_text());ri={r['file']:i+1 for i,r in enumerate(rel)}
    assert not rt.allowed_package('source',t,f'scores/{t["id"]}.npz',ri,99)
    assert not rt.allowed_package('source',t,f'years/2014/{t["target"]}/labels.csv',ri,99)
    assert not rt.allowed_package('predict',t,rel[1]['file'],ri,1)
    assert rt.allowed_package('predict',t,rel[1]['file'],ri,2)
    assert not rt.allowed_package('predict',t,f'scores/{t["id"]}.npz',ri,99)
    # Validate actual hook rejects forbidden opens and seal tampering, in fresh processes.
    probe=root/'probe.py'
    probe.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,sys.argv[1])\nimport runtime as rt, worker\nrt.BASE=Path(sys.argv[1]);rt.OUT=Path(sys.argv[2]);rt.SERVER=rt.OUT.parent\nt=rt.task(sys.argv[3]);rt.restrict("predict",t,{})\ntry:worker.load(rt.BASE/"scores"/(t["id"]+".npz"))\nexcept RuntimeError:print("DENIED")\nelse:raise AssertionError("leak")\n',encoding='utf-8')
    q=subprocess.run([sys.executable,'-B',str(probe),str(base),str(out),t['id']],env=env,capture_output=True,text=True)
    assert q.returncode==0 and 'DENIED' in q.stdout,q.stderr
    original=rt.BASE,rt.OUT;rt.BASE,rt.OUT=base,out
    try:
        fn=out/'tasks'/t['id']/'predict'/'predictions.npz';raw=fn.read_bytes();fn.write_bytes(raw+b'x')
        try:rt.prior_file('predict',t['id'],'predictions.npz');assert False
        except RuntimeError:pass
        fn.write_bytes(raw)
    finally:rt.BASE,rt.OUT=original
    return dict(directions=60,stages=180,actual_read_denial=True,seal_tamper_denial=True,full_synthetic_pipeline=True)
def main():
    numeric=numeric_tests();print('NUMERIC_PASS',flush=True)
    root=Path(tempfile.mkdtemp(prefix='model_matched_artificial_'));pipeline=pipeline_tests(root);print('PIPELINE_PASS',flush=True)
    for p in P.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
    try:rt.require_server();assert False
    except RuntimeError:pass
    put(H/'validation.json',dict(passed=True,numeric=numeric,pipeline=pipeline,fixture=str(root),
        local_real_compute_calls=0,python_310_syntax=True,code_sha256={p.name:sha(p) for p in P.glob('*.py')}))
if __name__=='__main__':main()
