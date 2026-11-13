"""Deployment guard and stage seals for the context-weighted atom screen."""
from pathlib import Path
import hashlib,json,os,sys
NAME='swap_water_context_atoms_20261113_v1';SERVER=Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source')
BASE=Path(__file__).resolve().parent;OUT=SERVER/NAME
BLAS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='ascii',newline='\n') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def filemap(root):
    paths=sorted(Path(root).rglob('*'));need(not any(p.is_symlink() for p in paths),'No symlinks')
    return {p.relative_to(root).as_posix():digest(p) for p in paths if p.is_file()}
def protocol():return read(BASE/'protocol.json')
def require_server():
    need(sys.platform.startswith('linux') and BASE==SERVER/'deploy'/NAME,'Real computation requires exact Linux server deployment')
    need(BASE.resolve()==BASE and OUT.resolve()==OUT,'No symlink deployment')
    need(all(os.environ.get(k)=='1' for k in BLAS),'BLAS1 required')
def jobs(p):
    return [('source_freeze.json',[('source',s) for s in p['sites']]),
        ('issue_freeze.json',[('issue',x['id']) for x in p['pairs']]),
        ('history_freeze.json',[('history',x['id']+'_b'+str(a)) for x in p['pairs'] for a in p['doses']])]
def meta(p,role,ident):
    if role=='issue':return next(x for x in p['pairs'] if x['id']==ident),None
    pid,b=ident.rsplit('_b',1);return next(x for x in p['pairs'] if x['id']==pid),int(b)
def verify_package():
    expected=read(BASE/'manifest.json')['files'];actual=filemap(BASE);actual.pop('manifest.json',None)
    need(actual==expected,'Exact package members/hashes');p=protocol()
    need(p['name']==NAME and p['sites']==['P1','P15','P2','P3','P4'] and p['years']==[2014,2015],'Identity')
    need(p['doses']==[0,10,15,20,25,30,40,60],'Actions')
    need([(x['site'],x['year']) for x in p['pairs']]==[(s,y) for s in p['sites'] for y in p['years']],'Ten pairs')
    need(p['candidate']=='context_weighted_paired_response_atoms' and p['feedback_floor_mm']==.05,'Fixed estimator')
    need(p['context_features']==['cwdm','cwso','dvs','lai','root_cm','rain_sum','rad_mean','tmax_mean','tmin_mean','vapor_mean','wind_mean'],'Fixed public context')
    need(p['score_pairs_per_behavior']==21 and p['score_pairs_total']==1680,'Unobserved action pairs')
    need(p['contract_sha256']==digest(BASE/'CONTRACT.md'),'Contract binding')
    need(p['workers']==8 and p['blas_threads']==1 and p['physical_cycles']==20,'Ledger')
    need([len(v) for _,v in jobs(p)]==[5,10,80] and p['stages']==dict(source=5,issue=10,history=80,score=1),'96 tasks')
    need(p['improvement_epsilon_mm']==1e-10 and p['sign_zero_mm']==.01,'Fixed numerical gates')
    need(all(p[k]==0 for k in ('native_runs','base_model_fit_calls','base_model_prediction_calls','recovery_QPs','crop_runs')) and not p['automatic_promotion'] and not p['economic_scoring'],'Scope')
    need(p['worker_timeout_seconds']==1800 and p['total_timeout_seconds']==10800,'Time limits')
    need(p['adoption']=='shared_advantage_gate / swap_a1_advantage_gate_20261019_v1','Adoption')
    return p
def root(role,ident):return OUT/role/ident
def seal(role,ident,reads):
    r=root(role,ident);put(r/'access.json',dict(opened=sorted(set(reads)),role=role,ident=ident))
    files=filemap(r);put(r/'sealed.json',dict(passed=True,files=files,sha256=digest(r/'result.json'),access_sha256=digest(r/'access.json')))
def frozen(role,ident):
    r=root(role,ident);freeze=read(OUT/(role+'_freeze.json'));need(freeze['passed'],'Stage barrier')
    s=read(r/'sealed.json');need(set(filemap(r))==set(s['files'])|{'sealed.json'},'Exact frozen members')
    for n,h in s['files'].items():need(digest(r/n)==h==freeze['files'][(r/n).relative_to(OUT).as_posix()],'Frozen output '+n)
    need(digest(r/'sealed.json')==freeze['files'][(r/'sealed.json').relative_to(OUT).as_posix()],'Seal binding')
    return read(r/'result.json')
def frozen_paths(role,ident):return [OUT/(role+'_freeze.json')]+[q for q in root(role,ident).iterdir() if q.is_file()]
