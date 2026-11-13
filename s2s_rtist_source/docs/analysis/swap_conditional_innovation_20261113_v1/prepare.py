"""Create-only byte projections; no real model, fit or prediction calls."""
from pathlib import Path
import ast,hashlib,io,json,shutil,sys,zipfile
import numpy as np
H=Path(__file__).resolve().parent;NAME=H.name;P=H/NAME
ROOT=Path(r'D:\study\s2s_rtist_source');AROOT=ROOT/'docs/analysis'
OLDNAME='swap_model_matched_feedback_20261113_v1'
OLDH=Path(r'C:\Users\LEGION\.codex\worktrees\water-model-matched\study\s2s_rtist_source\docs\analysis')/OLDNAME
OLD=OLDH/OLDNAME
A=AROOT/'swap_learned_water_metric_multiclip_20261105_v2'/'swap_learned_water_metric_multiclip_20261105_v2'
O=AROOT/'swap_null_component_transport_20261110_v1r1'/'swap_null_component_transport_20261110_v1r1'
PP=AROOT/'swap_expert_water_prequential_20261110_v1'/'swap_expert_water_prequential_20261110_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=True,allow_nan=False);f.write('\n')
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
def arr(b):
    with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def build():
    assert sha(OLD/'manifest.json')=='9a86fa76790c1f36c3d2976025fb44ce9a2ba18b6de6e1bf3c1f26bcea111020'
    om=json.loads((OLD/'manifest.json').read_bytes())['files'];am=json.loads((A/'manifest.json').read_bytes())['files'];bindings={}
    def copy(base,m,n):
        b=(base/n).read_bytes();assert hashlib.sha256(b).hexdigest()==m[n],n
        bindings[str(base/n)]=m[n];write(P/n,b)
    for n in om:
        if n.startswith(('models/','inputs/','scores/','feedback/','years/','soil/')):copy(OLD,om,n)
    for site in ('P1','P15','P2','P3','P4'):
        for fn in ('inputs.csv','rootstate.csv'):copy(A,am,f'years/2016/{site}/{fn}')
    for n in ('data.py','schema.py','rootstate.py','soil_features.py','bspline_basis.py','inference_model.py','optimizer.py','aggregate.py','run.py','launch.py','start.sh','runtime.py'):
        b=(OLD/n).read_bytes();assert hashlib.sha256(b).hexdigest()==om[n];bindings[str(OLD/n)]=om[n]
        write(P/n,b.replace(OLDNAME.encode(),NAME.encode()))
    for n in ('validate.py','test_boundaries.py','test_lifecycle.py','audit_assets.py','verify_zip.py'):
        b=(OLDH/n).read_bytes();bindings[str(OLDH/n)]=hashlib.sha256(b).hexdigest();write(H/n,b)
    proto=json.loads((OLD/'protocol.json').read_bytes());proto.update(name=NAME,methods=['raw','static','on','reverse','uncentered'],
        nuisance_year=2014,correction_year=2015,context_dimension=112,static_features=113,nuisance_ridge=.01)
    put(P/'protocol.json',proto)
    put(P/'provenance.json',dict(parent_manifest_sha256=sha(OLD/'manifest.json'),bindings=bindings,
        local_real_compute_calls=0,excluded_year_numeric_payload_reads=0,old_results_used_for_fitting=False))
    put(H/'asset_validation.json',dict(passed=True,models=30,directions=60,source_years=[2014,2015],evaluation_year=2016,
        parent_assets_exact=True,target_2016_raw_context_added=True,local_real_compute_calls=0))
    print('ASSETS_READY: 30 models, 60 directions, authenticated raw context')
def package():
    v=json.loads((H/'validation.json').read_bytes());assert v['passed'] and v['code_sha256']=={p.name:sha(p) for p in P.glob('*.py')}
    life=json.loads((H/'lifecycle_validation.json').read_bytes());assert life['passed']
    assert life['code_sha256']=={n:sha(P/n) for n in ('runtime.py','launch.py','run.py')}
    for n in ('binding_validation.json','boundary_validation.json','split_validation.json','target_identity_validation.json'):assert json.loads((H/n).read_bytes())['passed']
    split=json.loads((H/'split_validation.json').read_bytes());assert split['code_sha256']==v['code_sha256']
    assert split['test_sha256']==sha(H/'test_split_and_identity.py')
    for n in ('DESIGN.md','validation.json','asset_validation.json','lifecycle_validation.json','binding_validation.json','boundary_validation.json','split_validation.json','target_identity_validation.json'):write(P/n,(H/n).read_bytes())
    for p in P.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
    assert b'\r' not in (P/'start.sh').read_bytes()
    files={p.relative_to(P).as_posix():sha(p) for p in sorted(P.rglob('*')) if p.is_file()}
    assert all(n.isascii() and '__pycache__' not in n and not n.endswith('.pyc') for n in files)
    put(P/'manifest.json',dict(name=NAME,files=files));a=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n in [*files,'manifest.json']:z.write(P/n,NAME+'/'+n)
    with zipfile.ZipFile(a) as z:
        assert z.testzip() is None and set(z.namelist())=={NAME+'/'+n for n in [*files,'manifest.json']}
        for n,h in files.items():assert hashlib.sha256(z.read(NAME+'/'+n)).hexdigest()==h
    for p,h in json.loads((P/'provenance.json').read_bytes())['bindings'].items():assert sha(p)==h
    put(H/'delivery.json',dict(archive=str(a),sha256=sha(a),bytes=a.stat().st_size,members=len(files)+1,
        manifest_sha256=sha(P/'manifest.json'),local_real_compute_calls=0))
    print((H/'delivery.json').read_text(encoding='utf-8'))
if __name__=='__main__':{'build':build,'package':package}[sys.argv[1]]()
