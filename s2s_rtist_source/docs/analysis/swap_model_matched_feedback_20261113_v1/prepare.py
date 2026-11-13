"""Create-only projections of authenticated saved arrays; no model computation."""
from pathlib import Path
import ast, hashlib, io, json, shutil, sys, zipfile
import numpy as np
H=Path(__file__).resolve().parent; NAME=H.name; P=H/NAME
ROOT=Path(r'D:\study\s2s_rtist_source'); AROOT=ROOT/'docs/analysis'
AN='swap_learned_water_metric_multiclip_20261105_v2'; A=AROOT/AN/AN
ON='swap_null_component_transport_20261110_v1r1'; O=AROOT/ON/ON
PN='swap_expert_water_prequential_20261110_v1'; PP=AROOT/PN/PN
ARCHIVE=ROOT/(AN+'_r001_results.zip')
AH='1c59fe762a5457716358c41fc9ae309a5876600d97eae371bab0a3312c450846'
OH='b968e6ec42b3e6a811c4da9802daf36018b295c02e2b3da033ed3ddc097489f7'
PREP=('mean','scale','weights','state_mean','state_scale','gain_scale','physical_scales','latent','offsets')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n')
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
def arr(b):
    with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,**v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**v)
def build():
    assert sha(ARCHIVE)==AH and sha(O/'manifest.json')==OH
    om=json.loads((O/'manifest.json').read_bytes())['files']; am=json.loads((A/'manifest.json').read_bytes())['files']
    pm=json.loads((PP/'manifest.json').read_bytes())['files']; bindings={};tasks=[];counts=[]
    def local(base,m,n):
        b=(base/n).read_bytes();h=hashlib.sha256(b).hexdigest();assert h==m[n],n
        bindings[str(base/n)]=h;return b
    original=json.loads(local(O,om,'protocol.json'))['tasks']
    with zipfile.ZipFile(ARCHIVE) as z:
        rm=json.loads(z.read('output/result_manifest.json'))['files']
        manifest=z.read('package/manifest.json');assert hashlib.sha256(manifest).hexdigest()==rm['package/manifest.json']
        assert manifest==(A/'manifest.json').read_bytes()
        for n in ('data.py','schema.py','rootstate.py','soil_features.py','bspline_basis.py','inference_model.py'):
            raw=local(A,am,n)
            if n=='data.py':raw=raw.replace(b'range(2014,2019)',b'range(2014,2017)')
            write(P/n,raw)
        for site in ('P1','P15','P2','P3','P4'):
            n=f'soil/{site}/soil.csv';write(P/n,local(A,am,n))
            for year in (2014,2015):
                for fn in ('inputs.csv','labels.csv','rootstate.csv'):
                    n=f'years/{year}/{site}/{fn}';write(P/n,local(A,am,n))
        for t in original:
            if t['scenario']!='unseen':continue
            tid=t['id'];assert t['train_years']==[2014,2015] and t['evaluation_year']==2016
            assert len(t['source_sites'])==3 and len(t['evaluation_sites'])==2
            n=f'package/models/{tid}/bundle.npz';raw=z.read(n);assert hashlib.sha256(raw).hexdigest()==rm[n]
            bindings[ARCHIVE.name+'::'+n]=rm[n];bundle=arr(raw)
            m=arr(local(O,om,f'inputs/{tid}/model.npz'));c=arr(local(O,om,f'inputs/{tid}/context.npz'))
            g=arr(local(O,om,f'gains/{tid}.npz'))
            assert str(bundle['parent_sha256'])==str(m['parent_sha256'])
            np.testing.assert_array_equal(bundle['source_sites'],t['source_sites'])
            np.testing.assert_array_equal(bundle['gain_scale'],m['gain_scale'])
            np.testing.assert_array_equal(bundle['offsets'][1:8],m['offset'])
            np.testing.assert_array_equal(c['sample_ids'],g['sample_ids'])
            assert set(c['years'].astype(int))=={2016}
            save(P/f'models/{tid}.npz',**{k:bundle[k] for k in (*PREP,'parameters','heads','source_sites','parent_sha256')},C=m['C'])
            releases=json.loads(local(PP,pm,f'inputs/{tid}/releases.json'))
            for target in t['evaluation_sites']:
                donor=next(s for s in t['evaluation_sites'] if s!=target);taskid=tid+'__donor_'+donor+'__target_'+target
                task=dict(id=taskid,base_id=tid,source_sites=t['source_sites'],donor=donor,target=target,seed=t['seed'])
                tasks.append(task);ix=np.flatnonzero(c['sites']==target);mapping={int(j):i for i,j in enumerate(ix)}
                support=np.array([-1 if c['support'][j]<0 else mapping[int(c['support'][j])] for j in ix])
                save(P/f'inputs/{taskid}/context.npz',**{k:c[k][ix] for k in ['pre','B','sample_ids','sites','dates','years','curves','actions','water0']},support=support)
                save(P/f'scores/{taskid}.npz',sample_ids=g['sample_ids'][ix],gain=g['gain'][ix])
                rel=[]
                for r in releases:
                    if r['history'] not in mapping:continue
                    h=mapping[r['history']];name=f'feedback/{taskid}/{h:04d}.npz'
                    raw=local(PP,pm,r['file']);fb=arr(raw)
                    assert str(fb['sample_id'])==c['sample_ids'][r['history'],r['action']]
                    write(P/name,raw);rel.append(dict(history=h,action=r['action'],file=name))
                put(P/f'inputs/{taskid}/releases.json',rel)
                counts.append(dict(task=taskid,target=target,total=len(ix),mature=int(np.sum(support>=0))))
    assert len(tasks)==60
    # Reuse only infrastructure; output semantics and roles are replaced below.
    for n in ('run.py','launch.py','start.sh'):
        s=local(PP,pm,n).decode().replace(PN,NAME)
        s=s.replace('offline_training=false online_water_fit=true gains=false','source_gain_fit=true frozen_expert=true')
        s=s.replace('offline_training_performed=False,online_water_fitting=True','source_gain_fitting=True,expert_fitting=False')
        write(P/n,s.encode())
    # Keep the tested convex solver, remove unrelated legacy water-step/lift helpers.
    dp=AROOT/'swap_crossfit_feedback_20261105_v1'/'swap_crossfit_feedback_20261105_v1'/'decoder.py'
    source=dp.read_text(encoding='utf-8');bindings[str(dp)]=sha(dp)
    tail=source[source.index('def project('):]
    prefix='import numpy as np\nfrom scipy.special import logsumexp\nRADIUS, REG, TAU, PG_TOL, MAX_STEPS = .25, .01, .05, 1e-6, 100000\nKINDS=("static","feedback")\ndef need(ok,msg):\n    if not ok:raise ValueError(msg)\n\n'
    write(P/'optimizer.py',(prefix+tail).encode())
    put(P/'protocol.json',dict(name=NAME,tasks=tasks,worker_timeout_seconds=1800,total_timeout_seconds=10800,
        years=[2014,2015,2016],roles=['source','predict','score'],methods=['raw','static','on','reverse'],
        source_years=[2014,2015],evaluation_year=2016,primary='on-static',radius=.125,ridge=.01,tau=.05,
        automatic_promotion=False,counts=counts))
    put(P/'provenance.json',dict(archive_sha256=AH,parent_manifest_sha256=OH,bindings=bindings,
        local_real_compute_calls=0,excluded_year_numeric_payload_reads=0))
    put(H/'asset_validation.json',dict(passed=True,tasks=60,models=30,counts=counts,
        saved_fields_only=True,source_target_disjoint=True,local_real_compute_calls=0))
    print('ASSETS_READY directions=60 models=30')
def package():
    v=json.loads((H/'validation.json').read_bytes());assert v['passed']
    assert v['code_sha256']=={p.name:sha(p) for p in P.glob('*.py')}
    life=json.loads((H/'lifecycle_validation.json').read_bytes());assert life['passed']
    assert life['code_sha256']=={n:sha(P/n) for n in ['runtime.py','launch.py','run.py']}
    assert json.loads((H/'binding_validation.json').read_bytes())['passed']
    assert json.loads((H/'boundary_validation.json').read_bytes())['passed']
    for src in ('DESIGN.md','validation.json','asset_validation.json','lifecycle_validation.json','binding_validation.json','boundary_validation.json'):write(P/src,(H/src).read_bytes())
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
    for p,h in json.loads((P/'provenance.json').read_bytes())['bindings'].items():
        if '::' not in p:assert sha(p)==h
    put(H/'delivery.json',dict(archive=str(a),sha256=sha(a),bytes=a.stat().st_size,members=len(files)+1,
        manifest_sha256=sha(P/'manifest.json'),local_real_compute_calls=0))
    print((H/'delivery.json').read_text())
if __name__=='__main__':{'build':build,'package':package}[sys.argv[1]]()
