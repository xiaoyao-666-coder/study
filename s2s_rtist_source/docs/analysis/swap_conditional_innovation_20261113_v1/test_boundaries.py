"""Future feedback, aggregation, and adversarial access checks on invented data."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile
H=Path(__file__).resolve().parent;P=H/H.name;sys.path.insert(0,str(P))
import numpy as np
import pandas as pd
from aggregate import summarize,FIELDS
from kernel import METHODS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v),encoding='utf-8')
def main():
    v=json.loads((H/'validation.json').read_text());root=Path(v['fixture']);source=root/'package';t=json.loads((source/'protocol.json').read_text())['tasks'][0]
    # Two fresh harness outputs with identical source fit. Only a later capsule changes.
    tmp=Path(tempfile.mkdtemp(prefix='model_matched_boundaries_'));curves=[]
    for number in [0,1]:
        base=tmp/('p'+str(number));shutil.copytree(source,base);out=tmp/('o'+str(number));out.mkdir()
        src=out/'tasks'/t['id']/'source';shutil.copytree(root/'output/tasks'/t['id']/'source',src)
        if number:
            rel=json.loads((base/f'inputs/{t["id"]}/releases.json').read_text());path=base/rel[1]['file']
            with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
            arrays['observed']=np.full(7,.65);np.savez_compressed(path,**arrays)
        manifest={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file() and p.name!='manifest.json'};put(base/'manifest.json',dict(files=manifest))
        # Test-only rebind preserves the same fit bytes, intentionally changes the synthetic capsule.
        seal=json.loads((src/'seal.json').read_text());seal['manifest_sha256']=sha(base/'manifest.json');put(src/'seal.json',seal)
        put(out/'source_freeze.json',dict(tasks={t['id']:sha(src/'seal.json')}))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        r=subprocess.run([sys.executable,'-B',str(root/'harness.py'),str(base),str(out),'predict',t['id']],capture_output=True,text=True,env=env)
        assert r.returncode==0,r.stdout+r.stderr
        with np.load(out/'tasks'/t['id']/'predict/predictions.npz') as z:curves.append({k:z[k].copy() for k in z.files})
    np.testing.assert_array_equal(curves[0]['curves'][:2],curves[1]['curves'][:2])
    assert not np.array_equal(curves[0]['f'][2],curves[1]['f'][2])
    # Hand-computable balanced positive and negative economic examples.
    rows=[]
    for site in ['A','B','C','D','E']:
        for day in ['2016-06-08','2016-06-15']:
            for seed in [1,2,3]:
                for donor in ['d1','d2','d3','d4']:
                    for method,gain in [('raw',1.),('static',2.),('on',3.),('reverse',1.5),('uncentered',2.5)]:
                        row={c:0. for c in FIELDS};row.update(site=site,date=day,seed=seed,donor=donor,method=method,mature=1,gain=gain,regret=4.-gain,vs_raw=gain-1.,vs_static=gain-2.)
                        rows.append(row)
    frame=pd.DataFrame(rows);result=summarize(frame);assert all(result['gates'].values())
    assert np.all(result['contrasts'][result['contrasts'].reference=='static'].effect==1.)
    assert np.all(result['contrasts'][result['contrasts'].reference=='static'].remove_best==1.)
    doubled=summarize(pd.concat([frame,frame],ignore_index=True));pd.testing.assert_frame_equal(result['summary'],doubled['summary'])
    wrong=frame.copy();wrong.loc[wrong.method=='on','gain']=0.;wrong.loc[wrong.method=='on','regret']=4.
    assert not all(summarize(wrong)['gates'].values())
    put(H/'boundary_validation.json',dict(passed=True,future_capsule_no_earlier_influence=True,future_feature_actually_changed=True,
        hand_computed_economic_aggregation=True,version_duplication_invariance=True,negative_case_fails=True,synthetic_only=True))
    print('BOUNDARIES_PASS')
if __name__=='__main__':main()
