"""Linux stdlib-only invented worker lifecycle tests; no real assets copied."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,time,zipfile
H=Path(__file__).resolve().parent;P=H/H.name
ROOT=Path(tempfile.mkdtemp(prefix='model_matched_lifecycle_'));NAME=H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v),encoding='utf-8')
def setup(label,mode,timeout=5):
    server=ROOT/label;server.mkdir();base=server/'deploy'/NAME;base.mkdir(parents=True)
    for n in ['runtime.py','launch.py','run.py']:
        text=(P/n).read_text();text=text.replace("SERVER=Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source')",'SERVER=Path('+repr(str(server))+')')
        if n=='run.py':text=text.replace('began+30','began+.3').replace('now+30','now+.3')
        (base/n).write_text(text,encoding='utf-8')
    (base/'worker.py').write_text('import sys,time\nimport runtime as rt\nrt.require_server()\nrole,tid=sys.argv[1:]\nd=rt.directory(role,tid);d.mkdir(parents=True)\nmode=rt.protocol()["test_mode"]\nprint("ARTIFICIAL_WORKER",flush=True)\ntime.sleep(2 if mode=="timeout" else .7)\nif mode=="fail":raise RuntimeError("invented failure")\nrt.put(d/"value.json",dict(artificial=True));rt.seal(role,tid)\n',encoding='utf-8')
    (base/'aggregate.py').write_text('import runtime as rt\ndef run():rt.put(rt.OUT/"artificial_aggregate.json",dict(passed=True))\n',encoding='utf-8')
    put(base/'protocol.json',dict(tasks=[dict(id='task_'+str(i)) for i in range(9)],worker_timeout_seconds=timeout,total_timeout_seconds=30,test_mode=mode))
    put(base/'manifest.json',dict(files={p.relative_to(base).as_posix():sha(p) for p in base.iterdir() if p.is_file()}))
    return server,base
def launch(base):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    p=subprocess.run([sys.executable,'-B',str(base/'launch.py')],capture_output=True,text=True,env=env)
    assert p.returncode==0,p.stderr;return p.stdout
def finished(server,n=1):
    end=time.monotonic()+45
    archive=server/(NAME+f'_r{n:03d}_results.zip')
    while time.monotonic()<end:
        if archive.exists() and (server/(NAME+'.exit')).read_text().strip() in ['0','1','2']:
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                manifest=json.loads(z.read('output/result_manifest.json'))['files']
                assert set(z.namelist())==set(manifest)|{'output/result_manifest.json'}
                for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
                return archive,json.loads(z.read('output/completion.json'))
        time.sleep(.1)
    raise AssertionError('lifecycle wait exceeded')
def main():
    server,base=setup('success','success');assert 'STARTED' in launch(base);assert 'ALREADY_RUNNING' in launch(base)
    a,c=finished(server);assert c['exit_code']==0;assert 'ALREADY_COMPLETE' in launch(base)
    assert 'HEARTBEAT' in (server/(NAME+'.log')).read_text();print('SUCCESS_DUPLICATE_HEARTBEAT_PASS',flush=True)
    server,base=setup('failure','fail');assert 'STARTED' in launch(base);a,c=finished(server);assert c['exit_code']==1;old=sha(a)
    assert 'STARTED' in launch(base);b,d=finished(server,2);assert d['exit_code']==1 and sha(a)==old and b!=a
    assert (server/(NAME+'_attempts')/'r001'/'completion.json').exists();print('FAILURE_RETRY_PRESERVED_PASS',flush=True)
    server,base=setup('timeout','timeout',.4);launch(base);a,c=finished(server);assert c['exit_code']==1
    with zipfile.ZipFile(a) as z:
        statuses=[json.loads(z.read(n)) for n in z.namelist() if n.startswith('output/worker_status/')]
        assert statuses and all(s['reason']=='timeout' for s in statuses)
    print('TIMEOUT_RETURN_PASS',flush=True)
    put(H/'lifecycle_validation.json',dict(passed=True,synthetic_only=True,root=str(ROOT),linux_processes=True,
        duplicate_lock=True,heartbeat=True,success_idempotence=True,failure_zip=True,retry_preserved=True,timeout_zip=True,
        code_sha256={n:sha(P/n) for n in ['runtime.py','launch.py','run.py']}))
if __name__=='__main__':main()
