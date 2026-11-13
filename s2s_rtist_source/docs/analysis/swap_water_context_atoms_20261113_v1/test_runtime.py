"""Linux lifecycle using artificial workers; no real inputs or numerical work."""
from pathlib import Path
import contextlib, hashlib, io, json, os, subprocess, sys, tempfile, time, types, zipfile

H = Path(__file__).resolve().parent
P = H / H.name
sys.path.insert(0, str(P))
import runtime as rt
import run, launch


def verify(path, code):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        m = json.loads(z.read('output/result_manifest.json'))['files']
        assert set(z.namelist()) == set(m) | {'output/result_manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in m.items())
        assert json.loads(z.read('output/completion.json'))['exit_code'] == code


def main():
    assert sys.platform == 'linux'
    assert not (H / 'runtime_validation.json').exists()
    # run.py uses NumPy only for the version metadata. Numerical production
    # code is separately tested on synthetic arrays; do not install packages.
    sys.modules['numpy'] = types.SimpleNamespace(__version__='ARTIFICIAL_METADATA_ONLY')
    checks = []
    temp = Path(tempfile.mkdtemp(prefix='synthetic_lifecycle_', dir=H))
    base = temp / 'deploy/pkg'
    base.mkdir(parents=True)
    subprocess.run(['bash', '-n', str(P / 'start.sh')], check=True, timeout=5)
    for n in ('runtime.py', 'run.py', 'launch.py', 'access.py'):
        (base / n).write_bytes((P / n).read_bytes())
    (base / 'manifest.json').write_text('{"files":{}}')
    rt.BASE = base
    rt.SERVER = temp
    rt.require_server = lambda: None
    dummy = '''from pathlib import Path
import sys,json,hashlib,time,subprocess
c=json.loads(Path('mode.json').read_text());out=Path(c['out']);mode=c['mode'];role,ident=sys.argv[1:3]
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
if mode==role+'_fail':raise RuntimeError('SYNTHETIC_ROOT_CAUSE_'+role)
if role=='source' and mode in ('timeout','total_timeout'):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);put(out/'descendant.json',dict(pid=child.pid));time.sleep(30)
for dep in {'source':[], 'issue':['source'], 'history':['source','issue'], 'score':['source','issue','history']}[role]:assert json.loads((out/(dep+'_freeze.json')).read_text())['passed']
r=out/role/ident;put(r/'result.json',{'synthetic':True});put(r/'access.json',{'synthetic':True});put(r/'certificate.json',{'synthetic':True})
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.iterdir()}
put(r/'sealed.json',dict(files=files,passed=True,sha256='bad' if mode=='bad_seal' and role=='source' else files['result.json'],access_sha256=files['access.json']))
if mode=='score_tamper' and role=='score':put(r/'certificate.json',{'tampered':True})
'''
    (base / 'worker.py').write_text(dummy)
    for mode in ('pass', 'source_fail', 'issue_fail', 'history_fail', 'score_fail',
                 'bad_seal', 'timeout', 'total_timeout', 'score_tamper'):
        rt.NAME = 'dummy_' + mode
        rt.OUT = temp / rt.NAME
        rt.OUT.mkdir()
        rt.put(rt.OUT / 'launch.json', dict(attempt=1, manifest_sha256=rt.digest(base / 'manifest.json')))
        (base / 'mode.json').write_text(json.dumps(dict(mode=mode, out=str(rt.OUT))))
        counts = (5, 10, 80) if mode == 'pass' else (1, 1, 1)
        rt.jobs = lambda p: [(r+'_freeze.json', [(r, str(i)) for i in range(n)])
                             for r, n in zip(('source', 'issue', 'history'), counts)]
        rt.verify_package = lambda: dict(total_timeout_seconds=1.5 if mode == 'total_timeout' else 50,
                                         min_free_bytes=0)
        rt.protocol = lambda: dict(worker_timeout_seconds=1.5 if mode == 'timeout' else 50)
        run.STOP.clear()
        output = io.StringIO()
        with (temp / (rt.NAME + '.lock')).open('a') as lock, contextlib.redirect_stdout(output):
            sys.argv = ['run.py', str(lock.fileno())]
            code = run.main()
        transcript = output.getvalue()
        (temp / (mode + '.log')).write_text(transcript)
        expected = 0 if mode == 'pass' else 1
        assert code == expected, transcript
        verify(temp / (rt.NAME + '_r001_results.zip'), expected)
        if mode == 'pass':
            completion = rt.read(rt.OUT / 'completion.json')
            assert completion['sealed_tasks'] == dict(source=5, issue=10, history=80)
            assert completion['score_sealed'] and transcript.count('START ') == 96
        if mode in ('source_fail', 'issue_fail', 'history_fail', 'score_fail'):
            assert 'WORKER_FAILURE_DETAIL' in transcript and 'SYNTHETIC_ROOT_CAUSE' in transcript
        if mode in ('source_fail', 'issue_fail', 'history_fail', 'bad_seal'):
            assert not (rt.OUT / 'score').exists()
        if mode in ('timeout', 'total_timeout'):
            assert ('worker_timeout' if mode == 'timeout' else 'total_timeout') in transcript
            pid = rt.read(rt.OUT / 'descendant.json')['pid']
            stat = Path('/proc') / str(pid) / 'stat'
            assert not stat.exists() or stat.read_text().split()[2] == 'Z'
        try:
            run.package_results()
        except ValueError:
            pass
        else:
            raise AssertionError('Archive overwritten')
        checks.append(mode + ': stage barriers, exact archive and root-cause/timeout reporting')
        print('LIFECYCLE_PASS ' + mode, flush=True)

    rt.NAME = 'retry'
    rt.OUT = temp / 'retry'
    rt.OUT.mkdir()
    rt.put(rt.OUT / 'launch.json', dict(attempt=1, manifest_sha256=rt.digest(base / 'manifest.json')))
    rt.put(rt.OUT / 'completion.json', dict(exit_code=1))
    (base / 'run.py').write_text('import time;time.sleep(1)\n')
    launch.main()
    pid = int((temp / 'retry.pid').read_text())
    launch.main()
    assert rt.read(rt.OUT / 'launch.json')['attempt'] == 2
    time.sleep(1.2)
    os.waitpid(pid, 0)
    rt.put(rt.OUT / 'completion.json', dict(exit_code=0))
    (temp / 'retry_r002_results.zip').write_bytes(b'SYNTHETIC')
    launch.main()
    assert (temp / 'retry_attempts/r001/completion.json').exists()
    checks.append('lifetime lock, failed attempt preservation, duplicate-complete guard')

    rt.NAME = 'launchfail'
    rt.OUT = temp / rt.NAME
    old = launch.subprocess.Popen
    def fail(*a, **kw):
        raise OSError('SYNTHETIC_SPAWN_FAILURE')
    launch.subprocess.Popen = fail
    try:
        try:
            launch.main()
        except OSError:
            pass
        else:
            raise AssertionError('No launch failure')
    finally:
        launch.subprocess.Popen = old
    verify(temp / 'launchfail_r001_results.zip', 1)
    checks.append('launch failure returns exact archive')
    data = dict(passed=True, artificial_workers_only=True, real_model_calls=0,
                numpy_metadata_stub=True, fixture_path=str(temp), checks=checks,
                tested_code={n: hashlib.sha256((P/n).read_bytes()).hexdigest()
                             for n in ('runtime.py', 'run.py', 'launch.py', 'access.py', 'start.sh')})
    with (H / 'runtime_validation.json').open('x') as f:
        json.dump(data, f, indent=2)
    print('LIFECYCLE_ALL_PASS', flush=True)


if __name__ == '__main__':
    main()
