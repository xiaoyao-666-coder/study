"""Eight supervised subprocesses, lifetime lock, bounded waits and failure archives."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
import runtime as rt
STOP=threading.Event();LOCKFD=None;PRINT=threading.Lock()
def log(s):
    with PRINT:print(s,flush=True)
def terminate(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def execute(role,ident,deadline):
    rt.need(not STOP.is_set() and time.monotonic()<deadline,'Stopped before '+ident)
    logs=rt.OUT/'logs';logs.mkdir(exist_ok=True);path=logs/(role+'_'+ident+'.log')
    started=time.monotonic();limit=rt.protocol()['worker_timeout_seconds']
    with path.open('xb') as f:
        p=subprocess.Popen([sys.executable,'-u','-B',str(rt.BASE/'worker.py'),role,ident],
            cwd=rt.BASE,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(LOCKFD,))
        log('START '+role+' '+ident+' pid='+str(p.pid));heartbeat=started+30;reason=None
        try:
            while p.poll() is None:
                now=time.monotonic()
                if STOP.is_set() or now>=deadline or now-started>=limit:
                    reason='supervisor_stop' if STOP.is_set() else ('total_timeout' if now>=deadline else 'worker_timeout');terminate(p);break
                if now>=heartbeat:
                    status=path.read_text(encoding='utf-8',errors='replace').splitlines()
                    log('HEARTBEAT '+ident+' elapsed='+str(round(now-started))+' stage='+(status[-1] if status else 'starting'));heartbeat=now+30
                STOP.wait(.2)
        finally:terminate(p)
    if p.returncode!=0 or reason is not None:
        tail=path.read_text(encoding='utf-8',errors='replace').splitlines()[-14:]
        if tail:log('WORKER_FAILURE_DETAIL '+role+' '+ident+'\n'+'\n'.join(tail))
    rt.need(p.returncode==0 and reason is None,role+' '+ident+' failed rc='+str(p.returncode)+' '+str(reason or 'worker_exit'))
    log('PASS '+role+' '+ident)
def package_results():
    attempt=rt.read(rt.OUT/'launch.json')['attempt'];stem=rt.NAME+f'_r{attempt:03d}_results.zip';archive=rt.SERVER/stem;partial=rt.SERVER/(stem+'.partial')
    rt.need(not archive.exists() and not partial.exists(),'Existing archive preserved')
    output={};omitted=[]
    for p in sorted(rt.OUT.rglob('*')):
        if p.is_symlink():continue
        if not p.is_file():continue
        n=p.relative_to(rt.OUT).as_posix()
        if n.startswith('build/toolchain/') or n.startswith('native_runtime/') or n.endswith('.o') or n.endswith('.mod'):continue
        if n.startswith('native_work/') and (p.suffix=='.vap' or p.name=='swap_test'):
            omitted.append(dict(path=n,bytes=p.stat().st_size,sha256=rt.digest(p)));continue
        output[n]=rt.digest(p)
    rt.put(rt.OUT/'failure_scratch_omissions.json',omitted);output['failure_scratch_omissions.json']=rt.digest(rt.OUT/'failure_scratch_omissions.json')
    files={'output/'+n:h for n,h in output.items()};files.update({'package/'+n:h for n,h in rt.filemap(rt.BASE).items()})
    rt.put(rt.OUT/'result_manifest.json',dict(files=files,package_manifest_sha256=rt.digest(rt.BASE/'manifest.json')))
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n in sorted(files):z.write((rt.OUT if n.startswith('output/') else rt.BASE)/n.split('/',1)[1],n)
        z.write(rt.OUT/'result_manifest.json','output/result_manifest.json')
    with zipfile.ZipFile(partial) as z:rt.need(z.testzip() is None,'Return archive CRC')
    os.link(partial,archive);partial.unlink();return archive
def main():
    global LOCKFD
    rt.require_server();LOCKFD=int(sys.argv[1]);os.fstat(LOCKFD)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
    started=time.monotonic();code=1;failures=[]
    try:
        p=rt.verify_package();
        import numpy as np
        rt.put(rt.OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,blas={k:os.environ.get(k) for k in rt.BLAS}))
        deadline=time.monotonic()+p['total_timeout_seconds']
        rt.need(shutil.disk_usage(rt.OUT).free>=p['min_free_bytes'],'Insufficient free disk')
        stages=rt.jobs(p)
        for filename,jobs in stages:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(execute,role,ident,deadline) for role,ident in jobs]
                for f in as_completed(futures):
                    try:f.result()
                    except BaseException as e:failures.append(str(e));log('FAILED '+str(e));STOP.set()
            rt.need(not failures,'; '.join(failures))
            freeze={}
            for role,ident in jobs:
                root=rt.root(role,ident)
                name='result.json';seal=rt.read(root/'sealed.json')
                rt.need(rt.digest(root/name)==seal['sha256'] and seal['passed'],'Output seal')
                rt.need(rt.digest(root/'access.json')==seal['access_sha256'],'Access seal')
                rt.need(set(rt.filemap(root))==set(seal['files'])|{'sealed.json'},'Exact stage members')
                for n,h in seal['files'].items():
                    rt.need(rt.digest(root/n)==h,'All certificate files sealed')
                    freeze[(root/n).relative_to(rt.OUT).as_posix()]=h
                freeze[(root/'sealed.json').relative_to(rt.OUT).as_posix()]=rt.digest(root/'sealed.json')
            rt.put(rt.OUT/filename,dict(files=freeze,passed=True))
            log('FROZEN '+filename+' tasks='+str(len(jobs))+' files='+str(len(freeze)))
        execute('score','global',deadline)
        final=rt.root('score','global');seal=rt.read(final/'sealed.json')
        rt.need(seal['passed'] and rt.digest(final/'result.json')==seal['sha256'] and rt.digest(final/'access.json')==seal['access_sha256'],'Final score seal')
        rt.need(set(rt.filemap(final))==set(seal['files'])|{'sealed.json'},'Exact final members')
        for n,h in seal['files'].items():rt.need(rt.digest(final/n)==h,'Final file seal')
        rt.verify_package();code=0
    except BaseException:
        failure=traceback.format_exc();log(failure);(rt.OUT/'failure.txt').write_text(failure,encoding='utf-8')
    finally:
        try:
            rt.put(rt.OUT/'completion.json',dict(exit_code=code,elapsed_seconds=time.monotonic()-started,workers=8,blas_threads=1,failures=failures,
                automatic_promotion=False,planned_physical_cycles=20,planned_source_atom_sets=5,planned_issues=10,planned_feedback_versions=80,
                sealed_tasks={role:len(list((rt.OUT/role).glob('*/sealed.json'))) for role in ('source','issue','history')},
                base_model_fit_calls=0,base_model_prediction_calls=0,recovery_QPs=0,
                score_sealed=(rt.OUT/'score/global/sealed.json').is_file(),feedback_adaptation_enabled=True,crop_runs=0,native_runs=0))
            log('PACKING_RESULTS');log('RESULTS_SAVED '+str(package_results()))
        except BaseException:code=2;log('PACK_FAILED '+traceback.format_exc())
        (rt.SERVER/(rt.NAME+'.exit')).write_text(str(code)+'\n',encoding='ascii');log('RUN_FINISHED exit='+str(code))
    return code
if __name__=='__main__':sys.exit(main())
