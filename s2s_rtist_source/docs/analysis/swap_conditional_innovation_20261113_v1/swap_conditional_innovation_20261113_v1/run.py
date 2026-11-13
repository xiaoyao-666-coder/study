"""Bounded workers, global stage barriers, and return packaging on failure."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, signal, subprocess, sys, threading, time, traceback, zipfile
import runtime as rt

STOP=threading.Event();PRINT=threading.Lock();LOCKFDS=()

def log(message):
    with PRINT: print(message,flush=True)

def terminate(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-4096));raw=f.read()
        return raw.splitlines()[-1].decode('utf-8','replace')[-240:] if raw else 'starting'
    except OSError:return 'starting'

def execute(role,tid,deadline):
    path=rt.OUT/'worker_logs'/(role+'__'+tid+'.log');path.parent.mkdir(exist_ok=True)
    began=time.monotonic();reason=None
    with path.open('xb') as output:
        proc=subprocess.Popen([sys.executable,'-u','-B',str(rt.BASE/'worker.py'),role,tid],cwd=rt.BASE,
                              stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=LOCKFDS)
        log('START '+role+' '+tid+' pid='+str(proc.pid));beat=began+30
        try:
            while proc.poll() is None:
                now=time.monotonic()
                if STOP.is_set():reason='cancelled';break
                if now>=deadline or now-began>=rt.protocol()['worker_timeout_seconds']:reason='timeout';break
                if now>=beat:log('HEARTBEAT '+role+' '+tid+' stage='+tail(path));beat=now+30
                STOP.wait(.2)
        finally:terminate(proc)
    ok=proc.returncode==0 and reason is None
    rt.put(rt.OUT/'worker_status'/(role+'__'+tid+'.json'),dict(passed=ok,returncode=proc.returncode,reason=reason,elapsed_seconds=time.monotonic()-began))
    rt.need(ok,role+' '+tid+' failed: '+str(reason)+' '+tail(path))
    log('PASS '+role+' '+tid)

def stage(role,deadline):
    errors=[];tasks=rt.protocol()['tasks']
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(execute,role,t['id'],deadline) for t in tasks]
        for f in as_completed(futures):
            try:f.result()
            except BaseException as e:errors.append(str(e));log('FAILED '+str(e))
    rt.need(not errors,'Stage failed: '+repr(errors))
    rt.put(rt.OUT/(role+'_freeze.json'),dict(role=role,tasks={t['id']:rt.digest(rt.directory(role,t['id'])/'seal.json') for t in tasks}))
    log('FROZEN '+role+' tasks='+str(len(tasks)))

def package_results():
    attempt=rt.read(rt.OUT/'launch.json')['attempt']
    archive=rt.SERVER/(rt.NAME+f'_r{attempt:03d}_results.zip');partial=archive.with_suffix('.zip.partial')
    rt.need(not archive.exists() and not partial.exists(),'Return path already exists')
    # Copy a final log snapshot into the result before taking its manifest.
    (rt.OUT/'run.log').write_bytes((rt.SERVER/(rt.NAME+'.log')).read_bytes())
    files={'output/'+n:h for n,h in rt.filemap(rt.OUT).items()}
    files.update({'package/'+n:h for n,h in rt.filemap(rt.BASE).items()})
    rt.put(rt.OUT/'result_manifest.json',dict(package_manifest_sha256=rt.digest(rt.BASE/'manifest.json'),files=files))
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files):
            root=rt.OUT if name.startswith('output/') else rt.BASE
            z.write(root/name.split('/',1)[1],name)
        z.write(rt.OUT/'result_manifest.json','output/result_manifest.json')
    with zipfile.ZipFile(partial) as z:
        rt.need(z.testzip() is None,'Return CRC')
        import hashlib
        for name,h in files.items(): rt.need(hashlib.sha256(z.read(name)).hexdigest()==h,'Return hash: '+name)
    os.link(partial,archive);partial.unlink();return archive

def main():
    global LOCKFDS
    rt.require_server();LOCKFDS=tuple(map(int,sys.argv[1:]));started=time.monotonic();errors=[];code=0
    def on_signal(signum,frame):STOP.set()
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    try:
        rt.verify_package();deadline=started+rt.protocol()['total_timeout_seconds']
        print('RUN_BEGIN '+rt.NAME,flush=True)
        for role in rt.ROLES:stage(role,deadline)
        import aggregate
        aggregate.run()
    except BaseException:code=1;errors.append(traceback.format_exc());log('RUN_FAILED '+errors[-1])
    try:
        rt.put(rt.OUT/'completion.json',dict(exit_code=code,elapsed_seconds=time.monotonic()-started,errors=errors,
                                            workers=8,blas_threads=1,source_gain_fitting=True,expert_fitting=False,swap_runs=0,automatic_promotion=False))
        log('PACKING_RESULTS');log('RESULTS_SAVED '+str(package_results()))
    except BaseException:code=2;log('PACK_FAILED '+traceback.format_exc())
    (rt.SERVER/(rt.NAME+'.exit')).write_text(str(code)+'\n',encoding='ascii')
    log('RUN_END exit='+str(code));return code

if __name__=='__main__':sys.exit(main())
