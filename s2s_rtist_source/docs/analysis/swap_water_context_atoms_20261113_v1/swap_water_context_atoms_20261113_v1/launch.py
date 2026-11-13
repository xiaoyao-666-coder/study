"""Detach once; hold lock for the entire run; preserve every numbered attempt."""
import fcntl,subprocess,sys
import runtime as rt
def main():
    rt.require_server()
    handle=(rt.SERVER/(rt.NAME+'.lock')).open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('ALREADY_RUNNING '+rt.NAME,flush=True);return
    rt.verify_package();attempt=1
    while (rt.SERVER/(rt.NAME+f'_r{attempt:03d}_results.zip')).exists() or (rt.SERVER/(rt.NAME+f'_r{attempt:03d}_results.zip.partial')).exists():attempt+=1
    if rt.OUT.exists():
        previous=rt.read(rt.OUT/'launch.json');rt.need(previous['manifest_sha256']==rt.digest(rt.BASE/'manifest.json'),'Different existing package')
        prior=previous['attempt'];complete=rt.OUT/'completion.json';archive=rt.SERVER/(rt.NAME+f'_r{prior:03d}_results.zip')
        if complete.exists() and rt.read(complete)['exit_code']==0 and archive.is_file():print('ALREADY_COMPLETE '+str(archive),flush=True);return
        attempt=max(attempt,prior+1);destination=rt.SERVER/(rt.NAME+'_attempts')/f'r{prior:03d}'
        rt.need(rt.OUT.resolve()==rt.SERVER/rt.NAME and destination.resolve().is_relative_to(rt.SERVER) and not destination.exists(),'Attempt archive path')
        destination.parent.mkdir(exist_ok=True);rt.OUT.rename(destination);print('PRESERVED_ATTEMPT '+str(destination),flush=True)
    rt.OUT.mkdir();rt.put(rt.OUT/'launch.json',dict(manifest_sha256=rt.digest(rt.BASE/'manifest.json'),attempt=attempt))
    (rt.SERVER/(rt.NAME+'.exit')).write_text('RUNNING\n',encoding='ascii')
    try:
        with (rt.SERVER/(rt.NAME+'.log')).open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-u','-B',str(rt.BASE/'run.py'),str(handle.fileno())],cwd=rt.BASE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(handle.fileno(),))
    except BaseException:
        import traceback,run
        (rt.OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        rt.put(rt.OUT/'completion.json',dict(exit_code=1,stage='launch_failed',automatic_promotion=False))
        try:print('RESULTS_SAVED '+str(run.package_results()),flush=True)
        finally:(rt.SERVER/(rt.NAME+'.exit')).write_text('1\n',encoding='ascii')
        raise
    (rt.SERVER/(rt.NAME+'.pid')).write_text(str(proc.pid)+'\n',encoding='ascii')
    print('STARTED workers=8 BLAS=1 physical_cycles=20 source_atom_sets=5 issues=10 feedback_versions=80 base_fits=0 base_predictions=0 crop_runs=0 native_runs=0 pid='+str(proc.pid),flush=True)
if __name__=='__main__':main()
