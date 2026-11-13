"""Detach with lifetime lock; preserve attempts and never overwrite returns."""
import fcntl, subprocess, sys
import runtime as rt

def main():
    rt.require_server()
    lock=(rt.SERVER/(rt.NAME+'.lock')).open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print('ALREADY_RUNNING '+rt.NAME,flush=True);return
    rt.verify_package()
    attempt=1
    while (rt.SERVER/(rt.NAME+f'_r{attempt:03d}_results.zip')).exists() or (rt.SERVER/(rt.NAME+f'_r{attempt:03d}_results.zip.partial')).exists(): attempt+=1
    if rt.OUT.exists():
        previous=rt.read(rt.OUT/'launch.json')
        rt.need(previous['manifest_sha256']==rt.digest(rt.BASE/'manifest.json'),'Existing package differs')
        prior=previous['attempt'];archive=rt.SERVER/(rt.NAME+f'_r{prior:03d}_results.zip')
        if (rt.OUT/'completion.json').exists() and rt.read(rt.OUT/'completion.json')['exit_code']==0 and archive.is_file():
            print('ALREADY_COMPLETE '+str(archive),flush=True);return
        attempt=max(attempt,prior+1)
        destination=rt.SERVER/(rt.NAME+'_attempts')/f'r{prior:03d}'
        rt.need(rt.OUT.resolve()==rt.SERVER/rt.NAME and destination.resolve().is_relative_to(rt.SERVER) and not destination.exists(),'Attempt archive path')
        destination.parent.mkdir(exist_ok=True);rt.OUT.rename(destination)
        print('PRESERVED_ATTEMPT '+str(destination),flush=True)
    rt.OUT.mkdir();rt.put(rt.OUT/'launch.json',dict(attempt=attempt,manifest_sha256=rt.digest(rt.BASE/'manifest.json')))
    (rt.SERVER/(rt.NAME+'.exit')).write_text('RUNNING\n',encoding='ascii')
    with (rt.SERVER/(rt.NAME+'.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u','-B',str(rt.BASE/'run.py'),str(lock.fileno())],cwd=rt.BASE,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    (rt.SERVER/(rt.NAME+'.pid')).write_text(str(p.pid)+'\n',encoding='ascii')
    print('STARTED workers=8 BLAS=1 source_gain_fit=true frozen_expert=true pid='+str(p.pid),flush=True)

if __name__=='__main__': main()
