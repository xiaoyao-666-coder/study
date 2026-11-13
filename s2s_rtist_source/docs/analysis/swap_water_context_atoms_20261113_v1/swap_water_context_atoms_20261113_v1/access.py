"""Audit exact data reads. Numerical assets outside each role's allowlist fail."""
import os,sys
from pathlib import Path

def install(base,out,allowed,write_root,feedback_allowed=None):
    base=Path(base).resolve();out=Path(out).resolve();allowed={Path(p).resolve() for p in allowed}
    write_root=Path(write_root).resolve();runtime_roots={Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve()}
    opened=[]
    def hook(event,args):
        if event!='open' or isinstance(args[0],int):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            if not p.is_relative_to(write_root):raise PermissionError('Role write denied: '+str(p))
            return
        if feedback_allowed is not None and p.is_relative_to(base/'feedback') and p not in feedback_allowed:
            raise PermissionError('Feedback not yet released: '+str(p))
        if p in allowed:
            opened.append(str(p));return
        if p.is_relative_to(write_root):return
        if p.parent==base and p.suffix=='.py':return
        if p.is_relative_to(base) or p.is_relative_to(out):raise PermissionError('Role data read denied: '+str(p))
        if any(p.is_relative_to(r) for r in runtime_roots):return
        if str(p) in ('/dev/urandom','/dev/null'):return
        raise PermissionError('External data read denied: '+str(p))
    sys.addaudithook(hook);return opened
