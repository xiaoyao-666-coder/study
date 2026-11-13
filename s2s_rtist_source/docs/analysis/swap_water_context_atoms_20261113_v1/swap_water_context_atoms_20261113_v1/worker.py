"""Source fit, off, single-action on, and globally delayed scoring roles."""
import sys
import runtime as rt

def allowed_paths(role,ident,p):
    paths=[rt.BASE/'protocol.json',rt.BASE/'manifest.json']
    if role=='source':return paths+[rt.BASE/'source'/f'{ident}.json']
    if role=='score':
        for _,jobs in rt.jobs(p):
            for r,i in jobs:paths+=rt.frozen_paths(r,i)
        return paths+[rt.BASE/'truth'/f"{x['id']}.json" for x in p['pairs']]
    pair,b=rt.meta(p,role,ident);paths+=rt.frozen_paths('source',pair['site'])
    if role=='issue':return paths+[rt.BASE/'public'/f"{pair['id']}.json"]
    rt.need(role=='history' and b in p['doses'],'Known role and behavior')
    return paths+rt.frozen_paths('issue',pair['id'])+[rt.BASE/'feedback'/f"{pair['id']}_b{b}.json"]

def main():
    rt.require_server();role,ident=sys.argv[1:];p=rt.protocol()
    rt.need(rt.read(rt.OUT/'launch.json')['manifest_sha256']==rt.digest(rt.BASE/'manifest.json'),'Parent package verified')
    import numpy as np
    import transfer,scoring,access
    r=rt.root(role,ident);r.mkdir(parents=True,exist_ok=False)
    allowed=allowed_paths(role,ident,p);reads=access.install(rt.BASE,rt.OUT,allowed,r)
    rt.need(rt.read(rt.BASE/'protocol.json')==p,'Protocol identity');rt.read(rt.BASE/'manifest.json')
    if role=='source':
        rt.need(ident in p['sites'],'Known outer site');result=transfer.fit(rt.read(rt.BASE/'source'/f'{ident}.json'),p)
    elif role=='score':
        models={s:rt.frozen('source',s) for s in p['sites']}
        issues={x['id']:rt.frozen('issue',x['id']) for x in p['pairs']}
        histories={x['id']+'_b'+str(b):rt.frozen('history',x['id']+'_b'+str(b)) for x in p['pairs'] for b in p['doses']}
        truths={x['id']:rt.read(rt.BASE/'truth'/f"{x['id']}.json") for x in p['pairs']}
        result=scoring.evaluate(p,models,issues,histories,truths)
    else:
        pair,b=rt.meta(p,role,ident);model=rt.frozen('source',pair['site'])
        if role=='issue':
            public=rt.read(rt.BASE/'public'/f"{pair['id']}.json");rt.need(public['pair']==pair,'Public binding')
            result=transfer.issue(model,public,p)
        else:
            issued=rt.frozen('issue',pair['id']);rt.need(issued['pair']==pair,'Issued binding')
            feedback=rt.read(rt.BASE/'feedback'/f"{pair['id']}_b{b}.json");rt.need(feedback['dose_mm']==b,'Requested behavior only')
            result=transfer.adapt(model,issued,feedback,b,p)
    rt.put(r/'result.json',result)
    rt.need(set(reads)=={str(x.resolve()) for x in allowed},'Exact role read ledger')
    rt.seal(role,ident,reads);print('SEALED '+role+' '+ident,flush=True)
if __name__=='__main__':main()
