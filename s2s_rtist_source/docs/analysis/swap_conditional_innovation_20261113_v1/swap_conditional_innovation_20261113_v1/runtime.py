"""Stage and feedback permissions. This audit is not an OS security boundary."""
from pathlib import Path
import hashlib,json,os,sys
NAME='swap_conditional_innovation_20261113_v1'
SERVER=Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source')
BASE=Path(__file__).resolve().parent;OUT=SERVER/NAME
ROLES=['source','predict','score'];CURRENT_INDEX=-1
def need(v,msg):
    if not v:raise RuntimeError(msg)
def require_server():need(sys.platform=='linux' and BASE==SERVER/'deploy'/NAME,'Real computation requires exact Linux server deployment')
def read(p):return json.loads(Path(p).read_bytes())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2,ensure_ascii=True,allow_nan=False);f.write('\n')
def filemap(p):return {f.relative_to(p).as_posix():digest(f) for f in sorted(Path(p).rglob('*')) if f.is_file()}
def verify_package():
    a=filemap(BASE);a.pop('manifest.json');need(a==read(BASE/'manifest.json')['files'],'Package mismatch')
def protocol():return read(BASE/'protocol.json')
def task(tid):return next(t for t in protocol()['tasks'] if t['id']==tid)
def directory(role,tid):return OUT/'tasks'/tid/role
def checked(name):
    p=BASE/name;need(p.resolve()==p and digest(p)==read(BASE/'manifest.json')['files'][name],'Input binding '+name);return p
def seal(role,tid):
    d=directory(role,tid);put(d/'seal.json',dict(role=role,task=tid,manifest_sha256=digest(BASE/'manifest.json'),files=filemap(d)))
def prior_file(role,tid,name):
    d=directory(role,tid);s=read(d/'seal.json')
    need(s['role']==role and s['task']==tid and s['manifest_sha256']==digest(BASE/'manifest.json'),'Stage identity')
    need(read(OUT/(role+'_freeze.json'))['tasks'][tid]==digest(d/'seal.json'),'Global barrier mismatch')
    need(digest(d/name)==s['files'][name],'Sealed file mismatch');return d/name
def advance(index):
    global CURRENT_INDEX
    need(index==CURRENT_INDEX+1,'Nonsequential decision');CURRENT_INDEX=index
def allowed_package(role,t,name,releases,index):
    tid=t['id']
    if name in ('protocol.json','manifest.json') or ('/' not in name and name.endswith('.py')):return True
    if name==f'models/{t["base_id"]}.npz':return role in ('source','predict')
    if name.startswith('years/'):
        bits=name.split('/')
        if len(bits)!=4:return False
        if role=='source':return bits[1] in ('2014','2015') and bits[2]==t['donor'] and bits[3] in ('inputs.csv','labels.csv','rootstate.csv')
        return role=='predict' and bits[1]=='2016' and bits[2]==t['target'] and bits[3] in ('inputs.csv','rootstate.csv')
    if name==f'soil/{t["donor"]}/soil.csv':return role=='source'
    if name==f'soil/{t["target"]}/soil.csv':return role=='predict'
    if name in (f'inputs/{tid}/context.npz',f'inputs/{tid}/releases.json'):return role in ('predict','score')
    if name in releases:return role=='predict' and releases[name]<=index
    if name==f'scores/{tid}.npz':return role=='score'
    return False
def restrict(role,t,releases):
    tid=t['id'];own=directory(role,tid);events=[]
    def audit(event,args):
        if event.startswith(('socket.','subprocess.')) or event in ('os.system','os.posix_spawn'):raise RuntimeError('Worker external execution/network forbidden')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute().resolve();mode,flags=args[1:3]
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if write:need(p.is_relative_to(own) or str(p)=='/dev/null','Write outside own stage');return
        if p.is_relative_to(BASE):
            name=p.relative_to(BASE).as_posix();need(allowed_package(role,t,name,releases,CURRENT_INDEX),'Forbidden package read '+name)
            if not name.endswith('.py') and name!='manifest.json':events.append(dict(file='package/'+name,index=CURRENT_INDEX))
            return
        if p.is_relative_to(own):return
        allowed=[]
        if role=='predict':allowed=[directory('source',tid)/n for n in ('fit.npz','seal.json')]+[OUT/'source_freeze.json']
        if role=='score':allowed=[directory('predict',tid)/n for n in ('predictions.npz','seal.json')]+[OUT/'predict_freeze.json']
        if p in allowed:events.append(dict(file=str(p.relative_to(OUT)),index=CURRENT_INDEX));return
        if p.is_relative_to(SERVER):raise RuntimeError('Forbidden server read '+str(p))
        # Libraries are imported before this hook. Permit runtime/library resources only.
        if any(p.is_relative_to(Path(prefix).resolve()) for prefix in (sys.prefix,sys.base_prefix)) or str(p).startswith(('/usr/','/lib/','/proc/','/dev/')):return
        raise RuntimeError('Forbidden external read '+str(p))
    sys.addaudithook(audit);return events
