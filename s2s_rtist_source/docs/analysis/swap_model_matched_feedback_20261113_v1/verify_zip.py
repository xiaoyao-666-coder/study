"""Independent clean extraction and inventory validation; no numerical execution."""
import ast,hashlib,json,tempfile,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;NAME=H.name;ROOT=Path(r'D:\study\s2s_rtist_source')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    archive=ROOT/(NAME+'.zip');dest=Path(tempfile.mkdtemp(prefix='model_matched_zip_'))
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
        assert all(n.isascii() and n.startswith(NAME+'/') and '..' not in Path(n).parts for n in names)
        z.extractall(dest)
    p=dest/NAME;m=json.loads((p/'manifest.json').read_text());actual={f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'}
    assert actual==m['files']
    for f in p.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,10))
    assert b'\r' not in (p/'start.sh').read_bytes()
    assert all('__pycache__' not in n and not n.endswith('.pyc') for n in actual)
    assert not any(n.startswith(('years/2016','years/2017','years/2018')) for n in actual)
    for parent in ['models','inputs','scores','feedback']:assert (p/parent).exists()
    protocol=json.loads((p/'protocol.json').read_text());assert len(protocol['tasks'])==60 and protocol['years']==[2014,2015,2016]
    result=dict(passed=True,archive_sha256=sha(archive),clean_directory=str(p),members=len(names),exact_inventory=True,python310=True,LF=True,no_real_compute=True)
    (H/'clean_zip_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
