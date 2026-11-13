"""Clean extraction, byte identity and static production verifier; no model calls."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import tempfile
import zipfile

H = Path(__file__).resolve().parent


def main():
    build = json.loads((H/'build.json').read_text(encoding='utf-8'))
    archive = Path(build['archive_path'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == build['archive_sha256']
    target = Path(tempfile.mkdtemp(prefix='release_check_', dir=H))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 117
        for n in z.namelist():
            n.encode('ascii')
            assert n.startswith(H.name+'/') and '..' not in n.split('/') and '\\' not in n
        z.extractall(target)
    base = target/H.name
    spec = importlib.util.spec_from_file_location('release_runtime', base/'runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    protocol = runtime.verify_package()  # Hash/schema check only. Not require_server or worker.
    assert runtime.digest(base/'manifest.json') == build['manifest_sha256']
    assert sum(len(jobs) for _, jobs in runtime.jobs(protocol))+1 == 96
    for f in base.rglob('*'):
        if f.is_file():
            assert f.read_bytes() == (H/H.name/f.relative_to(base)).read_bytes()
        if f.suffix == '.py':
            ast.parse(f.read_text(encoding='utf-8'), feature_version=(3, 10))
    assert b'\r' not in (base/'start.sh').read_bytes()
    runtime.put(H/'release_validation.json', dict(passed=True, real_local_model_calls=0,
        archive_sha256=build['archive_sha256'], manifest_sha256=build['manifest_sha256'],
        members=117, clean_extraction_byte_identity=True, production_static_verifier=True,
        python310_syntax=True, ascii_member_paths=True, LF_shell=True))
    print('RELEASE_VERIFIED', build['archive_sha256'], flush=True)


if __name__ == '__main__':
    main()
