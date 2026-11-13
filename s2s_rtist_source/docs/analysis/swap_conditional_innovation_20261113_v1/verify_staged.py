"""Verify exact staged blob bytes, sealed package and Python syntax; no models."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

H = Path(__file__).resolve().parent
P = H / H.name
ROOT = H.parents[3]


def main():
    relative = H.relative_to(ROOT).as_posix()
    output = subprocess.run(['rtk', 'proxy', 'git', 'ls-files', '--stage', '-z', '--', relative],
                            cwd=ROOT, check=True, capture_output=True).stdout
    staged = {}
    for record in output.split(b'\0'):
        if not record:
            continue
        meta, filename = record.split(b'\t', 1)
        mode, oid, stage = meta.split()
        assert stage == b'0'
        staged[filename.decode('utf-8')] = oid.decode('ascii')
    local = {p.relative_to(ROOT).as_posix(): p for p in H.rglob('*') if p.is_file()}
    assert staged.keys() == local.keys(), dict(extra=len(staged.keys() - local.keys()),
                                              missing=len(local.keys() - staged.keys()))
    for name, path in local.items():
        data = path.read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == staged[name], name
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8'), feature_version=(3, 10))
    manifest = json.loads((P / 'manifest.json').read_bytes())
    assert manifest['files'] == {p.relative_to(P).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in P.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    changed = subprocess.run(['rtk', 'proxy', 'git', 'diff', '--cached', '--name-only', '-z'],
                             cwd=ROOT, check=True, capture_output=True).stdout
    assert all(n.decode('utf-8').startswith(relative + '/') for n in changed.split(b'\0') if n)
    print('STAGED_BYTES_PASS files=' + str(len(local)) + ' sealed_members=' + str(len(manifest['files']) + 1))


if __name__ == '__main__':
    main()
