"""Create-only release after all artificial and saved-field checks pass."""
from pathlib import Path
import argparse, ast, hashlib, json, zipfile

H = Path(__file__).resolve().parent
P = H / H.name
ROOT = H.parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def put(p, data):
    with p.open('x', encoding='ascii', newline='\n') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=ROOT)
    args = ap.parse_args()
    ARCHIVE = args.output_dir / (H.name + '.zip')
    assert not ARCHIVE.exists() and not (P / 'manifest.json').exists()
    reports = ('synthetic_validation.json', 'runtime_validation.json', 'contract_validation.json', 'asset_validation.json')
    for name in reports:
        report = read(H / name)
        assert report['passed']
        assert report.get('real_model_calls', report.get('real_local_model_calls')) == 0
        for n, h in report.get('tested_code', {}).items():
            assert sha(P / n) == h, (name, n)
    assert read(H / 'synthetic_validation.json')['artificial_only']
    assert read(H / 'runtime_validation.json')['artificial_workers_only']
    expected = {'runtime.py', 'run.py', 'worker.py', 'transfer.py', 'scoring.py', 'access.py', 'launch.py',
                'start.sh', 'protocol.json', 'provenance.json', 'CONTRACT.md'}
    p = read(P / 'protocol.json')
    expected |= {'source/' + s + '.json' for s in p['sites']}
    for pair in p['pairs']:
        expected |= {'public/' + pair['id'] + '.json', 'truth/' + pair['id'] + '.json'}
        expected |= {'feedback/' + pair['id'] + '_b' + str(b) + '.json' for b in p['doses']}
    members = {}
    for f in P.rglob('*'):
        assert not f.is_symlink()
        if not f.is_file():
            continue
        name = f.relative_to(P).as_posix()
        name.encode('ascii')
        assert name in expected, name
        if f.suffix == '.py':
            ast.parse(f.read_text(encoding='utf-8'), feature_version=(3, 10))
        elif f.suffix == '.json':
            read(f)
        members[name] = sha(f)
    assert set(members) == expected and len(members) == 116
    shell = (P / 'start.sh').read_bytes()
    assert shell.startswith(b'#!/usr/bin/env bash\n') and b'\r' not in shell
    assert sha(P / 'CONTRACT.md') == p['contract_sha256']
    protected = read(H / 'protected_inputs.json')
    for n, h in protected.items():
        assert sha(Path(n)) == h, n
    put(P / 'manifest.json', dict(name=H.name, files=members))
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(set(members) | {'manifest.json'}):
            z.write(P / name, H.name + '/' + name)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        expected_zip = {H.name + '/' + n for n in set(members) | {'manifest.json'}}
        assert len(z.namelist()) == len(expected_zip) and set(z.namelist()) == expected_zip
        for name in set(members) | {'manifest.json'}:
            assert hashlib.sha256(z.read(H.name + '/' + name)).hexdigest() == sha(P / name)
    for n, h in protected.items():
        assert sha(Path(n)) == h, n
    put(H / 'build.json', dict(name=H.name, archive_path=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
                              archive_sha256=sha(ARCHIVE), manifest_sha256=sha(P / 'manifest.json'),
                              members=len(members) + 1, real_local_model_calls=0,
                              protected_files_unchanged=len(protected),
                              validation={n: sha(H / n) for n in reports}))
    print('PACKAGE_BUILT', ARCHIVE.stat().st_size, len(members) + 1, sha(ARCHIVE), flush=True)


if __name__ == '__main__':
    main()
