"""Artificial arrays and the complete production worker pipeline, without real data."""
from pathlib import Path
from datetime import date, timedelta
import copy
import hashlib
import os
import subprocess
import sys
import tempfile

H = Path(__file__).resolve().parent
P = H / H.name
sys.path.insert(0, str(P))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import transfer as tr
import scoring as sc
import runtime as rt


def must_fail(fn):
    try:
        fn()
    except (ValueError, KeyError, PermissionError):
        return
    raise AssertionError('Expected rejection')


def protocol():
    sites = ['P1', 'P15', 'P2', 'P3', 'P4']
    pairs = []
    for s in sites:
        for y in (2014, 2015):
            pid = s + '_' + str(y)
            pairs.append(dict(id=pid, site=s, year=y, early=pid+'_early', late=pid+'_late',
                              early_date=f'{y}-05-01', late_date=f'{y}-05-15'))
    return dict(sites=sites, years=[2014, 2015], doses=[0, 10, 15, 20, 25, 30, 40, 60],
                pairs=pairs, improvement_epsilon_mm=1e-10)


def artificial(p):
    rng = np.random.default_rng(741)
    sources, public, feedback, truth = {}, {}, {}, {}
    doses, sites, pairs = p['doses'], p['sites'], p['pairs']

    def pred(pair, period, s, h):
        identity = 'synthetic_model_' + '_'.join(sorted([s, h]))
        u = 3 + sites.index(pair['site'])*.1 + np.arange(8)[:, None]*.04 + np.arange(7)[None, :]*.05
        storage = np.ones((8, 7))*220 + np.arange(8)[:, None]*2 - np.arange(7)[None, :]
        return dict(cycle=pair[period], site=pair['site'], year=pair['year'], model_identity=identity,
                    fit_result_sha256=hashlib.sha256(identity.encode()).hexdigest(), model_id=identity,
                    source_sites=[v for v in sites if v not in (s, h)], doses=doses,
                    storage_mm=storage.tolist(), U_mm=u.tolist())

    contexts = {v['id']: (rng.normal(size=11) * np.arange(11)).tolist() for v in pairs}
    for s in sites:
        rows = []
        for pair in pairs:
            h = pair['site']
            if h == s:
                continue
            e, l = pred(pair, 'early', s, h), pred(pair, 'late', s, h)
            noise = rng.normal(0, .5, (8, 7)) * (np.arange(8)[:, None]+1)
            noise[0, 0] = .01
            rows.append(dict(pair=pair, early=e, late=l, context=contexts[pair['id']],
                early_storage_truth=(np.asarray(e['storage_mm'])+noise).tolist(),
                late_U_truth=(np.asarray(l['U_mm'])+rng.normal(0, .05, (8, 7))).tolist()))
        sources[s] = dict(outer_site=s, pairs=rows)
    for pair in pairs:
        s, pid = pair['site'], pair['id']
        branches = {h: {period: pred(pair, period, s, h) for period in ('early', 'late')} for h in sites if h != s}
        public[pid] = dict(pair=pair, branches=branches, context=contexts[pid])
        truth[pid] = dict(pair=pair, U_mm=(np.asarray(next(iter(branches.values()))['late']['U_mm'])+.03).tolist(),
                          definition='native crop_tra_mm daily')
        for j, b in enumerate(doses):
            obs = np.asarray(next(iter(branches.values()))['early']['storage_mm'])[j]+rng.normal(0, .4, 7)
            feedback[pid+'_b'+str(b)] = dict(pair=pid, cycle=pair['early'], dose_mm=b,
                release_at=(date.fromisoformat(pair['early_date'])+timedelta(days=7)).isoformat()+'T06:00:00', storage_mm=obs.tolist())
    return sources, public, feedback, truth


def main():
    checks = []
    p = protocol()
    sources, public, feedback, truth = artificial(p)
    models = {s: tr.fit(sources[s], p) for s in p['sites']}
    issues = {k: tr.issue(models[v['pair']['site']], v, p) for k, v in public.items()}
    history = {k: tr.adapt(models[issues[v['pair']]['pair']['site']], issues[v['pair']], v, v['dose_mm'], p) for k, v in feedback.items()}
    report = sc.evaluate(p, models, issues, history, truth)
    assert report['engineering_complete'] and report['total_pairs'] == 1680
    checks.append('Independent complete replay: 5 source sets, 10 physical windows, 80 versions, 1680 masked pairs')
    s, pid = p['sites'][0], p['pairs'][0]['id']
    fb = feedback[pid+'_b10']
    assert np.asarray(models[s]['feedback_sigma_mm']).shape == (8, 7)
    assert models[s]['feedback_sigma_mm'][0][0] == .05
    assert not np.allclose(models[s]['feedback_sigma_mm'][1], models[s]['feedback_sigma_mm'][2])

    # Independent scalar masked metrics, including b=0 where zero is not an endpoint.
    for b in p['doses']:
        keep, pairs = sc.mask(p['doses'], b)
        assert len(keep) == 7 and len(pairs) == 21
        assert all(p['doses'][a] != b and p['doses'][c] != b for a, c in pairs)
        pred, target = np.asarray(issues[pid]['off']), np.asarray(truth[pid]['U_mm'])
        expected = sum(abs(sum((pred[a, d]-pred[c, d])-(target[a, d]-target[c, d]) for d in range(7))) for a, c in pairs)/21
        np.testing.assert_allclose(sc.measure(pred, target, keep, pairs)['mae'], expected)
    macro = sum(sum(w['on']['mae'] for w in report['windows'] if w['site'] == site)/2 for site in p['sites'])/5
    np.testing.assert_allclose(report['macro']['on']['mae'], macro)
    remaining = [w for w in report['windows'] if w['id'] != report['drop_largest']['removed']]
    drop = sum(sum(w['on']['mae'] for w in remaining if w['site'] == site)/sum(w['site'] == site for w in remaining) for site in p['sites'])/5
    np.testing.assert_allclose(report['drop_largest']['on']['mae'], drop)
    assert sc.signs(.02, -.02, .02)['repaired'] == 1
    assert sc.signs(.02, .0, -.02)['new_wrong'] == 1
    assert sc.signs(.01, -.02, .02)['repaired'] == 0
    checks.append('Scalar 21-pair score; b=0 exclusion; physical/site-balanced macro and drop-largest; sign zero band')

    # Neutral feedback gives identical likelihoods and therefore on=off.
    neutral = copy.deepcopy(sources[s])
    for row in neutral['pairs']:
        row['context'] = [3.]*11
        row['early_storage_truth'] = (np.asarray(row['early']['storage_mm'])+.2).tolist()
    nm = tr.fit(neutral, p)
    ni = tr.issue(nm, public[pid], p)
    nv = tr.adapt(nm, ni, fb, 10, p)
    np.testing.assert_allclose(ni['prior_weights'], [.125]*8, atol=0)
    np.testing.assert_allclose(nv['on'], ni['off'], rtol=1e-13, atol=1e-13)
    sc.audit_model(nm, p)
    checks.append('All-constant contexts give uniform prior; equal feedback likelihood leaves off unchanged')

    # Matched residual receives the largest likelihood; all contexts equal.
    directed = copy.deepcopy(neutral)
    for i, row in enumerate(directed['pairs']):
        row['early_storage_truth'] = (np.asarray(row['early']['storage_mm'])+i*.2).tolist()
    dm = tr.fit(directed, p)
    di = tr.issue(dm, public[pid], p)
    match = copy.deepcopy(fb)
    match['storage_mm'] = np.asarray(di['atoms'][0]['target_early_storage'])[1].tolist()
    dv = tr.adapt(dm, di, match, 10, p)
    assert int(np.argmax(dv['posterior_weights'])) == 0 and dv['posterior_weights'][0] > .125

    far = copy.deepcopy(public[pid])
    far['context'] = (np.asarray(far['context']) + 1e5).tolist()
    fi = tr.issue(models[s], far, p)
    fv = [tr.adapt(models[s], fi, feedback[pid+'_b'+str(b)], b, p) for b in p['doses']]
    assert any(w == 0 for w in fi['prior_weights'])
    sc.audit_prediction(models[s], fi, fv, p)
    assert all(np.isfinite(v['on']).all() for v in fv)
    checks.append('Known residual directs posterior; log-domain underflow keeps global and conditional predictions finite')

    for key, val in [('release_at', '2099-01-01T06:00:00'), ('cycle', 'wrong'), ('storage_mm', [1]*6), ('dose_mm', 11), ('extra', 1)]:
        bad = copy.deepcopy(fb)
        bad[key] = val
        must_fail(lambda: tr.adapt(models[s], issues[pid], bad, 10, p))
    must_fail(lambda: tr.adapt(models[s], issues[pid], fb, 20, p))
    bad = copy.deepcopy(sources[s])
    bad['pairs'][0]['early']['source_sites'].append(s)
    must_fail(lambda: tr.fit(bad, p))
    bad = copy.deepcopy(public[pid])
    next(iter(bad['branches'].values()))['late']['fit_result_sha256'] = 'wrong'
    must_fail(lambda: tr.issue(models[s], bad, p))
    for field in ('on', 'posterior_weights', 'log_likelihood'):
        bad = copy.deepcopy(history)
        v = bad[pid+'_b10'][field]
        if isinstance(v[0], list):
            v[0][0] += 1
        else:
            v[0] += 1
        must_fail(lambda: sc.evaluate(p, models, issues, bad, truth))
    bad = copy.deepcopy(models)
    bad[s]['atoms'][0]['early_residual'][0][0] += 1
    must_fail(lambda: sc.evaluate(p, bad, issues, history, truth))
    bad = copy.deepcopy(issues)
    bad[pid]['atoms'][0]['target_early_storage'][0][0] += 1
    must_fail(lambda: sc.evaluate(p, models, bad, history, truth))
    checks.append('Reject illegal feedback, source leak, mismatched model, tampered residuals, target bases, weights and predictions')

    # Copy code ONLY and generate every numerical input in a separate artificial tree.
    temp = Path(tempfile.mkdtemp(prefix='synthetic_', dir=H))
    base, out = temp/'pkg', temp/'out'
    base.mkdir()
    out.mkdir()
    for f in P.glob('*.py'):
        (base/f.name).write_bytes(f.read_bytes())
    for sub, values in [('source', sources), ('public', public), ('feedback', feedback), ('truth', truth)]:
        for key, val in values.items():
            rt.put(base/sub/(key+'.json'), val)
    rt.put(base/'protocol.json', p)
    rt.put(base/'manifest.json', {'files': rt.filemap(base)})
    rt.put(out/'launch.json', dict(manifest_sha256=rt.digest(base/'manifest.json'), attempt=1))
    driver = temp/'driver.py'
    driver.write_text("import pathlib,sys\nbase=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(base))\nimport runtime as rt\nrt.BASE=base;rt.OUT=out;rt.require_server=lambda:None\nsys.argv=['worker.py']+sys.argv[3:]\nimport worker\nworker.main()\n", encoding='ascii')
    count = 0
    for freeze, jobs in rt.jobs(p)+[('score_freeze.json', [('score', 'global')])]:
        files = {}
        for role, ident in jobs:
            r = subprocess.run([sys.executable, '-u', '-B', str(driver), str(base), str(out), role, ident], capture_output=True, text=True, timeout=45)
            assert r.returncode == 0, (role, ident, r.stdout, r.stderr)
            root = out/role/ident
            seal = rt.read(root/'sealed.json')
            assert rt.digest(root/'result.json') == seal['sha256']
            for n, digest in rt.filemap(root).items():
                files[(root/n).relative_to(out).as_posix()] = digest
            count += 1
        rt.put(out/freeze, dict(passed=True, files=files))
        print('SYNTHETIC_STAGE', freeze, 'workers', count, flush=True)
    assert count == 96 and rt.read(out/'score/global/result.json') == report
    checks.append('96 production worker subprocesses, artificial inputs only; exact read ledgers, stage seals and scorer equality')
    for target in (base/'truth'/f'{pid}.json', base/'feedback'/(pid+'_b20.json'), base/'source'/'P15.json'):
        code = "import sys,pathlib;sys.path.insert(0,sys.argv[1]);import access;access.install(sys.argv[1],sys.argv[2],[],sys.argv[2]);pathlib.Path(sys.argv[3]).read_bytes()"
        r = subprocess.run([sys.executable, '-B', '-c', code, str(base), str(out), str(target)], capture_output=True, text=True, timeout=10)
        assert r.returncode != 0 and 'PermissionError' in r.stderr
    checks.append('Audit hook blocks target truth, other action feedback and other source reads')
    rt.put(H/'synthetic_validation.json', dict(passed=True, artificial_only=True, real_model_calls=0, checks=checks,
        fixture_path=str(temp), tested_code={f.name: rt.digest(f) for f in P.glob('*.py')}))
    print('SYNTHETIC_PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
