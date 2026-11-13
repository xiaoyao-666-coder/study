"""Project authenticated R1/B0/D1 saved fields; never import model code."""
from pathlib import Path
from datetime import date, timedelta
import argparse
import copy
import hashlib
import json
import math

H = Path(__file__).resolve().parent
P = H / H.name
SOURCES = {
    'R1': ('swap_water_residual_transfer_20261112_v1', '1d25322261775ecb688d2b10a06633f1639f81790d2b1307b0fb0ee896d7f90f'),
    'B0': ('swap_water_nested_error_20261111_v1', '1c55c4ec5eeaab4bad69dce24b8e6fbc217e92bae37ed00cf68f21cd88caa334'),
    'D1': ('swap_water_feedback_factorial_20261112_v1', '8ff998261b859f44c65af014ecf550f9bb76b96067a2812f00ff5ff72da5283f'),
}
FEATURES = ['cwdm', 'cwso', 'dvs', 'lai', 'root_cm', 'rain_sum', 'rad_mean', 'tmax_mean', 'tmin_mean', 'vapor_mean', 'wind_mean']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def put(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='ascii', newline='\n') as f:
        json.dump(x, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    args = ap.parse_args()
    assert not (P / 'protocol.json').exists(), 'Projection is create-only'
    bases, protected, provenance = {}, {}, {}
    for key, (name, expected) in SOURCES.items():
        base = args.source_root / 'docs/analysis' / name / name
        assert sha(base / 'manifest.json') == expected, key
        manifest = read(base / 'manifest.json')['files']
        files = [f for f in base.rglob('*') if f.is_file()]
        assert not any(f.is_symlink() for f in base.rglob('*'))
        actual = {f.relative_to(base).as_posix(): sha(f) for f in files if f.name != 'manifest.json'}
        assert actual == manifest, key
        protected.update({str(f): sha(f) for f in files})
        bases[key] = base

    def get(key, name):
        f = bases[key] / name
        provenance[key + '/' + name] = sha(f)
        return read(f)

    r1 = get('R1', 'protocol.json')
    d1 = get('D1', 'protocol.json')
    assert r1['pairs'] == d1['pairs'] and r1['doses'] == d1['doses']
    pairs, doses, sites = r1['pairs'], r1['doses'], r1['sites']
    assert r1['years'] == [2014, 2015]
    contexts = {}
    for pair in pairs:
        rows = get('B0', 'public/' + pair['early'] + '.json')
        assert len(rows) == 8 and [r['dose_mm'] for r in rows] == doses
        assert all(r['cycle'] == pair['early'] and r['site'] == pair['site'] and r['year'] == pair['year'] for r in rows)
        assert all(r['C'] == rows[0]['C'] and r['weather'] == rows[0]['weather'] for r in rows)
        c, w = rows[0]['C'], rows[0]['weather']
        assert len(w) == 7
        assert [r['date'] for r in w] == [(date.fromisoformat(pair['early_date']) + timedelta(days=d)).isoformat() for d in range(7)]
        context = [c[k] for k in FEATURES[:5]] + [sum(r['rain'] for r in w)]
        context += [sum(r[k] for r in w) / 7 for k in ('rad', 'tmax', 'tmin', 'vapor', 'wind')]
        assert len(context) == 11 and all(math.isfinite(v) for v in context)
        contexts[pair['id']] = context

    truth, feedback = {}, {}
    for pair in pairs:
        pid = pair['id']
        truth[pid] = get('D1', 'truth/' + pid + '.json')
        assert truth[pid]['pair'] == pair and truth[pid]['definition'] == 'native crop_tra_mm daily'
        assert truth[pid]['U_mm'] == get('R1', 'truth/' + pid + '.json')['U_mm']
        for b in doses:
            name = pid + '_b' + str(b)
            f = get('D1', 'legal_early_storage/' + name + '.json')
            release = (date.fromisoformat(pair['early_date']) + timedelta(days=7)).isoformat() + 'T06:00:00'
            assert f['condition'] == 'ES' and f['quantity'] == 'storage' and not f['diagnostic_only']
            assert (f['pair'], f['cycle'], f['dose_mm'], f['release_at']) == (pid, pair['early'], b, release)
            assert len(f['values_mm']) == 7 and release <= pair['late_date'] + 'T06:00:00'
            feedback[name] = dict(pair=pid, cycle=pair['early'], dose_mm=b, release_at=release, storage_mm=f['values_mm'])
            assert feedback[name] == get('R1', 'feedback/' + name + '.json')

    for site in sites:
        source = copy.deepcopy(get('R1', 'source/' + site + '.json'))
        assert source['outer_site'] == site
        assert [r['pair'] for r in source['pairs']] == [p for p in pairs if p['site'] != site]
        for row in source['pairs']:
            pair = row['pair']
            e, l = row['early'], row['late']
            for k in ('model_identity', 'model_id', 'fit_result_sha256', 'source_sites'):
                assert e[k] == l[k]
            assert e['source_sites'] == [s for s in sites if s not in (site, pair['site'])]
            assert row['late_U_truth'] == truth[pair['id']]['U_mm']
            assert row['early_storage_truth'] == [feedback[pair['id'] + '_b' + str(b)]['storage_mm'] for b in doses]
            row['context'] = contexts[pair['id']]
        put(P / 'source' / (site + '.json'), source)

    for pair in pairs:
        pid = pair['id']
        public = get('R1', 'public/' + pid + '.json')
        assert public['pair'] == pair and set(public['branches']) == set(sites) - {pair['site']}
        source = read(P / 'source' / (pair['site'] + '.json'))
        for donor, branch in public['branches'].items():
            ref = next(r for r in source['pairs'] if r['pair']['site'] == donor)
            for period in ('early', 'late'):
                for k in ('model_identity', 'fit_result_sha256', 'model_id', 'source_sites'):
                    assert branch[period][k] == ref[period][k]
                assert branch[period]['cycle'] == pair[period] and branch[period]['doses'] == doses
        public['context'] = contexts[pid]
        put(P / 'public' / (pid + '.json'), public)
        put(P / 'truth' / (pid + '.json'), truth[pid])
        for b in doses:
            name = pid + '_b' + str(b)
            put(P / 'feedback' / (name + '.json'), feedback[name])

    p = dict(name=H.name, candidate='context_weighted_paired_response_atoms',
             sites=sites, years=[2014, 2015], doses=doses, pairs=pairs, context_features=FEATURES,
             feedback_floor_mm=.05, sign_zero_mm=.01, improvement_epsilon_mm=1e-10,
             score_pairs_per_behavior=21, score_pairs_total=1680,
             physical_cycles=20, physical_pairs=10, workers=8, blas_threads=1,
             worker_timeout_seconds=1800, total_timeout_seconds=10800, min_free_bytes=2000000000,
             stages=dict(source=5, issue=10, history=80, score=1),
             native_runs=0, base_model_fit_calls=0, base_model_prediction_calls=0, recovery_QPs=0, crop_runs=0,
             economic_scoring=False, automatic_promotion=False, adoption=r1['adoption'],
             b0_archive_sha256=r1['b0_archive_sha256'], contract_sha256=sha(P / 'CONTRACT.md'))
    put(P / 'protocol.json', p)
    put(P / 'provenance.json', dict(inputs=provenance, source_manifests={k: v[1] for k, v in SOURCES.items()},
        projection='Authenticated saved fields and public context arithmetic only. No real fit, forward, residual calculation or scoring.'))
    for path, digest in protected.items():
        assert sha(path) == digest
    put(H / 'protected_inputs.json', protected)
    put(H / 'asset_validation.json', dict(passed=True, real_local_model_calls=0, source_files=5,
        source_pairs_per_file=8, public_pairs=10, feedback_files=80, truth_files=10, context_features=FEATURES,
        exact_R1_B0_D1_bindings=True, dose_invariant_contexts=10, protected_files=len(protected)))
    print('PREPARED R1 + B0 + D1; no real model calls', flush=True)


if __name__ == '__main__':
    main()
