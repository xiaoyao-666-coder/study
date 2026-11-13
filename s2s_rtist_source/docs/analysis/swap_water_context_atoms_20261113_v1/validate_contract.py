"""Static contract and saved-field schema checks. No numerical model imports."""
from pathlib import Path
from datetime import date, timedelta
import ast
import hashlib
import json
import math
import subprocess
import sys

H = Path(__file__).resolve().parent
P = H/H.name


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value, dimensions):
    if not dimensions:
        assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return
    assert isinstance(value, list) and len(value) == dimensions[0]
    for v in value:
        shape(v, dimensions[1:])


def main():
    p = read(P/'protocol.json')
    assert sha(P/'CONTRACT.md') == p['contract_sha256']
    assert p['name'] == H.name and p['years'] == [2014, 2015]
    assert p['sites'] == ['P1', 'P15', 'P2', 'P3', 'P4']
    assert p['doses'] == [0, 10, 15, 20, 25, 30, 40, 60]
    assert [(r['site'], r['year']) for r in p['pairs']] == [(s, y) for s in p['sites'] for y in p['years']]
    assert len({r['id'] for r in p['pairs']}) == 10
    assert p['stages'] == dict(source=5, issue=10, history=80, score=1)
    assert p['score_pairs_per_behavior'] == 21 and p['score_pairs_total'] == 1680
    assert p['feedback_floor_mm'] == .05 and p['improvement_epsilon_mm'] == 1e-10 and p['sign_zero_mm'] == .01
    fields = ['cwdm', 'cwso', 'dvs', 'lai', 'root_cm', 'rain_sum', 'rad_mean', 'tmax_mean', 'tmin_mean', 'vapor_mean', 'wind_mean']
    assert p['context_features'] == fields
    source_keys = {'pair', 'early', 'late', 'early_storage_truth', 'late_U_truth', 'context'}
    pred_keys = {'cycle', 'site', 'year', 'model_identity', 'fit_result_sha256', 'model_id', 'source_sites', 'doses', 'storage_mm', 'U_mm'}

    def prediction(pred, pair, period, outer, donor):
        assert set(pred) == pred_keys
        assert pred['cycle'] == pair[period] and pred['site'] == pair['site'] and pred['year'] == pair['year']
        assert pred['source_sites'] == [s for s in p['sites'] if s not in (outer, donor)] and pred['doses'] == p['doses']
        shape(pred['storage_mm'], (8, 7))
        shape(pred['U_mm'], (8, 7))
        return {k: pred[k] for k in ('model_identity', 'fit_result_sha256', 'model_id', 'source_sites')}

    identities = {}
    for outer in p['sites']:
        source = read(P/'source'/(outer+'.json'))
        assert set(source) == {'outer_site', 'pairs'} and source['outer_site'] == outer
        assert [r['pair'] for r in source['pairs']] == [r for r in p['pairs'] if r['site'] != outer]
        for row in source['pairs']:
            assert set(row) == source_keys
            pair = row['pair']
            identity = prediction(row['early'], pair, 'early', outer, pair['site'])
            assert prediction(row['late'], pair, 'late', outer, pair['site']) == identity
            key = (outer, pair['site'])
            assert key not in identities or identities[key] == identity
            identities[key] = identity
            shape(row['context'], (11,))
            shape(row['early_storage_truth'], (8, 7))
            shape(row['late_U_truth'], (8, 7))
    for pair in p['pairs']:
        pid = pair['id']
        public = read(P/'public'/(pid+'.json'))
        assert set(public) == {'pair', 'branches', 'context'} and public['pair'] == pair
        assert set(public['branches']) == set(p['sites']) - {pair['site']}
        shape(public['context'], (11,))
        for donor, branch in public['branches'].items():
            assert set(branch) == {'early', 'late'}
            for period in ('early', 'late'):
                assert prediction(branch[period], pair, period, pair['site'], donor) == identities[(pair['site'], donor)]
        for b in p['doses']:
            f = read(P/'feedback'/(pid+'_b'+str(b)+'.json'))
            assert set(f) == {'pair', 'cycle', 'dose_mm', 'release_at', 'storage_mm'}
            assert (f['pair'], f['cycle'], f['dose_mm']) == (pid, pair['early'], b)
            release = (date.fromisoformat(pair['early_date'])+timedelta(days=7)).isoformat()+'T06:00:00'
            assert f['release_at'] == release <= pair['late_date']+'T06:00:00'
            shape(f['storage_mm'], (7,))
        truth = read(P/'truth'/(pid+'.json'))
        assert set(truth) == {'pair', 'definition', 'U_mm'} and truth['pair'] == pair
        assert truth['definition'] == 'native crop_tra_mm daily'
        shape(truth['U_mm'], (8, 7))
    # Only syntax and an early guard rejection. Never call a real model locally.
    for path in list(P.glob('*.py')) + list(H.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    r = subprocess.run([sys.executable, '-B', str(P/'worker.py'), 'source', 'P1'], capture_output=True, text=True, timeout=10)
    assert r.returncode != 0 and 'Real computation requires exact Linux server deployment' in r.stderr
    start = (P/'start.sh').read_bytes()
    assert b'\r' not in start and ('/deploy/'+H.name+'\n').encode() in start
    checks = ['Exact source/public/feedback/truth schemas and finite saved arrays',
              '2014-2015 axes, same-model source exclusion, 10 physical windows and 80 mature feedback files',
              'Frozen contract identity, 96 tasks, 21 masked dose pairs per behavior',
              'Python 3.10 syntax and LF launcher; local production worker rejected before data access']
    report = dict(passed=True, real_local_model_calls=0, checks=checks,
        tested_code={f.name: sha(f) for f in P.iterdir() if f.is_file() and f.suffix in ('.py', '.sh', '.md')})
    with (H/'contract_validation.json').open('x', encoding='ascii', newline='\n') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('CONTRACT_PASS', flush=True)


if __name__ == '__main__':
    main()
