"""Artificial year interventions and saved target key checks; no real forward."""
from pathlib import Path
from datetime import date, timedelta
import csv, hashlib, json, os, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
P = H / H.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    validation = json.loads((H / 'validation.json').read_bytes())
    original = Path(validation['fixture'])
    fixture = original / 'package'
    code = {p.name: sha(p) for p in P.glob('*.py')}
    assert code == validation['code_sha256']
    assert code == {p.name: sha(p) for p in fixture.glob('*.py')}
    task = json.loads((fixture / 'protocol.json').read_bytes())['tasks'][0]
    prior = original / 'output' / 'tasks' / task['id'] / 'source'
    baseline = arrays(prior / 'fit.npz')
    baseline_proof = arrays(prior / 'nuisance_proof.npz')
    root = Path(tempfile.mkdtemp(prefix='conditional_year_boundaries_'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for case in ('late_context_water_gain', 'early_gain_only'):
        base = root / case / 'package'
        shutil.copytree(fixture, base)
        out = base.parent / 'output'
        out.mkdir()
        year = 2015 if case == 'late_context_water_gain' else 2014
        folder = base / 'years' / str(year) / task['donor']
        # Preserve untouched decimal strings exactly: pandas' default parse and
        # rewrite can perturb VWC even when the intervention is only on gain.
        with (folder / 'labels.csv').open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            labels = list(reader)
        for row in labels:
            row['target_net_gain_7d'] = str(-3 * float(row['irrigation_mm']))
        if case == 'late_context_water_gain':
            water = [c for c in fields if c.startswith('target_soil_vwc_')]
            assert len(water) == 7, fields
            for row in labels:
                for column in water:
                    row[column] = '0.61'
            inputs = pd.read_csv(folder / 'inputs.csv')
            inputs['predecision_lai'] += .7
            inputs.to_csv(folder / 'inputs.csv', index=False)
        with (folder / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(labels)
        (base / 'manifest.json').write_text(json.dumps(dict(files={
            p.relative_to(base).as_posix(): sha(p) for p in base.rglob('*')
            if p.is_file() and p.name != 'manifest.json'})), encoding='utf-8')
        run = subprocess.run([sys.executable, '-B', str(original / 'harness.py'),
                              str(base), str(out), 'source', task['id']],
                             env=env, text=True, encoding='utf-8', capture_output=True)
        assert run.returncode == 0, run.stdout + run.stderr
        source = out / 'tasks' / task['id'] / 'source'
        fit = arrays(source / 'fit.npz')
        proof = arrays(source / 'nuisance_proof.npz')
        for key in baseline_proof:
            np.testing.assert_array_equal(proof[key], baseline_proof[key], err_msg=case + ':' + key)
        for key in ('mu', 'sd', 'A', 'rms'):
            np.testing.assert_array_equal(fit[key], baseline[key], err_msg=case + ':' + key)
        if case == 'early_gain_only':
            for key in baseline:
                np.testing.assert_array_equal(fit[key], baseline[key], err_msg=case + ':' + key)
        else:
            assert any(not np.array_equal(fit[key], baseline[key]) for key in ('c', 'W', 'UW'))
    write(H / 'split_validation.json', dict(passed=True, synthetic_only=True,
          late_context_water_gain_cannot_change_early_nuisance=True,
          early_gain_cannot_change_any_fit=True, late_intervention_changes_correction=True,
          fixture=str(root), code_sha256=code, test_sha256=sha(Path(__file__))))

    # Only compare saved keys and hashes for the real package. Never transform,
    # fit, encode, evaluate a model, or score a new policy on local real data.
    protocol = json.loads((P / 'protocol.json').read_bytes())
    provenance = json.loads((P / 'provenance.json').read_bytes())['bindings']
    sites = sorted({t['target'] for t in protocol['tasks']})
    saved = {}
    for site in sites:
        folder = P / 'years' / '2016' / site
        for name in ('inputs.csv', 'rootstate.csv'):
            path = folder / name
            suffix = str(Path('years') / '2016' / site / name)
            ancestry = [h for key, h in provenance.items() if key.endswith(suffix)]
            assert len(ancestry) == 1 and ancestry[0] == sha(path)
        with (folder / 'inputs.csv').open(encoding='utf-8', newline='') as stream:
            inputs = list(csv.DictReader(stream))
        with (folder / 'rootstate.csv').open(encoding='utf-8', newline='') as stream:
            roots = list(csv.DictReader(stream))
        keys = ('target_year', 'site_id', 'decision_date')
        cycles = {}
        for row in inputs:
            assert row['target_year'] == '2016' and row['site_id'] == site
            cycles.setdefault(tuple(row[k] for k in keys), []).append(row)
        assert len(roots) == len(cycles)
        assert len({tuple(row[k] for k in keys) for row in roots}) == len(roots)
        for row in roots:
            group = sorted(cycles[tuple(row[k] for k in keys)], key=lambda r: float(r['irrigation_mm']))
            assert [float(r['irrigation_mm']) for r in group] == [0, 10, 15, 20, 25, 30, 40, 60]
            assert row['sample_id'] == group[0]['sample_id']
            day = date.fromisoformat(row['decision_date'])
            assert row['state_timestamp'] == str(day) + 'T06:00:00'
            assert row['crop_timestamp'] == row['prior_end'] == str(day - timedelta(days=1))
            assert row['prior_start'] == str(day - timedelta(days=3))
        ordered = sorted(inputs, key=lambda r: tuple(r[k] for k in keys) + (float(r['irrigation_mm']),))
        saved[site] = np.array([r['sample_id'] for r in ordered]).reshape(-1, 8)
    for task in protocol['tasks']:
        context = arrays(P / 'inputs' / task['id'] / 'context.npz')
        np.testing.assert_array_equal(saved[task['target']], context['sample_ids'])
    write(H / 'target_identity_validation.json', dict(passed=True, sites=len(sites),
          directions=len(protocol['tasks']), authenticated_2016_input_and_rootstate_files=10,
          saved_target_keys_match=True, rootstate_clock_and_keys_match=True,
          local_real_compute_calls=0, test_sha256=sha(Path(__file__))))
    print('SPLIT_AND_TARGET_IDENTITY_PASS', flush=True)


if __name__ == '__main__':
    main()
