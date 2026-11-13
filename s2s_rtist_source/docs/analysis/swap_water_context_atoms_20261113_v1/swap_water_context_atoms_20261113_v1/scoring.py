"""Independent arithmetic replay, then physical-window-first masked scoring."""
from datetime import date, timedelta
from itertools import combinations
import numpy as np
from runtime import need

METRICS = ('mae', 'daily_delta', 'daily_u')
BRANCHES = ('raw', 'off', 'on')


def array(v, shape):
    a = np.asarray(v, dtype=float)
    need(a.shape == shape and np.isfinite(a).all(), 'Scoring finite shape ' + str(shape))
    return a


def close(a, b, message):
    x, y = np.asarray(a), np.asarray(b)
    need(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
         and np.allclose(x, y, rtol=1e-11, atol=1e-10), message)


def normalized_log(v):
    # Independent scalar normalization, without calling the estimator.
    from math import exp, fsum, log
    values = [float(x) for x in v]
    top = max(values)
    normalizer = log(fsum(exp(x - top) for x in values))
    return np.array([x - top - normalizer for x in values])


def audit_model(m, p):
    rows = m['source_snapshot']['pairs']
    expected = [v for v in p['pairs'] if v['site'] != m['outer_site']]
    need(m['source_snapshot']['outer_site'] == m['outer_site'] and [r['pair'] for r in rows] == expected, 'Source exclusion')
    need(m['physical_source_pairs'] == m['atom_count'] == 8 and m['source_pair_ids'] == [v['id'] for v in expected], 'Eight source units')
    contexts = array([r['context'] for r in rows], (8, 11))
    center = contexts.mean(0)
    scale0 = contexts.std(0)
    active = scale0 > 1e-12
    scale = np.where(active, scale0, 1.)
    distances = []
    for i, j in combinations(range(8), 2):
        d = float(np.sqrt(np.mean(((contexts[i, active] - contexts[j, active]) / scale[active]) ** 2))) if active.any() else 0.
        if d > 1e-12:
            distances.append(d)
    bandwidth = float(np.median(distances)) if distances else 1.
    for key, value in [('center', center), ('scale', scale), ('active', active), ('source_contexts', contexts), ('bandwidth', bandwidth)]:
        close(m['context'][key], value, 'Source context ' + key)
    errors = []
    need(len(m['atoms']) == 8, 'Exact atom count')
    for row, atom in zip(rows, m['atoms']):
        pair = row['pair']
        identity = {k: row['early'][k] for k in ('model_identity', 'fit_result_sha256', 'model_id', 'source_sites')}
        need(atom['pair'] == pair and atom['donor'] == pair['site'], 'Atom identity')
        need(identity == atom['model_identity'] == m['identities'][pair['site']], 'Matched base identity')
        for period in ('early', 'late'):
            pred = row[period]
            need({k: pred[k] for k in identity} == identity, 'Paired model identity')
            need(pred['source_sites'] == [s for s in p['sites'] if s not in (m['outer_site'], pair['site'])], 'Nested source exclusion')
            need(pred['cycle'] == pair[period] and pred['doses'] == p['doses']
                 and pred['site'] == pair['site'] and pred['year'] == pair['year'], 'Source period and action axis')
        early = array(row['early_storage_truth'], (8, 7)) - array(row['early']['storage_mm'], (8, 7))
        late = array(row['late_U_truth'], (8, 7)) - array(row['late']['U_mm'], (8, 7))
        close(atom['early_residual'], early, 'Source early residual')
        close(atom['late_residual'], late, 'Source late residual')
        errors.append(early)
    close(m['feedback_sigma_mm'], np.maximum(np.std(errors, axis=0), .05), 'Action/day feedback scale')
    return dict(passed=True, source_pairs=8, source_statistics_and_residuals_replayed=True)


def audit_prediction(m, issued, versions, p):
    public, pair = issued['public_snapshot'], issued['pair']
    need(public['pair'] == pair and pair['site'] == m['outer_site'], 'Public snapshot binding')
    need(set(public['branches']) == set(p['sites']) - {pair['site']}, 'Four public branches')
    close(issued['context'], public['context'], 'Unchanged public context')
    need([v['behavior'] for v in versions] == p['doses'], 'All behavior versions')
    ctx = array(issued['context'], (11,))
    source = array(m['context']['source_contexts'], (8, 11))
    active = np.asarray(m['context']['active'], dtype=bool)
    scale = array(m['context']['scale'], (11,))
    distance = [float(np.mean(((r[active] - source[0, active]) / scale[active]) *
        ((r[active] + source[0, active] - 2 * ctx[active]) / scale[active]))) if active.any() else 0. for r in source]
    logprior = normalized_log([-d / (2 * m['context']['bandwidth'] ** 2) for d in distance])
    prior = np.exp(logprior)
    close(issued['prior_log_weights'], logprior, 'Prior log weights')
    close(issued['prior_weights'], prior, 'Prior weights')
    close(issued['prior_effective_atoms'], 1 / sum(prior ** 2), 'Prior effective atoms')
    need(len(issued['atoms']) == 8 and not issued['feedback_used'], 'No feedback before issuance')
    corrected, raw = [], {}
    for index, (source_atom, target) in enumerate(zip(m['atoms'], issued['atoms'])):
        need(target['source_pair'] == source_atom['pair']['id'] and target['donor'] == source_atom['donor']
             and target['model_identity'] == source_atom['model_identity'], 'Atom/model alignment')
        close(target['early_residual'], source_atom['early_residual'], 'Unchanged early atom')
        close(target['late_residual'], source_atom['late_residual'], 'Unchanged late atom')
        base = array(target['target_late_base'], (8, 7))
        h = target['donor']
        close(target['prior'], prior[index], 'Atom prior')
        for period in ('early', 'late'):
            pred = public['branches'][h][period]
            need({k: pred[k] for k in source_atom['model_identity']} == source_atom['model_identity'], 'Public model identity')
            need(pred['cycle'] == pair[period] and pred['site'] == pair['site'] and pred['year'] == pair['year']
                 and pred['doses'] == p['doses'], 'Public forecast axes')
        close(base, public['branches'][h]['late']['U_mm'], 'Unchanged late base')
        close(target['target_early_storage'], public['branches'][h]['early']['storage_mm'], 'Unchanged early base')
        close(issued['branches'][h]['raw'], base, 'Raw donor branch')
        if h in raw:
            close(base, raw[h], 'One target branch per donor')
        raw[h] = base
        corrected.append(base + array(target['late_residual'], (8, 7)))
    corrected = np.asarray(corrected)
    close(issued['raw'], np.mean(list(raw.values()), axis=0), 'Equal-weight raw baseline')
    close(issued['off'], np.einsum('i,ijk->jk', prior, corrected), 'Context-only off reconstruction')

    def check_branches(output, logs, weights, mode, weight_key):
        need(set(output['branches']) == set(raw), 'All donor branches')
        for h in raw:
            idx = [i for i, a in enumerate(issued['atoms']) if a['donor'] == h]
            w = np.exp(normalized_log(logs[idx]))
            close(output['branches'][h][mode], np.einsum('i,ijk->jk', w, corrected[idx]), 'Conditional donor ' + mode)
            close(output[weight_key][h], sum(weights[idx]), 'Donor mass')
    check_branches(issued, logprior, prior, 'off', 'prior_donor_weights')
    for v in versions:
        pair, b, f = issued['pair'], v['behavior'], v['feedback']
        need(v['pair'] == pair and v['feedback_used'], 'Adapted pair binding')
        need(set(f) == {'pair', 'cycle', 'dose_mm', 'release_at', 'storage_mm'}, 'Only legal storage feedback')
        release = (date.fromisoformat(pair['early_date']) + timedelta(days=7)).isoformat() + 'T06:00:00'
        need(f['pair'] == pair['id'] and f['cycle'] == pair['early'] and f['dose_mm'] == b, 'Executed early action')
        need(f['release_at'] == v['release_at'] == release and release <= pair['late_date'] + 'T06:00:00', 'Mature feedback')
        j = p['doses'].index(b)
        obs = array(f['storage_mm'], (7,))
        sigma = array(m['feedback_sigma_mm'], (8, 7))[j]
        likelihood = []
        for atom in issued['atoms']:
            target = obs - array(atom['target_early_storage'], (8, 7))[j]
            mismatch = (target - array(atom['early_residual'], (8, 7))[j]) / sigma
            likelihood.append(-sum(float(x) ** 2 for x in mismatch) / 14)
        logs = normalized_log(logprior + likelihood)
        weights = np.exp(logs)
        close(v['log_likelihood'], likelihood, 'Action-specific likelihood replay')
        close(v['posterior_log_weights'], logs, 'Posterior log weights')
        close(v['posterior_weights'], weights, 'Posterior weights')
        close(v['effective_atoms'], 1 / sum(weights ** 2), 'Posterior effective atoms')
        close(v['on'], np.einsum('i,ijk->jk', weights, corrected), 'Posterior readout replay')
        check_branches(v, logs, weights, 'on', 'donor_weights')


def mask(doses, b):
    need(doses == [0, 10, 15, 20, 25, 30, 40, 60] and b in doses, 'Exact action axis')
    keep = [i for i, dose in enumerate(doses) if dose != b]
    pairs = [(j, i) for i, j in combinations(keep, 2)]
    need(len(pairs) == 21 and all(doses[i] != b and doses[j] != b for i, j in pairs), 'Both endpoints unobserved')
    return keep, pairs


def measure(pred, true, keep, pairs):
    pred, true = array(pred, (8, 7)), array(true, (8, 7))
    error = np.array([(pred[a] - pred[c]) - (true[a] - true[c]) for a, c in pairs])
    return dict(mae=float(abs(error.sum(-1)).mean()), daily_delta=float(abs(error).mean()),
                daily_u=float(abs(pred[keep] - true[keep]).mean()))


def signs(true, off, on):
    def sg(x):
        return 1 if x > .01 else -1 if x < -.01 else 0
    t, o, n = map(sg, (true, off, on))
    return dict(repaired=int(t != 0 and o == -t and n == t), new_wrong=int(t != 0 and o != -t and n == -t),
                entered_zero=int(o != 0 and n == 0), exited_zero=int(o == 0 and n != 0))


def avg(rows):
    need(bool(rows), 'Nonempty scoring group')
    return {b: {k: float(np.mean([r[b][k] for r in rows])) for k in METRICS} for b in BRANCHES}


def balanced(rows, sites):
    return avg([avg([r for r in rows if r['site'] == s]) for s in sites])


def evaluate(p, models, issues, histories, truths):
    need(set(models) == set(p['sites']), 'All source models')
    need(set(issues) == set(truths) == {v['id'] for v in p['pairs']}, 'Complete windows')
    need(set(histories) == {v['id'] + '_b' + str(b) for v in p['pairs'] for b in p['doses']}, 'Complete feedback grid')
    certificates = {s: audit_model(models[s], p) for s in p['sites']}
    windows, details, endpoints = [], [], []
    minimum = dict(off=float('inf'), on=float('inf'))
    total_signs = dict(repaired=0, new_wrong=0, entered_zero=0, exited_zero=0)
    for pair in p['pairs']:
        pid = pair['id']
        issued = issues[pid]
        need(issued['pair'] == truths[pid]['pair'] == pair, 'Scoring identity')
        need(truths[pid]['definition'] == 'native crop_tra_mm daily', 'Native uptake target')
        true = array(truths[pid]['U_mm'], (8, 7))
        off = array(issued['off'], (8, 7))
        versions = [histories[pid + '_b' + str(b)] for b in p['doses']]
        audit_prediction(models[pair['site']], issued, versions, p)
        rows = []
        for b, v in zip(p['doses'], versions):
            keep, pairs = mask(p['doses'], b)
            on = array(v['on'], (8, 7))
            row = dict(pair=pid, site=pair['site'], year=pair['year'], behavior=b,
                       unobserved_doses=[p['doses'][j] for j in keep],
                       action_pairs=[[p['doses'][a], p['doses'][c]] for a, c in pairs],
                       raw=measure(issued['raw'], true, keep, pairs), off=measure(off, true, keep, pairs),
                       on=measure(on, true, keep, pairs), true_U=true.tolist(), off_U=off.tolist(), on_U=on.tolist(),
                       prior_effective_atoms=issued['prior_effective_atoms'], posterior_effective_atoms=v['effective_atoms'])
            for a, c in pairs:
                dt, do, dn = true[a] - true[c], off[a] - off[c], on[a] - on[c]
                ts, os, ns = float(dt.sum()), float(do.sum()), float(dn.sum())
                count = signs(ts, os, ns)
                for k, value in count.items():
                    total_signs[k] += value
                endpoints.append(dict(pair=pid, behavior=b, a=p['doses'][a], c=p['doses'][c],
                    true_daily=dt.tolist(), off_daily=do.tolist(), on_daily=dn.tolist(),
                    true_7d=ts, off_7d=os, on_7d=ns, signs=count))
            for h in v['branches']:
                minimum['off'] = min(minimum['off'], float(array(issued['branches'][h]['off'], (8, 7))[keep].min()))
                minimum['on'] = min(minimum['on'], float(array(v['branches'][h]['on'], (8, 7))[keep].min()))
            rows.append(row)
            details.append(row)
        grouped = avg(rows)
        windows.append(dict(id=pid, site=pair['site'], year=pair['year'], **grouped,
                            improvement=grouped['off']['mae'] - grouped['on']['mae']))
    macro = balanced(windows, p['sites'])
    by_site = {s: avg([w for w in windows if w['site'] == s]) for s in p['sites']}
    by_year = {str(y): balanced([w for w in windows if w['year'] == y], p['sites']) for y in p['years']}
    largest = max(windows, key=lambda w: (w['improvement'], w['id']))
    drop = balanced([w for w in windows if w['id'] != largest['id']], p['sites'])
    eps = p['improvement_epsilon_mm']
    def improved(x):
        return x['off']['mae'] - x['on']['mae'] > eps
    gates = dict(macro=improved(macro), three_sites=sum(improved(x) for x in by_site.values()) >= 3,
        both_years=all(improved(x) for x in by_year.values()), drop_largest=improved(drop),
        daily_u=macro['on']['daily_u'] <= macro['off']['daily_u'] + eps,
        daily_delta=macro['on']['daily_delta'] <= macro['off']['daily_delta'] + eps,
        nonnegative_unobserved_branch_u=all(v >= -.01 for v in minimum.values()),
        signs=total_signs['new_wrong'] <= total_signs['repaired'])
    need(len(windows) == 10 and len(details) == 80 and len(endpoints) == 1680, 'Complete masked score')
    return dict(engineering_complete=True, method_passed=all(gates.values()), gates=gates, macro=macro,
        by_site=by_site, by_year=by_year, drop_largest=dict(removed=largest['id'], **drop),
        minimum_unobserved_branch_daily_U=minimum, signs=total_signs, windows=windows, details=details,
        endpoints=endpoints, independent_replay=certificates, physical_windows=10, feedback_versions=80,
        pairs_per_behavior=21, total_pairs=1680, economic_scoring=False, automatic_promotion=False,
        scope='Fixed 2014-2015 empirical screening. Only ten physical late windows; no economic or physical validation.')
