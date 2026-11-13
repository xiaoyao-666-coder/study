"""Context-weighted paired response atoms with a one-action likelihood update."""
from datetime import date, timedelta
import numpy as np
from runtime import need

FLOOR_MM = 0.05


def array(value, shape):
    out = np.asarray(value, dtype=float)
    need(out.shape == shape and np.isfinite(out).all(), 'Finite shape ' + str(shape))
    return out


def identity(prediction):
    return {key: prediction[key] for key in ('model_identity', 'fit_result_sha256', 'model_id', 'source_sites')}


def softmax(values):
    x = np.asarray(values, dtype=float)
    need(x.ndim == 1 and np.isfinite(x).all(), 'Finite log weights')
    x = np.exp(x - np.max(x))
    return x / x.sum()


def log_softmax(values):
    x = np.asarray(values, dtype=float)
    need(x.ndim == 1 and np.isfinite(x).all(), 'Finite log weights')
    shifted = x - np.max(x)
    return shifted - np.log(np.exp(shifted).sum())


def context_scale(contexts):
    x = array(contexts, (8, 11))
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    active = scale > 1e-12
    scale[~active] = 1.0
    z = (x - center) / scale
    distances = (np.sqrt(((z[:, None, active] - z[None, :, active]) ** 2).mean(axis=2))
                 if np.any(active) else np.zeros((8, 8)))
    positive = distances[distances > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return dict(center=center.tolist(), scale=scale.tolist(), active=active.tolist(),
                bandwidth=bandwidth, source_contexts=x.tolist())


def context_weights(model, context):
    ctx = array(context, (11,))
    center = array(model['context']['center'], (11,))
    scale = array(model['context']['scale'], (11,))
    active = np.asarray(model['context']['active'], dtype=bool)
    need(active.shape == (11,) and np.all(scale > 0), 'Valid context features')
    src = array(model['context']['source_contexts'], (8, 11))
    # Subtract the first atom's squared distance algebraically before softmax.
    # The target-only squared term cancels, avoiding loss of precision far away.
    relative_distance = (np.mean(((src[:, active] - src[0, active]) / scale[active]) *
        ((src[:, active] + src[0, active] - 2 * ctx[active]) / scale[active]), axis=1)
        if np.any(active) else np.zeros(8))
    return log_softmax(-relative_distance / (2 * float(model['context']['bandwidth']) ** 2))


def fit(source, protocol):
    outer = source['outer_site']
    expected = [pair for pair in protocol['pairs'] if pair['site'] != outer]
    need(outer in protocol['sites'] and [row['pair'] for row in source['pairs']] == expected,
         'Eight source station-years excluding target')
    atoms = []
    contexts = []
    identities = {}
    for row in source['pairs']:
        pair = row['pair']
        donor = pair['site']
        early, late = row['early'], row['late']
        for period, pred in (('early', early), ('late', late)):
            need(pred['cycle'] == pair[period] and pred['site'] == donor and pred['year'] == pair['year'],
                 'Source period identity')
            need(pred['doses'] == protocol['doses'] and pred['source_sites'] == [s for s in protocol['sites'] if s not in (outer, donor)],
                 'Source action axis and exclusion')
        need(identity(early) == identity(late), 'Same source model across periods')
        if donor in identities:
            need(identities[donor] == identity(early), 'Same donor model across years')
        identities[donor] = identity(early)
        contexts.append(array(row['context'], (11,)))
        early_error = array(row['early_storage_truth'], (8, 7)) - array(early['storage_mm'], (8, 7))
        late_error = array(row['late_U_truth'], (8, 7)) - array(late['U_mm'], (8, 7))
        atoms.append(dict(pair=pair, donor=donor, model_identity=identity(early),
                          early_residual=early_error.tolist(), late_residual=late_error.tolist()))
    context = context_scale(contexts)
    early = np.asarray([a['early_residual'] for a in atoms], dtype=float)
    sigma = np.maximum(early.std(axis=0), FLOOR_MM)
    return dict(outer_site=outer, atoms=atoms, identities=identities, context=context, source_snapshot=source,
                feedback_sigma_mm=sigma.tolist(), source_pair_ids=[p['id'] for p in expected],
                physical_source_pairs=8, atom_count=8, bandwidth_rule='median positive source standardized RMS distance',
                likelihood_rule='exp(-0.5 * mean over seven days of squared standardized residual mismatch)')


def issue(model, public, protocol):
    pair = public['pair']
    outer = pair['site']
    donors = {site for site in protocol['sites'] if site != outer}
    need(model['outer_site'] == outer and set(public['branches']) == donors, 'Target/outer identity')
    logprior = context_weights(model, public['context'])
    prior = np.exp(logprior)
    raw_branches = {}
    atom_targets = []
    for atom, weight in zip(model['atoms'], prior):
        donor = atom['donor']
        branch = public['branches'][donor]
        for period in ('early', 'late'):
            pred = branch[period]
            need(pred['cycle'] == pair[period] and pred['site'] == outer and pred['year'] == pair['year'],
                 'Target forecast identity')
            need(identity(pred) == atom['model_identity'] == model['identities'][donor],
                 'Each residual atom stays with its base model')
            need(pred['doses'] == protocol['doses'], 'Target action axis')
        raw = array(branch['late']['U_mm'], (8, 7))
        raw_branches[donor] = raw
        atom_targets.append(dict(donor=donor, source_pair=atom['pair']['id'], prior=float(weight),
            model_identity=atom['model_identity'], target_late_base=raw.tolist(),
            target_early_storage=branch['early']['storage_mm'], late_residual=atom['late_residual'],
            early_residual=atom['early_residual']))
    raw = np.mean(list(raw_branches.values()), axis=0)
    off = np.sum(np.asarray([v['target_late_base'] for v in atom_targets]) * prior[:, None, None], axis=0)
    off += np.sum(np.asarray([v['late_residual'] for v in atom_targets]) * prior[:, None, None], axis=0)
    ess = float(1.0 / np.sum(prior ** 2))
    by_donor = {h: float(sum(w for a, w in zip(model['atoms'], prior) if a['donor'] == h)) for h in sorted(donors)}
    corrected = np.asarray([a['target_late_base'] for a in atom_targets]) + np.asarray([a['late_residual'] for a in atom_targets])
    branches = {}
    for h in sorted(donors):
        indices = [i for i, a in enumerate(atom_targets) if a['donor'] == h]
        conditional = softmax(logprior[indices])
        branches[h] = dict(raw=raw_branches[h].tolist(), off=np.sum(corrected[indices] * conditional[:, None, None], axis=0).tolist())
    return dict(pair=pair, raw=raw.tolist(), off=off.tolist(), atoms=atom_targets,
                prior_log_weights=logprior.tolist(),
                prior_weights=prior.tolist(), prior_effective_atoms=ess, prior_donor_weights=by_donor,
                context=public['context'], public_snapshot=public, feedback_used=False, branches=branches)


def validate_feedback(feedback, pair, behavior, protocol):
    need(set(feedback) == {'pair', 'cycle', 'dose_mm', 'release_at', 'storage_mm'}, 'Exact feedback schema')
    need(behavior in protocol['doses'] and feedback['pair'] == pair['id'] and
         feedback['cycle'] == pair['early'] and feedback['dose_mm'] == behavior, 'Executed action identity')
    release = (date.fromisoformat(pair['early_date']) + timedelta(days=7)).isoformat() + 'T06:00:00'
    need(feedback['release_at'] == release and release <= pair['late_date'] + 'T06:00:00', 'Mature seven-day feedback')
    return array(feedback['storage_mm'], (7,)), release


def adapt(model, issued, feedback, behavior, protocol):
    pair = issued['pair']
    need(model['outer_site'] == pair['site'], 'Adaptation outer identity')
    obs, release = validate_feedback(feedback, pair, behavior, protocol)
    action = protocol['doses'].index(behavior)
    sigma = array(model['feedback_sigma_mm'], (8, 7))[action]
    loglike = []
    for atom in issued['atoms']:
        target_residual = obs - array(atom['target_early_storage'], (8, 7))[action]
        source_residual = array(atom['early_residual'], (8, 7))[action]
        mismatch = (source_residual - target_residual) / sigma
        loglike.append(-0.5 * float(np.mean(mismatch ** 2)))
    logprior = array(issued['prior_log_weights'], (8,))
    logposterior = log_softmax(logprior + np.asarray(loglike))
    posterior = np.exp(logposterior)
    targets = np.asarray([a['target_late_base'] for a in issued['atoms']], dtype=float)
    residuals = np.asarray([a['late_residual'] for a in issued['atoms']], dtype=float)
    on = np.sum(targets * posterior[:, None, None], axis=0) + np.sum(residuals * posterior[:, None, None], axis=0)
    by_donor = {h: float(sum(w for a, w in zip(issued['atoms'], posterior) if a['donor'] == h))
                for h in sorted({a['donor'] for a in issued['atoms']})}
    corrected = targets + residuals
    branches = {}
    for h in sorted(by_donor):
        indices = [i for i, a in enumerate(issued['atoms']) if a['donor'] == h]
        conditional = softmax(logposterior[indices])
        branches[h] = dict(on=np.sum(corrected[indices] * conditional[:, None, None], axis=0).tolist())
    return dict(pair=pair, behavior=behavior, on=on.tolist(), posterior_weights=posterior.tolist(), posterior_log_weights=logposterior.tolist(),
                log_likelihood=loglike, effective_atoms=float(1.0 / np.sum(posterior ** 2)),
                donor_weights=by_donor, feedback_used=True, feedback=feedback, release_at=release,
                branches=branches)
