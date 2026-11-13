"""Label-free soil descriptors from one receiving site's frozen profile."""
import numpy as np
import pandas as pd

SOIL_FEATURES = ('soil_relative_saturation_0_100', 'soil_log10_ksat_cm_day_0_100')
SITE_KEYS = {'P1':'code_N1', 'P2':'code_N2', 'P3':'code_N3', 'P4':'code_N4', 'P15':'code_active'}
DEPTHS = [0., 5., 15., 30., 60., 100.]
WEIGHTS = np.array([5., 10., 15., 30., 40.]) / 100
COLUMNS = ['paper_site_id','feature_site_key','profile_source','depth_start_cm','theta_r_','theta_s_','alpha_','n_','ksat_','alphaw_']

def descriptors(profile, site):
    if site not in SITE_KEYS or profile.columns.tolist() != COLUMNS:
        raise ValueError('Unknown site or soil profile columns')
    if len(profile) != 6 or set(profile.paper_site_id) != {site} or set(profile.feature_site_key) != {SITE_KEYS[site]}:
        raise ValueError('Soil profile site binding differs')
    if set(profile.profile_source) != {'embedded_confirmed_profiles'}:
        raise ValueError('Unknown soil provenance')
    a = profile[COLUMNS[3:]].to_numpy(float)
    if not np.isfinite(a).all() or a[:, 0].tolist() != DEPTHS:
        raise ValueError('Soil layer coverage differs')
    tr, ts, alpha, n, ks, aw = a[:, 1:].T
    if not ((tr > 0).all() and (tr < ts).all() and (ts < 1).all() and (alpha > 0).all() and (n > 1).all() and (ks > 0).all() and (aw > 0).all()):
        raise ValueError('Invalid hydraulic domain')
    return {'theta_r_0_100':float(WEIGHTS @ tr[:5]),
            'theta_s_0_100':float(WEIGHTS @ ts[:5]),
            'log10_ksat_cm_day_0_100':float(WEIGHTS @ np.log10(ks[:5]))}

def augment(frame, profile):
    sites = frame.site_id.astype(str).unique()
    if len(sites) != 1 or set(SOIL_FEATURES) & set(frame.columns):
        raise ValueError('Expect one receiving site and unaugmented input')
    info = descriptors(profile, sites[0])
    vwc = pd.to_numeric(frame.predecision_soil_vwc_0_100cm, errors='raise').to_numpy(float)
    if not np.isfinite(vwc).all() or ((vwc < 0) | (vwc > 1)).any():
        raise ValueError('Invalid predecision VWC')
    result = frame.copy()
    result[SOIL_FEATURES[0]] = (vwc-info['theta_r_0_100'])/(info['theta_s_0_100']-info['theta_r_0_100'])
    result[SOIL_FEATURES[1]] = info['log10_ksat_cm_day_0_100']
    return result, info

