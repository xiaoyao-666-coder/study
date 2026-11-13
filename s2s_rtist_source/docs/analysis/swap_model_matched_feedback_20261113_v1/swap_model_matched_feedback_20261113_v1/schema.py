"""Frozen experimental constants and deterministic task graph; no data access."""
from itertools import combinations

NAME = 'swap_auxiliary_pretraining_20261103_v3'
SITES = ['P1', 'P15', 'P2', 'P3', 'P4']
ARMS = ['G', 'GV', 'GAV', 'GAVW-old']
SEEDS = [1729, 1730, 1731]
GRID = [0, 10, 15, 20, 25, 30, 40, 60]
KEYS = ['sample_id', 'target_year', 'site_id', 'decision_date', 'irrigation_mm']
CYCLE = KEYS[1:4]
WEATHER = ['precipitation_mm', 'temperature_min_c', 'temperature_max_c',
           'actual_vapor_pressure_kpa', 'wind_speed_m_s', 'solar_kj_m2_day']
STATES = ['decision_doy_sin', 'decision_doy_cos', 'predecision_dvs', 'predecision_lai',
          'predecision_crop_root_depth_cm', 'predecision_cwdm_kg_ha', 'predecision_cwso_kg_ha',
          'predecision_soil_vwc_0_100cm'] + [f'weather_{w}_day{d:02d}' for w in WEATHER for d in range(1, 8)] + ['soil_relative_saturation_0_100', 'soil_log10_ksat_cm_day_0_100']
INPUTS = KEYS[:4] + STATES[:2] + ['irrigation_mm'] + STATES[2:50] + ['physics_initial_soil_storage_0_100cm_mm']
PHYSICAL = ['target_aet_7d_mm'] + [f'target_soil_vwc_0_100cm_day{d:02d}' for d in range(1, 8)]
LABELS = KEYS + ['target_net_gain_7d'] + PHYSICAL
INITIAL = 'physics_initial_soil_storage_0_100cm_mm'
RAIN = [f'weather_precipitation_mm_day{d:02d}' for d in range(1, 8)]
OPTIMIZER = dict(maxiter=1000000, maxfun=1500000, maxls=50, maxcor=20, gtol=1e-6, ftol=0.)
COUNTS = dict(P1=[13,14,12,13,13], P2=[14,14,13,13,12], P3=[16,15,13,15,13], P4=[14,15,13,13,13], P15=[14,15,13,13,14])

def config_id(sites, cutoff):
    return '_'.join(sorted(sites)) + '_through_' + str(cutoff)

def graph():
    configs = {}
    def add(sites, cutoff):
        sites = sorted(sites); key = config_id(sites, cutoff)
        configs.setdefault(key, dict(config_id=key, source_sites=sites, cutoff=cutoff,
                                     train_years=list(range(2014, cutoff+1)), outer=False, probes=[]))
        return configs[key]
    outer = []
    for year in (2017, 2018):
        for heldout in [None] + SITES:
            sites = SITES if heldout is None else [s for s in SITES if s != heldout]
            c = add(sites, year-1); c['outer'] = True
            folds = []
            for v in range(2015, year):
                for probe in sites:
                    pool = sites if heldout is None else [s for s in sites if s != probe]
                    inner = add(pool, v-1)
                    if probe not in inner['probes']: inner['probes'].append(probe)
                    folds.append(dict(config_id=inner['config_id'], probe=probe, year=v))
            outer.append(dict(config_id=c['config_id'], scenario='seen' if heldout is None else 'unseen',
                              heldout=heldout, evaluation_year=year,
                              evaluation_sites=SITES if heldout is None else [heldout], oof=folds))
    bases=[]; heads=[]; evaluations=[]
    for c in sorted(configs.values(), key=lambda c:(c['cutoff'], c['config_id'])):
        c['probes'].sort()
        for arm in ARMS:
            for seed in SEEDS:
                tid=c['config_id']+'__'+arm+'__s'+str(seed)
                bases.append(dict(c, id=tid, arm=arm, seed=seed))
                for head in ['Unified']+c['source_sites']:
                    heads.append(dict(id=tid+'__head_'+head, base_id=tid, head=head,
                                      source_sites=c['source_sites'] if head=='Unified' else [head],
                                      train_years=c['train_years'], arm=arm, seed=seed))
                for e in outer:
                    if e['config_id']==c['config_id']: evaluations.append(dict(e, id=tid, base_id=tid, arm=arm, seed=seed, source_sites=c['source_sites']))
    assert len(configs)==44 and len(bases)==528 and len(heads)==2328 and len(evaluations)==144
    return dict(bases=bases, heads=heads, evaluations=evaluations,
                counts=dict(configurations=44, pretrain=528, unified=528, private=1800, probe=144, gate=144, nonlinear_fits=3000))

def protocol():
    return dict(name=NAME, schema='auxiliary_pretraining_v1', sites=SITES, arms=ARMS, seeds=SEEDS,
                years=list(range(2014,2019)), excluded_years=[2019,2021,2024], grid=GRID,
                input_columns=INPUTS, label_columns=LABELS, cycle_counts=COUNTS,
                optimizer=OPTIMIZER, acceptance_gradient_inf=1e-5,
                recovery=dict(enabled=True,plan='recovery_plan.json',accepted_pretrains=527,
                              rerun_pretrains=1,failed_restart='original_seed',
                              parent_manifest_sha256='f2d00125b270812baa68922e64b7ab46349152e505e98134ba132b6c74091124'),
                pretrain=dict(state_ridge=.003,ridge=.0003,curvature=.0003,temperature=.25),
                readout=dict(state_ridge=.001,ridge=.0001,curvature=.0003,temperature=.25),
                physical=dict(ridge=1e-5,smooth=1e-3,weight=1.),
                gate=dict(ridge=.1,temperature=.25,sensitivity=[.1,.5,1.]),
                execution=dict(parallel_workers=4,per_fit_timeout_seconds=3600,gate_timeout_seconds=300,
                               other_stage_timeout_seconds=600,aggregate_timeout_seconds=1800,total_seconds=172800),
                required_versions=dict(numpy='2.2.6',pandas='2.3.2',scipy='1.15.3'),
                local_real_compute_allowed=False,automatic_promotion=False,tta=False,swap_runs=0,**graph())
