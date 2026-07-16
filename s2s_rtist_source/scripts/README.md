# Scripts

## Purpose
Categorized research and formal runners for the S2S RTIST project. The only root-level Python entry point is `project_cli.py`.

## Categories
| Category | Path | Count |
|---|---|---|
| Data preparation | `scripts/data_preparation/` | 28 |
| Simulation | `scripts/simulation/` | 12 |
| Diagnostics | `scripts/diagnostics/` | 26 |
| Training | `scripts/training/` | 29 |
| Evaluation | `scripts/evaluation/` | 31 |
| Visualization | `scripts/visualization/` | 5 |
| Archive | `scripts/archive/` | 4 |

Reusable formal libraries live under `src/s2s_rtist/` and are also listed in `scripts/script_catalog.csv`.

## Formal Runnable IDs
| ID | Status | Script | Purpose |
|---|---|---|---|
| `confirmed-5site-smoke` | formal | `run_confirmed_5site_restart_generation_smoke_v1.py` | run confirmed 5site restart generation smoke v1 |
| `continuous-12site-generation` | formal | `run_continuous_ir_12site_restart_generation_v1.py` | run continuous ir 12site restart generation v1 |
| `gefs-gridmet-bias` | formal | `run_gefs_gridmet_bias_validation_v1.py` | run gefs gridmet bias validation v1 |
| `restart-raw-audit` | formal | `restart_raw_audit_v1.py` | restart raw audit v1 |
| `rootzone-frequency` | formal | `run_rootzone_flux_frequency_validation_v1.py` | run rootzone flux frequency validation v1 |

## Usage
`python project_cli.py list`
`python project_cli.py find <query>`
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Catalog
All 140 original root scripts are tracked in `scripts/script_catalog.csv` with original path, current path, status, purpose, and SHA256 hashes.
