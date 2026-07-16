# Simulation

## Purpose
Scripts that launch SWAP restart generation, decision smokes, and related candidate simulation workflows.

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
| `confirmed-5site-smoke` | formal | `run_confirmed_5site_restart_generation_smoke_v1.py` | run confirmed 5site restart generation smoke v1 |
| `continuous-12site-generation` | formal | `run_continuous_ir_12site_restart_generation_v1.py` | run continuous ir 12site restart generation v1 |
| `decision-candidate-worker` | active | `decision_candidate_worker.py` | decision candidate worker |
| `decision-restart-8ir-compare` | active | `decision_restart_8ir_compare.py` | decision restart 8ir compare |
| `decision-restart-multiday-compare` | active | `decision_restart_multiday_compare.py` | decision restart multiday compare |
| `decision-restart-probe` | active | `decision_restart_probe.py` | decision restart probe |
| `decision-smoke-8ir` | active | `decision_smoke_8ir.py` | decision smoke 8ir |
| `decision-smoke-8ir-parallel` | active | `decision_smoke_8ir_parallel.py` | decision smoke 8ir parallel |
| `generate-restart-decision-dataset-dense-proxy` | active | `generate_restart_decision_dataset_dense_proxy.py` | generate restart decision dataset dense proxy |
| `generate-restart-decision-dataset-expanded` | active | `generate_restart_decision_dataset_expanded.py` | generate restart decision dataset expanded |
| `run-confirmed-5site-true-input-multidate-smoke-v1` | active | `run_confirmed_5site_true_input_multidate_smoke_v1.py` | run confirmed 5site true input multidate smoke v1 |
| `run-controlled-swap-completion-check-v1` | active | `run_controlled_swap_completion_check_v1.py` | run controlled swap completion check v1 |

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one.
