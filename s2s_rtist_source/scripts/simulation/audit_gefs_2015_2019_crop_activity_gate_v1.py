#!/usr/bin/env python3
"""Screen historical pilot dates for five-site seven-day crop activity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.simulation import run_confirmed_5site_restart_generation_smoke_v1 as base
from scripts.simulation.run_gefs_2015_2019_scenario_consistent_swap_pilot_v1 import (
    DEFAULT_CONTRACT,
    DEFAULT_HYBRID_WEATHER,
    load_contract,
    load_hybrid_weather,
    prepare_site_workspace,
    selected_cycles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_crop_activity_gate_v1"
)


def candidate_cycles(contract: dict[str, Any]) -> pd.DataFrame:
    selected = selected_cycles(contract).set_index("target_year")
    rows = []
    for year in sorted(selected.index.astype(int)):
        metadata = selected.loc[year]
        for month_day in contract["cycle_selection"]["candidate_month_days"]:
            decision = pd.Timestamp(f"{year}-{month_day}")
            rows.append(
                {
                    "target_year": year,
                    "decision_date": decision.strftime("%Y-%m-%d"),
                    "date_t": decision.strftime("%d-%b-%Y"),
                    "decision_doy": int(decision.dayofyear),
                    "split": str(metadata["split"]),
                    "fit_first_year": int(metadata["fit_first_year"]),
                    "fit_last_year": int(metadata["fit_last_year"]),
                    "previously_selected": bool(
                        decision.strftime("%Y-%m-%d")
                        == str(metadata["decision_date"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_last_crop_row(path: Path) -> dict[str, Any]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not re.match(r"^\s*\d{4}-\d{2}-\d{2}", line):
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) >= 4:
                rows.append(values)
    if not rows:
        return {"last_crop_date": "", "last_crop_dvs": None}
    last = rows[-1]
    return {
        "last_crop_date": pd.Timestamp(last[0]).strftime("%Y-%m-%d"),
        "last_crop_dvs": float(last[3]),
    }


def parse_restart_crop_flags(path: Path) -> dict[str, int | None]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""

    def value(name: str) -> int | None:
        match = re.search(rf"^\s*{name}\s*=\s*(-?\d+)", text, flags=re.MULTILINE)
        return int(match.group(1)) if match else None

    return {
        "sw_crop_emergence": value("swCropEmergence"),
        "sw_crop_harvest": value("swcropharvest"),
    }


def evaluate_site_gate(
    *,
    cycle: pd.Series,
    site: str,
    run_result: dict[str, Any],
    workspace: Path,
    hybrid: pd.DataFrame,
    maximum_dvs: float,
    required_crop_rows: int,
) -> dict[str, Any]:
    crop = parse_last_crop_row(workspace / "result_forec.crp")
    flags = parse_restart_crop_flags(workspace / "restart_initial.end")
    expected_predecision = (
        pd.Timestamp(cycle.decision_date) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    dataset_path = Path(str(run_result.get("dataset_csv", "")))
    dataset = pd.read_csv(dataset_path) if dataset_path.is_file() else pd.DataFrame()
    crop_record_pass = crop["last_crop_date"] == expected_predecision
    dvs_pass = crop["last_crop_dvs"] is not None and float(
        crop["last_crop_dvs"]
    ) < maximum_dvs
    harvest_pass = flags["sw_crop_harvest"] == 0
    horizon_rows = (
        int(pd.to_numeric(dataset["horizon_days_actual"], errors="coerce").min())
        if not dataset.empty and "horizon_days_actual" in dataset
        else 0
    )
    horizon_pass = (
        run_result["status"] == "completed"
        and len(dataset) == 1
        and horizon_rows == required_crop_rows
    )
    future_dates = pd.date_range(cycle.decision_date, periods=7, freq="D").strftime(
        "%Y-%m-%d"
    )
    future = hybrid.loc[
        (hybrid["target_year"].astype(int) == int(cycle.target_year))
        & (hybrid["site_id"].astype(str) == str(site))
        & (hybrid["local_date"].isin(future_dates))
    ]
    future_role = (
        "corrected_gefs_existing"
        if len(future) == 7 and future["lead_day"].notna().all()
        else "era5_screening_only"
    )
    return {
        "target_year": int(cycle.target_year),
        "decision_date": str(cycle.decision_date),
        "site_id": site,
        "previously_selected": bool(cycle.previously_selected),
        "expected_predecision_date": expected_predecision,
        **crop,
        **flags,
        "predecision_crop_record_pass": bool(crop_record_pass),
        "dvs_below_2_pass": bool(dvs_pass),
        "not_harvested_pass": bool(harvest_pass),
        "zero_ir_restart_status": str(run_result["status"]),
        "zero_ir_candidate_rows": int(len(dataset)),
        "zero_ir_horizon_crop_rows": int(horizon_rows),
        "seven_day_crop_coverage_pass": bool(horizon_pass),
        "future_weather_role": future_role,
        "screening_gate_passed": bool(
            crop_record_pass and dvs_pass and harvest_pass and horizon_pass
        ),
        "stdout_tail": str(run_result.get("stdout_tail", "")),
        "stderr_tail": str(run_result.get("stderr_tail", "")),
    }


def summarize_cycle_gate(
    site_results: pd.DataFrame, required_site_count: int
) -> pd.DataFrame:
    rows = []
    for (year, decision), group in site_results.groupby(
        ["target_year", "decision_date"], sort=True
    ):
        passed = int(group["screening_gate_passed"].sum())
        rows.append(
            {
                "target_year": int(year),
                "decision_date": str(decision),
                "previously_selected": bool(group["previously_selected"].all()),
                "site_count": int(group["site_id"].nunique()),
                "eligible_site_count": passed,
                "failed_sites": ";".join(
                    sorted(group.loc[~group["screening_gate_passed"], "site_id"])
                ),
                "uses_provisional_era5_future": bool(
                    group["future_weather_role"].eq("era5_screening_only").any()
                ),
                "all_five_sites_screening_eligible": bool(
                    passed == required_site_count
                    and group["site_id"].nunique() == required_site_count
                ),
            }
        )
    return pd.DataFrame(rows)


def run_zero_ir_screen(
    *,
    site: str,
    workspace: Path,
    year: int,
    decision: str,
    date_t: str,
    decision_doy: int,
    timeout: int,
    python_exe: str,
) -> dict[str, Any]:
    for name in (
        "site_restart_generation_smoke.csv",
        "site_restart_generation_smoke_best_by_date.csv",
    ):
        path = workspace / name
        if path.exists():
            path.unlink()
    plan = workspace / base.SITE_PLAN_FILE
    pd.DataFrame(
        [
            {
                "site_id": site,
                "date_t": date_t,
                "decision_doy": int(decision_doy),
                "irrigation_mm": 0.0,
            }
        ]
    ).to_csv(plan, index=False)
    cmd = [
        python_exe,
        "run_restart_smoke_one_site.py",
        "--site",
        site,
        "--year",
        str(int(year)),
        "--sampling-plan",
        plan.name,
    ]
    env = base.prepend_server_runtime_library(dict(base.os.environ))
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        status = "completed" if result.returncode == 0 else "returned_nonzero"
        returncode: str | int = result.returncode
        stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = ""
        stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
    dataset = workspace / "site_restart_generation_smoke.csv"
    return {
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "decision_dates": decision,
        "dataset_csv": str(dataset) if dataset.exists() else "",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--hybrid-weather", type=Path, default=DEFAULT_HYBRID_WEATHER)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-per-site-date", type=int, default=1800)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Screen only the five cycles currently selected by the contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    hybrid = load_hybrid_weather(args.hybrid_weather, contract)
    gate = contract["crop_activity_gate"]
    cycles = candidate_cycles(contract)
    if args.selected_only:
        cycles = cycles.loc[cycles["previously_selected"]].reset_index(drop=True)
    run_dir = args.run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    result_dir = run_dir / "site_gate_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for site in contract["sites"]:
        workspace = run_dir / "workspaces" / site
        if not (args.resume and workspace.is_dir()):
            item = prepare_site_workspace(run_dir, site, hybrid, contract)
            prepared[site] = Path(item["run_workspace"])
        else:
            prepared[site] = workspace

    site_rows = []
    for cycle in cycles.itertuples(index=False):
        cycle_series = pd.Series(cycle._asdict())
        for site in contract["sites"]:
            result_path = result_dir / (
                f"{cycle.target_year}_{cycle.decision_date}_{site}_gate_v1.json"
            )
            if args.resume and result_path.is_file():
                site_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            print(
                f"[CROP-GATE] {cycle.decision_date}/{site}: prestate + 0 mm restart",
                flush=True,
            )
            result = run_zero_ir_screen(
                site=site,
                workspace=prepared[site],
                year=int(cycle.target_year),
                decision=str(cycle.decision_date),
                date_t=str(cycle.date_t),
                decision_doy=int(cycle.decision_doy),
                timeout=args.timeout_per_site_date,
                python_exe=args.python,
            )
            row = evaluate_site_gate(
                cycle=cycle_series,
                site=site,
                run_result=result,
                workspace=prepared[site],
                hybrid=hybrid,
                maximum_dvs=float(gate["maximum_dvs_exclusive"]),
                required_crop_rows=int(gate["required_restart_crop_rows"]),
            )
            result_path.write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            site_rows.append(row)

    site_results = pd.DataFrame(site_rows).sort_values(
        ["target_year", "decision_date", "site_id"]
    )
    cycle_results = summarize_cycle_gate(
        site_results, int(gate["site_count_required"])
    )
    selected = cycle_results.loc[cycle_results["previously_selected"]]
    selected_all_passed = bool(
        len(selected) == 5 and selected["all_five_sites_screening_eligible"].all()
    )
    selected_has_provisional_future = bool(
        selected["uses_provisional_era5_future"].any()
    )
    if selected_all_passed and not selected_has_provisional_future:
        status = "crop_activity_final_corrected_gefs_gate_passed"
    elif selected_all_passed:
        status = "crop_activity_screening_passed_final_gefs_recheck_required"
    else:
        status = "crop_activity_screening_completed_cycle_reselection_required"
    audit = {
        "status": status,
        "site_date_rows": int(len(site_results)),
        "candidate_cycle_rows": int(len(cycle_results)),
        "screening_eligible_cycle_count": int(
            cycle_results["all_five_sites_screening_eligible"].sum()
        ),
        "previously_selected_cycle_count": int(len(selected)),
        "previously_selected_cycle_failure_count": int(
            (~selected["all_five_sites_screening_eligible"]).sum()
        ),
        "selected_cycle_count_required": 5,
        "selected_cycles_use_provisional_future": selected_has_provisional_future,
        "screening_scope": "selected_cycles_only" if args.selected_only else "all_candidates",
        "irrigation_values_used": [0.0],
        "candidate_selection_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    site_results.to_csv(run_dir / "crop_activity_site_gate_v1.csv", index=False)
    cycle_results.to_csv(run_dir / "crop_activity_cycle_gate_v1.csv", index=False)
    (run_dir / "crop_activity_gate_audit_v1.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), **audit}, indent=2))


if __name__ == "__main__":
    main()
