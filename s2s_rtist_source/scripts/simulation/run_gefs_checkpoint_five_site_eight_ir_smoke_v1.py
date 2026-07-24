#!/usr/bin/env python3
"""Expand the verified checkpoint branch smoke to five paper sites."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
    build_ensemble_mean_weather,
    run as run_one_site_branch,
    sha256_file,
)
from s2s_rtist.pipelines.season_decision_schedule import read_crop_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_ORDER = ("P1", "P2", "P3", "P4", "P15")
SITE_TO_SOURCE_WORKSPACE = {
    "P1": "code_N1_Maize",
    "P2": "code_N2_Maize",
    "P3": "code_N3_Maize",
    "P4": "code_N4_Maize",
    "P15": "code_active_Maize",
}
FORMAL_WORKSPACE_COPIES = (
    (
        PROJECT_ROOT / "src" / "s2s_rtist" / "pipelines" / "restart_decision_dataset.py",
        "generate_restart_decision_dataset.py",
    ),
    (
        PROJECT_ROOT / "src" / "s2s_rtist" / "labels" / "swap_three_output_labels.py",
        "swap_three_output_labels_v1.py",
    ),
    (
        PROJECT_ROOT / "src" / "s2s_rtist" / "physics" / "rootzone_flux_frequency.py",
        "rootzone_flux_frequency_diagnostic_v1.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "diagnostics" / "restart_raw_audit_v1.py",
        "restart_raw_audit_v1.py",
    ),
)
REQUIRED_SOURCE_FILES = ("Swap1.swp", "real_ir_update.py")


def prepend_runtime_environment() -> dict[str, str]:
    env = dict(os.environ)
    python_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        python_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_parts)
    if not platform.system().lower().startswith("win"):
        runtime = (
            PROJECT_ROOT
            / "local_libs"
            / "gcc_runtime"
            / "usr"
            / "lib"
            / "x86_64-linux-gnu"
        )
        existing_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(runtime) + (
            ":" + existing_library_path if existing_library_path else ""
        )
    return env


def choose_swap_executable(workspace: Path) -> Path:
    names = ("Swap.exe", "swap_test", "swap") if platform.system().lower().startswith(
        "win"
    ) else ("swap_test", "swap", "Swap.exe")
    for name in names:
        path = workspace / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing SWAP executable in {workspace}")


def validate_source_workspace(path: Path, *, year: int = 2015) -> None:
    suffix = f".{int(year) % 100:03d}"
    required = (
        *REQUIRED_SOURCE_FILES,
        f"weather{suffix}",
        f"WeatherOriginal{suffix}",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is missing source files: {missing}")
    choose_swap_executable(path)


def copy_formal_dependencies(workspace: Path) -> None:
    for source, target_name in FORMAL_WORKSPACE_COPIES:
        if not source.is_file():
            raise FileNotFoundError(f"missing formal dependency: {source}")
        shutil.copy2(source, workspace / target_name)


def run_logged(command: list[str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=prepend_runtime_environment(),
    )
    log_path.write_text(
        result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with return code {result.returncode}; see {log_path}"
        )


def build_target_checkpoint_schedule(
    crop: pd.DataFrame,
    *,
    decision_date: str,
    dvs_threshold: float = 0.1,
) -> pd.DataFrame:
    decision = pd.Timestamp(decision_date)
    checkpoint = decision - pd.Timedelta(days=1)
    horizon_dates = pd.date_range(decision, periods=7, freq="D")
    data = crop.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    checkpoint_rows = data.loc[data["Date"].eq(checkpoint)]
    if len(checkpoint_rows) != 1:
        raise ValueError(
            f"expected one crop checkpoint row on {checkpoint.date()}, got {len(checkpoint_rows)}"
        )
    state_dvs = float(checkpoint_rows["DVS"].iloc[0])
    if not np.isfinite(state_dvs) or state_dvs < dvs_threshold or state_dvs >= 2.0:
        raise ValueError(
            f"decision checkpoint is not crop-active: date={checkpoint.date()}, DVS={state_dvs}"
        )
    available = set(data["Date"].dt.normalize())
    missing_horizon = [
        date.strftime("%Y-%m-%d")
        for date in horizon_dates
        if date.normalize() not in available
    ]
    if missing_horizon:
        raise ValueError(f"full-season trunk is missing branch horizon dates: {missing_horizon}")
    return pd.DataFrame(
        [
            {
                "schedule_index": 0,
                "decision_date": decision.strftime("%Y-%m-%d"),
                "decision_doy": int(decision.dayofyear),
                "state_checkpoint_date": checkpoint.strftime("%Y-%m-%d"),
                "state_dvs": state_dvs,
                "horizon_end_date": horizon_dates[-1].strftime("%Y-%m-%d"),
            }
        ]
    )


def load_passed_site(site_root: Path) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    branch = site_root / "branch"
    audit_path = branch / "gefs_checkpoint_one_date_eight_ir_audit_v1.json"
    candidates_path = branch / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv"
    if not audit_path.is_file() or not candidates_path.is_file():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("mandatory_gate_passed", False):
        return None
    return pd.read_csv(candidates_path), audit


def run_site(
    *,
    site_id: str,
    source_workspace_root: Path,
    all_variable_weather: Path,
    output_dir: Path,
    year: int,
    decision_date: str,
    sowing_month_day: str,
    harvest_month_day: str,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    site_root = output_dir / site_id
    if resume:
        passed = load_passed_site(site_root)
        if passed is not None:
            candidates, audit = passed
            return candidates, audit, {
                "site_id": site_id,
                "status": "reused_passed_site",
                "candidate_rows": int(len(candidates)),
                "site_root": str(site_root),
                "error": "",
            }
    if site_root.exists():
        raise FileExistsError(
            f"site output already exists without a reusable pass: {site_root}"
        )
    site_root.mkdir(parents=True)
    source = source_workspace_root / SITE_TO_SOURCE_WORKSPACE[site_id]
    validate_source_workspace(source, year=year)
    workspace = site_root / "workspace"
    shutil.copytree(source, workspace)
    copy_formal_dependencies(workspace)

    trunk_dir = site_root / "trunk"
    trunk_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "simulation" / "run_swap_season_trunk_smoke_v1.py"),
        "--workspace",
        str(workspace),
        "--output-dir",
        str(trunk_dir),
        "--site-id",
        site_id,
        "--year",
        str(year),
        "--sowing-month-day",
        sowing_month_day,
        "--harvest-month-day",
        harvest_month_day,
        "--output-prefix",
        f"trunk{year}",
    ]
    run_logged(trunk_command, site_root / "trunk_runner.log")

    crop = read_crop_trajectory(workspace / f"trunk{year}.crp")
    target_schedule = build_target_checkpoint_schedule(
        crop,
        decision_date=decision_date,
    )
    target_schedule_path = site_root / "target_checkpoint_schedule_v1.csv"
    target_schedule.to_csv(target_schedule_path, index=False)

    checkpoint_root = site_root / "target_checkpoint_equivalence_v1"
    checkpoint_command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "simulation"
            / "audit_swap_season_checkpoint_equivalence_v1.py"
        ),
        "--workspace",
        str(workspace),
        "--schedule",
        str(target_schedule_path),
        "--output-dir",
        str(checkpoint_root),
        "--year",
        str(year),
        "--sowing-month-day",
        sowing_month_day,
        "--trunk-prefix",
        f"trunk{year}",
    ]
    run_logged(checkpoint_command, site_root / "checkpoint_runner.log")

    checkpoint_dir = checkpoint_root / "checkpoints" / pd.Timestamp(
        decision_date
    ).strftime("%Y%m%d")
    checkpoint_audit = checkpoint_root / "swap_season_checkpoint_equivalence_v1.csv"
    branch_dir = site_root / "branch"
    generated = run_one_site_branch(
        SimpleNamespace(
            source_workspace=workspace,
            checkpoint_dir=checkpoint_dir,
            checkpoint_audit_csv=checkpoint_audit,
            all_variable_weather=all_variable_weather,
            output_dir=branch_dir,
            site_id=site_id,
            year=year,
            decision_date=decision_date,
            sowing_month_day=sowing_month_day,
        )
    )
    candidates = pd.read_csv(generated["candidates"])
    audit = json.loads(generated["audit"].read_text(encoding="utf-8"))
    return candidates, audit, {
        "site_id": site_id,
        "status": audit["status"],
        "candidate_rows": int(len(candidates)),
        "site_root": str(site_root),
        "error": "",
    }


def build_five_site_audit(
    *,
    candidates: pd.DataFrame,
    site_audits: list[dict[str, Any]],
    expected_sites: tuple[str, ...] = SITE_ORDER,
) -> dict[str, Any]:
    expected = set(expected_sites)
    required_columns = {"site", "date_t", "ir", "is_best_ir"}
    candidate_schema_ok = required_columns.issubset(candidates.columns)
    actual = (
        set(candidates["site"].astype(str)) if candidate_schema_ok else set()
    )
    counts = (
        candidates.groupby("site").size().to_dict()
        if candidate_schema_ok and not candidates.empty
        else {}
    )
    duplicate_count = (
        int(candidates[["site", "date_t", "ir"]].duplicated().sum())
        if candidate_schema_ok and not candidates.empty
        else 0
    )
    irrigation_ok = bool(candidate_schema_ok) and all(
        sorted(
            candidates.loc[candidates["site"].astype(str).eq(site), "ir"]
            .astype(float)
            .tolist()
        )
        == IRRIGATION_OPTIONS_MM
        for site in expected_sites
    )
    audit_by_site = {str(row.get("site_id")): row for row in site_audits}
    passed_sites = {
        site
        for site, row in audit_by_site.items()
        if bool(row.get("mandatory_gate_passed", False))
    }
    maximum_crop_error = max(
        [float(row.get("maximum_absolute_checkpoint_crop_state_error", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_profile_error = max(
        [float(row.get("maximum_absolute_checkpoint_profile_state_error", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_rain_error = max(
        [float(row.get("maximum_absolute_swap_rain_error_mm", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    maximum_residual = max(
        [float(row.get("maximum_absolute_water_balance_residual_mm", np.inf)) for row in site_audits],
        default=float("inf"),
    )
    missing_primary = sum(
        int(row.get("primary_output_missing_value_count", 0)) for row in site_audits
    )
    best_mask = (
        candidates["is_best_ir"].fillna(False)
        if candidate_schema_ok
        and pd.api.types.is_bool_dtype(candidates["is_best_ir"].dtype)
        else candidates["is_best_ir"].astype(str).str.strip().str.lower().eq("true")
        if candidate_schema_ok
        else pd.Series(False, index=candidates.index)
    )
    best = (
        candidates.loc[best_mask].copy()
        if candidate_schema_ok and not candidates.empty
        else pd.DataFrame()
    )
    best_counts = best.groupby("site").size().to_dict() if not best.empty else {}
    best_ir = {
        str(row.site): float(row.ir)
        for row in best[["site", "ir"]].itertuples(index=False)
    } if not best.empty else {}
    passed = all(
        [
            actual == expected,
            candidate_schema_ok,
            len(candidates) == len(expected_sites) * len(IRRIGATION_OPTIONS_MM),
            all(int(counts.get(site, 0)) == 8 for site in expected_sites),
            duplicate_count == 0,
            irrigation_ok,
            passed_sites == expected,
            len(best) == len(expected_sites),
            all(int(best_counts.get(site, 0)) == 1 for site in expected_sites),
            maximum_crop_error <= 1e-6,
            maximum_profile_error <= 1e-6,
            maximum_rain_error <= 0.01,
            maximum_residual <= 0.5,
            missing_primary == 0,
        ]
    )
    return {
        "status": (
            "verified_checkpoint_five_site_eight_ir_swap_smoke_passed"
            if passed
            else "verified_checkpoint_five_site_eight_ir_swap_smoke_failed"
        ),
        "mandatory_gate_passed": passed,
        "site_count": int(len(actual)),
        "expected_sites": list(expected_sites),
        "passed_sites": sorted(passed_sites),
        "candidate_rows": int(len(candidates)),
        "candidate_rows_by_site": {site: int(counts.get(site, 0)) for site in expected_sites},
        "duplicate_candidate_key_count": duplicate_count,
        "best_row_count": int(len(best)),
        "best_rows_by_site": {
            site: int(best_counts.get(site, 0)) for site in expected_sites
        },
        "best_ir_by_site_mm": best_ir,
        "nonzero_best_ir_site_count": int(sum(value > 0.0 for value in best_ir.values())),
        "maximum_absolute_checkpoint_crop_state_error": maximum_crop_error,
        "maximum_absolute_checkpoint_profile_state_error": maximum_profile_error,
        "maximum_absolute_swap_rain_error_mm": maximum_rain_error,
        "maximum_absolute_water_balance_residual_mm": maximum_residual,
        "primary_output_missing_value_count": missing_primary,
        "prestate_swap_rerun_count": 0,
        "weather_driver_source": "frozen_corrected_GEFS_5member_ensemble_mean",
        "weather_label_scenario_consistent": True,
        "training_eligible": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "review_five_site_response_before_bounded_2015_2019_label_generation"
            if passed
            else "repair_five_site_checkpoint_branch_smoke"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace-root", type=Path, required=True)
    parser.add_argument("--all-variable-weather", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", default=list(SITE_ORDER))
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--decision-date", default="2015-07-06")
    parser.add_argument("--sowing-month-day", default="04-26")
    parser.add_argument("--harvest-month-day", default="10-10")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    sites = tuple(args.sites)
    if sites != SITE_ORDER:
        raise ValueError(f"formal five-site smoke requires sites in order: {SITE_ORDER}")
    if pd.Timestamp(args.decision_date).year != int(args.year):
        raise ValueError("decision date year does not match --year")
    weather = pd.read_csv(args.all_variable_weather)
    for site in sites:
        build_ensemble_mean_weather(
            weather,
            site_id=site,
            decision_date=args.decision_date,
        )
        validate_source_workspace(
            args.source_workspace_root / SITE_TO_SOURCE_WORKSPACE[site],
            year=args.year,
        )
    for source, _ in FORMAL_WORKSPACE_COPIES:
        if not source.is_file():
            raise FileNotFoundError(f"missing formal dependency: {source}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    candidates_frames: list[pd.DataFrame] = []
    site_audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for site in sites:
        print(f"[{site}] starting full-season trunk, checkpoint, and branch smoke", flush=True)
        try:
            candidates, audit, summary = run_site(
                site_id=site,
                source_workspace_root=args.source_workspace_root,
                all_variable_weather=args.all_variable_weather,
                output_dir=args.output_dir,
                year=args.year,
                decision_date=args.decision_date,
                sowing_month_day=args.sowing_month_day,
                harvest_month_day=args.harvest_month_day,
                resume=args.resume,
            )
            candidates_frames.append(candidates)
            site_audits.append(audit)
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    "site_id": site,
                    "status": "failed",
                    "candidate_rows": 0,
                    "site_root": str(args.output_dir / site),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{site}] failed: {type(exc).__name__}: {exc}", flush=True)

    candidates = (
        pd.concat(candidates_frames, ignore_index=True)
        if candidates_frames
        else pd.DataFrame()
    )
    summary = pd.DataFrame(summaries)
    five_site_audit = build_five_site_audit(
        candidates=candidates,
        site_audits=site_audits,
        expected_sites=sites,
    )
    best = (
        candidates.loc[candidates["is_best_ir"].astype(bool)].copy()
        if not candidates.empty
        else pd.DataFrame()
    )
    outputs = {
        "candidates": args.output_dir / "gefs_checkpoint_five_site_eight_ir_candidates_v1.csv",
        "best": args.output_dir / "gefs_checkpoint_five_site_eight_ir_best_by_site_v1.csv",
        "summary": args.output_dir / "gefs_checkpoint_five_site_run_summary_v1.csv",
        "audit": args.output_dir / "gefs_checkpoint_five_site_eight_ir_audit_v1.json",
        "manifest": args.output_dir / "gefs_checkpoint_five_site_eight_ir_manifest_v1.json",
    }
    candidates.to_csv(outputs["candidates"], index=False)
    best.to_csv(outputs["best"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    outputs["audit"].write_text(
        json.dumps(five_site_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": five_site_audit["status"],
        "inputs": {
            "all_variable_weather_sha256": sha256_file(args.all_variable_weather),
            "source_workspace_root": str(args.source_workspace_root),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "network_download_performed": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not five_site_audit["mandatory_gate_passed"]:
        raise RuntimeError(f"five-site smoke failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
