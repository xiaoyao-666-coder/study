#!/usr/bin/env python3
"""Generate resumable SWAP labels for the formal site-specific GEFS schedule."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_checkpoint_2015_2019_bounded_pilot_v1 import (
    boolean_mask,
    build_response_summary,
    normalize_candidates,
)
from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    FORMAL_WORKSPACE_COPIES,
    SITE_ORDER,
    SITE_TO_SOURCE_WORKSPACE,
    copy_formal_dependencies,
    run_logged,
    validate_source_workspace,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    IRRIGATION_OPTIONS_MM,
    build_ensemble_mean_weather,
    endpoint_fallback_metadata,
    run as run_one_site_branch,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_YEARS = (2015, 2016, 2017, 2018, 2019)
EXPECTED_SITE_CYCLES = 338
EXPECTED_CANDIDATE_ROWS = EXPECTED_SITE_CYCLES * len(IRRIGATION_OPTIONS_MM)
FORMAL_RESTART_NPRINTDAY = 48
FORMAL_SWAP_DTMAX_DAYS = 0.01
FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM = {
    float(value): (
        round(float(value) - 0.1, 1),
        round(float(value) - 0.2, 1),
    )
    for value in IRRIGATION_OPTIONS_MM
    if float(value) > 0.0
}
STALE_AGGREGATE_NAMES = (
    "gefs_exact_schedule_formal_candidates_v1.csv",
    "gefs_exact_schedule_formal_best_v1.csv",
    "gefs_exact_schedule_formal_response_summary_v1.csv",
    "gefs_exact_schedule_formal_label_generation_audit_v1.json",
    "gefs_exact_schedule_formal_label_generation_manifest_v1.json",
)


def split_for_year(year: int) -> str:
    return "training_oof" if int(year) <= 2018 else "validation"


def build_formal_plan(
    weather: pd.DataFrame,
    *,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
    expected_sites: tuple[str, ...] = SITE_ORDER,
    expected_site_cycles: int | None = EXPECTED_SITE_CYCLES,
) -> pd.DataFrame:
    required = {
        "site_id",
        "decision_date",
        "gefs_member",
        "local_date",
        "lead_day",
        "target_year",
    }
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(f"formal weather is missing fields: {missing}")
    data = weather.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"], errors="raise")
    data["local_date"] = pd.to_datetime(data["local_date"], errors="raise")
    data["target_year"] = pd.to_numeric(data["target_year"], errors="raise").astype(int)
    if not data["decision_date"].dt.year.eq(data["target_year"]).all():
        raise ValueError("formal weather target year differs from decision-date year")
    if tuple(sorted(data["target_year"].unique().tolist())) != expected_years:
        raise ValueError(f"formal weather must contain years {expected_years}")
    if set(data["site_id"].astype(str)) != set(expected_sites):
        raise ValueError(f"formal weather must contain sites {expected_sites}")

    key = ["target_year", "site_id", "decision_date"]
    rows: list[dict[str, Any]] = []
    for values, group in data.groupby(key, sort=False):
        year, site, decision = values
        decision = pd.Timestamp(decision)
        build_ensemble_mean_weather(
            data,
            site_id=str(site),
            decision_date=decision.strftime("%Y-%m-%d"),
        )
        expected_dates = pd.date_range(decision, periods=7, freq="D")
        if set(group["local_date"].dt.normalize()) != set(expected_dates):
            raise ValueError(f"formal weather horizon mismatch for {site}/{decision.date()}")
        rows.append(
            {
                "target_year": int(year),
                "site_id": str(site),
                "decision_date": decision.strftime("%Y-%m-%d"),
                "decision_doy": int(decision.dayofyear),
                "checkpoint_date": (decision - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "horizon_end_date": (decision + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                "member_day_rows": int(len(group)),
                "split": split_for_year(int(year)),
            }
        )
    plan = pd.DataFrame(rows)
    site_order = {site: index for index, site in enumerate(expected_sites)}
    plan["site_order"] = plan["site_id"].map(site_order)
    plan = plan.sort_values(
        ["target_year", "site_order", "decision_date"], kind="stable"
    ).drop(columns="site_order").reset_index(drop=True)
    if plan[["target_year", "site_id", "decision_date"]].duplicated().any():
        raise ValueError("formal plan contains duplicate site-cycle keys")
    if expected_site_cycles is not None and len(plan) != int(expected_site_cycles):
        raise ValueError(
            f"formal plan contains {len(plan)} site-cycles, expected {expected_site_cycles}"
        )
    if not plan["member_day_rows"].eq(35).all():
        raise ValueError("every formal site-cycle must contain 35 member-day rows")
    return plan


def unit_path(output_dir: Path, year: int, site: str) -> Path:
    return output_dir / "units" / f"Y{int(year)}" / str(site)


def branch_path(output_dir: Path, year: int, site: str, decision_date: str) -> Path:
    return unit_path(output_dir, year, site) / "branches" / pd.Timestamp(
        decision_date
    ).strftime("%Y%m%d")


def ensure_formal_endpoint_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    if result.empty:
        return result
    if "requested_ir_mm" not in result.columns:
        result["requested_ir_mm"] = pd.to_numeric(result["ir"], errors="raise")
    if "simulated_ir_mm" not in result.columns:
        result["simulated_ir_mm"] = result["requested_ir_mm"]
    if "numerical_endpoint_fallback" not in result.columns:
        result["numerical_endpoint_fallback"] = result.get(
            "numerical_irrigation_fallback", False
        )
    if "numerical_endpoint_fallback_delta_mm" not in result.columns:
        result["numerical_endpoint_fallback_delta_mm"] = (
            pd.to_numeric(result["simulated_ir_mm"], errors="raise")
            - pd.to_numeric(result["requested_ir_mm"], errors="raise")
        )
    if "numerical_endpoint_fallback_policy" not in result.columns:
        result["numerical_endpoint_fallback_policy"] = result.get(
            "numerical_irrigation_fallback_policy", "not_triggered"
        )
    if "numerical_endpoint_fallback_trigger_error_type" not in result.columns:
        result["numerical_endpoint_fallback_trigger_error_type"] = ""
    if "numerical_endpoint_fallback_trigger_error" not in result.columns:
        result["numerical_endpoint_fallback_trigger_error"] = ""
    if "numerical_irrigation_fallback" not in result.columns:
        result["numerical_irrigation_fallback"] = result[
            "numerical_endpoint_fallback"
        ]
    if "numerical_irrigation_fallback_delta_mm" not in result.columns:
        result["numerical_irrigation_fallback_delta_mm"] = result[
            "numerical_endpoint_fallback_delta_mm"
        ]
    if "numerical_irrigation_fallback_policy" not in result.columns:
        result["numerical_irrigation_fallback_policy"] = result[
            "numerical_endpoint_fallback_policy"
        ]
    if "numerical_irrigation_fallback_trigger_error_type" not in result.columns:
        result["numerical_irrigation_fallback_trigger_error_type"] = result[
            "numerical_endpoint_fallback_trigger_error_type"
        ]
    if "numerical_irrigation_fallback_trigger_error" not in result.columns:
        result["numerical_irrigation_fallback_trigger_error"] = result[
            "numerical_endpoint_fallback_trigger_error"
        ]
    fallback_mask = result["numerical_irrigation_fallback"].map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
    )
    if "numerical_irrigation_fallback_attempt_count" not in result.columns:
        result["numerical_irrigation_fallback_attempt_count"] = fallback_mask.astype(
            int
        )
    if "numerical_irrigation_fallback_attempted_values_mm" not in result.columns:
        result["numerical_irrigation_fallback_attempted_values_mm"] = [
            json.dumps([float(simulated)]) if used else "[]"
            for simulated, used in zip(
                result["simulated_ir_mm"], fallback_mask, strict=True
            )
        ]
    if "numerical_irrigation_fallback_failed_values_mm" not in result.columns:
        result["numerical_irrigation_fallback_failed_values_mm"] = "[]"
    return result


def validate_schedule_match(unit_plan: pd.DataFrame, schedule: pd.DataFrame) -> None:
    required_plan = {"decision_date", "checkpoint_date", "horizon_end_date"}
    required_schedule = {
        "decision_date",
        "state_checkpoint_date",
        "horizon_end_date",
    }
    if missing := sorted(required_plan - set(unit_plan.columns)):
        raise ValueError(f"formal unit plan is missing fields: {missing}")
    if missing := sorted(required_schedule - set(schedule.columns)):
        raise ValueError(f"SWAP trunk schedule is missing fields: {missing}")
    expected_keys = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(unit_plan["decision_date"]).dt.strftime("%Y-%m-%d"),
            "state_checkpoint_date": pd.to_datetime(unit_plan["checkpoint_date"]).dt.strftime("%Y-%m-%d"),
            "horizon_end_date": pd.to_datetime(unit_plan["horizon_end_date"]).dt.strftime("%Y-%m-%d"),
        }
    ).sort_values("decision_date").reset_index(drop=True)
    actual_keys = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(schedule["decision_date"]).dt.strftime("%Y-%m-%d"),
            "state_checkpoint_date": pd.to_datetime(schedule["state_checkpoint_date"]).dt.strftime("%Y-%m-%d"),
            "horizon_end_date": pd.to_datetime(schedule["horizon_end_date"]).dt.strftime("%Y-%m-%d"),
        }
    ).sort_values("decision_date").reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise ValueError("SWAP trunk schedule differs from frozen GEFS formal schedule")


def setup_is_reusable(unit_root: Path, unit_plan: pd.DataFrame) -> bool:
    audit_path = unit_root / "setup_audit_v1.json"
    schedule_path = unit_root / "trunk" / "swap_season_decision_schedule_v1.csv"
    checkpoint_audit = unit_root / "all_checkpoints_v1" / "swap_season_checkpoint_equivalence_v1.csv"
    if not all(path.is_file() for path in (audit_path, schedule_path, checkpoint_audit)):
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("mandatory_gate_passed", False):
        return False
    validate_schedule_match(unit_plan, pd.read_csv(schedule_path))
    checkpoints = pd.read_csv(checkpoint_audit)
    return bool(
        len(checkpoints) == len(unit_plan)
        and boolean_mask(checkpoints["checkpoint_equivalence_passed"]).all()
        and float(checkpoints["maximum_absolute_crop_state_error"].max()) <= 1e-6
        and float(checkpoints["maximum_absolute_profile_state_error"].max()) <= 1e-6
    )


def prepare_site_year(
    *,
    source_workspace_root: Path,
    output_dir: Path,
    year: int,
    site: str,
    unit_plan: pd.DataFrame,
    sowing_month_day: str,
    harvest_month_day: str,
    resume: bool,
) -> tuple[Path, Path, Path]:
    root = unit_path(output_dir, year, site)
    if resume and setup_is_reusable(root, unit_plan):
        return (
            root / "workspace",
            root / "all_checkpoints_v1" / "checkpoints",
            root / "all_checkpoints_v1" / "swap_season_checkpoint_equivalence_v1.csv",
        )
    if root.exists():
        raise FileExistsError(f"site-year setup exists without a reusable pass: {root}")
    root.mkdir(parents=True)
    source = source_workspace_root / SITE_TO_SOURCE_WORKSPACE[site]
    validate_source_workspace(source, year=year)
    workspace = root / "workspace"
    shutil.copytree(source, workspace)
    copy_formal_dependencies(workspace)

    trunk_dir = root / "trunk"
    run_logged(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "run_swap_season_trunk_smoke_v1.py"),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(trunk_dir),
            "--site-id",
            site,
            "--year",
            str(year),
            "--split",
            split_for_year(year),
            "--sowing-month-day",
            sowing_month_day,
            "--harvest-month-day",
            harvest_month_day,
            "--output-prefix",
            f"trunk{year}",
        ],
        root / "trunk_runner.log",
    )
    schedule_path = trunk_dir / "swap_season_decision_schedule_v1.csv"
    schedule = pd.read_csv(schedule_path)
    validate_schedule_match(unit_plan, schedule)

    checkpoints_root = root / "all_checkpoints_v1"
    run_logged(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "audit_swap_season_checkpoint_equivalence_v1.py"),
            "--workspace",
            str(workspace),
            "--schedule",
            str(schedule_path),
            "--output-dir",
            str(checkpoints_root),
            "--year",
            str(year),
            "--sowing-month-day",
            sowing_month_day,
            "--trunk-prefix",
            f"trunk{year}",
            "--all-checkpoints",
        ],
        root / "checkpoint_runner.log",
    )
    checkpoint_audit = checkpoints_root / "swap_season_checkpoint_equivalence_v1.csv"
    checkpoints = pd.read_csv(checkpoint_audit)
    mandatory = bool(
        len(checkpoints) == len(unit_plan)
        and boolean_mask(checkpoints["checkpoint_equivalence_passed"]).all()
        and float(checkpoints["maximum_absolute_crop_state_error"].max()) <= 1e-6
        and float(checkpoints["maximum_absolute_profile_state_error"].max()) <= 1e-6
    )
    setup_audit = {
        "status": "exact_schedule_site_year_setup_passed" if mandatory else "exact_schedule_site_year_setup_failed",
        "mandatory_gate_passed": mandatory,
        "target_year": int(year),
        "site_id": site,
        "scheduled_checkpoint_count": int(len(unit_plan)),
        "saved_checkpoint_count": int(len(checkpoints)),
        "maximum_absolute_crop_state_error": float(checkpoints["maximum_absolute_crop_state_error"].max()),
        "maximum_absolute_profile_state_error": float(checkpoints["maximum_absolute_profile_state_error"].max()),
        "schedule_sha256": sha256_file(schedule_path),
        "checkpoint_audit_sha256": sha256_file(checkpoint_audit),
    }
    (root / "setup_audit_v1.json").write_text(
        json.dumps(setup_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not mandatory:
        raise RuntimeError(f"site-year setup gate failed: {root}")
    return workspace, checkpoints_root / "checkpoints", checkpoint_audit


def load_passed_branch(
    path: Path,
    *,
    site: str,
    year: int,
    decision_date: str,
    weather_sha256: str,
    restart_nprintday: int = FORMAL_RESTART_NPRINTDAY,
    swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    audit_path = path / "gefs_checkpoint_one_date_eight_ir_audit_v1.json"
    candidates_path = path / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv"
    manifest_path = path / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json"
    if not all(item.is_file() for item in (audit_path, candidates_path, manifest_path)):
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = ensure_formal_endpoint_metadata(pd.read_csv(candidates_path))
    frequency_values = (
        sorted(
            pd.to_numeric(candidates["nprintday"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if "nprintday" in candidates.columns
        else []
    )
    audit_frequency = audit.get("restart_nprintday")
    if audit_frequency is not None and int(audit_frequency) != int(restart_nprintday):
        return None
    if frequency_values != [int(restart_nprintday)]:
        return None
    dtmax_values = (
        sorted(
            pd.to_numeric(candidates["swap_dtmax_days"], errors="coerce")
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        if "swap_dtmax_days" in candidates.columns
        else []
    )
    audit_dtmax = audit.get("swap_dtmax_days")
    manifest_dtmax = manifest.get("inputs", {}).get("swap_dtmax_days")
    if len(dtmax_values) != 1 or not np.isclose(
        dtmax_values[0], float(swap_dtmax_days), rtol=0.0, atol=1.0e-12
    ):
        return None
    if audit_dtmax is None or not np.isclose(
        float(audit_dtmax), float(swap_dtmax_days), rtol=0.0, atol=1.0e-12
    ):
        return None
    if manifest_dtmax is None or not np.isclose(
        float(manifest_dtmax), float(swap_dtmax_days), rtol=0.0, atol=1.0e-12
    ):
        return None
    if not audit.get("mandatory_gate_passed", False):
        return None
    if (
        str(audit.get("site_id")) != site
        or int(audit.get("target_year", -1)) != int(year)
        or str(audit.get("decision_date")) != decision_date
        or manifest.get("inputs", {}).get("all_variable_weather_sha256") != weather_sha256
    ):
        return None
    if len(candidates) != 8:
        return None
    if not endpoint_fallback_metadata(
        candidates, FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM
    )["valid"]:
        return None
    return candidates, audit


def candidate_restart_nprintday(path: Path) -> int | None:
    candidates_path = path / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv"
    if not candidates_path.is_file():
        return None
    candidates = pd.read_csv(candidates_path)
    if "nprintday" not in candidates.columns or candidates.empty:
        return None
    values = sorted(
        pd.to_numeric(candidates["nprintday"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return values[0] if len(values) == 1 else None


def candidate_swap_dtmax_days(path: Path) -> float | None:
    candidates_path = path / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv"
    if not candidates_path.is_file():
        return None
    candidates = pd.read_csv(candidates_path)
    if "swap_dtmax_days" not in candidates.columns or candidates.empty:
        return None
    values = sorted(
        pd.to_numeric(candidates["swap_dtmax_days"], errors="coerce")
        .dropna()
        .astype(float)
        .unique()
        .tolist()
    )
    return values[0] if len(values) == 1 else None


def dtmax_label(value: float | None) -> str:
    if value is None:
        return "unknown_or_incomplete"
    return f"{float(value):g}".replace(".", "p")


def archive_nonreusable_branch(
    path: Path,
    *,
    output_dir: Path,
    year: int,
    site: str,
    decision_date: str,
    required_restart_nprintday: int,
    required_swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> Path:
    observed = candidate_restart_nprintday(path)
    observed_dtmax = candidate_swap_dtmax_days(path)
    label = str(observed) if observed is not None else "unknown_or_incomplete"
    destination_base = (
        output_dir
        / "archived_nonreusable_branches_v1"
        / f"restart_nprintday_{label}_swap_dtmax_{dtmax_label(observed_dtmax)}"
        / f"Y{int(year)}"
        / str(site)
        / pd.Timestamp(decision_date).strftime("%Y%m%d")
    )
    destination = destination_base
    attempt = 1
    while destination.exists():
        attempt += 1
        destination = destination_base.with_name(
            f"{destination_base.name}_attempt_{attempt:03d}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    archive_audit = {
        "status": "formal_branch_archived_as_nonreusable",
        "target_year": int(year),
        "site_id": str(site),
        "decision_date": str(decision_date),
        "observed_restart_nprintday": observed,
        "required_restart_nprintday": int(required_restart_nprintday),
        "observed_swap_dtmax_days": observed_dtmax,
        "required_swap_dtmax_days": float(required_swap_dtmax_days),
        "archive_attempt": int(attempt),
    }
    (destination / "formal_branch_archive_audit_v1.json").write_text(
        json.dumps(archive_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def archive_stale_aggregate_outputs(
    output_dir: Path,
    *,
    required_restart_nprintday: int,
    required_swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> Path | None:
    candidates_path = output_dir / "gefs_exact_schedule_formal_candidates_v1.csv"
    if not candidates_path.is_file():
        return None
    candidates = pd.read_csv(candidates_path)
    if candidates.empty or "nprintday" not in candidates.columns:
        return None
    values = sorted(
        pd.to_numeric(candidates["nprintday"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    dtmax_values = (
        sorted(
            pd.to_numeric(candidates["swap_dtmax_days"], errors="coerce")
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        if "swap_dtmax_days" in candidates.columns
        else []
    )
    dtmax_matches = len(dtmax_values) == 1 and np.isclose(
        dtmax_values[0], float(required_swap_dtmax_days), rtol=0.0, atol=1.0e-12
    )
    if values == [int(required_restart_nprintday)] and dtmax_matches:
        return None
    label = "_".join(str(value) for value in values) if values else "unknown"
    dtmax_values_label = (
        "_".join(dtmax_label(value) for value in dtmax_values)
        if dtmax_values
        else "unknown"
    )
    archive_root = (
        output_dir
        / "archived_stale_aggregates_v1"
        / f"restart_nprintday_{label}_swap_dtmax_{dtmax_values_label}"
    )
    if archive_root.exists():
        raise FileExistsError(f"aggregate archive destination already exists: {archive_root}")
    archive_root.mkdir(parents=True)
    moved = []
    for name in STALE_AGGREGATE_NAMES:
        source = output_dir / name
        if source.exists():
            shutil.move(str(source), str(archive_root / name))
            moved.append(name)
    audit = {
        "status": "formal_stale_aggregates_archived",
        "observed_restart_nprintday_values": values,
        "required_restart_nprintday": int(required_restart_nprintday),
        "observed_swap_dtmax_days_values": dtmax_values,
        "required_swap_dtmax_days": float(required_swap_dtmax_days),
        "moved_files": moved,
    }
    (archive_root / "formal_stale_aggregate_archive_audit_v1.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive_root


def import_verified_site_year_seed(
    *,
    seed_root: Path,
    output_dir: Path,
    plan: pd.DataFrame,
    weather_sha256: str,
    restart_nprintday: int = FORMAL_RESTART_NPRINTDAY,
    swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> Path:
    seed_audit_path = seed_root / "gefs_site_year_dtmax_stability_audit_v1.json"
    if not seed_audit_path.is_file():
        raise FileNotFoundError(f"missing verified seed audit: {seed_audit_path}")
    seed_audit = json.loads(seed_audit_path.read_text(encoding="utf-8"))
    site = str(seed_audit.get("site_id", ""))
    year = int(seed_audit.get("target_year", -1))
    if not seed_audit.get("mandatory_gate_passed", False):
        raise ValueError("site-year seed did not pass its mandatory gate")
    if seed_audit.get("formal_output_modified") is not False:
        raise ValueError("site-year seed must declare that formal output was not modified")
    if int(seed_audit.get("restart_nprintday", -1)) != int(restart_nprintday):
        raise ValueError("site-year seed restart frequency differs from formal frequency")
    if not np.isclose(
        float(seed_audit.get("swap_dtmax_days", np.nan)),
        float(swap_dtmax_days),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("site-year seed DTMAX differs from formal DTMAX")

    unit_plan = plan.loc[
        plan["target_year"].eq(year) & plan["site_id"].astype(str).eq(site)
    ].copy()
    if unit_plan.empty:
        raise ValueError(f"site-year seed {site}/{year} is absent from the formal plan")
    expected_dates = sorted(unit_plan["decision_date"].astype(str).tolist())
    observed_dates = sorted(
        str(value) for value in seed_audit.get("completed_decision_dates", [])
    )
    if observed_dates != expected_dates:
        raise ValueError("site-year seed decision dates differ from the formal plan")
    if int(seed_audit.get("completed_site_cycle_count", -1)) != len(unit_plan):
        raise ValueError("site-year seed completed cycle count differs from the formal plan")
    if int(seed_audit.get("candidate_rows", -1)) != len(unit_plan) * len(
        IRRIGATION_OPTIONS_MM
    ):
        raise ValueError("site-year seed candidate count is incomplete")

    imported_dates: list[str] = []
    reused_dates: list[str] = []
    archived_destinations: list[str] = []
    for row in unit_plan.itertuples(index=False):
        decision_date = str(row.decision_date)
        source = seed_root / "branches" / pd.Timestamp(decision_date).strftime("%Y%m%d")
        source_loaded = load_passed_branch(
            source,
            site=site,
            year=year,
            decision_date=decision_date,
            weather_sha256=weather_sha256,
            restart_nprintday=restart_nprintday,
            swap_dtmax_days=swap_dtmax_days,
        )
        if source_loaded is None:
            raise ValueError(f"site-year seed branch is not reusable: {source}")
        destination = branch_path(output_dir, year, site, decision_date)
        destination_loaded = load_passed_branch(
            destination,
            site=site,
            year=year,
            decision_date=decision_date,
            weather_sha256=weather_sha256,
            restart_nprintday=restart_nprintday,
            swap_dtmax_days=swap_dtmax_days,
        )
        if destination_loaded is not None:
            reused_dates.append(decision_date)
            continue
        if destination.exists():
            archived = archive_nonreusable_branch(
                destination,
                output_dir=output_dir,
                year=year,
                site=site,
                decision_date=decision_date,
                required_restart_nprintday=restart_nprintday,
                required_swap_dtmax_days=swap_dtmax_days,
            )
            archived_destinations.append(str(archived))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        if load_passed_branch(
            destination,
            site=site,
            year=year,
            decision_date=decision_date,
            weather_sha256=weather_sha256,
            restart_nprintday=restart_nprintday,
            swap_dtmax_days=swap_dtmax_days,
        ) is None:
            raise RuntimeError(f"copied seed branch failed formal validation: {destination}")
        imported_dates.append(decision_date)

    audit_root = (
        output_dir
        / "verified_site_year_seed_imports_v1"
        / f"Y{year}"
        / site
    )
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_path = audit_root / "verified_site_year_seed_import_audit_v1.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "verified_site_year_seed_import_passed",
                "site_id": site,
                "target_year": year,
                "restart_nprintday": int(restart_nprintday),
                "swap_dtmax_days": float(swap_dtmax_days),
                "planned_site_cycle_count": int(len(unit_plan)),
                "imported_site_cycle_count": int(len(imported_dates)),
                "reused_site_cycle_count": int(len(reused_dates)),
                "imported_decision_dates": imported_dates,
                "reused_decision_dates": reused_dates,
                "archived_nonreusable_destinations": archived_destinations,
                "source_seed_root": str(seed_root),
                "source_seed_audit_sha256": sha256_file(seed_audit_path),
                "weather_sha256": weather_sha256,
                "mandatory_gate_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return audit_path


def collect_passed(
    output_dir: Path,
    plan: pd.DataFrame,
    *,
    weather_sha256: str,
    restart_nprintday: int = FORMAL_RESTART_NPRINTDAY,
    swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for row in plan.itertuples(index=False):
        loaded = load_passed_branch(
            branch_path(output_dir, row.target_year, row.site_id, row.decision_date),
            site=str(row.site_id),
            year=int(row.target_year),
            decision_date=str(row.decision_date),
            weather_sha256=weather_sha256,
            restart_nprintday=restart_nprintday,
            swap_dtmax_days=swap_dtmax_days,
        )
        if loaded is None:
            continue
        frame, audit = loaded
        candidates.append(ensure_formal_endpoint_metadata(frame))
        audits.append(audit)
    combined = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return combined, audits


def build_formal_audit(
    *,
    plan: pd.DataFrame,
    candidates: pd.DataFrame,
    branch_audits: list[dict[str, Any]],
    restart_nprintday: int = FORMAL_RESTART_NPRINTDAY,
    swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> dict[str, Any]:
    full_keys = set(
        plan[["target_year", "site_id", "decision_date"]].itertuples(index=False, name=None)
    )
    data = (
        normalize_candidates(ensure_formal_endpoint_metadata(candidates))
        if not candidates.empty
        else pd.DataFrame()
    )
    fallback = (
        endpoint_fallback_metadata(data, FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM)
        if not data.empty
        else {
            "valid": True,
            "count": 0,
            "requested_values_mm": [],
            "simulated_values_mm": [],
            "maximum_absolute_adjustment_mm": 0.0,
            "maximum_attempt_count": 0,
            "second_level_count": 0,
        }
    )
    actual_keys = (
        set(
            data[["target_year", "site", "decision_date"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not data.empty
        else set()
    )
    duplicate_count = (
        int(data[["target_year", "site", "decision_date", "ir"]].duplicated().sum())
        if not data.empty
        else 0
    )
    counts = (
        data.groupby(["target_year", "site", "decision_date"]).size().to_dict()
        if not data.empty
        else {}
    )
    best = data.loc[boolean_mask(data["is_best_ir"])].copy() if not data.empty else pd.DataFrame()
    best_counts = (
        best.groupby(["target_year", "site", "decision_date"]).size().to_dict()
        if not best.empty
        else {}
    )
    irrigation_ok = all(
        sorted(
            data.loc[
                data["target_year"].eq(year)
                & data["site"].astype(str).eq(site)
                & data["decision_date"].eq(decision),
                "ir",
            ].astype(float).tolist()
        ) == IRRIGATION_OPTIONS_MM
        for year, site, decision in actual_keys
    )
    audit_keys = {
        (int(row.get("target_year", -1)), str(row.get("site_id", "")), str(row.get("decision_date", "")))
        for row in branch_audits
        if bool(row.get("mandatory_gate_passed", False))
    }
    finite_metrics = bool(branch_audits) and all(
        np.isfinite(
            [
                row.get("maximum_absolute_checkpoint_crop_state_error", np.inf),
                row.get("maximum_absolute_checkpoint_profile_state_error", np.inf),
                row.get("maximum_absolute_swap_rain_error_mm", np.inf),
                row.get("maximum_absolute_water_balance_residual_mm", np.inf),
            ]
        ).all()
        for row in branch_audits
    )
    completed_gate = all(
        [
            actual_keys.issubset(full_keys),
            actual_keys == audit_keys,
            len(data) == len(actual_keys) * 8,
            all(int(counts.get(key, 0)) == 8 for key in actual_keys),
            all(int(best_counts.get(key, 0)) == 1 for key in actual_keys),
            duplicate_count == 0,
            irrigation_ok,
            fallback["valid"],
            finite_metrics,
            all(float(row["maximum_absolute_checkpoint_crop_state_error"]) <= 1e-6 for row in branch_audits),
            all(float(row["maximum_absolute_checkpoint_profile_state_error"]) <= 1e-6 for row in branch_audits),
            all(float(row["maximum_absolute_swap_rain_error_mm"]) <= 0.01 for row in branch_audits),
            all(float(row["maximum_absolute_water_balance_residual_mm"]) <= 0.5 for row in branch_audits),
            all(int(row.get("primary_output_missing_value_count", 0)) == 0 for row in branch_audits),
            all(int(row.get("prestate_swap_rerun_count", 0)) == 0 for row in branch_audits),
            all(
                int(row.get("restart_nprintday", -1)) == int(restart_nprintday)
                for row in branch_audits
            ),
            all(
                np.isclose(
                    float(row.get("swap_dtmax_days", np.nan)),
                    float(swap_dtmax_days),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for row in branch_audits
            ),
        ]
    )
    complete = completed_gate and actual_keys == full_keys
    status = (
        "exact_schedule_2015_2019_formal_label_generation_complete_passed_with_numerical_irrigation_fallbacks"
        if complete and fallback["count"] > 0
        else "exact_schedule_2015_2019_formal_label_generation_complete_passed"
        if complete
        else "exact_schedule_2015_2019_formal_label_generation_partial_passed"
        if completed_gate
        else "exact_schedule_2015_2019_formal_label_generation_failed"
    )
    return {
        "status": status,
        "completed_output_gate_passed": completed_gate,
        "full_generation_gate_passed": complete,
        "planned_site_cycle_count": int(len(full_keys)),
        "completed_site_cycle_count": int(len(actual_keys)),
        "remaining_site_cycle_count": int(len(full_keys - actual_keys)),
        "planned_candidate_rows": int(len(full_keys) * 8),
        "candidate_rows": int(len(data)),
        "duplicate_candidate_key_count": duplicate_count,
        "best_row_count": int(len(best)),
        "completed_site_year_count": int(
            len({(year, site) for year, site, _ in actual_keys})
        ),
        "maximum_absolute_checkpoint_crop_state_error": max(
            [float(row.get("maximum_absolute_checkpoint_crop_state_error", np.inf)) for row in branch_audits],
            default=None,
        ),
        "maximum_absolute_checkpoint_profile_state_error": max(
            [float(row.get("maximum_absolute_checkpoint_profile_state_error", np.inf)) for row in branch_audits],
            default=None,
        ),
        "maximum_absolute_swap_rain_error_mm": max(
            [float(row.get("maximum_absolute_swap_rain_error_mm", np.inf)) for row in branch_audits],
            default=None,
        ),
        "maximum_absolute_water_balance_residual_mm": max(
            [float(row.get("maximum_absolute_water_balance_residual_mm", np.inf)) for row in branch_audits],
            default=None,
        ),
        "prestate_swap_rerun_count": int(
            sum(int(row.get("prestate_swap_rerun_count", 0)) for row in branch_audits)
        ),
        "primary_output_missing_value_count": int(
            sum(int(row.get("primary_output_missing_value_count", 0)) for row in branch_audits)
        ),
        "numerical_endpoint_fallback_policy": {
            f"{requested:g}": simulated
            for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
        },
        "numerical_endpoint_fallback_metadata_valid": fallback["valid"],
        "numerical_endpoint_fallback_count": fallback["count"],
        "numerical_endpoint_fallback_requested_values_mm": fallback[
            "requested_values_mm"
        ],
        "numerical_endpoint_fallback_simulated_values_mm": fallback[
            "simulated_values_mm"
        ],
        "maximum_absolute_numerical_endpoint_adjustment_mm": fallback[
            "maximum_absolute_adjustment_mm"
        ],
        "numerical_irrigation_fallback_policy": {
            f"{requested:g}": simulated
            for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
        },
        "numerical_irrigation_fallback_metadata_valid": fallback["valid"],
        "numerical_irrigation_fallback_count": fallback["count"],
        "numerical_irrigation_fallback_requested_values_mm": fallback[
            "requested_values_mm"
        ],
        "numerical_irrigation_fallback_simulated_values_mm": fallback[
            "simulated_values_mm"
        ],
        "maximum_absolute_numerical_irrigation_adjustment_mm": fallback[
            "maximum_absolute_adjustment_mm"
        ],
        "maximum_numerical_irrigation_fallback_attempt_count": fallback[
            "maximum_attempt_count"
        ],
        "second_level_numerical_irrigation_fallback_count": fallback[
            "second_level_count"
        ],
        "site_decision_dates_independent": True,
        "restart_nprintday": int(restart_nprintday),
        "observed_restart_nprintday_values": sorted(
            {int(row.get("restart_nprintday", -1)) for row in branch_audits}
        ),
        "swap_dtmax_days": float(swap_dtmax_days),
        "observed_swap_dtmax_days_values": sorted(
            {float(row.get("swap_dtmax_days", np.nan)) for row in branch_audits}
        ),
        "weather_driver_source": "frozen_corrected_GEFS_5member_ensemble_mean",
        "weather_label_scenario_consistent": True,
        "training_eligible": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "review_complete_formal_labels_and_numerical_irrigation_fallbacks_before_training"
            if complete and fallback["count"] > 0
            else "review_complete_formal_labels_before_training"
            if complete
            else "resume_remaining_site_year_units"
            if completed_gate
            else "repair_failed_formal_label_unit"
        ),
    }


def write_aggregate_outputs(
    output_dir: Path,
    *,
    plan: pd.DataFrame,
    candidates: pd.DataFrame,
    branch_audits: list[dict[str, Any]],
    weather_path: Path,
    plan_only: bool = False,
    restart_nprintday: int = FORMAL_RESTART_NPRINTDAY,
    swap_dtmax_days: float = FORMAL_SWAP_DTMAX_DAYS,
) -> dict[str, Path]:
    normalized = (
        normalize_candidates(ensure_formal_endpoint_metadata(candidates))
        if not candidates.empty
        else candidates
    )
    if not normalized.empty:
        normalized = normalized.sort_values(
            ["target_year", "site", "decision_date", "ir"]
        ).reset_index(drop=True)
    best = (
        normalized.loc[boolean_mask(normalized["is_best_ir"])].copy()
        if not normalized.empty
        else pd.DataFrame()
    )
    response = build_response_summary(normalized) if not normalized.empty else pd.DataFrame()
    audit = (
        {
            "status": "exact_schedule_2015_2019_formal_label_plan_passed",
            "completed_output_gate_passed": True,
            "full_generation_gate_passed": False,
            "plan_only": True,
            "planned_site_cycle_count": int(len(plan)),
            "completed_site_cycle_count": 0,
            "remaining_site_cycle_count": int(len(plan)),
            "planned_candidate_rows": int(len(plan) * len(IRRIGATION_OPTIONS_MM)),
            "candidate_rows": 0,
            "site_year_unit_count": int(
                len(plan[["target_year", "site_id"]].drop_duplicates())
            ),
            "site_decision_dates_independent": True,
            "weather_driver_source": "frozen_corrected_GEFS_5member_ensemble_mean",
            "weather_label_scenario_consistent": True,
            "label_generation_performed": False,
            "restart_nprintday": int(restart_nprintday),
            "swap_dtmax_days": float(swap_dtmax_days),
            "numerical_endpoint_fallbacks_mm": {
                f"{requested:g}": simulated
                for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
            },
            "numerical_irrigation_fallbacks_mm": {
                f"{requested:g}": simulated
                for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
            },
            "training_eligible": False,
            "surrogate_training_performed": False,
            "tta_performed": False,
            "next_gate": "run_one_site_year_recovery_smoke",
        }
        if plan_only
        else build_formal_audit(
            plan=plan,
            candidates=normalized,
            branch_audits=branch_audits,
            restart_nprintday=restart_nprintday,
            swap_dtmax_days=swap_dtmax_days,
        )
    )
    outputs = {
        "plan": output_dir / "gefs_exact_schedule_formal_label_plan_v1.csv",
        "candidates": output_dir / "gefs_exact_schedule_formal_candidates_v1.csv",
        "best": output_dir / "gefs_exact_schedule_formal_best_v1.csv",
        "response": output_dir / "gefs_exact_schedule_formal_response_summary_v1.csv",
        "audit": output_dir / "gefs_exact_schedule_formal_label_generation_audit_v1.json",
        "manifest": output_dir / "gefs_exact_schedule_formal_label_generation_manifest_v1.json",
    }
    plan.to_csv(outputs["plan"], index=False)
    normalized.to_csv(outputs["candidates"], index=False)
    best.to_csv(outputs["best"], index=False)
    response.to_csv(outputs["response"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "all_variable_weather": str(weather_path),
            "all_variable_weather_sha256": sha256_file(weather_path),
            "restart_nprintday": int(restart_nprintday),
            "swap_dtmax_days": float(swap_dtmax_days),
            "numerical_endpoint_fallbacks_mm": {
                f"{requested:g}": simulated
                for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
            },
            "numerical_irrigation_fallbacks_mm": {
                f"{requested:g}": simulated
                for requested, simulated in FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM.items()
            },
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "network_download_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace-root", type=Path, required=True)
    parser.add_argument("--weather-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sowing-month-day", default="04-26")
    parser.add_argument("--harvest-month-day", default="10-10")
    parser.add_argument("--max-site-years", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verified-site-year-seed-root", type=Path)
    return parser.parse_args()


def weather_files(weather_root: Path) -> list[Path]:
    return [
        weather_root
        / f"gefs_exact_schedule_{year}_frozen_weather_v1"
        / "04_frozen_all_variable_weather"
        / "gefs_2015_2019_frozen_all_variable_member_weather_v1.csv"
        for year in EXPECTED_YEARS
    ]


def run(args: argparse.Namespace) -> dict[str, Path]:
    paths = weather_files(args.weather_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing annual formal weather files: {missing}")
    weather = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    plan = build_formal_plan(weather)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    combined_weather = args.output_dir / "gefs_exact_schedule_2015_2019_formal_weather_v1.csv"
    if not combined_weather.exists():
        weather.to_csv(combined_weather, index=False)
    weather_hash = sha256_file(combined_weather)
    if args.plan_only:
        return write_aggregate_outputs(
            args.output_dir,
            plan=plan,
            candidates=pd.DataFrame(),
            branch_audits=[],
            weather_path=combined_weather,
            plan_only=True,
            restart_nprintday=FORMAL_RESTART_NPRINTDAY,
            swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
        )

    if args.verified_site_year_seed_root is not None:
        seed_import_audit = import_verified_site_year_seed(
            seed_root=args.verified_site_year_seed_root,
            output_dir=args.output_dir,
            plan=plan,
            weather_sha256=weather_hash,
            restart_nprintday=FORMAL_RESTART_NPRINTDAY,
            swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
        )
        print(f"imported verified site-year seed; see {seed_import_audit}", flush=True)

    if args.resume:
        archived_aggregates = archive_stale_aggregate_outputs(
            args.output_dir,
            required_restart_nprintday=FORMAL_RESTART_NPRINTDAY,
            required_swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
        )
        if archived_aggregates is not None:
            print(
                f"archived stale aggregate outputs to {archived_aggregates}",
                flush=True,
            )

    completed_candidates, _ = collect_passed(
        args.output_dir,
        plan,
        weather_sha256=weather_hash,
        restart_nprintday=FORMAL_RESTART_NPRINTDAY,
        swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
    )
    completed_keys = (
        set(
            normalize_candidates(completed_candidates)[
                ["target_year", "site", "decision_date"]
            ].drop_duplicates().itertuples(index=False, name=None)
        )
        if not completed_candidates.empty
        else set()
    )
    units = plan[["target_year", "site_id"]].drop_duplicates().reset_index(drop=True)
    remaining_units = []
    for row in units.itertuples(index=False):
        unit_plan = plan.loc[
            plan["target_year"].eq(int(row.target_year))
            & plan["site_id"].eq(str(row.site_id))
        ]
        unit_keys = set(
            unit_plan[["target_year", "site_id", "decision_date"]].itertuples(index=False, name=None)
        )
        if not unit_keys.issubset(completed_keys):
            remaining_units.append((int(row.target_year), str(row.site_id)))
    if args.max_site_years is not None:
        if args.max_site_years <= 0:
            raise ValueError("--max-site-years must be positive")
        remaining_units = remaining_units[: args.max_site_years]

    for year, site in remaining_units:
        unit_plan = plan.loc[
            plan["target_year"].eq(year) & plan["site_id"].eq(site)
        ].copy()
        print(f"[{year}/{site}] preparing trunk and {len(unit_plan)} checkpoints", flush=True)
        workspace, checkpoints_root, checkpoint_audit = prepare_site_year(
            source_workspace_root=args.source_workspace_root,
            output_dir=args.output_dir,
            year=year,
            site=site,
            unit_plan=unit_plan,
            sowing_month_day=args.sowing_month_day,
            harvest_month_day=args.harvest_month_day,
            resume=args.resume,
        )
        copy_formal_dependencies(workspace)
        for row in unit_plan.itertuples(index=False):
            destination = branch_path(
                args.output_dir, year, site, str(row.decision_date)
            )
            if args.resume and load_passed_branch(
                destination,
                site=site,
                year=year,
                decision_date=str(row.decision_date),
                weather_sha256=weather_hash,
                restart_nprintday=FORMAL_RESTART_NPRINTDAY,
                swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
            ) is not None:
                print(f"[{year}/{site}/{row.decision_date}] reusing passed branch", flush=True)
                continue
            if destination.exists():
                archived = archive_nonreusable_branch(
                    destination,
                    output_dir=args.output_dir,
                    year=year,
                    site=site,
                    decision_date=str(row.decision_date),
                    required_restart_nprintday=FORMAL_RESTART_NPRINTDAY,
                    required_swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
                )
                print(
                    f"[{year}/{site}/{row.decision_date}] archived nonreusable branch to {archived}",
                    flush=True,
                )
            print(f"[{year}/{site}/{row.decision_date}] running eight branches", flush=True)
            run_one_site_branch(
                SimpleNamespace(
                    source_workspace=workspace,
                    checkpoint_dir=checkpoints_root / pd.Timestamp(row.decision_date).strftime("%Y%m%d"),
                    checkpoint_audit_csv=checkpoint_audit,
                    all_variable_weather=combined_weather,
                    output_dir=destination,
                    site_id=site,
                    year=year,
                    decision_date=str(row.decision_date),
                    sowing_month_day=args.sowing_month_day,
                    restart_nprintday=FORMAL_RESTART_NPRINTDAY,
                    swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
                    numerical_endpoint_fallbacks_mm=(
                        FORMAL_NUMERICAL_ENDPOINT_FALLBACKS_MM
                    ),
                )
            )

    candidates, audits = collect_passed(
        args.output_dir,
        plan,
        weather_sha256=weather_hash,
        restart_nprintday=FORMAL_RESTART_NPRINTDAY,
        swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
    )
    return write_aggregate_outputs(
        args.output_dir,
        plan=plan,
        candidates=candidates,
        branch_audits=audits,
        weather_path=combined_weather,
        restart_nprintday=FORMAL_RESTART_NPRINTDAY,
        swap_dtmax_days=FORMAL_SWAP_DTMAX_DAYS,
    )


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
