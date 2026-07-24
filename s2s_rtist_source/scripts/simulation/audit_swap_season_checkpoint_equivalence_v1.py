#!/usr/bin/env python3
"""Compare prefix-run restart checkpoints with one full-season SWAP trunk."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_rtist.pipelines.season_decision_schedule import read_crop_trajectory


CROP_FIELDS = ("DVS", "LAI", "Rootd", "CWDM", "CWSO")
PROFILE_FIELDS = ("wcontent", "phead", "rootext", "waterflux")
PROFILE_KEYS = ("depth", "top", "bottom")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def select_checkpoint_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        raise ValueError("decision schedule is empty")
    ordered = schedule.sort_values("schedule_index").reset_index(drop=True)
    indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
    return ordered.iloc[indices].reset_index(drop=True)


def checkpoint_rows(schedule: pd.DataFrame, *, all_checkpoints: bool) -> pd.DataFrame:
    if all_checkpoints:
        if schedule.empty:
            raise ValueError("decision schedule is empty")
        return schedule.sort_values("schedule_index").reset_index(drop=True)
    return select_checkpoint_rows(schedule)


def read_profile_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().split(",", 1)[0].strip().lower() == "date":
            start = index
            break
    if start is None:
        raise RuntimeError(f"Missing date header in {path}")
    frame = pd.read_csv(StringIO("\n".join(lines[start:])), skipinitialspace=True)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"].astype(str).str.strip())
    for column in set(PROFILE_KEYS).union(PROFILE_FIELDS):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def compare_crop_state(
    full_crop: pd.DataFrame,
    prefix_crop: pd.DataFrame,
    checkpoint_date: str,
) -> dict[str, float | int | str]:
    checkpoint = pd.Timestamp(checkpoint_date)
    full = full_crop.loc[full_crop["Date"] == checkpoint]
    prefix = prefix_crop.loc[prefix_crop["Date"] == checkpoint]
    if len(full) != 1 or len(prefix) != 1:
        raise ValueError(
            f"expected one crop row for {checkpoint_date}; full={len(full)}, prefix={len(prefix)}"
        )
    diffs: dict[str, float] = {}
    for field in CROP_FIELDS:
        full_value = float(pd.to_numeric(full[field], errors="raise").iloc[0])
        prefix_value = float(pd.to_numeric(prefix[field], errors="raise").iloc[0])
        diffs[field] = abs(prefix_value - full_value)
    return {
        "checkpoint_date": checkpoint.strftime("%Y-%m-%d"),
        **{f"absolute_{field.lower()}_error": value for field, value in diffs.items()},
        "maximum_absolute_crop_state_error": max(diffs.values()),
    }


def compare_profile_state(
    full_profile: pd.DataFrame,
    prefix_profile: pd.DataFrame,
    checkpoint_date: str,
) -> dict[str, float | int]:
    checkpoint = pd.Timestamp(checkpoint_date)
    full = full_profile.loc[full_profile["date"] == checkpoint].copy()
    prefix = prefix_profile.loc[prefix_profile["date"] == checkpoint].copy()
    if full.empty or prefix.empty:
        raise ValueError(
            f"missing profile rows for {checkpoint_date}; full={len(full)}, prefix={len(prefix)}"
        )
    keys = [key for key in PROFILE_KEYS if key in full and key in prefix]
    if not keys:
        raise ValueError("profile output has no common depth keys")
    fields = [field for field in PROFILE_FIELDS if field in full and field in prefix]
    if not fields:
        raise ValueError("profile output has no comparable state fields")
    merged = full[keys + fields].merge(
        prefix[keys + fields],
        on=keys,
        how="outer",
        suffixes=("_full", "_prefix"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"profile layer mismatch for {checkpoint_date}")
    errors: dict[str, float] = {}
    for field in fields:
        full_values = merged[f"{field}_full"]
        prefix_values = merged[f"{field}_prefix"]
        missing_mismatch = full_values.isna() ^ prefix_values.isna()
        if missing_mismatch.any():
            raise ValueError(
                f"{field} missing-value pattern differs for {checkpoint_date}"
            )
        comparable = full_values.notna() & prefix_values.notna()
        if not comparable.any():
            continue
        delta = (prefix_values.loc[comparable] - full_values.loc[comparable]).abs()
        errors[field] = float(delta.max())
    if not errors:
        raise ValueError(f"profile has no numeric comparable states for {checkpoint_date}")
    return {
        "profile_layer_rows": int(len(merged)),
        **{f"maximum_absolute_{field}_error": value for field, value in errors.items()},
        "maximum_absolute_profile_state_error": max(errors.values()),
    }


def load_workspace_generator(workspace: Path):
    module_path = workspace / "generate_restart_decision_dataset.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing workspace generator: {module_path}")
    sys.path.insert(0, str(workspace))
    spec = importlib.util.spec_from_file_location("checkpoint_restart_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--sowing-month-day", default="04-26")
    parser.add_argument("--trunk-prefix", default="trunk2015")
    parser.add_argument("--crop-tolerance", type=float, default=1e-6)
    parser.add_argument("--profile-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Generate and verify every decision checkpoint instead of first/middle/last.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()

    schedule = pd.read_csv(args.schedule)
    selected = checkpoint_rows(schedule, all_checkpoints=args.all_checkpoints)
    full_crop = read_crop_trajectory(workspace / f"{args.trunk_prefix}.crp")
    full_profile = read_profile_table(workspace / f"{args.trunk_prefix}.vap")
    generator = load_workspace_generator(workspace)
    generator.YEAR = int(args.year)
    generator.START_DOY = int(pd.Timestamp(f"{args.year}-{args.sowing_month_day}").dayofyear)

    rows: list[dict[str, object]] = []
    with working_directory(workspace):
        for item in selected.itertuples(index=False):
            checkpoint_date = str(item.state_checkpoint_date)
            decision_date = pd.Timestamp(item.decision_date)
            date_t = decision_date.strftime("%d-%b-%Y")
            label = decision_date.strftime("%Y%m%d")
            generator.configure_irrigation(date_t, None)
            generator.run_pre_state(
                str(output_dir / f"prefix_to_{label}.log"),
                int(item.decision_doy),
            )
            prefix_crop_path = workspace / "result_forec.crp"
            prefix_profile_path = workspace / "result_forec.vap"
            prefix_end_path = workspace / "result_forec.end"
            prefix_crop = read_crop_trajectory(prefix_crop_path)
            prefix_profile = read_profile_table(prefix_profile_path)
            prefix_last_crop_date = prefix_crop["Date"].max().strftime("%Y-%m-%d")
            if prefix_last_crop_date != checkpoint_date:
                raise ValueError(
                    f"prefix crop ends on {prefix_last_crop_date}, expected {checkpoint_date}"
                )
            crop_errors = compare_crop_state(full_crop, prefix_crop, checkpoint_date)
            profile_errors = compare_profile_state(
                full_profile, prefix_profile, checkpoint_date
            )
            passed = bool(
                crop_errors["maximum_absolute_crop_state_error"]
                <= args.crop_tolerance
                and profile_errors["maximum_absolute_profile_state_error"]
                <= args.profile_tolerance
            )
            destination = checkpoint_dir / label
            destination.mkdir()
            for source in (prefix_crop_path, prefix_profile_path, prefix_end_path):
                shutil.copy2(source, destination / source.name)
            rows.append(
                {
                    "schedule_index": int(item.schedule_index),
                    "decision_date": str(item.decision_date),
                    "decision_doy": int(item.decision_doy),
                    **crop_errors,
                    **profile_errors,
                    "crop_tolerance": float(args.crop_tolerance),
                    "profile_tolerance": float(args.profile_tolerance),
                    "checkpoint_equivalence_passed": passed,
                    "saved_restart_end": str(destination / "result_forec.end"),
                }
            )

    result = pd.DataFrame(rows)
    result_path = output_dir / "swap_season_checkpoint_equivalence_v1.csv"
    result.to_csv(result_path, index=False)
    passed = bool(result["checkpoint_equivalence_passed"].all())
    complete_schedule = bool(args.all_checkpoints and len(result) == len(schedule))
    audit = {
        "status": (
            "all_season_checkpoints_generated_and_verified"
            if passed and complete_schedule
            else "checkpoint_equivalence_smoke_passed"
            if passed
            else "checkpoint_equivalence_smoke_failed"
        ),
        "target_year": int(args.year),
        "checkpoint_scope": (
            "all_schedule_rows" if args.all_checkpoints else "first_middle_last"
        ),
        "scheduled_checkpoint_count": int(len(schedule)),
        "tested_checkpoint_count": int(len(result)),
        "tested_schedule_indices": result["schedule_index"].astype(int).tolist(),
        "maximum_absolute_crop_state_error": float(
            result["maximum_absolute_crop_state_error"].max()
        ),
        "maximum_absolute_profile_state_error": float(
            result["maximum_absolute_profile_state_error"].max()
        ),
        "all_checkpoints_passed": passed,
        "formal_checkpoint_generation_allowed": passed,
        "all_scheduled_checkpoints_saved": complete_schedule,
        "formal_label_generation_allowed": False,
        "branch_label_generation_allowed": False,
        "next_gate": (
            "all_variable_gefs_correction_before_one_date_eight_ir_branch_smoke"
            if complete_schedule
            else "generate_all_season_checkpoints"
        ),
    }
    audit_path = output_dir / "swap_season_checkpoint_equivalence_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
