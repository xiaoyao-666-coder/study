#!/usr/bin/env python3
"""Run one isolated ERA5-driven full-season SWAP trunk technical smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.data_preparation.build_swap_season_decision_schedule_v1 import (
    build_from_manifest,
)
from s2s_rtist.pipelines.season_decision_schedule import read_crop_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def swap_date(year: int, month_day: str) -> str:
    value = datetime.strptime(f"{year}-{month_day}", "%Y-%m-%d")
    return value.strftime("%d-%b-%Y").lower()


def patch_trunk_swp_text(
    text: str,
    *,
    year: int,
    sowing_month_day: str,
    harvest_month_day: str,
    output_prefix: str,
) -> str:
    if len(output_prefix) > 16:
        raise ValueError("SWAP output_prefix must be at most 16 characters")
    sowing = swap_date(year, sowing_month_day)
    harvest = swap_date(year, harvest_month_day)
    lines = text.splitlines(keepends=True)
    replacements = {"TSTART": 0, "TEND": 0, "OUTFIL": 0, "METFIL": 0}
    crop_row_replaced = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("TSTART") and "=" in line:
            lines[index] = re.sub(
                r"(?i)(TSTART\s*=\s*)\S+",
                rf"\g<1>{sowing}",
                line,
                count=1,
            )
            replacements["TSTART"] += 1
        elif stripped.startswith("TEND") and "=" in line:
            lines[index] = re.sub(
                r"(?i)(TEND\s*=\s*)\S+",
                rf"\g<1>{harvest}",
                line,
                count=1,
            )
            replacements["TEND"] += 1
        elif stripped.startswith("OUTFIL") and "=" in line:
            lines[index] = re.sub(
                r"(?i)(OUTFIL\s*=\s*)'[^']*'",
                rf"\g<1>'{output_prefix}'",
                line,
                count=1,
            )
            replacements["OUTFIL"] += 1
        elif stripped.startswith("METFIL") and "=" in line:
            lines[index] = re.sub(
                r"(?i)(METFIL\s*=\s*)'[^']*'",
                r"\g<1>'weather'",
                line,
                count=1,
            )
            replacements["METFIL"] += 1

    for index, line in enumerate(lines):
        if not re.match(r"^\s*\d+\s+\d{1,2}-[A-Za-z]{3}-\d{4}\s+", line):
            continue
        match = re.match(
            r"^(\s*\d+\s+)\S+(\s+)\S+(\s+.*'mais'.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            newline = "\n" if line.endswith("\n") else ""
            tail = match.group(3).rstrip("\r\n")
            lines[index] = (
                f"{match.group(1)}{sowing}{match.group(2)}{harvest}{tail}{newline}"
            )
            crop_row_replaced = True
            break

    bad = [name for name, count in replacements.items() if count != 1]
    if bad:
        raise ValueError(f"expected exactly one SWP assignment for: {bad}")
    if not crop_row_replaced:
        raise ValueError("could not locate the detailed maize crop row")

    patched = "".join(lines)
    swinco = re.search(r"(?im)^\s*SWINCO\s*=\s*(-?\d+)", patched)
    if swinco is None:
        raise ValueError("could not locate SWINCO")
    if int(swinco.group(1)) == 3:
        raise ValueError("full-season trunk cannot start from a restart .end state")
    return patched


def weather_record_years(path: Path) -> tuple[set[int], int]:
    years: set[int] = set()
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].strip("'") == "Weather":
            years.add(int(parts[3]))
            rows += 1
    return years, rows


def disable_irrigation(template: Path) -> None:
    helper = template.parent / "real_ir_update.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Missing irrigation helper: {helper}")
    spec = importlib.util.spec_from_file_location("trunk_real_ir_update", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import irrigation helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.modify_irrigation_swp(str(template), 0)


def choose_swap_executable(workspace: Path) -> Path:
    names = ("Swap.exe", "swap_test", "swap") if platform.system().lower().startswith(
        "win"
    ) else ("swap_test", "swap", "Swap.exe")
    for name in names:
        candidate = workspace / name
        if candidate.is_file():
            if not platform.system().lower().startswith("win"):
                candidate.chmod(candidate.stat().st_mode | 0o111)
            return candidate
    raise FileNotFoundError(f"No SWAP executable found in {workspace}")


def run_swap(workspace: Path, log_path: Path) -> tuple[int, bool]:
    executable = choose_swap_executable(workspace)
    env = dict(os.environ)
    runtime = PROJECT_ROOT / "local_libs" / "gcc_runtime" / "usr" / "lib" / "x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = f"{runtime}:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(
        [str(executable.resolve())],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
    )
    log_path.write_text(
        result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    normal = "normal completion" in (result.stdout + result.stderr).lower()
    if not normal:
        raise RuntimeError(
            f"SWAP did not report normal completion (returncode={result.returncode}); "
            f"see {log_path}"
        )
    return int(result.returncode), normal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-id", default="P1")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--split", default="training")
    parser.add_argument("--sowing-month-day", default="04-26")
    parser.add_argument("--harvest-month-day", default="10-10")
    parser.add_argument("--output-prefix", default="trunk2015")
    parser.add_argument("--dvs-threshold", type=float, default=0.1)
    parser.add_argument("--interval-days", type=int, default=7)
    parser.add_argument("--horizon-days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template = workspace / "Swap1.swp"
    if not template.is_file():
        raise FileNotFoundError(f"Missing SWAP template: {template}")

    suffix = f".{args.year % 100:03d}"
    weather_path = workspace / f"weather{suffix}"
    weather_years, weather_rows = weather_record_years(weather_path)
    if weather_years != {int(args.year)} or weather_rows < 365 - 1:
        raise ValueError(
            f"Weather year gate failed: years={sorted(weather_years)}, rows={weather_rows}"
        )

    backup = workspace / "Swap1.pre_trunk_smoke.swp"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    shutil.copy2(template, backup)
    disable_irrigation(template)
    patched = patch_trunk_swp_text(
        template.read_text(encoding="utf-8", errors="ignore"),
        year=args.year,
        sowing_month_day=args.sowing_month_day,
        harvest_month_day=args.harvest_month_day,
        output_prefix=args.output_prefix,
    )
    template.write_text(patched, encoding="utf-8")
    active_swp = workspace / "swap.swp"
    active_swp.write_text(patched, encoding="utf-8")

    log_path = output_dir / "swap_season_trunk_smoke_v1.log"
    returncode, normal = run_swap(workspace, log_path)
    crop_path = workspace / f"{args.output_prefix}.crp"
    crop = read_crop_trajectory(crop_path)
    configured_harvest = pd.Timestamp(f"{args.year}-{args.harvest_month_day}")
    effective_harvest = pd.Timestamp(crop["Date"].max())
    if effective_harvest > configured_harvest:
        raise ValueError(
            f"Crop output ends after configured harvest: {effective_harvest.date()} "
            f"> {configured_harvest.date()}"
        )

    trunk_manifest = pd.DataFrame(
        [
            {
                "site_id": args.site_id,
                "target_year": args.year,
                "split": args.split,
                "crop_output_path": str(crop_path),
                "harvest_date": effective_harvest.strftime("%Y-%m-%d"),
            }
        ]
    )
    manifest_path = output_dir / "swap_season_trunk_manifest_v1.csv"
    trunk_manifest.to_csv(manifest_path, index=False)
    schedule, sources = build_from_manifest(
        trunk_manifest,
        dvs_threshold=args.dvs_threshold,
        interval_days=args.interval_days,
        horizon_days=args.horizon_days,
    )
    schedule_path = output_dir / "swap_season_decision_schedule_v1.csv"
    source_path = output_dir / "swap_season_trunk_source_manifest_v1.csv"
    schedule.to_csv(schedule_path, index=False)
    sources.to_csv(source_path, index=False)

    audit = {
        "status": (
            "p1_2015_full_season_trunk_smoke_passed"
            if args.site_id == "P1" and args.year == 2015
            else "full_season_trunk_smoke_passed"
        ),
        "technical_smoke_only": True,
        "formal_label_generation_allowed": False,
        "site_id": args.site_id,
        "target_year": args.year,
        "sowing_date": f"{args.year}-{args.sowing_month_day}",
        "configured_harvest_date": configured_harvest.strftime("%Y-%m-%d"),
        "effective_crop_end_date": effective_harvest.strftime("%Y-%m-%d"),
        "crop_terminated_before_configured_harvest": bool(
            effective_harvest < configured_harvest
        ),
        "weather_file": str(weather_path),
        "weather_file_sha256": sha256_file(weather_path),
        "weather_rows": weather_rows,
        "weather_years": sorted(weather_years),
        "irrigation_policy": "disabled_for_checkpoint_technical_smoke_only",
        "swap_returncode": returncode,
        "swap_normal_completion": normal,
        "crop_output": str(crop_path),
        "crop_output_sha256": sha256_file(crop_path),
        "crop_daily_rows": int(len(crop)),
        "minimum_crop_dvs": float(crop["DVS"].min()),
        "maximum_crop_dvs": float(crop["DVS"].max()),
        "decision_rows": int(len(schedule)),
        "first_decision_date": str(schedule["decision_date"].iloc[0]),
        "last_decision_date": str(schedule["decision_date"].iloc[-1]),
        "minimum_state_dvs": float(schedule["state_dvs"].min()),
        "maximum_horizon_end_date": str(schedule["horizon_end_date"].max()),
        "next_gate": "daily_restart_checkpoint_equivalence_smoke",
    }
    audit_path = output_dir / "swap_season_trunk_smoke_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
