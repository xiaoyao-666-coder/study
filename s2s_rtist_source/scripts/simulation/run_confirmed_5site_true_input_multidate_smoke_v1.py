#!/usr/bin/env python3
"""Run confirmed 5-site true-input multi-date restart smoke.

This wrapper chains the validated true site-specific input layers:

1. static SWP inputs
2. POLARIS soil hydraulic curves
3. gridMET weather
4. multi-date restart-generation smoke
5. curve audit

Default dates stay within the currently available 2024 gridMET file coverage
(213 days): 16-Jul, 20-Jul, and 24-Jul. With a 7-day horizon, 24-Jul ends at
DOY 213.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
RUN_ROOT = OUT_DIR / "confirmed_5site_restart_generation_smoke_v1"
DEFAULT_SITES = ["P1", "P15", "P2", "P3", "P4"]
DEFAULT_DECISION_DATES = ["16-Jul-2024:198", "20-Jul-2024:202", "24-Jul-2024:206"]
DEFAULT_HORIZON_DAYS = 7


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def validate_dates(decision_dates: list[str], weather_days: int, horizon_days: int) -> None:
    for raw in decision_dates:
        if ":" not in raw:
            raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
        date_text, doy_text = raw.split(":", 1)
        if not doy_text.isdigit():
            raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
        decision_doy = int(doy_text)
        parsed = datetime.strptime(date_text, "%d-%b-%Y")
        actual_doy = int(parsed.timetuple().tm_yday)
        if actual_doy != decision_doy:
            raise ValueError(
                f"{raw} has mismatched DOY: {date_text} is DOY {actual_doy}, "
                f"but got {decision_doy}. Use {date_text}:{actual_doy}."
            )
        end_doy = decision_doy + horizon_days
        if end_doy > weather_days:
            raise ValueError(
                f"{raw} ends at DOY {end_doy}, beyond available gridMET days {weather_days}. "
                "Use earlier dates or provide fuller weather files."
            )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_wrapper_report(run_dir: Path, decision_dates: list[str], sites: list[str]) -> Path:
    report = OUT_DIR / "confirmed_5site_true_input_multidate_smoke_v1.md"
    summary_path = run_dir / "confirmed_5site_restart_generation_smoke_summary_v1.csv"
    best_path = run_dir / "confirmed_5site_restart_generation_smoke_best_by_date_v1.csv"
    audit_summary_path = run_dir / "confirmed_5site_restart_curve_audit_v1_summary.csv"
    best_audit_path = run_dir / "confirmed_5site_restart_curve_audit_v1_best_by_site.csv"

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()
    audit_summary = pd.read_csv(audit_summary_path) if audit_summary_path.exists() else pd.DataFrame()
    best_audit = pd.read_csv(best_audit_path) if best_audit_path.exists() else pd.DataFrame()

    lines = [
        "# Confirmed 5-Site True-Input Multidate Smoke V1",
        "",
        "## Scope",
        "",
        f"- Sites: `{', '.join(sites)}`",
        f"- Decision dates: `{'; '.join(decision_dates)}`",
        "- Input layers: static SWP + POLARIS soil + gridMET weather.",
        "- This is a bounded smoke and curve audit, not surrogate training.",
        "",
        "## Run Directory",
        "",
        f"`{run_dir}`",
        "",
        "## Smoke Summary",
        "",
        markdown_table(summary[["site", "status", "returncode", "candidate_rows", "best_rows"]] if not summary.empty else summary),
        "",
        "## Best By Site/Date",
        "",
        markdown_table(best),
        "",
        "## Curve Audit Summary",
        "",
        markdown_table(audit_summary),
        "",
        "## Curve Audit Best Rows",
        "",
        markdown_table(best_audit),
        "",
        "## Interpretation",
        "",
        "If all sites completed and the curve audit status is passed, the true-input generation chain is stable across multiple decision dates. "
        "That is the recommended gate before small multi-site surrogate training.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--decision-dates", nargs="+", default=DEFAULT_DECISION_DATES)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--weather-days", type=int, default=213)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--run-id", default="true_input_multidate_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-per-site", type=int, default=7200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dates(args.decision_dates, weather_days=args.weather_days, horizon_days=args.horizon_days)

    run_cmd([args.python, "apply_confirmed_5site_static_swp_inputs_v1.py", "--create-missing", "--sites", *args.sites], args.dry_run)
    run_cmd(
        [
            args.python,
            "apply_confirmed_5site_polaris_soil_inputs_v1.py",
            "--create-missing",
            "--use-embedded-profiles",
            "--sites",
            *args.sites,
        ],
        args.dry_run,
    )
    run_cmd(
        [
            args.python,
            "apply_confirmed_5site_gridmet_weather_inputs_v1.py",
            "--create-missing",
            "--year",
            str(args.year),
            "--days",
            str(args.weather_days),
            "--sites",
            *args.sites,
        ],
        args.dry_run,
    )
    run_cmd(
        [
            args.python,
            "run_confirmed_5site_restart_generation_smoke_v1.py",
            "--run-id",
            args.run_id,
            "--timeout-per-site",
            str(args.timeout_per_site),
            "--sites",
            *args.sites,
            "--decision-dates",
            *args.decision_dates,
        ],
        args.dry_run,
    )

    run_dir = RUN_ROOT / args.run_id
    run_cmd(
        [
            args.python,
            "audit_confirmed_5site_restart_curves_v1.py",
            "--run-dir",
            str(run_dir),
            "--sites",
            *args.sites,
        ],
        args.dry_run,
    )
    if not args.dry_run:
        report = write_wrapper_report(run_dir, args.decision_dates, args.sites)
        print(f"wrapper_report: {report}", flush=True)


if __name__ == "__main__":
    main()
