#!/usr/bin/env python3
"""Create a SWAP sampling plan focused on date/state coverage for TTA work.

The current per-site experts pass in-sample capacity but fail held-out dates.
This planner shifts the next data-generation step toward the teacher's TTA and
coverage direction: add dense date/state coverage, especially around the sites
and seasonal windows where held-out CV failed catastrophically.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from plan_continuous_irrigation_sampling_v1 import uniform_values
from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


OUT_DIR = Path("site_general_surrogate_eval")
DEFAULT_OUT_DIR = OUT_DIR / "continuous_ir_12site_tta_date_coverage_sampling_v1"
DEFAULT_FAILURE_DIAG = (
    OUT_DIR
    / "continuous_ir_12site_10k_persite_curve_mlp_ranker_cv_v1"
    / "failure_diagnostic_v1"
    / "persite_curve_ranker_cv_failure_by_site_v1.csv"
)
DEFAULT_SITE_FEATURES = OUT_DIR / "site_feature_screening_12_code_sites.csv"
DEFAULT_SITES = "code_B1,code_C2,code_N1,code_N2"
DEFAULT_START = "01-Jun-2024"
DEFAULT_END = "31-Jul-2024"


def parse_sites(text: str) -> list[str]:
    sites = [part.strip() for part in text.split(",") if part.strip()]
    if not sites:
        raise ValueError("At least one site is required")
    return list(dict.fromkeys(sites))


def format_decision_date(dt: datetime) -> tuple[str, int]:
    return dt.strftime("%d-%b-%Y"), int(dt.timetuple().tm_yday)


def date_range(start: str, end: str, step_days: int) -> list[tuple[str, int]]:
    if step_days <= 0:
        raise ValueError("--date-step-days must be positive")
    cur = datetime.strptime(start, "%d-%b-%Y")
    stop = datetime.strptime(end, "%d-%b-%Y")
    if stop < cur:
        raise ValueError("--date-end must not be earlier than --date-start")
    out = []
    while cur <= stop:
        out.append(format_decision_date(cur))
        cur += timedelta(days=step_days)
    return out


def read_site_caps(site_feature_csv: Path, default_ir_max: float) -> dict[str, float]:
    if not site_feature_csv.exists():
        return {}
    df = pd.read_csv(site_feature_csv)
    if "site" not in df.columns:
        return {}
    caps = {}
    for _, row in df.iterrows():
        site = str(row["site"])
        cap = default_ir_max
        for col in ["ir_max", "site_ir_max", "max_ir", "max_irrigation_mm"]:
            if col in row.index and pd.notna(row[col]):
                try:
                    cap = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass
        caps[site] = cap
    return caps


def load_failure_sites(path: Path, top_n: int) -> list[str]:
    if top_n <= 0 or not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"site_id", "continuous_ranker_mean_regret"}
    if not required.issubset(df.columns):
        return []
    return (
        df.sort_values("continuous_ranker_mean_regret", ascending=False)
        .head(top_n)["site_id"]
        .astype(str)
        .tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", default=DEFAULT_SITES)
    parser.add_argument("--failure-diagnostic", default=str(DEFAULT_FAILURE_DIAG))
    parser.add_argument("--top-failure-sites", type=int, default=0)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURES))
    parser.add_argument("--date-start", default=DEFAULT_START)
    parser.add_argument("--date-end", default=DEFAULT_END)
    parser.add_argument("--date-step-days", type=int, default=1)
    parser.add_argument("--ir-min", type=float, default=0.0)
    parser.add_argument("--ir-max", type=float, default=60.0)
    parser.add_argument("--samples-per-site-date", type=int, default=31)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    sites = parse_sites(args.sites)
    failure_sites = load_failure_sites(Path(args.failure_diagnostic), args.top_failure_sites)
    if failure_sites:
        sites = list(dict.fromkeys(failure_sites + sites))
    dates = date_range(args.date_start, args.date_end, args.date_step_days)
    site_caps = read_site_caps(Path(args.site_feature_csv), args.ir_max)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "tta_date_coverage_sampling_plan_v1.csv"
    summary_path = out_dir / "tta_date_coverage_sampling_plan_v1.md"

    rows = []
    for site in sites:
        site_ir_max = float(site_caps.get(site, args.ir_max))
        if site_ir_max <= args.ir_min:
            raise ValueError(f"Irrigation max for {site} must be larger than --ir-min")
        irrigation_values = uniform_values(args.ir_min, site_ir_max, args.samples_per_site_date)
        for date_t, doy in dates:
            for idx, irrigation_mm in enumerate(irrigation_values):
                rows.append(
                    {
                        "sample_id": f"{site}_{date_t.replace('-', '').lower()}_tta_{idx:04d}",
                        "site_id": site,
                        "date_t": date_t,
                        "decision_doy": doy,
                        "horizon_days": int(args.horizon_days),
                        "irrigation_mm": float(irrigation_mm),
                        "ir_min": float(args.ir_min),
                        "ir_max": site_ir_max,
                        "samples_per_site_date": int(args.samples_per_site_date),
                        "sampling_method": "tta_dense_date_coverage_uniform_ir",
                    }
                )

    with plan_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_site = pd.DataFrame(rows).groupby("site_id").agg(
        rows=("sample_id", "count"),
        dates=("date_t", "nunique"),
        ir_min=("irrigation_mm", "min"),
        ir_max=("irrigation_mm", "max"),
    ).reset_index()
    summary = pd.DataFrame(
        [
            {
                "sites": len(sites),
                "date_start": args.date_start,
                "date_end": args.date_end,
                "date_step_days": int(args.date_step_days),
                "decision_dates": len(dates),
                "samples_per_site_date": int(args.samples_per_site_date),
                "rows": len(rows),
                "horizon_days": int(args.horizon_days),
                "plan": str(plan_path),
            }
        ]
    )
    lines = [
        "# TTA Date Coverage Sampling Plan V1",
        "",
        "## Purpose",
        "",
        "Add SWAP-labeled date/state coverage before MoE. This plan targets sites and dates where per-site experts failed held-out-date CV.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Notes",
        "",
        "- This is a sampling plan only; run the existing 12-site restart generator with this plan.",
        "- The default sites are the worst held-out-date ranker sites: code_B1, code_C2, code_N1, code_N2.",
        "- Use this expanded data to test rolling/few-shot/TTA-style updates before any MoE stage.",
        "",
        "## Output",
        "",
        f"- `{plan_path}`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("TTA date coverage sampling plan v1")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print(f"plan: {plan_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
