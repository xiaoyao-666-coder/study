#!/usr/bin/env python3
"""Build old-year sampling plans aligned to paper cached schedule DOYs.

The plan uses the public `day_scheduled.csv` dates only to recover the paper
schedule decision DOYs, then maps those DOYs into older years. It keeps the
paper fixed irrigation candidate list.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_PAPER_SCHEDULE = Path("model3_opt_sto_upload") / "Maize" / "day_scheduled.csv"
DEFAULT_OUT_ROOT = Path("site_general_surrogate_eval") / "older_year_swap_label_generation_v1"
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019]
DEFAULT_SITES = ["code_C2", "code_N1", "code_N2", "code_N4"]
PAPER_IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-schedule", default=str(DEFAULT_PAPER_SCHEDULE))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--irrigation-options", nargs="+", type=float, default=PAPER_IRRIGATION_OPTIONS_MM)
    parser.add_argument(
        "--alignment",
        choices=["doy", "calendar"],
        default="doy",
        help=(
            "doy: preserve the paper schedule day-of-year exactly; "
            "calendar: preserve month-day labels from day_scheduled.csv."
        ),
    )
    parser.add_argument("--plan-dir-template", default="failure_sites_{year}_paper_schedule_doy_aligned_plan_v1")
    return parser.parse_args()


def parse_paper_schedule(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing paper schedule CSV: {path}")
    schedule = pd.read_csv(path)
    if "date_t" not in schedule.columns:
        raise ValueError(f"{path} is missing required column: date_t")
    out = schedule[["date_t"]].drop_duplicates().copy()
    out["paper_date"] = pd.to_datetime(out["date_t"], format="%d-%b-%Y", errors="raise")
    out["paper_decision_doy"] = out["paper_date"].dt.dayofyear.astype(int)
    out["paper_month"] = out["paper_date"].dt.month.astype(int)
    out["paper_day"] = out["paper_date"].dt.day.astype(int)
    return out.sort_values("paper_decision_doy").reset_index(drop=True)


def doy_to_date_label(year: int, doy: int) -> str:
    return datetime.strptime(f"{year}-{doy}", "%Y-%j").strftime("%d-%b-%Y")


def calendar_to_date_label(year: int, month: int, day: int) -> tuple[str, int]:
    date = datetime(year, month, day)
    return date.strftime("%d-%b-%Y"), int(date.strftime("%j"))


def build_year_plan(
    *,
    year: int,
    paper_schedule: pd.DataFrame,
    sites: list[str],
    irrigation_options: list[float],
    alignment: str,
) -> pd.DataFrame:
    rows = []
    for sched in paper_schedule.itertuples(index=False):
        if alignment == "doy":
            decision_doy = int(sched.paper_decision_doy)
            date_t = doy_to_date_label(year, decision_doy)
        else:
            date_t, decision_doy = calendar_to_date_label(year, int(sched.paper_month), int(sched.paper_day))
        for site in sites:
            for ir in irrigation_options:
                rows.append(
                    {
                        "site_id": site,
                        "year": year,
                        "date_t": date_t,
                        "decision_doy": decision_doy,
                        "irrigation_mm": float(ir),
                        "paper_schedule_date_t": sched.date_t,
                        "paper_schedule_decision_doy": int(sched.paper_decision_doy),
                        "alignment": alignment,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    paper_schedule = parse_paper_schedule(Path(args.paper_schedule))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for year in args.years:
        plan = build_year_plan(
            year=year,
            paper_schedule=paper_schedule,
            sites=args.sites,
            irrigation_options=args.irrigation_options,
            alignment=args.alignment,
        )
        plan_dir = out_root / args.plan_dir_template.format(year=year)
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "tta_date_coverage_sampling_plan_v1.csv"
        plan.to_csv(plan_path, index=False)
        manifest_rows.append(
            {
                "year": year,
                "alignment": args.alignment,
                "sites": len(args.sites),
                "dates": plan[["date_t", "decision_doy"]].drop_duplicates().shape[0],
                "irrigation_options": len(args.irrigation_options),
                "rows": len(plan),
                "plan_path": str(plan_path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / f"paper_schedule_{args.alignment}_aligned_plan_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False)

    schedule_path = out_root / f"paper_schedule_{args.alignment}_aligned_decision_dates_v1.csv"
    paper_schedule.to_csv(schedule_path, index=False)

    report = out_root / f"paper_schedule_{args.alignment}_aligned_plan_manifest_v1.md"
    lines = [
        f"# Paper Schedule {args.alignment.upper()}-Aligned Old-Year Sampling Plans",
        "",
        f"- paper schedule source: `{args.paper_schedule}`",
        f"- alignment: `{args.alignment}`",
        f"- irrigation options: `{[float(v) for v in args.irrigation_options]}`",
        f"- sites: `{args.sites}`",
        "",
        "The `doy` alignment preserves the paper schedule day-of-year exactly. For non-leap years this can shift the calendar month-day relative to 2024.",
        "The `calendar` alignment preserves month-day and lets DOY vary by leap year.",
        "",
        "## Paper Schedule Dates",
        markdown_table(paper_schedule[["date_t", "paper_decision_doy", "paper_month", "paper_day"]]),
        "",
        "## Generated Plans",
        markdown_table(manifest),
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote:")
    print(manifest_path)
    print(schedule_path)
    print(report)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
