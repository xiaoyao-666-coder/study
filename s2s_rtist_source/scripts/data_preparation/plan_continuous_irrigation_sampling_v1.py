#!/usr/bin/env python3
"""Create an offline SWAP sampling plan for continuous irrigation surrogate data.

This script does not run SWAP. It creates a deterministic site/date/irrigation
plan that can be used by the server-side restart runner to generate dense
training samples for a continuous-input surrogate model.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_SITE_FEATURES = Path("site_general_surrogate_eval") / "site_feature_screening_12_code_sites.csv"
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "continuous_irrigation_sampling_v1"
DEFAULT_DECISION_DATES = [
    "01-Jul-2024:183",
    "03-Jul-2024:185",
    "05-Jul-2024:187",
    "07-Jul-2024:189",
    "09-Jul-2024:191",
    "11-Jul-2024:193",
    "13-Jul-2024:195",
    "15-Jul-2024:197",
    "17-Jul-2024:199",
    "19-Jul-2024:201",
    "21-Jul-2024:203",
    "24-Jul-2024:206",
]


def parse_decision_date(raw: str) -> tuple[str, int]:
    if ":" not in raw:
        raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
    date_t, doy_text = raw.split(":", 1)
    if not doy_text.isdigit():
        raise ValueError(f"Decision date must be DATE:DOY, got {raw!r}")
    doy = int(doy_text)
    actual = datetime.strptime(date_t, "%d-%b-%Y").timetuple().tm_yday
    if actual != doy:
        raise ValueError(f"{raw} has mismatched DOY: {date_t} is DOY {actual}, got {doy}")
    return date_t, doy


def format_decision_date(dt: datetime) -> tuple[str, int]:
    return dt.strftime("%d-%b-%Y"), int(dt.timetuple().tm_yday)


def date_range(start: str, end: str, step_days: int) -> list[tuple[str, int]]:
    if step_days <= 0:
        raise ValueError("--decision-date-step-days must be positive")
    start_dt = datetime.strptime(start, "%d-%b-%Y")
    end_dt = datetime.strptime(end, "%d-%b-%Y")
    if end_dt < start_dt:
        raise ValueError("--decision-date-end must not be earlier than --decision-date-start")
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(format_decision_date(cur))
        cur += timedelta(days=step_days)
    return out


def read_sites(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "site" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'site' column")
        sites = [str(row["site"]) for row in reader if str(row.get("site", "")).strip()]
    return sorted(dict.fromkeys(sites))


def uniform_values(ir_min: float, ir_max: float, n: int) -> list[float]:
    if n <= 1:
        return [float(ir_min)]
    step = (ir_max - ir_min) / float(n - 1)
    return [round(ir_min + i * step, 6) for i in range(n)]


def parse_site_float_overrides(raw_values: list[str] | None, option_name: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for raw in raw_values or []:
        if "=" not in raw:
            raise ValueError(f"{option_name} values must be SITE=VALUE, got {raw!r}")
        site, value = raw.split("=", 1)
        site = site.strip()
        if not site:
            raise ValueError(f"{option_name} has an empty site in {raw!r}")
        overrides[site] = float(value)
    return overrides


def write_markdown_summary(
    path: Path,
    plan_path: Path,
    sites: list[str],
    dates: list[tuple[str, int]],
    samples_per_site_date: int,
    ir_min: float,
    ir_max: float,
    site_ir_max: dict[str, float],
    total_rows: int,
) -> None:
    override_lines = (
        [f"- {site}: max {value} mm" for site, value in sorted(site_ir_max.items())]
        if site_ir_max
        else ["- none"]
    )
    lines = [
        "# Continuous Irrigation Sampling Plan V1",
        "",
        "## Purpose",
        "",
        "Generate dense offline SWAP runs for a continuous-input surrogate model.",
        "",
        "## Scope",
        "",
        f"- Sites: {len(sites)}",
        f"- Decision dates: {len(dates)}",
        f"- Samples per site-date: {samples_per_site_date}",
        f"- Irrigation range: {ir_min} to {ir_max} mm",
        "- Site-specific max irrigation overrides:",
        *override_lines,
        f"- Total planned rows: {total_rows}",
        "",
        "## Output",
        "",
        f"- `{plan_path}`",
        "",
        "## Notes",
        "",
        "- This is a sampling plan only; it does not run SWAP.",
        "- The resulting table should be consumed by a restart-based SWAP runner.",
        "- Use the generated SWAP outputs to train a surrogate with irrigation amount as a continuous input.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURES))
    parser.add_argument("--sites", nargs="+", help="Optional explicit site list. Defaults to the site column in --site-feature-csv.")
    parser.add_argument("--decision-dates", nargs="+", default=DEFAULT_DECISION_DATES)
    parser.add_argument("--decision-date-start", help="Optional start date like 01-Jun-2024. Overrides --decision-dates when paired with --decision-date-end.")
    parser.add_argument("--decision-date-end", help="Optional end date like 31-Jul-2024. Overrides --decision-dates when paired with --decision-date-start.")
    parser.add_argument("--decision-date-step-days", type=int, default=2)
    parser.add_argument("--target-samples", type=int, default=10000)
    parser.add_argument("--samples-per-site-date", type=int, help="Override computed samples per site-date.")
    parser.add_argument("--ir-min", type=float, default=0.0)
    parser.add_argument("--ir-max", type=float, default=60.0)
    parser.add_argument(
        "--site-ir-max",
        nargs="+",
        help="Optional per-site max irrigation overrides, e.g. code_C1=15.",
    )
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ir_max <= args.ir_min:
        raise ValueError("--ir-max must be larger than --ir-min")

    sites = args.sites if args.sites else read_sites(Path(args.site_feature_csv))
    if not sites:
        raise ValueError("No sites provided")
    if args.decision_date_start or args.decision_date_end:
        if not args.decision_date_start or not args.decision_date_end:
            raise ValueError("--decision-date-start and --decision-date-end must be provided together")
        dates = date_range(args.decision_date_start, args.decision_date_end, args.decision_date_step_days)
    else:
        dates = [parse_decision_date(raw) for raw in args.decision_dates]
    if not dates:
        raise ValueError("No decision dates provided")

    groups = len(sites) * len(dates)
    samples_per_site_date = args.samples_per_site_date or int(math.ceil(args.target_samples / groups))
    site_ir_max = parse_site_float_overrides(args.site_ir_max, "--site-ir-max")
    missing_override_sites = sorted(set(site_ir_max).difference(sites))
    if missing_override_sites:
        raise ValueError(f"--site-ir-max references sites not in the plan: {missing_override_sites}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "continuous_irrigation_sampling_plan_v1.csv"
    summary_path = out_dir / "continuous_irrigation_sampling_plan_v1.md"

    rows = []
    for site in sites:
        site_ir_max_value = site_ir_max.get(site, args.ir_max)
        if site_ir_max_value <= args.ir_min:
            raise ValueError(f"Irrigation max for {site} must be larger than --ir-min")
        irrigation_values = uniform_values(args.ir_min, site_ir_max_value, samples_per_site_date)
        for date_t, doy in dates:
            for idx, irrigation_mm in enumerate(irrigation_values):
                rows.append(
                    {
                        "sample_id": f"{site}_{date_t.replace('-', '').lower()}_{idx:04d}",
                        "site_id": site,
                        "date_t": date_t,
                        "decision_doy": doy,
                        "horizon_days": args.horizon_days,
                        "irrigation_mm": irrigation_mm,
                        "ir_min": args.ir_min,
                        "ir_max": site_ir_max_value,
                        "samples_per_site_date": samples_per_site_date,
                        "sampling_method": "deterministic_uniform_grid",
                    }
                )

    with plan_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_markdown_summary(
        summary_path,
        plan_path,
        sites,
        dates,
        samples_per_site_date,
        args.ir_min,
        args.ir_max,
        site_ir_max,
        len(rows),
    )

    print("Continuous irrigation sampling plan v1")
    print(f"sites: {len(sites)}")
    print(f"decision_dates: {len(dates)}")
    print(f"samples_per_site_date: {samples_per_site_date}")
    print(f"rows: {len(rows)}")
    print(f"plan: {plan_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
