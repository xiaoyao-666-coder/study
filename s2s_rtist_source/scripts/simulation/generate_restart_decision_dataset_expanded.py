#!/usr/bin/env python3
"""Generate an expanded restart decision dataset on a denser date grid.

Run inside the Maize/SWAP experiment directory that already contains
generate_restart_decision_dataset.py, swap_test, ForecastStep.py and inputs.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import generate_restart_decision_dataset as base


def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%d-%b-%Y")


def format_date(dt: datetime) -> str:
    return dt.strftime("%d-%b-%Y")


def date_to_doy(dt: datetime) -> int:
    return int(dt.strftime("%j"))


def make_decision_dates(start: str, end: str, step_days: int) -> list[tuple[str, int]]:
    cur = parse_date(start)
    last = parse_date(end)
    out = []
    while cur <= last:
        out.append((format_date(cur), date_to_doy(cur)))
        cur += timedelta(days=step_days)
    return out


def run_one_date_expanded(date_t: str, decision_doy: int, output_prefix: str) -> pd.DataFrame:
    label = base.safe_label(date_t)
    end_doy = decision_doy + base.HORIZON_DAYS

    print(f"\n=== {date_t}: pre-decision state ===", flush=True)
    base.configure_irrigation(date_t, None)
    base.run_pre_state(f"{output_prefix}_{label}_pre.log", decision_doy)
    shutil.copyfile("result_forec.end", "restart_initial.end")
    shutil.copyfile("result_forec.end", f"restart_initial_{label}.end")
    shutil.copyfile("result_forec.end", f"result_pre_{label}.end")
    shutil.copyfile("result_forec.crp", f"result_pre_{label}.crp")

    rows = []
    for ir in base.IRRIGATION_OPTIONS_MM:
        print(f"{date_t}: running restart candidate {ir} mm", flush=True)
        base.configure_irrigation(date_t, ir)
        base.set_swp_for_restart(decision_doy, end_doy, outfil="result_restart")
        base.run_swap(f"{output_prefix}_{label}_restart_ir_{ir}.log")
        rows.append(
            {
                "date_t": date_t,
                "decision_doy": decision_doy,
                "horizon_end_doy": end_doy,
                "ir": ir,
                **base.read_last("result_restart.crp"),
            }
        )

    scored = base.score_one_date(rows)
    scored.to_csv(f"{output_prefix}_{label}.csv", index=False)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="16-Jul-2024")
    parser.add_argument("--end", default="21-Aug-2024")
    parser.add_argument("--step-days", type=int, default=2)
    parser.add_argument("--output-prefix", default="restart_decision_dataset_expanded")
    args = parser.parse_args()

    if not Path("swap_test").exists():
        raise FileNotFoundError("Run inside a copied Maize/SWAP directory containing swap_test.")
    if not Path("generate_restart_decision_dataset.py").exists():
        raise FileNotFoundError("This wrapper expects generate_restart_decision_dataset.py in the same directory.")

    dates = make_decision_dates(args.start, args.end, args.step_days)
    print("Expanded decision dates:", flush=True)
    for date_t, doy in dates:
        print(f"  {date_t} DOY={doy}", flush=True)

    all_rows = []
    for i, (date_t, decision_doy) in enumerate(dates, start=1):
        print(f"\n[{i}/{len(dates)}] processing {date_t}", flush=True)
        all_rows.append(run_one_date_expanded(date_t, decision_doy, args.output_prefix))

    dataset = pd.concat(all_rows, ignore_index=True)
    best = dataset[dataset["is_best_ir"]][
        ["date_t", "decision_doy", "best_ir_for_date", "best_target_for_date"]
    ].drop_duplicates()

    dataset_path = f"{args.output_prefix}.csv"
    best_path = f"{args.output_prefix}_best_by_date.csv"
    dataset.to_csv(dataset_path, index=False)
    best.to_csv(best_path, index=False)

    print(f"\nwrote {dataset_path}", flush=True)
    print(f"wrote {best_path}", flush=True)
    print(f"rows: {len(dataset)}", flush=True)
    print(f"decision dates: {len(dates)}", flush=True)


if __name__ == "__main__":
    main()
