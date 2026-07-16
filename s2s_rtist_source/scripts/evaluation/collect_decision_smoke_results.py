"""Collect candidate_result.csv files from a parallel smoke-test run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]
YIELD_PRICE_PER_KG = 0.20
WATER_COST_PER_HA_PER_MM = 2.0
WEIGHT_INDEX = 0.7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory, e.g. decision_smoke_parallel_20260530_103455")
    parser.add_argument("--out", default="decision_smoke_8ir_parallel_latest.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rows = []
    missing = []
    for ir in IRRIGATION_OPTIONS_MM:
        result_path = run_dir / f"ir_{ir}" / "candidate_result.csv"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        rows.append(pd.read_csv(result_path))

    if missing:
        raise FileNotFoundError("Missing candidate result files:\n" + "\n".join(missing))

    out = pd.concat(rows, ignore_index=True).sort_values("ir")
    cwdm_ir0 = float(out.loc[out["ir"] == 0, "cwdm_value"].iloc[0])
    out["target_value"] = (
        (out["cwdm_value"] - cwdm_ir0) * YIELD_PRICE_PER_KG
        - out["ir"] * WATER_COST_PER_HA_PER_MM * WEIGHT_INDEX
    )
    out.loc[out["ir"] == 0, "target_value"] = 0.0
    out = out[
        [
            "date_t",
            "ir",
            "end_daynr",
            "dvs",
            "cwdm_value",
            "cwso_value",
            "target_value",
        ]
    ]

    out_path = Path(args.out).resolve()
    out.to_csv(out_path, index=False)
    run_out_path = run_dir / "decision_smoke_8ir_parallel.csv"
    out.to_csv(run_out_path, index=False)

    print(out.to_string(index=False))
    print("\nbest candidate:")
    print(out.loc[out["target_value"].idxmax()].to_string())
    print(f"\nwrote {out_path}")
    print(f"wrote {run_out_path}")


if __name__ == "__main__":
    main()
