#!/usr/bin/env python3
"""Diagnose GEFS UTC-day versus GHCN station-record date alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_ensemble_validation import _ensemble_crps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBER_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
    / "gefs_reforecast_member_daily_precipitation_utc_v2.csv"
)
DEFAULT_SELECTION_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "station_quality_audit_final"
    / "ghcnd_primary_station_selection_v1.json"
)
DEFAULT_STATION_FILES_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "candidate_station_files"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_ghcnd_day_alignment_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-file", type=Path, default=DEFAULT_MEMBER_FILE)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    parser.add_argument(
        "--station-files-dir", type=Path, default=DEFAULT_STATION_FILES_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_selected_station_reference(
    selection_file: Path, station_files_dir: Path
) -> pd.DataFrame:
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    columns = (
        "station_id",
        "date",
        "element",
        "value_tenths_mm",
        "m_flag",
        "q_flag",
        "s_flag",
        "observation_time",
    )
    parts = []
    for site_id, selected in sorted(selection["sites"].items()):
        station_id = str(selected["station_id"])
        station = pd.read_csv(
            station_files_dir / f"{station_id}.csv.gz",
            compression="gzip",
            header=None,
            names=columns,
            dtype={"date": "string", "element": "string", "q_flag": "string"},
        )
        station = station.loc[station["element"].eq("PRCP")].copy()
        station["station_record_date"] = pd.to_datetime(
            station["date"], format="%Y%m%d"
        )
        station["precipitation_mm_reference"] = (
            pd.to_numeric(station["value_tenths_mm"], errors="raise") * 0.1
        )
        station["reference_valid_unflagged"] = station["q_flag"].fillna("").str.strip().eq("")
        station["site_id"] = site_id
        station["ghcnd_station_id"] = station_id
        parts.append(
            station[
                [
                    "site_id",
                    "ghcnd_station_id",
                    "station_record_date",
                    "precipitation_mm_reference",
                    "reference_valid_unflagged",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True)


def pair(member: pd.DataFrame, reference: pd.DataFrame, offset_days: int) -> pd.DataFrame:
    reference = reference.loc[reference["reference_valid_unflagged"]].copy()
    reference["station_record_date"] = pd.to_datetime(reference["station_record_date"])
    reference["valid_date_utc"] = reference["station_record_date"] - pd.Timedelta(
        days=int(offset_days)
    )
    columns = [
        "site_id",
        "valid_date_utc",
        "ghcnd_station_id",
        "station_record_date",
        "precipitation_mm_reference",
    ]
    result = member.merge(
        reference[columns],
        on=["site_id", "valid_date_utc"],
        how="left",
        validate="many_to_one",
    )
    result["station_date_offset_from_gefs_valid_date_days"] = int(offset_days)
    return result


def daily_metrics(paired: pd.DataFrame, split: str, offset_days: int) -> pd.DataFrame:
    valid = paired.loc[paired["precipitation_mm_reference"].notna()].copy()
    keys = ["site_id", "forecast_init_utc", "valid_date_utc"]
    rows = []
    for key, group in valid.groupby(keys, sort=True):
        raw = group["precipitation_mm_raw"].to_numpy(dtype=float)
        reference = float(group["precipitation_mm_reference"].iloc[0])
        rows.append(
            {
                "site_id": key[0],
                "forecast_init_utc": key[1],
                "valid_date_utc": key[2],
                "reference": reference,
                "ensemble_mean": float(raw.mean()),
                "crps": float(_ensemble_crps(raw, reference)),
                "covered_p10_p90": bool(
                    np.quantile(raw, 0.1) <= reference <= np.quantile(raw, 0.9)
                ),
                "covered_min_max": bool(raw.min() <= reference <= raw.max()),
            }
        )
    observations = pd.DataFrame(rows)
    output_rows = []
    scopes = [("pooled", observations)] + [
        (site_id, group) for site_id, group in observations.groupby("site_id", sort=True)
    ]
    for scope, group in scopes:
        error = group["ensemble_mean"] - group["reference"]
        output_rows.append(
            {
                "split": split,
                "station_date_offset_from_gefs_valid_date_days": offset_days,
                "scope": scope,
                "paired_daily_observation_count": int(len(group)),
                "ensemble_mean_bias_mm": float(error.mean()),
                "ensemble_mean_mae_mm": float(error.abs().mean()),
                "ensemble_mean_rmse_mm": float(np.sqrt(np.mean(error**2))),
                "mean_crps_mm": float(group["crps"].mean()),
                "p10_p90_coverage": float(group["covered_p10_p90"].mean()),
                "min_max_coverage": float(group["covered_min_max"].mean()),
            }
        )
    return pd.DataFrame(output_rows)


def seven_day_metrics(paired: pd.DataFrame, split: str, offset_days: int) -> dict[str, object]:
    valid = paired.loc[paired["precipitation_mm_reference"].notna()].copy()
    member_sums = valid.groupby(
        ["site_id", "forecast_init_utc", "gefs_member"], as_index=False
    ).agg(
        forecast_7d_mm=("precipitation_mm_raw", "sum"),
        valid_days=("valid_date_utc", "nunique"),
    )
    reference_sums = valid.drop_duplicates(
        ["site_id", "forecast_init_utc", "valid_date_utc"]
    ).groupby(["site_id", "forecast_init_utc"], as_index=False).agg(
        reference_7d_mm=("precipitation_mm_reference", "sum"),
        valid_days=("valid_date_utc", "nunique"),
    )
    complete = reference_sums.loc[reference_sums["valid_days"].eq(7)].copy()
    member_sums = member_sums.loc[member_sums["valid_days"].eq(7)]
    ensemble = member_sums.groupby(
        ["site_id", "forecast_init_utc"], as_index=False
    )["forecast_7d_mm"].mean()
    joined = complete.merge(
        ensemble,
        on=["site_id", "forecast_init_utc"],
        how="inner",
        validate="one_to_one",
    )
    error = joined["forecast_7d_mm"] - joined["reference_7d_mm"]
    return {
        "split": split,
        "station_date_offset_from_gefs_valid_date_days": offset_days,
        "complete_site_cycle_count": int(len(joined)),
        "seven_day_bias_mm": float(error.mean()),
        "seven_day_mae_mm": float(error.abs().mean()),
        "seven_day_rmse_mm": float(np.sqrt(np.mean(error**2))),
    }


def run(
    member_file: Path,
    selection_file: Path,
    station_files_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    member = pd.read_csv(member_file)
    member["forecast_init_utc"] = pd.to_datetime(member["forecast_init_utc"], utc=True)
    member["decision_date"] = pd.to_datetime(member["decision_date"])
    member["valid_date_utc"] = pd.to_datetime(member["valid_date_utc"])
    reference = load_selected_station_reference(selection_file, station_files_dir)
    daily_parts = []
    seven_rows = []
    for offset in (-1, 0, 1):
        paired = pair(member, reference, offset)
        for split, years in (("training_2015_2018", range(2015, 2019)), ("validation_2019", (2019,))):
            subset = paired.loc[paired["decision_date"].dt.year.isin(years)].copy()
            daily_parts.append(daily_metrics(subset, split, offset))
            seven_rows.append(seven_day_metrics(subset, split, offset))
    daily = pd.concat(daily_parts, ignore_index=True)
    seven = pd.DataFrame(seven_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "gefs_ghcnd_day_alignment_daily_metrics_v1.csv"
    seven_path = output_dir / "gefs_ghcnd_day_alignment_seven_day_metrics_v1.csv"
    report_path = output_dir / "gefs_ghcnd_day_alignment_conclusion_v1.md"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    seven.to_csv(seven_path, index=False, encoding="utf-8-sig")
    pooled = daily.loc[daily["scope"].eq("pooled")]
    report = [
        "# GEFS UTC 日与 GHCN-D 站点记录日对齐诊断",
        "",
        "偏移定义：`station_record_date = GEFS valid_date_utc + offset_days`。",
        "本诊断只报告敏感性，不自动选择偏移，也不把站点记录日声明为 UTC 日。",
        "",
        "| 分割 | 偏移 | 日MAE | CRPS | P10-P90覆盖率 | min-max覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        report.append(
            f"| {row.split} | {row.station_date_offset_from_gefs_valid_date_days:+d} | "
            f"{row.ensemble_mean_mae_mm:.4f} | {row.mean_crps_mm:.4f} | "
            f"{row.p10_p90_coverage:.4f} | {row.min_max_coverage:.4f} |"
        )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    return {"daily": daily_path, "seven_day": seven_path, "report": report_path}


def main() -> None:
    args = parse_args()
    outputs = run(
        args.member_file,
        args.selection_file,
        args.station_files_dir,
        args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
