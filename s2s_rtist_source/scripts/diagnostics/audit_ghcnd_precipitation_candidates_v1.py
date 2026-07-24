#!/usr/bin/env python3
"""Audit GHCN-Daily candidate precipitation completeness on required dates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CYCLE_MONTH_DAYS = ("06-01", "06-15", "07-01", "07-15", "08-01", "08-15")
GHCND_COLUMNS = (
    "station_id",
    "date",
    "element",
    "value_tenths_mm",
    "m_flag",
    "q_flag",
    "s_flag",
    "observation_time",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_dates(first_year: int, last_year: int) -> pd.DatetimeIndex:
    dates = []
    for year in range(int(first_year), int(last_year) + 1):
        for month_day in CYCLE_MONTH_DAYS:
            cycle = pd.Timestamp(f"{year}-{month_day}")
            dates.extend(cycle + pd.Timedelta(days=offset) for offset in range(7))
    index = pd.DatetimeIndex(sorted(set(dates)))
    expected = (int(last_year) - int(first_year) + 1) * len(CYCLE_MONTH_DAYS) * 7
    if len(index) != expected:
        raise ValueError("required GHCN target dates overlap unexpectedly")
    return index


def longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def read_station_prcp(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=GHCND_COLUMNS,
        dtype={
            "station_id": "string",
            "date": "string",
            "element": "string",
            "m_flag": "string",
            "q_flag": "string",
            "s_flag": "string",
            "observation_time": "string",
        },
    )
    frame = frame.loc[frame["element"] == "PRCP"].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    frame["precipitation_mm"] = (
        pd.to_numeric(frame["value_tenths_mm"], errors="raise") * 0.1
    )
    if (frame["precipitation_mm"] < 0.0).any():
        raise ValueError(f"negative GHCN PRCP value in {path}")
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError(f"duplicate GHCN PRCP station-date in {path}")
    return frame


def audit_candidate(
    candidate: pd.Series,
    *,
    station_file: Path,
    target_dates: pd.DatetimeIndex,
    minimum_overall: float,
    minimum_annual: float,
) -> dict[str, object]:
    station = read_station_prcp(station_file)
    station = station.set_index("date").reindex(target_dates)
    present = station["precipitation_mm"].notna()
    q_flag_blank = station["q_flag"].fillna("").str.strip().eq("")
    unflagged = present & q_flag_blank
    annual = pd.DataFrame({"unflagged": unflagged}, index=target_dates)
    annual_rates = annual.groupby(annual.index.year)["unflagged"].mean()
    overall_rate = float(unflagged.mean())
    minimum_annual_rate = float(annual_rates.min())
    return {
        "project_site_id": str(candidate["project_site_id"]),
        "candidate_rank_by_distance": int(candidate["candidate_rank_by_distance"]),
        "station_id": str(candidate["station_id"]),
        "station_name": str(candidate["station_name"]),
        "state": str(candidate["state"]),
        "distance_km": float(candidate["distance_km"]),
        "required_target_date_count": int(len(target_dates)),
        "present_target_date_count": int(present.sum()),
        "quality_flagged_target_date_count": int((present & ~q_flag_blank).sum()),
        "unflagged_target_date_count": int(unflagged.sum()),
        "overall_unflagged_completeness": overall_rate,
        "minimum_annual_unflagged_completeness": minimum_annual_rate,
        "mean_annual_unflagged_completeness": float(annual_rates.mean()),
        "longest_missing_target_slot_run": int(longest_false_run(unflagged.to_numpy())),
        "trace_measurement_count": int(
            (present & station["m_flag"].fillna("").str.strip().eq("T")).sum()
        ),
        "observation_time_present_count": int(
            (present & station["observation_time"].notna()).sum()
        ),
        "passes_overall_completeness": overall_rate >= float(minimum_overall),
        "passes_annual_completeness": minimum_annual_rate >= float(minimum_annual),
        "eligible_primary_station": (
            overall_rate >= float(minimum_overall)
            and minimum_annual_rate >= float(minimum_annual)
        ),
        "station_file": str(station_file.resolve()),
        "station_file_bytes": int(station_file.stat().st_size),
        "station_file_sha256": sha256_file(station_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--station-files-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--first-year", type=int, default=2000)
    parser.add_argument("--last-year", type=int, default=2019)
    parser.add_argument("--project-sites", nargs="+")
    parser.add_argument("--minimum-candidate-rank", type=int, default=1)
    parser.add_argument("--maximum-candidate-rank", type=int, default=3)
    parser.add_argument("--stop-after-first-eligible", action="store_true")
    parser.add_argument("--minimum-overall", type=float, default=0.95)
    parser.add_argument("--minimum-annual", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    if args.project_sites:
        candidates = candidates.loc[
            candidates["project_site_id"].isin(args.project_sites)
        ].copy()
    candidates = candidates.loc[
        (candidates["candidate_rank_by_distance"] >= args.minimum_candidate_rank)
        & (candidates["candidate_rank_by_distance"] <= args.maximum_candidate_rank)
    ].copy()
    target_dates = required_dates(args.first_year, args.last_year)
    rows = []
    selected_sites: set[str] = set()
    for candidate in candidates.sort_values(
        ["project_site_id", "candidate_rank_by_distance"]
    ).itertuples(index=False):
        if args.stop_after_first_eligible and candidate.project_site_id in selected_sites:
            continue
        candidate_series = pd.Series(candidate._asdict())
        station_file = args.station_files_dir / f"{candidate.station_id}.csv.gz"
        if not station_file.is_file():
            raise FileNotFoundError(f"missing candidate station file: {station_file}")
        result = audit_candidate(
            candidate_series,
            station_file=station_file,
            target_dates=target_dates,
            minimum_overall=args.minimum_overall,
            minimum_annual=args.minimum_annual,
        )
        rows.append(result)
        if bool(result["eligible_primary_station"]):
            selected_sites.add(str(candidate.project_site_id))
    audit = pd.DataFrame(rows).sort_values(
        ["project_site_id", "candidate_rank_by_distance"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "ghcnd_candidate_prcp_completeness_v1.csv"
    selection_path = args.output_dir / "ghcnd_primary_station_selection_v1.json"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    selection: dict[str, object] = {
        "contract_id": "ghcnd-primary-precipitation-station-selection-v1",
        "required_years": [args.first_year, args.last_year],
        "required_target_date_count_per_site": int(len(target_dates)),
        "minimum_overall_unflagged_completeness": args.minimum_overall,
        "minimum_annual_unflagged_completeness": args.minimum_annual,
        "audited_candidate_rank_maximum": args.maximum_candidate_rank,
        "sites": {},
    }
    for site_id, group in audit.groupby("project_site_id", sort=True):
        eligible = group.loc[group["eligible_primary_station"]].sort_values(
            ["distance_km", "station_id"]
        )
        if eligible.empty:
            site_selection = {
                "status": "no_eligible_station_in_audited_candidates",
                "next_action": "download_and_audit_candidate_ranks_4_to_10",
            }
        else:
            selected = eligible.iloc[0]
            site_selection = {
                "status": "eligible_primary_station_selected",
                "candidate_rank_by_distance": int(
                    selected["candidate_rank_by_distance"]
                ),
                "station_id": str(selected["station_id"]),
                "station_name": str(selected["station_name"]),
                "state": str(selected["state"]),
                "distance_km": float(selected["distance_km"]),
                "overall_unflagged_completeness": float(
                    selected["overall_unflagged_completeness"]
                ),
                "minimum_annual_unflagged_completeness": float(
                    selected["minimum_annual_unflagged_completeness"]
                ),
            }
        site_selection["audited_candidate_count"] = int(len(group))
        selection["sites"][str(site_id)] = site_selection
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit": str(audit_path), "selection": str(selection_path)}, indent=2))


if __name__ == "__main__":
    main()
