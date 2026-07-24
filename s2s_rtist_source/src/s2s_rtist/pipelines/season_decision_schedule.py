"""Build leakage-safe decision dates from a full-season SWAP crop trajectory."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from s2s_rtist.labels.swap_three_output_labels import CRP_COLUMNS


def read_crop_trajectory(path: str | Path) -> pd.DataFrame:
    """Read daily crop states from a SWAP ``result.crp`` output."""
    source = Path(path)
    rows: list[list[str]] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not re.match(r"^\s*\d{4}-\d{2}-\d{2}", line):
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(CRP_COLUMNS):
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No complete crop rows found in {source}")

    frame = pd.DataFrame(rows, columns=CRP_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["DVS"] = pd.to_numeric(frame["DVS"], errors="coerce")
    if frame["DVS"].isna().any():
        raise ValueError(f"Crop trajectory contains non-numeric DVS values: {source}")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise ValueError(f"Crop trajectory contains duplicate dates: {source}")

    expected = pd.date_range(frame["Date"].iloc[0], frame["Date"].iloc[-1], freq="D")
    if len(frame) != len(expected) or not frame["Date"].reset_index(drop=True).equals(
        pd.Series(expected)
    ):
        raise ValueError(f"Crop trajectory is not daily-contiguous: {source}")
    return frame


def build_decision_schedule(
    crop: pd.DataFrame,
    *,
    site_id: str,
    target_year: int,
    split: str,
    dvs_threshold: float = 0.1,
    interval_days: int = 7,
    horizon_days: int = 7,
    harvest_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Sample independent seven-day decisions from one seasonal trunk.

    SWAP branches start at the beginning of the decision day, so the restart
    checkpoint is the end of the preceding day. The DVS threshold is therefore
    evaluated on ``decision_date - 1``.
    """
    if dvs_threshold < 0:
        raise ValueError("dvs_threshold must be non-negative")
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    required = {"Date", "DVS"}
    missing = required.difference(crop.columns)
    if missing:
        raise ValueError(f"crop trajectory missing columns: {sorted(missing)}")

    daily = crop[["Date", "DVS"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    daily["DVS"] = pd.to_numeric(daily["DVS"], errors="coerce")
    if daily["DVS"].isna().any():
        raise ValueError("crop trajectory contains missing DVS values")
    daily = daily.sort_values("Date").reset_index(drop=True)
    if daily["Date"].duplicated().any():
        raise ValueError("crop trajectory contains duplicate dates")

    inferred_harvest = daily["Date"].iloc[-1]
    harvest = pd.Timestamp(harvest_date) if harvest_date is not None else inferred_harvest
    if harvest not in set(daily["Date"]):
        raise ValueError("harvest_date must be present in the crop trajectory")
    if harvest > inferred_harvest:
        raise ValueError("harvest_date is later than the crop trajectory")
    daily = daily.loc[daily["Date"] <= harvest].copy()

    eligible = daily.loc[daily["DVS"] >= float(dvs_threshold)]
    if eligible.empty:
        raise ValueError(f"crop trajectory never reaches DVS >= {dvs_threshold}")
    first_checkpoint = pd.Timestamp(eligible["Date"].iloc[0])
    first_decision = first_checkpoint + pd.Timedelta(days=1)
    latest_decision = harvest - pd.Timedelta(days=horizon_days - 1)
    if first_decision > latest_decision:
        raise ValueError("no decision date has a complete post-emergence horizon")

    dates = set(daily["Date"])
    dvs_by_date = daily.set_index("Date")["DVS"]
    rows: list[dict[str, object]] = []
    decision = first_decision
    schedule_index = 0
    while decision <= latest_decision:
        checkpoint = decision - pd.Timedelta(days=1)
        horizon_end = decision + pd.Timedelta(days=horizon_days - 1)
        horizon_dates = pd.date_range(decision, horizon_end, freq="D")
        if checkpoint not in dates:
            raise ValueError(f"missing checkpoint crop state for {checkpoint.date()}")
        if float(dvs_by_date.loc[checkpoint]) < float(dvs_threshold):
            raise ValueError(f"checkpoint DVS falls below threshold on {checkpoint.date()}")
        if any(date not in dates for date in horizon_dates):
            raise ValueError(f"incomplete crop horizon for {decision.date()}")
        rows.append(
            {
                "site_id": str(site_id),
                "target_year": int(target_year),
                "split": str(split),
                "schedule_index": int(schedule_index),
                "state_checkpoint_date": checkpoint.strftime("%Y-%m-%d"),
                "state_dvs": float(dvs_by_date.loc[checkpoint]),
                "decision_date": decision.strftime("%Y-%m-%d"),
                "decision_doy": int(decision.dayofyear),
                "horizon_start_date": decision.strftime("%Y-%m-%d"),
                "horizon_end_date": horizon_end.strftime("%Y-%m-%d"),
                "horizon_days": int(horizon_days),
                "harvest_date": harvest.strftime("%Y-%m-%d"),
                "days_from_decision_to_harvest": int((harvest - decision).days),
                "dvs_threshold": float(dvs_threshold),
                "sampling_interval_days": int(interval_days),
            }
        )
        schedule_index += 1
        decision += pd.Timedelta(days=interval_days)
    return pd.DataFrame(rows)
