"""Utilities for the root-zone flux-frequency diagnostic."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from swap_three_output_labels_v1 import _read_crop_table, _read_profile_table, _read_swap_csv


@dataclass(frozen=True)
class FluxDiagnosticResult:
    samples: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, float | int | str]


def trapezoid_integral(
    times_days: Sequence[float],
    rates_cm_day: Sequence[float],
) -> float:
    if len(times_days) != len(rates_cm_day):
        raise ValueError("times_days and rates_cm_day must have equal length")
    if len(times_days) < 2:
        raise ValueError("at least two samples are required")

    total = 0.0
    previous_time = float(times_days[0])
    previous_rate = float(rates_cm_day[0])
    for raw_time, raw_rate in zip(times_days[1:], rates_cm_day[1:]):
        current_time = float(raw_time)
        current_rate = float(raw_rate)
        interval = current_time - previous_time
        if interval <= 0.0:
            raise ValueError("times_days must be strictly increasing")
        total += 0.5 * (previous_rate + current_rate) * interval
        previous_time = current_time
        previous_rate = current_rate
    return total


def signed_flux_to_downward_outflow_mm(signed_flux_integral_cm: float) -> float:
    return -10.0 * float(signed_flux_integral_cm)


def split_profile_snapshots(profile_rows: pd.DataFrame) -> list[pd.DataFrame]:
    required = {"date", "top", "bottom"}
    missing = required.difference(profile_rows.columns)
    if missing:
        raise ValueError(f"profile rows are missing columns: {sorted(missing)}")
    if profile_rows.empty:
        raise ValueError("profile rows are empty")

    rows = profile_rows.reset_index(drop=True).copy()
    starts = [
        index
        for index, value in enumerate(pd.to_numeric(rows["top"], errors="coerce"))
        if abs(float(value)) < 1e-9
    ]
    if not starts or starts[0] != 0:
        raise ValueError("each profile snapshot must start at top=0")

    snapshots: list[pd.DataFrame] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(rows)
        snapshot = rows.iloc[start:end].reset_index(drop=True)
        dates = snapshot["date"].astype(str).str.strip().unique()
        if len(dates) != 1:
            raise ValueError("profile snapshot contains multiple dates")
        snapshots.append(snapshot)
    return snapshots


def assign_horizon_times(
    snapshots: Sequence[pd.DataFrame],
    *,
    decision_date: str,
    nprintday: int,
    horizon_days: int,
) -> list[tuple[float, pd.DataFrame]]:
    frequency = int(nprintday)
    if frequency <= 0:
        raise ValueError("nprintday must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    start = pd.Timestamp(decision_date).normalize()
    by_date: dict[str, list[pd.DataFrame]] = {}
    for snapshot in snapshots:
        date = pd.Timestamp(str(snapshot.iloc[0]["date"]).strip()).strftime("%Y-%m-%d")
        by_date.setdefault(date, []).append(snapshot)

    previous_dates = sorted(date for date in by_date if pd.Timestamp(date) < start)
    if not previous_dates:
        raise ValueError("missing pre-decision profile snapshot")
    timeline: list[tuple[float, pd.DataFrame]] = [
        (0.0, by_date[previous_dates[-1]][-1])
    ]

    for day_offset in range(horizon_days):
        date = (start + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        daily = by_date.get(date, [])
        if len(daily) != frequency:
            raise ValueError(
                f"expected {frequency} snapshots for {date}, found {len(daily)}"
            )
        for sample_index, snapshot in enumerate(daily, start=1):
            elapsed = day_offset + sample_index / frequency
            timeline.append((float(elapsed), snapshot))
    return timeline


def rootzone_snapshot_metrics(
    profile: pd.DataFrame,
    root_depth_cm: float,
) -> dict[str, float]:
    required = {"top", "bottom", "wcontent", "drainage", "waterflux"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"profile is missing columns: {sorted(missing)}")

    depth = float(root_depth_cm)
    day = profile.copy()
    day["shallow_cm"] = day[["top", "bottom"]].abs().min(axis=1)
    day["deep_cm"] = day[["top", "bottom"]].abs().max(axis=1)
    day["thickness_cm"] = day["deep_cm"] - day["shallow_cm"]
    day["overlap_cm"] = (
        day["deep_cm"].clip(upper=depth) - day["shallow_cm"].clip(lower=0.0)
    ).clip(lower=0.0)
    selected = day[day["overlap_cm"] > 0.0].copy()
    if selected.empty:
        raise ValueError(f"no compartments intersect root depth {depth}")
    selected["fraction"] = selected["overlap_cm"] / selected["thickness_cm"]

    storage_cm = float((selected["wcontent"] * selected["overlap_cm"]).sum())
    drainage_rate = float((selected["drainage"] * selected["fraction"]).sum())
    boundary_index = (day["top"].abs() - depth).abs().idxmin()
    boundary = day.loc[boundary_index]
    boundary_depth = abs(float(boundary["top"]))
    return {
        "rootzone_storage_mm": 10.0 * storage_cm,
        "rootzone_vwc": storage_cm / float(selected["overlap_cm"].sum()),
        "root_drainage_rate_cm_day": drainage_rate,
        "root_boundary_waterflux_cm_day": float(boundary["waterflux"]),
        "root_boundary_depth_cm": boundary_depth,
        "root_boundary_depth_error_cm": boundary_depth - depth,
    }


def _slice_storage_mm(
    profile: pd.DataFrame,
    shallow_cm: float,
    deep_cm: float,
) -> float:
    if deep_cm < shallow_cm:
        raise ValueError("deep_cm must be greater than or equal to shallow_cm")
    required = {"top", "bottom", "wcontent"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"profile is missing columns: {sorted(missing)}")

    storage_cm = 0.0
    for row in profile.itertuples(index=False):
        compartment_shallow = min(abs(float(row.top)), abs(float(row.bottom)))
        compartment_deep = max(abs(float(row.top)), abs(float(row.bottom)))
        overlap = max(
            0.0,
            min(compartment_deep, float(deep_cm))
            - max(compartment_shallow, float(shallow_cm)),
        )
        if overlap <= 0.0:
            continue
        if pd.isna(row.wcontent):
            raise ValueError(
                f"missing wcontent in overlapping slice {shallow_cm}-{deep_cm} cm"
            )
        storage_cm += float(row.wcontent) * overlap
    return 10.0 * storage_cm


def moving_boundary_term_mm(
    *,
    previous_profile: pd.DataFrame,
    current_profile: pd.DataFrame,
    previous_root_depth_cm: float,
    current_root_depth_cm: float,
) -> float:
    previous_depth = float(previous_root_depth_cm)
    current_depth = float(current_root_depth_cm)
    if current_depth == previous_depth:
        return 0.0

    shallow = min(previous_depth, current_depth)
    deep = max(previous_depth, current_depth)
    previous_storage = _slice_storage_mm(previous_profile, shallow, deep)
    current_storage = _slice_storage_mm(current_profile, shallow, deep)
    magnitude = 0.5 * (previous_storage + current_storage)
    return magnitude if current_depth > previous_depth else -magnitude


def patch_nprintday_text(text: str, nprintday: int) -> str:
    value = int(nprintday)
    if not 1 <= value <= 1000:
        raise ValueError("nprintday must be in [1, 1000]")

    pattern = re.compile(r"^(\s*NPrintDay\s*=\s*)\d+(.*)$", re.MULTILINE)
    patched, count = pattern.subn(rf"\g<1>{value}\g<2>", text)
    if count != 1:
        raise ValueError(f"expected one NPrintDay assignment, found {count}")
    return patched


def aggregate_increment_rows(
    increments: pd.DataFrame,
    *,
    dates: Sequence[str],
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    required = {"Date", *numeric_columns}
    missing = required.difference(increments.columns)
    if missing:
        raise ValueError(f"increment table is missing columns: {sorted(missing)}")

    grouping_column = "Dcum" if "Dcum" in increments.columns else "Date"
    frame = increments[[grouping_column, *numeric_columns]].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if grouping_column == "Dcum":
        frame["Dcum"] = pd.to_numeric(frame["Dcum"], errors="coerce")
        frame = frame[frame["Dcum"].between(1, len(dates))]
        daily = frame.groupby("Dcum", as_index=True)[list(numeric_columns)].sum(
            min_count=1
        )
        daily = daily.reindex(range(1, len(dates) + 1)).reset_index(drop=True)
        daily.insert(0, "Date", list(dates))
        return daily

    frame["Date"] = pd.to_datetime(
        frame["Date"].astype(str).str.strip()
    ).dt.strftime("%Y-%m-%d")
    daily = frame.groupby("Date", as_index=False)[list(numeric_columns)].sum(
        min_count=1
    )
    return daily.set_index("Date").reindex(list(dates)).reset_index()


def _root_depth_for_date(crop: pd.DataFrame, date: str) -> float:
    values = crop.loc[crop["Date"] == date, "Rootd"].dropna()
    if values.empty:
        raise ValueError(f"missing root depth for {date}")
    return float(values.iloc[-1])


def _last_snapshot_before(
    snapshots: Sequence[pd.DataFrame],
    date: pd.Timestamp,
) -> pd.DataFrame:
    candidates = [
        snapshot
        for snapshot in snapshots
        if pd.Timestamp(str(snapshot.iloc[0]["date"]).strip()) < date
    ]
    if not candidates:
        raise ValueError("missing pre-decision profile snapshot")
    return candidates[-1]


def analyze_case_outputs(
    *,
    pre_crop_path: Path,
    pre_profile_path: Path,
    restart_crop_path: Path,
    restart_profile_path: Path,
    restart_increment_path: Path,
    decision_date: str,
    nprintday: int,
    horizon_days: int = 7,
    control_depth_cm: float | None = None,
) -> FluxDiagnosticResult:
    if control_depth_cm is not None and float(control_depth_cm) <= 0.0:
        raise ValueError("control_depth_cm must be positive")
    start = pd.Timestamp(decision_date).normalize()
    dates = [
        (start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(horizon_days)
    ]

    pre_crop = _read_crop_table(Path(pre_crop_path))
    restart_crop = _read_crop_table(Path(restart_crop_path))
    pre_profiles = split_profile_snapshots(_read_profile_table(Path(pre_profile_path)))
    restart_profiles = split_profile_snapshots(
        _read_profile_table(Path(restart_profile_path))
    )
    initial_snapshot = _last_snapshot_before(pre_profiles, start)
    horizon_snapshots = [
        snapshot
        for snapshot in restart_profiles
        if str(snapshot.iloc[0]["date"]).strip() in dates
    ]
    timeline = assign_horizon_times(
        [initial_snapshot, *horizon_snapshots],
        decision_date=start.strftime("%Y-%m-%d"),
        nprintday=nprintday,
        horizon_days=horizon_days,
    )

    sample_rows: list[dict[str, float | str]] = []
    timeline_profiles: list[pd.DataFrame] = []
    for elapsed_days, snapshot in timeline:
        date = str(snapshot.iloc[0]["date"]).strip()
        if pd.Timestamp(date) < start:
            root_depth = _root_depth_for_date(pre_crop, date)
        else:
            root_depth = _root_depth_for_date(restart_crop, date)
        metrics_depth = (
            float(control_depth_cm)
            if control_depth_cm is not None
            else float(root_depth)
        )
        metrics = rootzone_snapshot_metrics(snapshot, metrics_depth)
        sample_rows.append(
            {
                "elapsed_days": elapsed_days,
                "date": date,
                "root_depth_cm": root_depth,
                "control_depth_cm": metrics_depth,
                **metrics,
            }
        )
        timeline_profiles.append(snapshot)
    samples = pd.DataFrame(sample_rows)

    signed_flux_integral_cm = trapezoid_integral(
        samples["elapsed_days"].tolist(),
        samples["root_boundary_waterflux_cm_day"].tolist(),
    )
    drainage_integral_cm = trapezoid_integral(
        samples["elapsed_days"].tolist(),
        samples["root_drainage_rate_cm_day"].tolist(),
    )
    root_boundary_outflow = signed_flux_to_downward_outflow_mm(
        signed_flux_integral_cm
    )
    root_drainage_outflow = 10.0 * drainage_integral_cm

    endpoint_indices = [0]
    for date in dates:
        matching = samples.index[samples["date"] == date].tolist()
        if not matching:
            raise ValueError(f"missing endpoint profile for {date}")
        endpoint_indices.append(matching[-1])

    daily_rows: list[dict[str, float | str]] = []
    moving_term = 0.0
    for day_index, date in enumerate(dates, start=1):
        previous_index = endpoint_indices[day_index - 1]
        current_index = endpoint_indices[day_index]
        previous_row = samples.loc[previous_index]
        current_row = samples.loc[current_index]
        if control_depth_cm is None:
            day_moving_term = moving_boundary_term_mm(
                previous_profile=timeline_profiles[previous_index],
                current_profile=timeline_profiles[current_index],
                previous_root_depth_cm=float(previous_row["root_depth_cm"]),
                current_root_depth_cm=float(current_row["root_depth_cm"]),
            )
        else:
            day_moving_term = 0.0
        moving_term += day_moving_term
        interval_start = float(day_index - 1)
        interval_end = float(day_index)
        interval = samples[
            samples["elapsed_days"].between(interval_start, interval_end)
        ]
        signed_day_cm = trapezoid_integral(
            interval["elapsed_days"].tolist(),
            interval["root_boundary_waterflux_cm_day"].tolist(),
        )
        drainage_day_cm = trapezoid_integral(
            interval["elapsed_days"].tolist(),
            interval["root_drainage_rate_cm_day"].tolist(),
        )
        daily_rows.append(
            {
                "date": date,
                "root_depth_cm": float(current_row["root_depth_cm"]),
                "root_depth_previous_cm": float(previous_row["root_depth_cm"]),
                "root_depth_current_cm": float(current_row["root_depth_cm"]),
                "moving_root_boundary_term_mm": day_moving_term,
                "rootzone_storage_mm": float(current_row["rootzone_storage_mm"]),
                "rootzone_vwc": float(current_row["rootzone_vwc"]),
                "root_boundary_flux_mm": 10.0 * signed_day_cm,
                "root_boundary_outflow_mm": signed_flux_to_downward_outflow_mm(
                    signed_day_cm
                ),
                "root_drainage_mm": 10.0 * drainage_day_cm,
                "root_boundary_depth_cm": float(
                    current_row["root_boundary_depth_cm"]
                ),
            }
        )
    daily = pd.DataFrame(daily_rows)

    increments = _read_swap_csv(Path(restart_increment_path), "Date")
    numeric_columns = [
        "Rain",
        "Snow",
        "Irrig",
        "Interc",
        "Runon",
        "Runoff",
        "Tact",
        "Eact",
    ]
    increments = aggregate_increment_rows(
        increments,
        dates=dates,
        numeric_columns=numeric_columns,
    )
    missing_mask = increments[list(numeric_columns)].isna().all(axis=1)
    if missing_mask.any():
        missing_dates = increments.loc[missing_mask, "Date"].tolist()
        raise ValueError(f"missing increment rows for dates: {missing_dates}")

    increment_daily = pd.DataFrame(
        {
            "date": increments["Date"],
            "rain_mm": 10.0 * increments["Rain"],
            "snow_mm": 10.0 * increments["Snow"],
            "irrigation_mm": 10.0 * increments["Irrig"],
            "interc_mm": 10.0 * increments["Interc"],
            "runon_mm": 10.0 * increments["Runon"],
            "runoff_mm": 10.0 * increments["Runoff"],
            "tact_mm": 10.0 * increments["Tact"],
            "eact_mm": 10.0 * increments["Eact"],
        }
    )
    increment_daily["aet_mm"] = (
        increment_daily["tact_mm"]
        + increment_daily["eact_mm"]
        + increment_daily["interc_mm"]
    )
    daily = daily.merge(increment_daily, on="date", how="left", validate="one_to_one")

    water_input = 10.0 * float(
        increments[["Rain", "Snow", "Irrig", "Runon"]].sum().sum()
    )
    aet = 10.0 * float(
        increments[["Tact", "Eact", "Interc"]].sum().sum()
    )
    runoff = 10.0 * float(increments["Runoff"].sum())
    delta_storage = float(samples.iloc[-1]["rootzone_storage_mm"]) - float(
        samples.iloc[0]["rootzone_storage_mm"]
    )
    direct_outflow = runoff + root_drainage_outflow + root_boundary_outflow
    balance_derived_without_moving = water_input - aet - delta_storage
    balance_derived_with_moving = (
        water_input - aet + moving_term - delta_storage
    )
    residual_without_moving = balance_derived_without_moving - direct_outflow
    residual_corrected = balance_derived_with_moving - direct_outflow

    summary: dict[str, float | int | str] = {
        "decision_date": start.strftime("%Y-%m-%d"),
        "horizon_days": int(horizon_days),
        "nprintday": int(nprintday),
        "control_volume_type": (
            "fixed" if control_depth_cm is not None else "dynamic_rootzone"
        ),
        "control_depth_cm": (
            float(control_depth_cm) if control_depth_cm is not None else "dynamic"
        ),
        "profile_sample_count": int(len(samples)),
        "flux_integration_method": "trapezoid_actual_subdaily_interval",
        "increment_grouping_method": "Dcum_1_to_horizon_days",
        "rain_7d_mm": 10.0 * float(increments["Rain"].sum()),
        "snow_7d_mm": 10.0 * float(increments["Snow"].sum()),
        "irrigation_7d_mm": 10.0 * float(increments["Irrig"].sum()),
        "runon_7d_mm": 10.0 * float(increments["Runon"].sum()),
        "tact_7d_mm": 10.0 * float(increments["Tact"].sum()),
        "eact_7d_mm": 10.0 * float(increments["Eact"].sum()),
        "interc_7d_mm": 10.0 * float(increments["Interc"].sum()),
        "water_input_7d_mm": water_input,
        "aet_7d_mm": aet,
        "runoff_7d_mm": runoff,
        "root_drainage_7d_mm": root_drainage_outflow,
        "root_boundary_signed_integral_mm": 10.0 * signed_flux_integral_cm,
        "root_boundary_outflow_7d_mm": root_boundary_outflow,
        "direct_component_outflow_7d_mm": direct_outflow,
        "predecision_root_depth_cm": float(samples.iloc[0]["root_depth_cm"]),
        "predecision_rootzone_vwc": float(samples.iloc[0]["rootzone_vwc"]),
        "final_root_depth_cm": float(samples.iloc[-1]["root_depth_cm"]),
        "root_depth_change_cm": float(samples.iloc[-1]["root_depth_cm"])
        - float(samples.iloc[0]["root_depth_cm"]),
        "predecision_rootzone_storage_mm": float(
            samples.iloc[0]["rootzone_storage_mm"]
        ),
        "final_rootzone_storage_mm": float(samples.iloc[-1]["rootzone_storage_mm"]),
        "delta_rootzone_storage_7d_mm": delta_storage,
        "moving_root_boundary_term_7d_mm": moving_term,
        "balance_derived_outflow_without_moving_7d_mm": balance_derived_without_moving,
        "balance_derived_outflow_with_moving_7d_mm": balance_derived_with_moving,
        "water_balance_residual_without_moving_7d_mm": residual_without_moving,
        "water_balance_residual_corrected_7d_mm": residual_corrected,
        "max_abs_root_boundary_depth_error_cm": float(
            samples["root_boundary_depth_error_cm"].abs().max()
        ),
    }
    return FluxDiagnosticResult(samples=samples, daily=daily, summary=summary)
