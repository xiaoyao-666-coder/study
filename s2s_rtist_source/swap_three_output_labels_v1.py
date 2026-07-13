"""Extract three-output surrogate labels from one SWAP restart candidate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


CRP_COLUMNS = [
    "Date",
    "Daynr",
    "Daycrp",
    "DVS",
    "TSUM",
    "LAIpot",
    "LAI",
    "Height",
    "CrpFac",
    "RootdPot",
    "Rootd",
    "PWLV",
    "WLV",
    "PWST",
    "WST",
    "PWRT",
    "WRT",
    "CPWDM",
    "CWDM",
    "CPWSO",
    "CWSO",
    "PGRASSDM",
    "GRASSDM",
    "PMOWDM",
    "MOWDM",
    "PGRAZDM",
    "GRAZDM",
    "DWLVCROP",
    "DWLVSOIL",
    "DWST",
    "DWRT",
    "DWSO",
    "HarLosOrm",
]


@dataclass(frozen=True)
class CandidateLabels:
    daily: pd.DataFrame
    summary: dict[str, float | int | str]


def flatten_candidate_labels(
    labels: CandidateLabels,
) -> dict[str, float | int | str]:
    flat = dict(labels.summary)
    daily_fields = {
        "root_depth_cm": "root_depth_day{day:02d}_cm",
        "rootzone_vwc": "rootzone_vwc_day{day:02d}",
        "rootzone_storage_mm": "rootzone_storage_day{day:02d}_mm",
        "tact_mm": "tact_day{day:02d}_mm",
        "eact_mm": "eact_day{day:02d}_mm",
        "interc_mm": "interc_day{day:02d}_mm",
        "aet_mm": "aet_day{day:02d}_mm",
        "runoff_mm": "runoff_day{day:02d}_mm",
        "root_drainage_mm": "root_drainage_day{day:02d}_mm",
        "root_boundary_flux_mm": "root_boundary_flux_day{day:02d}_mm",
        "root_boundary_depth_cm": "root_boundary_depth_day{day:02d}_cm",
    }
    for day, row in enumerate(labels.daily.to_dict(orient="records"), start=1):
        for source, template in daily_fields.items():
            flat[template.format(day=day)] = float(row[source])
    return flat


def inclusive_horizon_end_doy(decision_doy: int, horizon_days: int) -> int:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return int(decision_doy) + int(horizon_days) - 1


def _read_swap_csv(path: Path, header_name: str) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for index, line in enumerate(lines):
        first = line.strip().split(",", 1)[0].strip().lower()
        if first == header_name.lower():
            start = index
            break
    if start is None:
        raise RuntimeError(f"Missing {header_name} header in {path}")
    frame = pd.read_csv(StringIO("\n".join(lines[start:])), skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _read_crop_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not re.match(r"^\s*\d{4}-\d{2}-\d{2}", line):
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(CRP_COLUMNS):
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No crop rows found in {path}")
    frame = pd.DataFrame(rows, columns=CRP_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    for column in ["Rootd", "DVS", "LAI", "CWDM", "CWSO"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_profile_table(path: Path) -> pd.DataFrame:
    frame = _read_swap_csv(path, "date")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    for column in [
        "depth",
        "wcontent",
        "phead",
        "drainage",
        "rootext",
        "waterflux",
        "top",
        "bottom",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"].astype(str).str.strip()).dt.strftime(
        "%Y-%m-%d"
    )
    return frame


def _root_depth_for_date(crop: pd.DataFrame, date: str) -> float:
    values = crop.loc[crop["Date"] == date, "Rootd"].dropna()
    if values.empty:
        raise RuntimeError(f"Missing root depth for {date}")
    return float(values.iloc[-1])


def _rootzone_metrics(profile: pd.DataFrame, root_depth_cm: float) -> dict[str, float]:
    if profile.empty:
        raise RuntimeError("Empty daily soil profile")
    day = profile.copy()
    day["shallow_cm"] = np.minimum(day["top"].abs(), day["bottom"].abs())
    day["deep_cm"] = np.maximum(day["top"].abs(), day["bottom"].abs())
    day["thickness_cm"] = day["deep_cm"] - day["shallow_cm"]
    day["overlap_cm"] = (
        np.minimum(day["deep_cm"], float(root_depth_cm))
        - np.maximum(day["shallow_cm"], 0.0)
    ).clip(lower=0.0)
    selected = day[day["overlap_cm"] > 0].copy()
    if selected.empty:
        raise RuntimeError(f"No compartments intersect root depth {root_depth_cm}")
    selected["fraction"] = selected["overlap_cm"] / selected["thickness_cm"]
    covered_depth_cm = float(selected["overlap_cm"].sum())
    storage_cm = float((selected["wcontent"] * selected["overlap_cm"]).sum())
    boundary_index = (day["top"].abs() - float(root_depth_cm)).abs().idxmin()
    boundary = day.loc[boundary_index]
    return {
        "root_depth_cm": float(root_depth_cm),
        "rootzone_vwc": storage_cm / covered_depth_cm,
        "rootzone_storage_mm": 10.0 * storage_cm,
        "root_extraction_mm": 10.0
        * float((selected["rootext"] * selected["fraction"]).sum()),
        "root_drainage_mm": 10.0
        * float((selected["drainage"] * selected["fraction"]).sum()),
        "root_boundary_flux_mm": 10.0 * float(boundary["waterflux"]),
        "root_boundary_depth_cm": abs(float(boundary["top"])),
        "covered_depth_cm": covered_depth_cm,
    }


def extract_candidate_labels(
    *,
    pre_crop_path: Path,
    pre_profile_path: Path,
    restart_crop_path: Path,
    restart_profile_path: Path,
    restart_increment_path: Path,
    decision_date: str,
    horizon_days: int = 7,
) -> CandidateLabels:
    start = pd.Timestamp(decision_date).normalize()
    dates = [(start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(horizon_days)]

    increments = _read_swap_csv(Path(restart_increment_path), "Date")
    increments["Date"] = pd.to_datetime(
        increments["Date"].astype(str).str.strip()
    ).dt.strftime("%Y-%m-%d")
    increments = increments[increments["Date"].isin(dates)].copy()
    increments = increments.set_index("Date").reindex(dates).reset_index()
    if increments["Day"].isna().any():
        missing = increments.loc[increments["Day"].isna(), "Date"].tolist()
        raise RuntimeError(f"Missing increment rows for dates: {missing}")
    for column in [
        "Rain",
        "Snow",
        "Irrig",
        "Interc",
        "Runon",
        "Runoff",
        "Tact",
        "Eact",
        "Drainage",
        "QBottom",
        "dstorage",
        "baldev",
    ]:
        increments[column] = pd.to_numeric(increments[column], errors="coerce")

    pre_crop = _read_crop_table(Path(pre_crop_path))
    pre_profile = _read_profile_table(Path(pre_profile_path))
    pre_dates = sorted(date for date in pre_profile["date"].unique() if date < dates[0])
    if not pre_dates:
        raise RuntimeError("Missing pre-decision profile date")
    pre_date = pre_dates[-1]
    pre_root_depth = _root_depth_for_date(pre_crop, pre_date)
    pre_metrics = _rootzone_metrics(
        pre_profile[pre_profile["date"] == pre_date], pre_root_depth
    )

    restart_crop = _read_crop_table(Path(restart_crop_path))
    restart_profile = _read_profile_table(Path(restart_profile_path))
    daily_rows: list[dict[str, float | str]] = []
    for date in dates:
        root_depth = _root_depth_for_date(restart_crop, date)
        metrics = _rootzone_metrics(
            restart_profile[restart_profile["date"] == date], root_depth
        )
        flux = increments.loc[increments["Date"] == date].iloc[0]
        tact_mm = 10.0 * float(flux["Tact"])
        eact_mm = 10.0 * float(flux["Eact"])
        interc_mm = 10.0 * float(flux["Interc"])
        daily_rows.append(
            {
                "date": date,
                **metrics,
                "rain_mm": 10.0 * float(flux["Rain"]),
                "snow_mm": 10.0 * float(flux["Snow"]),
                "irrigation_mm": 10.0 * float(flux["Irrig"]),
                "interc_mm": interc_mm,
                "runon_mm": 10.0 * float(flux["Runon"]),
                "runoff_mm": 10.0 * float(flux["Runoff"]),
                "tact_mm": tact_mm,
                "eact_mm": eact_mm,
                "aet_mm": tact_mm + eact_mm + interc_mm,
                "profile_drainage_mm": 10.0 * float(flux["Drainage"]),
                "profile_qbottom_mm": 10.0 * float(flux["QBottom"]),
                "profile_dstorage_mm": 10.0 * float(flux["dstorage"]),
                "profile_baldev_mm": 10.0 * float(flux["baldev"]),
            }
        )
    daily = pd.DataFrame(daily_rows)

    delta_storage = float(daily.iloc[-1]["rootzone_storage_mm"]) - float(
        pre_metrics["rootzone_storage_mm"]
    )
    residual_flux = float(
        daily["runoff_mm"].sum()
        + daily["root_drainage_mm"].sum()
        + daily["root_boundary_flux_mm"].sum()
    )
    water_balance_residual = float(
        daily["rain_mm"].sum()
        + daily["snow_mm"].sum()
        + daily["irrigation_mm"].sum()
        + daily["runon_mm"].sum()
        - daily["aet_mm"].sum()
        - residual_flux
        - delta_storage
    )
    summary: dict[str, float | int | str] = {
        "horizon_days_actual": int(len(daily)),
        "horizon_start_date": dates[0],
        "horizon_end_date": dates[-1],
        "predecision_date": pre_date,
        "predecision_root_depth_cm": pre_root_depth,
        "predecision_rootzone_vwc": float(pre_metrics["rootzone_vwc"]),
        "predecision_rootzone_storage_mm": float(
            pre_metrics["rootzone_storage_mm"]
        ),
        "rain_7d_mm": float(daily["rain_mm"].sum()),
        "snow_7d_mm": float(daily["snow_mm"].sum()),
        "irrigation_7d_mm": float(daily["irrigation_mm"].sum()),
        "runon_7d_mm": float(daily["runon_mm"].sum()),
        "tact_7d_mm": float(daily["tact_mm"].sum()),
        "eact_7d_mm": float(daily["eact_mm"].sum()),
        "interc_7d_mm": float(daily["interc_mm"].sum()),
        "aet_7d_mm": float(daily["aet_mm"].sum()),
        "runoff_7d_mm": float(daily["runoff_mm"].sum()),
        "root_drainage_7d_mm": float(daily["root_drainage_mm"].sum()),
        "root_boundary_flux_7d_mm": float(
            daily["root_boundary_flux_mm"].sum()
        ),
        "residual_flux_7d_mm": residual_flux,
        "delta_rootzone_storage_7d_mm": delta_storage,
        "water_balance_residual_7d_mm": water_balance_residual,
    }
    return CandidateLabels(daily=daily, summary=summary)
