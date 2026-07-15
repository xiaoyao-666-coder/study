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
        "soil_vwc_0_100cm": "soil_vwc_0_100cm_day{day:02d}",
        "soil_storage_0_100cm_mm": "soil_storage_0_100cm_day{day:02d}_mm",
        "tact_mm": "tact_day{day:02d}_mm",
        "eact_mm": "eact_day{day:02d}_mm",
        "interc_mm": "interc_day{day:02d}_mm",
        "aet_mm": "aet_day{day:02d}_mm",
        "runoff_mm": "runoff_day{day:02d}_mm",
        "soil_drainage_0_100cm_mm": "soil_drainage_0_100cm_day{day:02d}_mm",
        "soil_boundary_waterflux_100cm_signed_mm": (
            "soil_boundary_waterflux_100cm_signed_day{day:02d}_mm"
        ),
        "soil_boundary_outflow_100cm_mm": (
            "soil_boundary_outflow_100cm_day{day:02d}_mm"
        ),
        "soil_boundary_depth_cm": "soil_boundary_depth_day{day:02d}_cm",
    }
    for day, row in enumerate(labels.daily.to_dict(orient="records"), start=1):
        for source, template in daily_fields.items():
            flat[template.format(day=day)] = float(row[source])
    return flat


def inclusive_horizon_end_doy(decision_doy: int, horizon_days: int) -> int:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return int(decision_doy) + int(horizon_days) - 1


def patch_nprintday_text(text: str, nprintday: int) -> str:
    from rootzone_flux_frequency_diagnostic_v1 import patch_nprintday_text as impl

    return impl(text, nprintday)


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
    nprintday: int = 24,
) -> CandidateLabels:
    from rootzone_flux_frequency_diagnostic_v1 import analyze_case_outputs

    result = analyze_case_outputs(
        pre_crop_path=Path(pre_crop_path),
        pre_profile_path=Path(pre_profile_path),
        restart_crop_path=Path(restart_crop_path),
        restart_profile_path=Path(restart_profile_path),
        restart_increment_path=Path(restart_increment_path),
        decision_date=decision_date,
        nprintday=nprintday,
        horizon_days=horizon_days,
        control_depth_cm=100.0,
    )
    daily = result.daily.rename(
        columns={
            "rootzone_vwc": "soil_vwc_0_100cm",
            "rootzone_storage_mm": "soil_storage_0_100cm_mm",
            "root_drainage_mm": "soil_drainage_0_100cm_mm",
            "root_boundary_flux_mm": (
                "soil_boundary_waterflux_100cm_signed_mm"
            ),
            "root_boundary_outflow_mm": "soil_boundary_outflow_100cm_mm",
            "root_boundary_depth_cm": "soil_boundary_depth_cm",
        }
    ).drop(columns=["moving_root_boundary_term_mm"])
    summary = dict(result.summary)
    summary_renames = {
        "root_drainage_7d_mm": "soil_drainage_0_100cm_7d_mm",
        "root_boundary_signed_integral_mm": (
            "soil_boundary_waterflux_100cm_signed_7d_mm"
        ),
        "root_boundary_outflow_7d_mm": "soil_boundary_outflow_100cm_7d_mm",
        "predecision_rootzone_vwc": "predecision_soil_vwc_0_100cm",
        "predecision_rootzone_storage_mm": (
            "predecision_soil_storage_0_100cm_mm"
        ),
        "final_rootzone_storage_mm": "final_soil_storage_0_100cm_mm",
        "delta_rootzone_storage_7d_mm": "delta_soil_storage_0_100cm_7d_mm",
        "max_abs_root_boundary_depth_error_cm": (
            "max_abs_soil_boundary_depth_error_cm"
        ),
    }
    for old_name, new_name in summary_renames.items():
        summary[new_name] = summary.pop(old_name)
    direct_outflow = float(summary.pop("direct_component_outflow_7d_mm"))
    fixed_residual = float(summary.pop("water_balance_residual_corrected_7d_mm"))
    for legacy_name in (
        "moving_root_boundary_term_7d_mm",
        "balance_derived_outflow_without_moving_7d_mm",
        "balance_derived_outflow_with_moving_7d_mm",
        "water_balance_residual_without_moving_7d_mm",
    ):
        summary.pop(legacy_name, None)
    summary.update(
        {
            "control_volume_type": "fixed_0_100cm",
            "control_depth_cm": 100.0,
            "horizon_days_actual": int(len(daily)),
            "horizon_start_date": str(daily.iloc[0]["date"]),
            "horizon_end_date": str(daily.iloc[-1]["date"]),
            "predecision_date": str(result.samples.iloc[0]["date"]),
            "residual_flux_7d_mm": direct_outflow,
            "water_balance_residual_0_100cm_7d_mm": fixed_residual,
        }
    )
    return CandidateLabels(daily=daily, summary=summary)
