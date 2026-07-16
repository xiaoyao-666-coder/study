#!/usr/bin/env python3
"""Apply confirmed 5-site gridMET weather inputs to SWAP workspaces.

This is the weather input layer after static SWP and POLARIS soil inputs. It
extracts 2024 daily gridMET weather from existing NetCDF files and writes:

- df_gridmet.csv
- weather_gridmet_out.csv
- weather.024
- WeatherOriginal.024

The script avoids network downloads. It expects the gridMET NetCDF files to be
present under `model3_opt_sto_upload/data/gridmet`.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
CONFIRMED_WORKSPACES = OUT_DIR / "confirmed_5site_workspaces"
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"
GRIDMET_DIR = Path("model3_opt_sto_upload") / "data" / "gridmet"
REPORT_CSV = OUT_DIR / "confirmed_5site_gridmet_weather_input_application_v1.csv"
REPORT_MD = OUT_DIR / "confirmed_5site_gridmet_weather_input_application_v1.md"

SITE_TO_WORKSPACE = {
    "P1": "P1_N1_Maize",
    "P2": "P2_N2_Maize",
    "P3": "P3_N3_Maize",
    "P4": "P4_N4_Maize",
    "P15": "P15_coord_12_Maize",
}

SITE_COORDINATES = {
    "P1": {"code_site_id": "N1", "longitude": -98.224144, "latitude": 42.015928},
    "P2": {"code_site_id": "N2", "longitude": -88.415, "latitude": 40.595},
    "P3": {"code_site_id": "N3", "longitude": -96.877, "latitude": 46.321},
    "P4": {"code_site_id": "N4", "longitude": -94.6686, "latitude": 42.6816},
    "P15": {"code_site_id": "coord_12", "longitude": -112.265, "latitude": 41.735},
}

GRIDMET_VARIABLES = {
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "pr": "precipitation_amount",
    "vs": "wind_speed",
    "vpd": "mean_vapor_pressure_deficit",
}

WEATHER_FILES = ["df_gridmet.csv", "weather_gridmet_out.csv", "weather.024", "WeatherOriginal.024"]


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ensure_workspace(site: str, create_missing: bool) -> Path:
    workspace = CONFIRMED_WORKSPACES / SITE_TO_WORKSPACE[site]
    if workspace.exists():
        return workspace
    if not create_missing:
        raise FileNotFoundError(f"Missing workspace: {workspace}")
    if not DEFAULT_SOURCE_MAIZE.exists():
        raise FileNotFoundError(f"Missing base Maize template: {DEFAULT_SOURCE_MAIZE}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEFAULT_SOURCE_MAIZE, workspace)
    return workspace


def available_gridmet_days(year: int) -> int:
    counts = []
    for variable in GRIDMET_VARIABLES:
        path = GRIDMET_DIR / f"{variable}_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing gridMET NetCDF: {path}")
        counts.append(read_gridmet_band_count(path))
    return min(counts)


def read_gridmet_band_count(path: Path) -> int:
    try:
        import rasterio
    except ModuleNotFoundError:
        rasterio = None

    if rasterio is not None:
        with rasterio.open(path) as src:
            return int(src.count)

    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("gridMET extraction requires rasterio or netCDF4") from exc

    with Dataset(path, mode="r") as nc:
        for name in GRIDMET_VARIABLES.values():
            if name in nc.variables:
                return int(nc.variables[name].shape[0])
    raise ValueError(f"No known gridMET variable found in {path}")


def apply_scale_offset(values: np.ndarray, scale_factor: str | float | None, add_offset: str | float | None) -> np.ndarray:
    scale = 1.0 if scale_factor in {None, ""} else float(scale_factor)
    offset = 0.0 if add_offset in {None, ""} else float(add_offset)
    return values.astype(float) * scale + offset


def read_gridmet_series_rasterio(path: Path, variable_name: str, lat: float, lon: float, days: int) -> list[float]:
    import rasterio

    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(f"Coordinate out of bounds for {path}: lat={lat}, lon={lon}")
        band_count = min(days, src.count)
        raw = []
        scale_factor = None
        add_offset = None
        for band in range(1, band_count + 1):
            tags = src.tags(band)
            scale_factor = tags.get("scale_factor", scale_factor)
            add_offset = tags.get("add_offset", add_offset)
            nc_var = tags.get("NETCDF_VARNAME")
            if nc_var and nc_var != variable_name:
                raise ValueError(f"Unexpected variable in {path}: expected {variable_name}, found {nc_var}")
            raw.append(src.read(band, window=((row, row + 1), (col, col + 1)))[0, 0])
    return apply_scale_offset(np.asarray(raw), scale_factor, add_offset).tolist()


def read_gridmet_series_netcdf4(path: Path, variable_name: str, lat: float, lon: float, days: int) -> list[float]:
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("gridMET extraction requires rasterio or netCDF4") from exc

    with Dataset(path, mode="r") as nc:
        lons = np.asarray(nc.variables["lon"][:])
        lats = np.asarray(nc.variables["lat"][:])
        lon_idx = int(np.abs(lons - lon).argmin())
        lat_idx = int(np.abs(lats - lat).argmin())
        series = np.asarray(nc.variables[variable_name][:days, lat_idx, lon_idx]).astype(float)
    return series.tolist()


def read_gridmet_series(year: int, variable: str, variable_name: str, lat: float, lon: float, days: int) -> list[float]:
    path = GRIDMET_DIR / f"{variable}_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing gridMET NetCDF: {path}")

    try:
        series = np.asarray(read_gridmet_series_rasterio(path, variable_name, lat, lon, days), dtype=float)
    except ModuleNotFoundError:
        series = np.asarray(read_gridmet_series_netcdf4(path, variable_name, lat, lon, days), dtype=float)

    if variable in {"tmmn", "tmmx"}:
        series = series - 273.15
    elif variable == "srad":
        series = series * 86.4
    return series.tolist()


def build_gridmet_dataframe(site: str, year: int, days: int) -> pd.DataFrame:
    meta = SITE_COORDINATES[site]
    lat = float(meta["latitude"])
    lon = float(meta["longitude"])
    values = {
        variable: read_gridmet_series(year, variable, nc_name, lat, lon, days)
        for variable, nc_name in GRIDMET_VARIABLES.items()
    }
    tmin = np.asarray(values["tmmn"], dtype=float)
    tmax = np.asarray(values["tmmx"], dtype=float)
    ta = 0.5 * (tmin + tmax)
    vpd = np.asarray(values["vpd"], dtype=float)
    es = 0.6108 * np.exp((17.27 * ta) / (ta + 237.3))
    humd = np.maximum(0.0, es - vpd)

    dates = pd.date_range(f"{year}-01-01", periods=days, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%m/%d/%Y"),
            "Year": year,
            "DOY": np.arange(1, days + 1),
            "Solar": values["srad"],
            "T-max": tmax,
            "T-min": tmin,
            "RelHum": humd,
            "Precip": np.maximum(0.0, np.asarray(values["pr"], dtype=float)),
            "WindSpeed": np.maximum(0.0, np.asarray(values["vs"], dtype=float)),
        }
    )
    df["T-min"] = np.minimum(df["T-min"], df["T-max"] * 0.95)
    return df


def to_weather_out(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], format="%m/%d/%Y")
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["year"] = out["Date"].dt.year
    out["station"] = "'Weather'"
    out["ETref"] = 2.0
    return out[
        ["station", "day", "month", "year", "Solar", "T-min", "T-max", "RelHum", "WindSpeed", "Precip", "ETref"]
    ]


def weather_header(template: Path) -> list[str]:
    if template.exists():
        return template.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)[:11]
    return [
        "*************************************************************************************************************************     \n",
        "* Filename: Hupsel.002                                                             \n",
        "* Contents: SWAP  4.0 - Daily meterorological data                \n",
        "*************************************************************************************************************************     \n",
        "* Comment area:                                                                 \n",
        "*                                                                               \n",
        "*                                                                               \n",
        "*************************************************************************************************************************     \n",
        " Station      DD      MM    YYYY         RAD       Tmin      Tmax        HUM      WIND      RAIN     ETref   \n",
        "*             nr      nr      nr       kJ/m2         C        C        kPa       m/s        mm        mm\n",
        "*************************************************************************************************************************\t\t\t\t\t\t\t\t\t\t\t\t\n",
    ]


def write_weather_file(path: Path, weather: pd.DataFrame, header: list[str]) -> None:
    lines = list(header)
    for _, row in weather.iterrows():
        lines.append(
            f" 'Weather'         {int(row['day'])}       {int(row['month'])}    {int(row['year'])}"
            f"      {float(row['Solar']):.1f}      {float(row['T-min']):.1f}"
            f"      {float(row['T-max']):.1f}      {float(row['RelHum']):.2f}"
            f"      {float(row['WindSpeed']):.1f}      {float(row['Precip']):.1f}"
            f"      {float(row['ETref']):.1f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(report: pd.DataFrame) -> None:
    status = "weather_inputs_differ_by_site" if report.groupby("file")["sha256_16"].nunique().min() > 1 else "weather_inputs_still_identical"
    lines = [
        "# Confirmed 5-Site gridMET Weather Input Application V1",
        "",
        "## Scope",
        "",
        "- Extracts 2024 gridMET weather for each confirmed site coordinate.",
        "- Updates `df_gridmet.csv`, `weather_gridmet_out.csv`, `weather.024`, and `WeatherOriginal.024`.",
        "- Does not apply S2S forecast files yet.",
        "",
        "## Status",
        "",
        f"- status: `{status}`",
        "",
        "## Updated Files",
        "",
        markdown_table(report),
        "",
        "## Interpretation",
        "",
        "The confirmed workspaces now have site-specific gridMET weather inputs. "
        "Next, rerun the restart-generation smoke and curve audit. S2S forecast weather remains a later layer.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=sorted(SITE_TO_WORKSPACE))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--strict-days", action="store_true", help="Fail if gridMET files have fewer bands than --days.")
    parser.add_argument("--create-missing", action="store_true")
    args = parser.parse_args()

    available_days = available_gridmet_days(args.year)
    if available_days < args.days and args.strict_days:
        raise ValueError(f"gridMET files only have {available_days} days, but --days requested {args.days}")
    effective_days = min(args.days, available_days)
    if effective_days != args.days:
        print(f"gridMET files have {available_days} days; using {effective_days} days for this smoke run", flush=True)

    report_rows = []
    for site in args.sites:
        if site not in SITE_COORDINATES:
            raise ValueError(f"Unknown site: {site}")
        workspace = ensure_workspace(site, create_missing=args.create_missing)
        df = build_gridmet_dataframe(site, args.year, effective_days)
        weather = to_weather_out(df)

        df.to_csv(workspace / "df_gridmet.csv")
        weather.to_csv(workspace / "weather_gridmet_out.csv")
        header = weather_header(workspace / "WeatherOriginal.024")
        write_weather_file(workspace / "weather.024", weather, header)
        write_weather_file(workspace / "WeatherOriginal.024", weather, header)

        for name in WEATHER_FILES:
            path = workspace / name
            report_rows.append(
                {
                    "paper_site_id": site,
                    "code_site_id": SITE_COORDINATES[site]["code_site_id"],
                    "file": name,
                    "rows": effective_days if name.endswith(".csv") else effective_days + 11,
                    "sha256_16": file_hash(path),
                }
            )

    report = pd.DataFrame(report_rows)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False)
    write_report(report)

    print("Confirmed 5-site gridMET weather input application v1")
    print(f"csv: {REPORT_CSV}")
    print(f"md: {REPORT_MD}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
