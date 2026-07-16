#!/usr/bin/env python3
"""Apply gridMET weather inputs to 12-site continuous-ir workspaces.

This is the weather input layer after static SWP and POLARIS soil inputs. It
extracts 2024 daily gridMET values from existing local NetCDF files and writes:

- df_gridmet.csv
- weather_gridmet_out.csv
- weather.024
- WeatherOriginal.024

The script does not download data and does not create missing workspaces. Run
prepare_continuous_ir_12site_workspaces_v1.py and
apply_continuous_ir_12site_static_polaris_inputs_v1.py first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
DEFAULT_SITE_FEATURE_CSV = OUT_DIR / "site_feature_screening_12_code_sites.csv"
DEFAULT_WORKSPACE_ROOT = OUT_DIR / "continuous_ir_12site_workspaces_v1"
DEFAULT_GRIDMET_DIR = Path("model3_opt_sto_upload") / "data" / "gridmet"
DEFAULT_REPORT_PREFIX = OUT_DIR / "continuous_ir_12site_gridmet_weather_input_application_v1"

GRIDMET_VARIABLES = {
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "pr": "precipitation_amount",
    "vs": "wind_speed",
    "vpd": "mean_vapor_pressure_deficit",
}

WEATHER_FILES = ["df_gridmet.csv", "weather_gridmet_out.csv", "weather.024", "WeatherOriginal.024"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--gridmet-dir", default=str(DEFAULT_GRIDMET_DIR))
    parser.add_argument("--output-prefix", default=str(DEFAULT_REPORT_PREFIX))
    parser.add_argument("--sites", nargs="+", help="Optional subset of site ids.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--strict-days", action="store_true", help="Fail if gridMET files have fewer bands than --days.")
    return parser.parse_args()


def workspace_name(site: str) -> str:
    return f"{site}_Maize"


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_features(df: pd.DataFrame) -> None:
    missing = [col for col in ["site", "lon", "lat"] if col not in df.columns]
    if missing:
        raise ValueError(f"Site feature CSV is missing required columns: {missing}")


def available_gridmet_days(gridmet_dir: Path, year: int) -> int:
    counts = []
    for variable in GRIDMET_VARIABLES:
        path = gridmet_dir / f"{variable}_{year}.nc"
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


def read_gridmet_series_rasterio(
    path: Path,
    variable_name: str,
    lat: float,
    lon: float,
    days: int,
) -> list[float]:
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


def read_gridmet_series_netcdf4(
    path: Path,
    variable_name: str,
    lat: float,
    lon: float,
    days: int,
) -> list[float]:
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


def read_gridmet_series(
    gridmet_dir: Path,
    year: int,
    variable: str,
    variable_name: str,
    lat: float,
    lon: float,
    days: int,
) -> list[float]:
    path = gridmet_dir / f"{variable}_{year}.nc"
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


def build_gridmet_dataframe(site_row: pd.Series, gridmet_dir: Path, year: int, days: int) -> pd.DataFrame:
    lat = float(site_row["lat"])
    lon = float(site_row["lon"])
    values = {
        variable: read_gridmet_series(gridmet_dir, year, variable, nc_name, lat, lon, days)
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


def update_site_config(workspace: Path, site: str, year: int, days: int) -> None:
    path = workspace / "site_config.json"
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}
    config.update(
        {
            "site_id": site,
            "workspace_stage": "static_polaris_gridmet_applied_ready_for_one_date_smoke",
            "gridmet_weather_input_application": "continuous_ir_12site_gridmet_weather_inputs_v1",
            "gridmet_year": year,
            "gridmet_days": days,
        }
    )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(report: pd.DataFrame, output_prefix: Path) -> None:
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    report.to_csv(csv_path, index=False)

    if report.empty:
        status = "no_sites"
    elif report.groupby("file")["sha256_16"].nunique().min() > 1:
        status = "weather_inputs_differ_by_site"
    else:
        status = "weather_inputs_still_identical"

    lines = [
        "# Continuous Irrigation 12-Site gridMET Weather Input Application V1",
        "",
        "## Scope",
        "",
        "- Extracts 2024 gridMET weather for each 12-site coordinate.",
        "- Updates df_gridmet.csv, weather_gridmet_out.csv, weather.024, and WeatherOriginal.024.",
        "- Does not apply S2S forecast weather.",
        "- Does not run SWAP.",
        "",
        "## Status",
        "",
        f"- status: `{status}`",
        "",
        "## Updated Files",
        "",
        markdown_table(report),
        "",
        "## Next Required Step",
        "",
        "Run a one-date 12-site SWAP smoke before launching the full 1620-row plan. "
        "If all one-date sites complete and produce non-empty candidate rows, then start the full run in the background.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    site_feature_csv = Path(args.site_feature_csv)
    workspace_root = Path(args.workspace_root)
    gridmet_dir = Path(args.gridmet_dir)
    output_prefix = Path(args.output_prefix)

    if not site_feature_csv.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {site_feature_csv}")
    if not workspace_root.exists():
        raise FileNotFoundError(
            f"Missing 12-site workspace root: {workspace_root}. "
            "Run prepare_continuous_ir_12site_workspaces_v1.py first."
        )
    if not gridmet_dir.exists():
        raise FileNotFoundError(f"Missing gridMET directory: {gridmet_dir}")

    features = pd.read_csv(site_feature_csv)
    validate_features(features)
    if args.sites:
        requested = set(args.sites)
        features = features[features["site"].astype(str).isin(requested)].copy()
        missing = sorted(requested.difference(set(features["site"].astype(str))))
        if missing:
            raise ValueError(f"Requested sites not found in feature CSV: {missing}")

    available_days = available_gridmet_days(gridmet_dir, args.year)
    if available_days < args.days and args.strict_days:
        raise ValueError(f"gridMET files only have {available_days} days, but --days requested {args.days}")
    effective_days = min(args.days, available_days)
    if effective_days != args.days:
        print(f"gridMET files have {available_days} days; using {effective_days} days for this run", flush=True)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for _, row in features.iterrows():
        site = str(row["site"])
        workspace = workspace_root / workspace_name(site)
        if not workspace.exists():
            raise FileNotFoundError(
                f"Missing prepared workspace for {site}: {workspace}. "
                "Run prepare_continuous_ir_12site_workspaces_v1.py first."
            )

        df = build_gridmet_dataframe(row, gridmet_dir, args.year, effective_days)
        weather = to_weather_out(df)
        df.to_csv(workspace / "df_gridmet.csv")
        weather.to_csv(workspace / "weather_gridmet_out.csv")
        header = weather_header(workspace / "WeatherOriginal.024")
        write_weather_file(workspace / "weather.024", weather, header)
        write_weather_file(workspace / "WeatherOriginal.024", weather, header)
        update_site_config(workspace, site, args.year, effective_days)

        for name in WEATHER_FILES:
            path = workspace / name
            report_rows.append(
                {
                    "site_id": site,
                    "longitude": float(row["lon"]),
                    "latitude": float(row["lat"]),
                    "file": name,
                    "rows": effective_days if name.endswith(".csv") else effective_days + 11,
                    "sha256_16": file_hash(path),
                }
            )

    report = pd.DataFrame(report_rows)
    write_report(report, output_prefix)

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    print("Continuous irrigation 12-site gridMET weather input application v1")
    print(f"sites: {features['site'].nunique()}")
    print(f"days: {effective_days}")
    print(f"csv: {csv_path}")
    print(f"report: {md_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
