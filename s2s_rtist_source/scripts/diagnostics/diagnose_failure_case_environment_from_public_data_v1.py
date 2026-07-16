#!/usr/bin/env python3
"""Diagnose failure-case environments using the public paper data package.

This script does not train a surrogate model. It only joins existing failure
diagnostics with environment variables recoverable from the released data
package and previously extracted site-feature table.

Outputs:
- site_environment_failure_summary_v1.csv
- environment_feature_contrast_v1.csv
- worst_date_environment_windows_v1.csv
- public_data_failure_environment_diagnostic_v1.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SITE_FEATURE_CSV = (
    Path("site_general_surrogate_eval") / "site_feature_screening_12_code_sites.csv"
)
DEFAULT_FAILURE_SITE_CSV = (
    Path("teacher_persite_expert_evidence_pack_20260608")
    / "teacher_persite_expert_evidence_pack_20260608"
    / "summary_failure_site_comparison.csv"
)
DEFAULT_WORST_DATE_CSV = (
    Path("teacher_persite_expert_evidence_pack_20260608")
    / "teacher_persite_expert_evidence_pack_20260608"
    / "summary_worst_date_examples.csv"
)
DEFAULT_GRIDMET_DIR = Path("model3_opt_sto_upload") / "data" / "gridmet"
DEFAULT_DTW_RASTER = Path("model3_opt_sto_upload") / "data" / "dtw" / "dtw250.tif"
DEFAULT_OUTPUT_DIR = (
    Path("site_general_surrogate_eval")
    / "public_data_failure_environment_diagnostic_v1"
)


GRIDMET_VARIABLES = {
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "pr": "precipitation_amount",
    "vs": "wind_speed",
    "vpd": "mean_vapor_pressure_deficit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--failure-site-csv", default=str(DEFAULT_FAILURE_SITE_CSV))
    parser.add_argument("--worst-date-csv", default=str(DEFAULT_WORST_DATE_CSV))
    parser.add_argument("--gridmet-dir", default=str(DEFAULT_GRIDMET_DIR))
    parser.add_argument("--dtw-raster", default=str(DEFAULT_DTW_RASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--pre-window-days", type=int, default=7)
    parser.add_argument("--post-window-days", type=int, default=7)
    parser.add_argument("--top-n-worst", type=int, default=15)
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def normalize_site_id(value: str) -> str:
    value = str(value)
    if value.startswith("code_"):
        return value
    return f"code_{value}"


def add_soil_and_climate_proxies(site_features: pd.DataFrame) -> pd.DataFrame:
    df = site_features.copy()
    df["site_id"] = df["site"].map(normalize_site_id)

    if {"theta_s_0_60_mean", "theta_r_0_60_mean"}.issubset(df.columns):
        df["plant_available_water_proxy_0_60"] = (
            df["theta_s_0_60_mean"] - df["theta_r_0_60_mean"]
        )
    if {"pet_sum", "precip_sum"}.issubset(df.columns):
        df["available_window_water_deficit_proxy"] = df["pet_sum"] - df["precip_sum"]
    if {"rad_mean", "precip_sum"}.issubset(df.columns):
        df["radiation_to_precip_proxy"] = df["rad_mean"] / (df["precip_sum"].abs() + 1e-6)
    if {"ksat_0_60_mean", "tile_drain"}.issubset(df.columns):
        df["drainage_permeability_proxy"] = df["ksat_0_60_mean"] + df["tile_drain"]

    return df


def sample_raster_at_sites(path: Path, sites: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = sites[["site_id", "lon", "lat"]].copy()
    out[value_name] = np.nan
    out[f"{value_name}_status"] = "not_attempted"
    if not path.exists():
        out[f"{value_name}_status"] = "missing_raster"
        return out

    try:
        import rasterio
    except ModuleNotFoundError:
        out[f"{value_name}_status"] = "missing_rasterio"
        return out

    try:
        with rasterio.open(path) as src:
            transformer = None
            if src.crs is not None:
                try:
                    from rasterio.warp import transform

                    transformer = transform
                except Exception:
                    transformer = None
            for idx, row in out.iterrows():
                try:
                    x = float(row["lon"])
                    y = float(row["lat"])
                    if src.crs is not None and transformer is not None:
                        xs, ys = transformer("EPSG:4326", src.crs, [x], [y])
                        x, y = float(xs[0]), float(ys[0])
                    r, c = src.index(x, y)
                    if not (0 <= r < src.height and 0 <= c < src.width):
                        out.loc[idx, f"{value_name}_status"] = "out_of_bounds"
                        continue
                    val = src.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]
                    if src.nodata is not None and float(val) == float(src.nodata):
                        out.loc[idx, f"{value_name}_status"] = "nodata"
                    else:
                        out.loc[idx, value_name] = float(val)
                        out.loc[idx, f"{value_name}_status"] = "sampled"
                except Exception as exc:  # pragma: no cover - diagnostic status
                    out.loc[idx, f"{value_name}_status"] = f"error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - diagnostic status
        out[f"{value_name}_status"] = f"open_error:{type(exc).__name__}"
    return out


def classify_failure_sites(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["main_failure_site"] = (
        (df["curve_top_cv_continuous_regret"] >= 20.0)
        | (df["fixedlist_ranker_regret_vs_SWAP_fixed_oracle"] >= 7.0)
    )
    df["stable_site"] = (
        (df["curve_top_cv_continuous_regret"] <= 2.0)
        & (df["fixedlist_ranker_regret_vs_SWAP_fixed_oracle"] <= 2.0)
    )
    df["failure_group"] = np.select(
        [df["main_failure_site"], df["stable_site"]],
        ["main_failure", "stable_or_low_failure"],
        default="intermediate",
    )
    return df


def numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    exclude_set = set(exclude)
    return [
        col
        for col in df.columns
        if col not in exclude_set and pd.api.types.is_numeric_dtype(df[col])
    ]


def feature_contrast(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {
        "site_dates",
        "paper_fixed_list_regret",
        "curve_top_cv_continuous_regret",
        "curve_top_cv_p90_regret",
        "curve_top_large_regret_gt5_rate",
        "offlist_large_failure_rate",
        "fixedgrid_large_failure_rate",
        "fixedlist_ranker_regret_vs_SWAP_fixed_oracle",
        "fixedlist_learned_decision_regret_vs_dense",
        "fixedlist_learned_large_regret_gt5_rate",
    }
    fail = df.loc[df["main_failure_site"]].copy()
    stable = df.loc[df["stable_site"]].copy()
    rows = []
    for col in numeric_feature_columns(df, exclude):
        fail_vals = fail[col].dropna().astype(float)
        stable_vals = stable[col].dropna().astype(float)
        if len(fail_vals) < 2 or len(stable_vals) < 2:
            continue
        pooled = float(np.sqrt(0.5 * (fail_vals.var(ddof=1) + stable_vals.var(ddof=1))))
        diff = float(fail_vals.mean() - stable_vals.mean())
        rows.append(
            {
                "feature": col,
                "main_failure_n": int(len(fail_vals)),
                "stable_n": int(len(stable_vals)),
                "main_failure_mean": float(fail_vals.mean()),
                "stable_mean": float(stable_vals.mean()),
                "mean_diff_failure_minus_stable": diff,
                "standardized_diff": np.nan if pooled == 0 else diff / pooled,
                "main_failure_median": float(fail_vals.median()),
                "stable_median": float(stable_vals.median()),
            }
        )
    contrast = pd.DataFrame(rows)
    if contrast.empty:
        return contrast
    contrast["abs_standardized_diff"] = contrast["standardized_diff"].abs()
    return contrast.sort_values("abs_standardized_diff", ascending=False)


def parse_public_date(value: str, year: int) -> datetime:
    parsed = datetime.strptime(str(value), "%d-%b-%Y")
    if parsed.year != year:
        return parsed
    return parsed


@dataclass
class GridmetCache:
    gridmet_dir: Path
    year: int
    series_by_site: dict[tuple[str, str], np.ndarray]

    def read_series(self, site_id: str, lon: float, lat: float, variable: str) -> np.ndarray:
        key = (site_id, variable)
        if key not in self.series_by_site:
            self.series_by_site[key] = read_gridmet_series(
                self.gridmet_dir,
                self.year,
                variable,
                GRIDMET_VARIABLES[variable],
                lat,
                lon,
            )
        return self.series_by_site[key]


def apply_gridmet_units(variable: str, values: np.ndarray) -> np.ndarray:
    series = values.astype(float)
    if variable in {"tmmn", "tmmx"}:
        series = series - 273.15
    elif variable == "srad":
        series = series * 86.4
    elif variable == "pr":
        series = np.maximum(0.0, series)
    elif variable in {"vs", "vpd"}:
        series = np.maximum(0.0, series)
    return series


def read_gridmet_series(
    gridmet_dir: Path,
    year: int,
    variable: str,
    variable_name: str,
    lat: float,
    lon: float,
) -> np.ndarray:
    path = gridmet_dir / f"{variable}_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing gridMET file: {path}")

    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("gridMET extraction requires rasterio") from exc

    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(f"Coordinate out of bounds for {path}: lat={lat}, lon={lon}")
        raw = []
        scale_factor = None
        add_offset = None
        for band in range(1, src.count + 1):
            tags = src.tags(band)
            scale_factor = tags.get("scale_factor", scale_factor)
            add_offset = tags.get("add_offset", add_offset)
            nc_var = tags.get("NETCDF_VARNAME")
            if nc_var and nc_var != variable_name:
                raise ValueError(
                    f"Unexpected variable in {path}: expected {variable_name}, found {nc_var}"
                )
            raw.append(src.read(band, window=((row, row + 1), (col, col + 1)))[0, 0])
    scale = 1.0 if scale_factor in {None, ""} else float(scale_factor)
    offset = 0.0 if add_offset in {None, ""} else float(add_offset)
    return apply_gridmet_units(variable, np.asarray(raw, dtype=float) * scale + offset)


def summarize_window(series: np.ndarray, start_idx: int, end_idx: int) -> dict[str, float]:
    start = max(0, start_idx)
    end = min(len(series), end_idx)
    values = series[start:end]
    if len(values) == 0:
        return {"mean": np.nan, "sum": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.nanmean(values)),
        "sum": float(np.nansum(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def add_worst_date_weather_windows(
    worst: pd.DataFrame,
    site_env: pd.DataFrame,
    gridmet_dir: Path,
    year: int,
    pre_window_days: int,
    post_window_days: int,
    top_n: int,
) -> pd.DataFrame:
    worst_top = (
        worst.sort_values("continuous_top_ranker_regret_vs_dense_oracle", ascending=False)
        .head(top_n)
        .copy()
    )
    site_lookup = site_env.set_index("site_id")
    cache = GridmetCache(gridmet_dir=gridmet_dir, year=year, series_by_site={})
    rows = []
    for _, row in worst_top.iterrows():
        site_id = normalize_site_id(row["site_id"])
        out = row.to_dict()
        if site_id not in site_lookup.index:
            out["gridmet_window_status"] = "site_missing"
            rows.append(out)
            continue

        site_row = site_lookup.loc[site_id]
        date_t = parse_public_date(row["date_t"], year)
        doy_idx = int(date_t.timetuple().tm_yday) - 1
        try:
            for variable in GRIDMET_VARIABLES:
                series = cache.read_series(
                    site_id,
                    float(site_row["lon"]),
                    float(site_row["lat"]),
                    variable,
                )
                pre = summarize_window(series, doy_idx - pre_window_days, doy_idx)
                post = summarize_window(series, doy_idx, doy_idx + post_window_days)
                for stat, value in pre.items():
                    out[f"pre{pre_window_days}_{variable}_{stat}"] = value
                for stat, value in post.items():
                    out[f"post{post_window_days}_{variable}_{stat}"] = value

            out["pre_window_start"] = (date_t - timedelta(days=pre_window_days)).strftime(
                "%Y-%m-%d"
            )
            out["pre_window_end"] = (date_t - timedelta(days=1)).strftime("%Y-%m-%d")
            out["post_window_start"] = date_t.strftime("%Y-%m-%d")
            out["post_window_end"] = (date_t + timedelta(days=post_window_days - 1)).strftime(
                "%Y-%m-%d"
            )
            out["gridmet_window_status"] = "sampled"
        except Exception as exc:  # pragma: no cover - diagnostic status
            out["gridmet_window_status"] = f"error:{type(exc).__name__}:{exc}"
        rows.append(out)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 12, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    columns = list(view.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in view.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                text = ""
            else:
                text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    out_dir: Path,
    site_env: pd.DataFrame,
    contrast: pd.DataFrame,
    worst_windows: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    main_sites = site_env.loc[site_env["main_failure_site"], "site_id"].tolist()
    stable_sites = site_env.loc[site_env["stable_site"], "site_id"].tolist()
    top_contrast_cols = [
        "feature",
        "main_failure_mean",
        "stable_mean",
        "mean_diff_failure_minus_stable",
        "standardized_diff",
    ]
    worst_cols = [
        "site_id",
        "date_t",
        "dense_oracle_ir",
        "continuous_top_ranker_ir",
        "continuous_top_ranker_regret_vs_dense_oracle",
        f"pre{args.pre_window_days}_pr_sum",
        f"post{args.post_window_days}_pr_sum",
        f"post{args.post_window_days}_tmmx_mean",
        f"post{args.post_window_days}_srad_mean",
        "gridmet_window_status",
    ]

    report = f"""# Failure cases 环境诊断（公开数据包）

## 目的

这份诊断只回答一个问题：当前 held-out-date failure 是否集中在某些站点环境或最坏日期天气背景下。

它不训练新模型，也不把 TinyForest/RandomForest 作为后续 TTA 模型；这里只把已有 failure 结果和原论文公开包可恢复的环境数据拼起来，用于定位过拟合/分布外泛化的可能来源。

## 数据来源

- 失败结果：`{args.failure_site_csv}` 和 `{args.worst_date_csv}`
- 站点静态环境：`{args.site_feature_csv}`，该表由公开包中的坐标、DEM、POLARIS、tile drainage 和可恢复气象输入整理得到
- 最坏日期天气窗口：`{args.gridmet_dir}` 中 2024 gridMET NetCDF
- 地下水深度尝试抽取：`{args.dtw_raster}`

## 分组定义

- main_failure_site：`curve_top_cv_continuous_regret >= 20` 或 `fixedlist_ranker_regret_vs_SWAP_fixed_oracle >= 7`
- stable_site：`curve_top_cv_continuous_regret <= 2` 且 `fixedlist_ranker_regret_vs_SWAP_fixed_oracle <= 2`

这两个阈值只用于诊断分组，不是模型训练规则。

## 站点分组

- main_failure_site：{', '.join(main_sites)}
- stable_site：{', '.join(stable_sites)}

## 失败站点 vs 稳定站点：环境差异最大的变量

{markdown_table(contrast[[c for c in top_contrast_cols if c in contrast.columns]], max_rows=15)}

## 最坏日期的天气窗口

下面窗口中，`pre{args.pre_window_days}` 表示决策日前 {args.pre_window_days} 天，不含决策日；`post{args.post_window_days}` 表示决策日开始后 {args.post_window_days} 天。

{markdown_table(worst_windows[[c for c in worst_cols if c in worst_windows.columns]], max_rows=15)}

## 初步解释

1. 这一步主要是 failure case 的环境定位，而不是证明某个新方法有效。
2. 如果 main_failure_site 在土壤持水能力、渗透/排水 proxy、降水-辐射-温度窗口上与 stable_site 明显不同，说明当前训练样本在这些环境组合上的覆盖不足。
3. 最坏日期普遍表现为 SWAP oracle 需要较大灌溉量，但学习模型选择 0 或接近 0；因此后续应优先检查这些日期的状态特征是否能表达作物需水、土壤亏缺和未来天气需求。
4. 当前本地 evidence pack 只有摘要级 failure 表；如果后续同步完整 decision/sample 表，可以把本诊断扩展到全部 324 个 site-date，而不只看最坏日期。
"""
    path = out_dir / "public_data_failure_environment_diagnostic_v1.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    site_feature_csv = Path(args.site_feature_csv)
    failure_site_csv = Path(args.failure_site_csv)
    worst_date_csv = Path(args.worst_date_csv)
    gridmet_dir = Path(args.gridmet_dir)
    dtw_raster = Path(args.dtw_raster)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    site_features = add_soil_and_climate_proxies(read_csv_required(site_feature_csv))
    failure_sites = read_csv_required(failure_site_csv).copy()
    failure_sites["site_id"] = failure_sites["site_id"].map(normalize_site_id)
    worst_dates = read_csv_required(worst_date_csv).copy()
    worst_dates["site_id"] = worst_dates["site_id"].map(normalize_site_id)

    dtw = sample_raster_at_sites(dtw_raster, site_features, "dtw250_value")
    site_features = site_features.merge(
        dtw[["site_id", "dtw250_value", "dtw250_value_status"]],
        on="site_id",
        how="left",
    )

    site_env = site_features.merge(failure_sites, on="site_id", how="left")
    site_env = classify_failure_sites(site_env)
    contrast = feature_contrast(site_env)
    worst_windows = add_worst_date_weather_windows(
        worst_dates,
        site_env,
        gridmet_dir,
        int(args.year),
        int(args.pre_window_days),
        int(args.post_window_days),
        int(args.top_n_worst),
    )
    worst_windows = worst_windows.merge(
        site_env[
            [
                "site_id",
                "lon",
                "lat",
                "dem_m",
                "tile_drain",
                "theta_s_0_60_mean",
                "theta_r_0_60_mean",
                "plant_available_water_proxy_0_60",
                "ksat_0_60_mean",
                "drainage_permeability_proxy",
                "dtw250_value",
                "dtw250_value_status",
                "failure_group",
            ]
        ],
        on="site_id",
        how="left",
    )

    site_env.to_csv(out_dir / "site_environment_failure_summary_v1.csv", index=False)
    contrast.to_csv(out_dir / "environment_feature_contrast_v1.csv", index=False)
    worst_windows.to_csv(out_dir / "worst_date_environment_windows_v1.csv", index=False)
    report = write_report(out_dir, site_env, contrast, worst_windows, args)

    print("Failure-case public-data environment diagnostic")
    print(f"output_dir: {out_dir}")
    print(f"site_summary: {out_dir / 'site_environment_failure_summary_v1.csv'}")
    print(f"feature_contrast: {out_dir / 'environment_feature_contrast_v1.csv'}")
    print(f"worst_windows: {out_dir / 'worst_date_environment_windows_v1.csv'}")
    print(f"report: {report}")
    print()
    print("Main failure sites:")
    print(site_env.loc[site_env["main_failure_site"], ["site_id", "curve_top_cv_continuous_regret", "fixedlist_ranker_regret_vs_SWAP_fixed_oracle"]].to_string(index=False))


if __name__ == "__main__":
    main()
