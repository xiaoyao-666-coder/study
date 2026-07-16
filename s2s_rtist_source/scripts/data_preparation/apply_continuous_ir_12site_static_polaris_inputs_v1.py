#!/usr/bin/env python3
"""Apply static SWP and POLARIS soil inputs to 12-site continuous-ir workspaces.

This is the required input layer after
prepare_continuous_ir_12site_workspaces_v1.py and before any 12-site SWAP
restart generation. It intentionally does not create missing workspaces or
fall back to the base Maize template: fallback runs would pollute the
site-general surrogate training set.

Applied inputs:
- LAT, ALT, SWETR, SWDRA in SWP files
- POLARIS soil hydraulic profile in df_polaris_soil_hydraulic.csv
- ISOILLAY1 soil hydraulic table in SwapOriginal.swp, Swap1.swp, and swap.swp

Groundwater level is only updated when a depth-to-water column is present in
the feature CSV. The current 12-site CSV does not contain that column, so this
script records `gwli_status=not_available` instead of inventing a value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
DEFAULT_SITE_FEATURE_CSV = OUT_DIR / "site_feature_screening_12_code_sites.csv"
DEFAULT_WORKSPACE_ROOT = OUT_DIR / "continuous_ir_12site_workspaces_v1"
DEFAULT_REPORT_PREFIX = OUT_DIR / "continuous_ir_12site_static_polaris_input_application_v1"

DEPTHS = [
    ("0_5", 0),
    ("5_15", 5),
    ("15_30", 15),
    ("30_60", 30),
    ("60_100", 60),
    ("100_200", 100),
]

SWP_FILES = ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--output-prefix", default=str(DEFAULT_REPORT_PREFIX))
    parser.add_argument("--sites", nargs="+", help="Optional subset of site ids.")
    return parser.parse_args()


def workspace_name(site: str) -> str:
    return f"{site}_Maize"


def required_columns() -> list[str]:
    cols = ["site", "lon", "lat", "dem_m", "tile_drain"]
    for prefix in ["theta_r", "theta_s", "alpha", "n", "ksat"]:
        for suffix, _ in DEPTHS:
            cols.append(f"{prefix}_{suffix}")
    return cols


def validate_features(df: pd.DataFrame) -> None:
    missing = [col for col in required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Site feature CSV is missing required columns: {missing}")


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def first_existing_value(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return None


def dtw_to_gwli(dtw_m: float) -> float:
    if dtw_m >= 0:
        return min((-100.0) * dtw_m, -100.0)
    return -100.0


def build_polaris_profile(row: pd.Series) -> pd.DataFrame:
    profile_rows = []
    for suffix, start_depth_cm in DEPTHS:
        alpha = 10 ** float(row[f"alpha_{suffix}"])
        ksat = 24.0 * (10 ** float(row[f"ksat_{suffix}"]))
        profile_rows.append(
            {
                "depth_start_cm": start_depth_cm,
                "theta_r_": float(row[f"theta_r_{suffix}"]),
                "theta_s_": float(row[f"theta_s_{suffix}"]),
                "alpha_": alpha,
                "n_": 5.0 * float(row[f"n_{suffix}"]),
                "ksat_": ksat,
                "alphaw_": 2.0 * alpha,
            }
        )
    return pd.DataFrame(profile_rows).set_index("depth_start_cm")


def profile_to_swp_rows(profile: pd.DataFrame) -> list[str]:
    rows = []
    for idx, row in enumerate(profile.itertuples(), start=1):
        rows.append(
            f"{idx} {row.theta_r_:.3f} {row.theta_s_:.3f} {row.alpha_:.4f} "
            f"{row.n_:.3f} {row.ksat_:.3f} 0.500 {row.alphaw_:.4f}  -4.0\n"
        )
    return rows


def update_swp_static_and_soil(path: Path, row: pd.Series, profile: pd.DataFrame) -> dict[str, object]:
    if not path.exists():
        return {"file": path.name, "status": "missing", "sha256_16": ""}

    lat = float(row["lat"])
    alt = round(float(row["dem_m"]), 1)
    swdra = int(round(float(row["tile_drain"])))
    dtw_m = first_existing_value(row, ["dtw_m", "dtw_value", "depth_to_water_m", "depth_to_water"])
    gwli = None if dtw_m is None else round(dtw_to_gwli(dtw_m), 1)

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    flags: dict[str, object] = {
        "set_lat": False,
        "set_alt": False,
        "set_swetr": False,
        "set_swdra": False,
        "set_gwli": False,
        "set_gw_2010": False,
        "set_gw_2030": False,
        "soil_rows_written": 0,
        "gwli_status": "not_available" if gwli is None else "updated",
    }

    out = []
    header_idx_in_out = None
    skip_until_soil_end = False
    soil_table_found = False
    soil_table_closed = False

    for line in lines:
        if skip_until_soil_end:
            if line.startswith("* --- end of table"):
                out.append(line)
                skip_until_soil_end = False
                soil_table_closed = True
            continue

        if line.lstrip().startswith("ISOILLAY1") and "ORES" in line and "KSAT" in line:
            out.append(line)
            header_idx_in_out = len(out) - 1
            out.extend(profile_to_swp_rows(profile))
            flags["soil_rows_written"] = len(profile)
            soil_table_found = True
            skip_until_soil_end = True
        elif "LAT    =" in line and "Latitude of meteo station" in line:
            out.append(f"  LAT    =   {lat}       ! Latitude of meteo station, [-60..60 degrees, R, North = +]\n")
            flags["set_lat"] = True
        elif "ALT    =" in line and "Altitude of meteo station" in line:
            out.append(f"  ALT    =   {alt}       ! Altitude of meteo station, [-400..3000 m, R]\n")
            flags["set_alt"] = True
        elif "SWETR  =" in line and "Switch, use reference ET values of meteo file" in line:
            out.append("  SWETR  =  0           ! Switch, use reference ET values of meteo file [Y=1, N=0]\n")
            flags["set_swetr"] = True
        elif gwli is not None and "GWLI   =" in line and "Initial groundwater level" in line:
            out.append(f"  GWLI   = {gwli}  ! Initial groundwater level, [-10000..100 cm, R]\n")
            flags["set_gwli"] = True
        elif gwli is not None and "  01-jan-2010     " in line:
            out.append(f"  01-jan-2010     {gwli}\n")
            flags["set_gw_2010"] = True
        elif gwli is not None and "  31-dec-2030     " in line:
            out.append(f"  31-dec-2030     {gwli}\n")
            flags["set_gw_2030"] = True
        elif "SWDRA =" in line and "Switch, simulation of lateral drainage" in line:
            out.append(f"  SWDRA = {swdra}  ! Switch, simulation of lateral drainage:\n")
            flags["set_swdra"] = True
        else:
            out.append(line)

    if not soil_table_found:
        flags["soil_status"] = "missing_isoillay1_header"
        status = "partial_static_only"
    elif header_idx_in_out is None or not soil_table_closed:
        flags["soil_status"] = "missing_isoillay1_end"
        status = "partial_static_only"
    else:
        flags["soil_status"] = "updated"
        status = "updated"

    path.write_text("".join(out), encoding="utf-8")
    return {
        "file": path.name,
        "status": status,
        "sha256_16": file_hash(path),
        "lat": lat,
        "alt_m": alt,
        "tile_drain": swdra,
        "dtw_m": "" if dtw_m is None else dtw_m,
        "gwli_cm": "" if gwli is None else gwli,
        **flags,
    }


def write_workspace_profile(workspace: Path, site: str, profile: pd.DataFrame) -> str:
    csv_profile = profile[["theta_r_", "theta_s_", "alpha_", "n_", "ksat_", "alphaw_"]]
    out = workspace / "df_polaris_soil_hydraulic.csv"
    csv_profile.to_csv(out)
    profile.reset_index().assign(site_id=site).to_csv(
        workspace / "continuous_ir_12site_polaris_soil_hydraulic_application_v1.csv",
        index=False,
    )
    return file_hash(out)


def update_site_config(workspace: Path, row: pd.Series, df_polaris_hash: str) -> None:
    path = workspace / "site_config.json"
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    dtw_m = first_existing_value(row, ["dtw_m", "dtw_value", "depth_to_water_m", "depth_to_water"])
    config.update(
        {
            "site_id": str(row["site"]),
            "code_site_id": str(row["site"]).replace("code_", ""),
            "longitude": float(row["lon"]),
            "latitude": float(row["lat"]),
            "dem_m": float(row["dem_m"]),
            "tile_drain": float(row["tile_drain"]),
            "workspace_stage": "static_polaris_applied_requires_gridmet_inputs",
            "static_polaris_input_application": "continuous_ir_12site_static_polaris_inputs_v1",
            "df_polaris_sha256_16": df_polaris_hash,
            "gwli_status": "not_available" if dtw_m is None else "updated",
        }
    )
    if dtw_m is not None:
        config["dtw_m"] = dtw_m
        config["gwli_cm"] = round(dtw_to_gwli(dtw_m), 1)

    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(report: pd.DataFrame, profiles: pd.DataFrame, output_prefix: Path) -> None:
    csv_path = output_prefix.with_suffix(".csv")
    profile_path = output_prefix.with_name(output_prefix.name + "_profiles").with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    report.to_csv(csv_path, index=False)
    profiles.to_csv(profile_path, index=False)

    site_summary = (
        profiles.groupby("site_id")
        .agg(
            theta_r_mean=("theta_r_", "mean"),
            theta_s_mean=("theta_s_", "mean"),
            alpha_mean=("alpha_", "mean"),
            n_mean=("n_", "mean"),
            ksat_mean=("ksat_", "mean"),
            ksat_min=("ksat_", "min"),
            ksat_max=("ksat_", "max"),
        )
        .reset_index()
    )
    status_summary = (
        report.groupby(["site_id", "status", "soil_status", "gwli_status"])
        .size()
        .reset_index(name="files")
    )
    view_cols = [
        "site_id",
        "file",
        "status",
        "soil_status",
        "gwli_status",
        "set_lat",
        "set_alt",
        "set_swdra",
        "soil_rows_written",
        "df_polaris_sha256_16",
        "sha256_16",
    ]
    lines = [
        "# Continuous Irrigation 12-Site Static + POLARIS Input Application V1",
        "",
        "## Scope",
        "",
        "- Applies site-specific LAT/ALT/SWETR/SWDRA from the 12-site feature CSV.",
        "- Rebuilds POLARIS soil hydraulic profiles using the original project transforms.",
        "- Rewrites the SWP ISOILLAY1 table in SwapOriginal.swp, Swap1.swp, and swap.swp.",
        "- Does not apply gridMET weather yet.",
        "- Does not update groundwater level unless a depth-to-water column exists.",
        "",
        "## Status Summary",
        "",
        markdown_table(status_summary),
        "",
        "## Site Soil Summary",
        "",
        markdown_table(site_summary),
        "",
        "## Updated Files",
        "",
        markdown_table(report[view_cols]),
        "",
        "## Next Required Step",
        "",
        "Run a short audit of this report. If all SWP files show `status=updated` and "
        "`soil_rows_written=6`, continue to the 12-site gridMET weather input layer. "
        "Do not start the 1620-row SWAP generation until gridMET is applied and a "
        "one-date smoke passes.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    site_feature_csv = Path(args.site_feature_csv)
    workspace_root = Path(args.workspace_root)
    output_prefix = Path(args.output_prefix)

    if not site_feature_csv.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {site_feature_csv}")
    if not workspace_root.exists():
        raise FileNotFoundError(
            f"Missing 12-site workspace root: {workspace_root}. "
            "Run prepare_continuous_ir_12site_workspaces_v1.py first."
        )

    features = pd.read_csv(site_feature_csv)
    validate_features(features)
    if args.sites:
        requested = set(args.sites)
        features = features[features["site"].astype(str).isin(requested)].copy()
        missing = sorted(requested.difference(set(features["site"].astype(str))))
        if missing:
            raise ValueError(f"Requested sites not found in feature CSV: {missing}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_rows = []
    profile_rows = []

    for _, row in features.iterrows():
        site = str(row["site"])
        workspace = workspace_root / workspace_name(site)
        if not workspace.exists():
            raise FileNotFoundError(
                f"Missing prepared workspace for {site}: {workspace}. "
                "Run prepare_continuous_ir_12site_workspaces_v1.py first."
            )
        if not os.name == "nt":
            for exe_name in ["swap_test", "swap"]:
                exe = workspace / exe_name
                if exe.exists():
                    os.chmod(exe, exe.stat().st_mode | 0o111)

        profile = build_polaris_profile(row)
        df_hash = write_workspace_profile(workspace, site, profile)
        update_site_config(workspace, row, df_hash)

        for profile_row in profile.reset_index().to_dict(orient="records"):
            profile_rows.append({"site_id": site, **profile_row})

        for swp_name in SWP_FILES:
            result = update_swp_static_and_soil(workspace / swp_name, row, profile)
            report_rows.append(
                {
                    "site_id": site,
                    "workspace": str(workspace),
                    "df_polaris_sha256_16": df_hash,
                    **result,
                }
            )

    report = pd.DataFrame(report_rows)
    profiles = pd.DataFrame(profile_rows)
    write_report(report, profiles, output_prefix)

    csv_path = output_prefix.with_suffix(".csv")
    profile_path = output_prefix.with_name(output_prefix.name + "_profiles").with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    updated = int((report["status"] == "updated").sum()) if not report.empty else 0
    print("Continuous irrigation 12-site static + POLARIS input application v1")
    print(f"sites: {features['site'].nunique()}")
    print(f"updated_swp_files: {updated} / {len(report)}")
    print(f"csv: {csv_path}")
    print(f"profiles: {profile_path}")
    print(f"report: {md_path}")
    print(report[["site_id", "file", "status", "soil_status", "gwli_status", "soil_rows_written"]].to_string(index=False))


if __name__ == "__main__":
    main()
