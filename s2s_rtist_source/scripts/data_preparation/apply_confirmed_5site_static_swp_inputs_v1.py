#!/usr/bin/env python3
"""Apply confirmed 5-site static SWP inputs to isolated Maize workspaces.

This is the first true site-specific input step after the restart-generation
smoke test. It does not train a model and does not regenerate weather or
POLARIS soil curves yet. It writes the site-specific static values that are
already confirmed from the input audit into the SWAP control files:

- LAT
- ALT
- SWETR
- GWLI and groundwater table lines
- SWDRA

The script is intentionally dependency-light so it can run on the server
without rasterio/geopandas/netCDF. It uses the confirmed audit values captured
on 2026-06-02.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
CONFIRMED_WORKSPACES = OUT_DIR / "confirmed_5site_workspaces"
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"
REPORT_CSV = OUT_DIR / "confirmed_5site_static_swp_input_application_v1.csv"
REPORT_MD = OUT_DIR / "confirmed_5site_static_swp_input_application_v1.md"

SITE_TO_WORKSPACE = {
    "P1": "P1_N1_Maize",
    "P2": "P2_N2_Maize",
    "P3": "P3_N3_Maize",
    "P4": "P4_N4_Maize",
    "P15": "P15_coord_12_Maize",
}

SITE_STATIC = {
    "P1": {
        "code_site_id": "N1",
        "longitude": -98.224144,
        "latitude": 42.015928,
        "dem_value": 613.0,
        "dtw_value": 22.083223342895508,
        "tiledrain_value": 0.0,
    },
    "P2": {
        "code_site_id": "N2",
        "longitude": -88.415,
        "latitude": 40.595,
        "dem_value": 245.0,
        "dtw_value": 7.827049255371094,
        "tiledrain_value": 1.0,
    },
    "P3": {
        "code_site_id": "N3",
        "longitude": -96.877,
        "latitude": 46.321,
        "dem_value": 297.0,
        "dtw_value": 0.6152263283729553,
        "tiledrain_value": 1.0,
    },
    "P4": {
        "code_site_id": "N4",
        "longitude": -94.6686,
        "latitude": 42.6816,
        "dem_value": 382.0,
        "dtw_value": 2.9925479888916016,
        "tiledrain_value": 1.0,
    },
    "P15": {
        "code_site_id": "coord_12",
        "longitude": -112.265,
        "latitude": 41.735,
        "dem_value": 1334.0,
        "dtw_value": -0.006484962999820709,
        "tiledrain_value": 0.0,
    },
}


def dtw_to_gwli(dtw_value: float) -> float:
    if dtw_value >= 0:
        return min((-100.0) * dtw_value, -100.0)
    if dtw_value < 0:
        return -100.0
    return -200.0


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


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def update_swp_file(path: Path, meta: dict) -> dict:
    if not path.exists():
        return {"file": path.name, "status": "missing", "sha256_16": ""}

    lat = float(meta["latitude"])
    alt = round(float(meta["dem_value"]), 1)
    gwli = round(dtw_to_gwli(float(meta["dtw_value"])), 1)
    swdra = int(round(float(meta["tiledrain_value"])))

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    flags = {
        "lat": False,
        "alt": False,
        "swetr": False,
        "gwli": False,
        "gw_2010": False,
        "gw_2030": False,
        "swdra": False,
    }
    out = []
    for line in lines:
        if "LAT    =" in line and "Latitude of meteo station" in line:
            out.append(f"  LAT    =   {lat}       ! Latitude of meteo station, [-60..60 degrees, R, North = +]\n")
            flags["lat"] = True
        elif "ALT    =" in line and "Altitude of meteo station" in line:
            out.append(f"  ALT    =   {alt}       ! Altitude of meteo station, [-400..3000 m, R]\n")
            flags["alt"] = True
        elif "SWETR  =" in line and "Switch, use reference ET values of meteo file" in line:
            out.append("  SWETR  =  0           ! Switch, use reference ET values of meteo file [Y=1, N=0]\n")
            flags["swetr"] = True
        elif "GWLI   =" in line and "Initial groundwater level" in line:
            out.append(f"  GWLI   = {gwli}  ! Initial groundwater level, [-10000..100 cm, R]\n")
            flags["gwli"] = True
        elif "  01-jan-2010     " in line:
            out.append(f"  01-jan-2010     {gwli}\n")
            flags["gw_2010"] = True
        elif "  31-dec-2030     " in line:
            out.append(f"  31-dec-2030     {gwli}\n")
            flags["gw_2030"] = True
        elif "SWDRA =" in line and "Switch, simulation of lateral drainage" in line:
            out.append(f"  SWDRA = {swdra}  ! Switch, simulation of lateral drainage:\n")
            flags["swdra"] = True
        else:
            out.append(line)

    path.write_text("".join(out), encoding="utf-8")
    return {
        "file": path.name,
        "status": "updated",
        "sha256_16": file_hash(path),
        **{f"set_{key}": value for key, value in flags.items()},
    }


def write_site_config(workspace: Path, site: str, meta: dict, rows: list[dict]) -> None:
    config = {
        "paper_site_id": site,
        **meta,
        "gwli_cm": round(dtw_to_gwli(float(meta["dtw_value"])), 1),
        "static_swp_input_application": "confirmed_5site_static_swp_inputs_v1",
    }
    (workspace / "site_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)
    pd.DataFrame(rows).to_csv(workspace / "static_swp_input_application_v1.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(df: pd.DataFrame) -> None:
    view = df[
        [
            "paper_site_id",
            "code_site_id",
            "file",
            "status",
            "latitude",
            "dem_value",
            "dtw_value",
            "gwli_cm",
            "tiledrain_value",
            "sha256_16",
        ]
    ]
    unique_hashes = df.groupby("paper_site_id")["sha256_16"].agg(lambda s: ",".join(sorted(set(s))))
    lines = [
        "# Confirmed 5-Site Static SWP Input Application V1",
        "",
        "## Scope",
        "",
        "- Applies site-specific LAT/ALT/GWLI/SWDRA to SWP files.",
        "- Does not regenerate weather or POLARIS soil hydraulic curves yet.",
        "- Does not train any surrogate model.",
        "",
        "## Updated SWP Files",
        "",
        markdown_table(view),
        "",
        "## Hash Check",
        "",
        markdown_table(unique_hashes.reset_index().rename(columns={"sha256_16": "site_swp_hashes"})),
        "",
        "## Interpretation",
        "",
        "The five isolated SWAP workspaces now contain site-specific static hydrology/meteo-location parameters. "
        "Next, rerun the restart-generation smoke and check whether candidate curves differ across sites. "
        "If they still do not differ enough, the next input layer to apply is POLARIS soil hydraulics and then weather.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=sorted(SITE_TO_WORKSPACE))
    parser.add_argument("--create-missing", action="store_true")
    args = parser.parse_args()

    rows = []
    for site in args.sites:
        if site not in SITE_STATIC:
            raise ValueError(f"Unknown site: {site}")
        meta = SITE_STATIC[site]
        workspace = ensure_workspace(site, create_missing=args.create_missing)
        if not os.name == "nt":
            for exe_name in ["swap_test", "swap"]:
                exe = workspace / exe_name
                if exe.exists():
                    os.chmod(exe, exe.stat().st_mode | 0o111)

        site_rows = []
        for swp_name in ["SwapOriginal.swp", "Swap1.swp", "swap.swp"]:
            result = update_swp_file(workspace / swp_name, meta)
            row = {
                "paper_site_id": site,
                "code_site_id": meta["code_site_id"],
                "workspace": str(workspace),
                "longitude": meta["longitude"],
                "latitude": meta["latitude"],
                "dem_value": meta["dem_value"],
                "dtw_value": meta["dtw_value"],
                "gwli_cm": round(dtw_to_gwli(float(meta["dtw_value"])), 1),
                "tiledrain_value": meta["tiledrain_value"],
                **result,
            }
            rows.append(row)
            site_rows.append(row)
        write_site_config(workspace, site, meta, site_rows)

    df = pd.DataFrame(rows)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_CSV, index=False)
    write_report(df)

    print("Confirmed 5-site static SWP input application v1")
    print(f"csv: {REPORT_CSV}")
    print(f"md: {REPORT_MD}")
    print(df[["paper_site_id", "file", "status", "sha256_16"]].to_string(index=False))


if __name__ == "__main__":
    main()
