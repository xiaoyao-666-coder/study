#!/usr/bin/env python3
"""Audit multi-site inputs before training any universal surrogate model.

The audit is intentionally lightweight: it does not extract raster values or
run SWAP. It checks which candidate site coordinates are visible in Main_win.py
and whether the data/source directories needed for multi-site generation exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import pandas as pd


DEFAULT_PACKAGE_ROOT = Path("model3_opt_sto_upload")
OUT_DIR = Path("site_general_surrogate_eval")

EXPECTED_DATA_DIRS = [
    "CropAT_US",
    "dem",
    "dtw",
    "gridmet",
    "polaris",
    "tiledrain",
    "era5_2015",
    "era5_2016",
    "era5_2017",
    "era5_2018",
    "era5_2019",
    "lai_2015",
    "lai_2016",
    "lai_2017",
    "lai_2018",
    "lai_2019",
]

EXPECTED_MAIZE_FILES = [
    "ForecastStep.py",
    "use_s2s.py",
    "Extract_tif.py",
    "download_extract_nc_gridmet.py",
    "swap_test",
    "swap",
    "Swap.exe",
    "swap.swp",
    "SwapOriginal.swp",
    "GmaizeDOriginal.crp",
    "df_gridmet.csv",
    "weather_s2s_out.csv",
    "df_polaris_soil_hydraulic.csv",
    "SoilPhysParam.csv",
]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def extension_counts(path: Path) -> str:
    if not path.exists() or not path.is_dir():
        return ""
    counter: Counter[str] = Counter()
    for item in path.rglob("*"):
        if item.is_file():
            suffix = item.suffix.lower() or "<no_ext>"
            counter[suffix] += 1
    if not counter:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def parse_sites_from_main(main_py: Path) -> pd.DataFrame:
    if not main_py.exists():
        raise FileNotFoundError(main_py)
    lines = main_py.read_text(encoding="utf-8", errors="ignore").splitlines()

    rows = []
    pending_label = None
    pending_lon = None
    source_line = None

    label_patterns = [
        re.compile(r"site\s+([A-Za-z]\d+)", re.IGNORECASE),
        re.compile(r"\((site\s+[A-Za-z]\d+)\)", re.IGNORECASE),
    ]

    def label_from_line(text: str) -> str | None:
        for pattern in label_patterns:
            m = pattern.search(text)
            if m:
                label = m.group(1)
                label = label.replace("site", "").replace("SITE", "").strip()
                return label
        return None

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        label = label_from_line(stripped)
        if label:
            pending_label = label

        lon_match = re.search(r"longitude\s*=\s*([-+]?\d+(?:\.\d+)?)", stripped)
        lat_match = re.search(r"latitude\s*=\s*([-+]?\d+(?:\.\d+)?)", stripped)

        if lon_match:
            pending_lon = float(lon_match.group(1))
            source_line = idx
        if lat_match and pending_lon is not None:
            lat = float(lat_match.group(1))
            lon = pending_lon
            is_commented = stripped.startswith("#")
            site_id = pending_label or f"coord_{len(rows) + 1:02d}"
            rows.append(
                {
                    "site_id": site_id,
                    "longitude": lon,
                    "latitude": lat,
                    "source": "commented_coordinate" if is_commented else "active_assignment",
                    "main_win_line": source_line,
                    "status": "coordinate_available",
                }
            )
            pending_lon = None
            source_line = None
            pending_label = None

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["duplicate_coordinate"] = df.duplicated(["longitude", "latitude"], keep=False)
    df["needs_paper_p_id_mapping"] = ~df["site_id"].astype(str).str.match(r"^P\d+$")
    return df


def audit_data_sources(package_root: Path) -> pd.DataFrame:
    data_dir = package_root / "data"
    rows = []
    for name in EXPECTED_DATA_DIRS:
        path = data_dir / name
        rows.append(
            {
                "category": "data_dir",
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "file_count_by_extension": extension_counts(path),
            }
        )

    maize_dir = package_root / "Maize"
    for name in EXPECTED_MAIZE_FILES:
        path = maize_dir / name
        rows.append(
            {
                "category": "maize_file",
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "file_count_by_extension": "",
            }
        )
    return pd.DataFrame(rows)


def build_report(site_df: pd.DataFrame, source_df: pd.DataFrame, package_root: Path) -> str:
    present_sources = int(source_df["exists"].sum())
    missing_sources = source_df[~source_df["exists"]]
    site_count = len(site_df)
    active = site_df[site_df["source"] == "active_assignment"] if not site_df.empty else site_df

    report = [
        "# Multi-Site Input Audit V1",
        "",
        f"Package root: `{package_root}`",
        "",
        "## Summary",
        "",
        f"- Candidate coordinates found in `Main_win.py`: {site_count}",
        f"- Active assignment coordinates: {len(active)}",
        f"- Expected data/source entries present: {present_sources}/{len(source_df)}",
        "- This audit does not train a model, extract raster values, or run SWAP.",
        "",
        "## Candidate Site Coordinates",
        "",
        markdown_table(site_df),
        "",
        "## Missing Data/Source Entries",
        "",
        markdown_table(missing_sources[["category", "name", "path"]]),
        "",
        "## Data Source Availability",
        "",
        markdown_table(source_df),
        "",
        "## Interpretation",
        "",
        "- The package appears to use one mutable `Maize` working directory, while multiple site coordinates are embedded in `Main_win.py` comments and active assignments.",
        "- Before leave-one-site-out, each coordinate should be mapped to the paper's P1-P15 site IDs.",
        "- Static attributes can be extracted later from `data/polaris`, `data/dem`, `data/dtw`, and `data/tiledrain`, but that requires the geospatial extraction stack used by `Maize/Extract_tif.py`.",
        "- The next safe step is not model training; it is building a canonical site inventory table with paper IDs, coordinates, and extracted static attributes.",
    ]
    return "\n".join(report) + "\n"


def build_inventory_template(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "paper_site_id": "",
            "code_site_id": site_df["site_id"],
            "longitude": site_df["longitude"],
            "latitude": site_df["latitude"],
            "coordinate_source": site_df["source"],
            "main_win_line": site_df["main_win_line"],
            "paper_id_mapping_status": "pending",
            "static_attribute_status": "pending_extract",
            "swap_input_status": "single_mutable_maize_workspace",
            "multisite_generation_status": "not_started",
            "notes": "",
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT), help="Path to model3_opt_sto_upload.")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Directory for audit outputs.")
    args = parser.parse_args()

    package_root = Path(args.package_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_df = parse_sites_from_main(package_root / "Main_win.py")
    source_df = audit_data_sources(package_root)
    inventory_df = build_inventory_template(site_df)

    site_out = output_dir / "multisite_input_audit_v1.csv"
    source_out = output_dir / "multisite_data_source_audit_v1.csv"
    inventory_out = output_dir / "multisite_canonical_site_inventory_template_v1.csv"
    report_out = output_dir / "multisite_input_audit_v1.md"

    site_df.to_csv(site_out, index=False)
    source_df.to_csv(source_out, index=False)
    inventory_df.to_csv(inventory_out, index=False)
    report_out.write_text(build_report(site_df, source_df, package_root), encoding="utf-8")

    print("Multi-site input audit v1")
    print("")
    print(f"candidate coordinates: {len(site_df)}")
    print(f"data/source entries present: {int(source_df['exists'].sum())}/{len(source_df)}")
    print("")
    print(f"wrote: {site_out}")
    print(f"wrote: {source_out}")
    print(f"wrote: {inventory_out}")
    print(f"wrote: {report_out}")


if __name__ == "__main__":
    main()
