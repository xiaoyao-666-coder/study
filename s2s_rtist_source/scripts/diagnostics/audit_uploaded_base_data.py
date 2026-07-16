"""Audit uploaded base data after extracting model3_opt_sto_upload.zip.

Run from:

    /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source

It produces a compact report that can be kept with experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("model3_opt_sto_upload")
DATA = BASE / "data"
OUT_DIR = Path("report_artifacts/2026-05-30_restart_surrogate_smoke")
CSV_OUT = OUT_DIR / "uploaded_base_data_inventory.csv"
MD_OUT = OUT_DIR / "uploaded_base_data_audit.md"

EXPECTED_TOP_LEVEL = [
    "CropAT_US",
    "dem",
    "dtw",
    "era5_2015",
    "era5_2016",
    "era5_2017",
    "era5_2018",
    "era5_2019",
    "gridmet",
    "lai_2015",
    "lai_2016",
    "lai_2017",
    "lai_2018",
    "lai_2019",
    "polaris",
    "tiledrain",
    "weather_era.xlsx",
    "weather_gridmet.xlsx",
    "weather_s2s.xlsx",
]

EXPECTED_POLARIS = [
    f"{var}_{depth}.tif"
    for var in ["alpha", "ksat", "lambda", "n", "theta_r", "theta_s"]
    for depth in ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]
]

EXPECTED_GRIDMET = [
    "pr_2024.nc",
    "srad_2024.nc",
    "tmmn_2024.nc",
    "tmmx_2024.nc",
    "vpd_2024.nc",
    "vs_2024.nc",
]


def bytes_to_gib(n: int) -> float:
    return round(n / 1024**3, 3)


def summarize_path(path: Path) -> dict:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    else:
        files = []
    return {
        "path": str(path),
        "exists": path.exists(),
        "kind": "file" if path.is_file() else "dir" if path.is_dir() else "missing",
        "file_count": len(files),
        "size_gib": bytes_to_gib(sum(p.stat().st_size for p in files)),
    }


def extension_counts(path: Path) -> pd.DataFrame:
    rows = []
    for p in path.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower() or "[no_ext]"
            rows.append({"extension": suffix, "size": p.stat().st_size})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["extension", "file_count", "size_gib"])
    out = (
        df.groupby("extension")
        .agg(file_count=("extension", "size"), size_bytes=("size", "sum"))
        .reset_index()
        .sort_values(["file_count", "size_bytes"], ascending=False)
    )
    out["size_gib"] = out["size_bytes"].map(bytes_to_gib)
    return out[["extension", "file_count", "size_gib"]]


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}; run after extracting model3_opt_sto_upload.zip")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.DataFrame([summarize_path(DATA / name) for name in EXPECTED_TOP_LEVEL])
    inventory.to_csv(CSV_OUT, index=False)

    polaris_missing = [name for name in EXPECTED_POLARIS if not (DATA / "polaris" / name).exists()]
    gridmet_missing = [name for name in EXPECTED_GRIDMET if not (DATA / "gridmet" / name).exists()]
    nc_files = sorted(str(p) for p in DATA.rglob("*.nc"))
    s2s_dirs = sorted(str(p) for p in BASE.rglob("*") if p.is_dir() and ("s2s" in p.name.lower() or "operational" in p.name.lower()))
    ext = extension_counts(DATA)

    lines = [
        "# Uploaded Base Data Audit",
        "",
        "## Summary",
        "",
        f"- Base path: `{BASE.resolve()}`",
        f"- Data path: `{DATA.resolve()}`",
        f"- Total files under data: {sum(inventory['file_count'])}",
        f"- Total size under expected top-level entries: {inventory['size_gib'].sum():.3f} GiB",
        f"- POLARIS expected files: {len(EXPECTED_POLARIS)}",
        f"- POLARIS missing files: {len(polaris_missing)}",
        f"- GRIDMET expected files: {len(EXPECTED_GRIDMET)}",
        f"- GRIDMET missing files: {len(gridmet_missing)}",
        f"- NetCDF files found under data: {len(nc_files)}",
        f"- S2S/Operational directories found under model3_opt_sto_upload: {len(s2s_dirs)}",
        "",
        "## Top-Level Inventory",
        "",
        markdown_table(inventory),
        "",
        "## File Types",
        "",
        markdown_table(ext),
        "",
        "## NetCDF Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in nc_files)
    lines.extend(
        [
            "",
            "## Missing Checks",
            "",
            f"- Missing POLARIS files: {polaris_missing if polaris_missing else 'none'}",
            f"- Missing GRIDMET files: {gridmet_missing if gridmet_missing else 'none'}",
            "",
            "## Interpretation",
            "",
            "- Uploaded base data are sufficient for many static/site-feature checks: POLARIS, DEM, DTW, CropAT-US, LAI, ERA5, and GRIDMET.",
            "- The extracted Figshare package does not include original S2S ensemble NetCDF files.",
            "- Therefore ensemble-aware candidate-level SWAP surrogate generation remains blocked until S2S forecast inputs are provided.",
            "",
            f"Wrote `{CSV_OUT}` and `{MD_OUT}`.",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
