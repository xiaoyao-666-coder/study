#!/usr/bin/env python3
"""Prepare isolated workspaces for 12-site continuous-irrigation SWAP runs.

This is the first 12-site step after the safe date-dense 5-site surrogate
check. It creates one workspace per code site, writes a site config, and audits
whether the basic Maize/SWAP files are present. It does not claim that
site-specific static, POLARIS, or gridMET inputs have been applied; those remain
explicit downstream steps before production sampling.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_SITE_FEATURE_CSV = (
    Path("site_general_surrogate_eval") / "site_feature_screening_12_code_sites.csv"
)
DEFAULT_SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"
DEFAULT_WORKSPACE_ROOT = (
    Path("site_general_surrogate_eval") / "continuous_ir_12site_workspaces_v1"
)
DEFAULT_REPORT_PREFIX = (
    Path("site_general_surrogate_eval") / "continuous_ir_12site_workspace_preflight_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-feature-csv", default=str(DEFAULT_SITE_FEATURE_CSV))
    parser.add_argument("--source-maize", default=str(DEFAULT_SOURCE_MAIZE))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--output-prefix", default=str(DEFAULT_REPORT_PREFIX))
    parser.add_argument("--sites", nargs="+", help="Optional subset of site ids.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-site workspaces.",
    )
    return parser.parse_args()


def required_columns(df: pd.DataFrame) -> None:
    required = {"site", "lon", "lat"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Site feature CSV is missing columns: {sorted(missing)}")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def workspace_name(site: str) -> str:
    return f"{site}_Maize"


def expected_executable(workspace: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return workspace / "Swap.exe"
    return workspace / "swap_test"


def audit_workspace(workspace: Path) -> dict[str, object]:
    exe = expected_executable(workspace)
    checks = {
        "workspace_exists": workspace.exists(),
        "swap_executable": str(exe),
        "swap_executable_exists": exe.exists(),
        "swp_exists": any(workspace.glob("*.swp")),
        "weather_exists": any(workspace.glob("weather.*")) or any(workspace.glob("WeatherOriginal.*")),
        "crop_exists": any(workspace.glob("*.crp")) or any(workspace.glob("Crop*")),
    }
    checks["basic_workspace_ready"] = all(
        bool(checks[name])
        for name in ["workspace_exists", "swap_executable_exists", "swp_exists"]
    )
    return checks


def copy_workspace(source: Path, destination: Path, overwrite: bool) -> str:
    if destination.exists():
        if not overwrite:
            return "already_exists"
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return "created"


def write_site_config(workspace: Path, row: pd.Series, source_maize: Path) -> None:
    config = {
        "site_id": str(row["site"]),
        "code_site_id": str(row["site"]).replace("code_", ""),
        "longitude": float(row["lon"]),
        "latitude": float(row["lat"]),
        "source_maize": str(source_maize),
        "workspace_stage": "copied_template_requires_site_specific_inputs",
        "continuous_irrigation_note": (
            "Prepared for 12-site continuous-irrigation surrogate sampling. "
            "Apply static, POLARIS, and gridMET inputs before production SWAP generation."
        ),
    }
    if "dem_m" in row:
        config["dem_m"] = float(row["dem_m"])
    if "tile_drain" in row:
        config["tile_drain"] = float(row["tile_drain"])

    (workspace / "site_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)
    (workspace / "README_continuous_ir_12site.txt").write_text(
        "\n".join(
            [
                f"Site: {config['site_id']}",
                f"Longitude: {config['longitude']}",
                f"Latitude: {config['latitude']}",
                "Purpose: isolated workspace for continuous-irrigation SWAP sampling.",
                "Status: template copy only until site-specific inputs are applied.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    site_feature_csv = Path(args.site_feature_csv)
    source_maize = Path(args.source_maize)
    workspace_root = Path(args.workspace_root)
    output_prefix = Path(args.output_prefix)

    if not site_feature_csv.exists():
        raise FileNotFoundError(f"Missing site feature CSV: {site_feature_csv}")
    if not source_maize.exists():
        raise FileNotFoundError(f"Missing source Maize workspace: {source_maize}")

    sites = pd.read_csv(site_feature_csv)
    required_columns(sites)
    if args.sites:
        requested = set(args.sites)
        sites = sites[sites["site"].astype(str).isin(requested)].copy()
        missing_requested = sorted(requested.difference(set(sites["site"].astype(str))))
        if missing_requested:
            raise ValueError(f"Requested sites not found in feature CSV: {missing_requested}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in sites.iterrows():
        site = str(row["site"])
        workspace = workspace_root / workspace_name(site)
        status = copy_workspace(source_maize, workspace, overwrite=args.overwrite)
        write_site_config(workspace, row, source_maize)
        audit = audit_workspace(workspace)
        rows.append(
            {
                "site_id": site,
                "workspace": str(workspace),
                "workspace_status": status,
                "longitude": float(row["lon"]),
                "latitude": float(row["lat"]),
                **audit,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    df.to_csv(csv_path, index=False)

    ready_count = int(df["basic_workspace_ready"].sum()) if not df.empty else 0
    lines = [
        "# Continuous Irrigation 12-Site Workspace Preflight V1",
        "",
        f"- site feature csv: `{site_feature_csv}`",
        f"- source maize: `{source_maize}`",
        f"- workspace root: `{workspace_root}`",
        f"- sites: `{len(df)}`",
        f"- basic workspace ready: `{ready_count} / {len(df)}`",
        "",
        "## Workspaces",
        markdown_table(df),
        "",
        "## Next Required Step",
        "- Apply site-specific static, POLARIS, and gridMET inputs before production SWAP sampling.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation 12-site workspace preflight v1")
    print(f"sites: {len(df)}")
    print(f"basic_workspace_ready: {ready_count} / {len(df)}")
    print(f"workspace_root: {workspace_root}")
    print(f"csv: {csv_path}")
    print(f"report: {md_path}")
    print(df[["site_id", "workspace_status", "basic_workspace_ready"]].to_string(index=False))


if __name__ == "__main__":
    main()
