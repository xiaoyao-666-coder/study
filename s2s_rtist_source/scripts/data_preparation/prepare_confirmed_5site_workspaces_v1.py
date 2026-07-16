"""Prepare isolated Maize workspaces and run input/SWAP smoke checks.

This script uses only the five high-confidence paper sites P1, P2, P3, P4, and
P15. It does not train a surrogate model. It creates isolated copies of the
Maize working directory, writes per-site configuration files, samples static
input rasters at each site, and attempts a short SWAP executable smoke run in
each isolated workspace.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform


OUT_DIR = Path("site_general_surrogate_eval")
MANIFEST = OUT_DIR / "confirmed_5site_generation_manifest_v1.csv"
SOURCE_MAIZE = Path("model3_opt_sto_upload") / "Maize"
DATA_DIR = Path("model3_opt_sto_upload") / "data"
WORKSPACE_ROOT = OUT_DIR / "confirmed_5site_workspaces"


def sample_tif(lon: float, lat: float, tif_path: Path) -> float | None:
    with rasterio.open(tif_path) as src:
        if src.crs:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        else:
            x, y = lon, lat
        value = next(src.sample([(x, y)]))[0]
        if src.nodata is not None and value == src.nodata:
            return None
        try:
            return float(value)
        except Exception:
            return None


def first_tif(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.tif"))
    return files[0] if files else None


def sample_static_inputs(row: pd.Series) -> dict:
    lon = float(row["longitude"])
    lat = float(row["latitude"])
    out: dict[str, object] = {
        "paper_site_id": row["paper_site_id"],
        "code_site_id": row["code_site_id"],
        "longitude": lon,
        "latitude": lat,
    }

    for name in ["dem", "dtw", "tiledrain"]:
        tif = first_tif(DATA_DIR / name)
        out[f"{name}_path"] = str(tif) if tif else ""
        out[f"{name}_value"] = sample_tif(lon, lat, tif) if tif else None

    polaris_dir = DATA_DIR / "polaris"
    for tif in sorted(polaris_dir.glob("*.tif")):
        out[f"polaris_{tif.stem}"] = sample_tif(lon, lat, tif)

    crop_dir = DATA_DIR / "CropAT_US"
    crop_year_values = {}
    for tif in sorted(crop_dir.glob("CropType_*.tif")):
        year = tif.stem.split("_")[-1]
        value = sample_tif(lon, lat, tif)
        crop_year_values[year] = value
        out[f"crop_type_{year}"] = value
    corn_years = [int(year) for year, value in crop_year_values.items() if value == 1.0]
    out["cropat_selected_year"] = max(corn_years) if corn_years else max(map(int, crop_year_values)) if crop_year_values else None
    out["static_extraction_status"] = "ok"
    return out


def copy_workspace(row: pd.Series) -> tuple[Path, str]:
    workspace = WORKSPACE_ROOT / str(row["recommended_workspace_name"])
    if workspace.exists():
        return workspace, "already_exists"
    shutil.copytree(SOURCE_MAIZE, workspace)
    return workspace, "created"


def write_site_config(workspace: Path, row: pd.Series, static_values: dict) -> None:
    config = {
        "paper_site_id": row["paper_site_id"],
        "code_site_id": row["code_site_id"],
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"]),
        "source_manifest": str(MANIFEST),
        "static_input_smoke_status": static_values.get("static_extraction_status", ""),
    }
    (workspace / "site_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame([config]).to_csv(workspace / "site_config.csv", index=False)
    pd.DataFrame([static_values]).to_csv(workspace / "static_input_smoke_values.csv", index=False)
    (workspace / "README_site_smoke.txt").write_text(
        "\n".join(
            [
                f"Paper site: {row['paper_site_id']}",
                f"Code site: {row['code_site_id']}",
                f"Longitude: {row['longitude']}",
                f"Latitude: {row['latitude']}",
                "Purpose: isolated Maize workspace for input extraction and SWAP smoke checks.",
                "This directory is not a trained surrogate dataset.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_swap_smoke(workspace: Path) -> dict:
    exe = workspace / ("Swap.exe" if platform.system().lower().startswith("win") else "swap_test")
    if not exe.exists():
        return {
            "swap_executable": str(exe),
            "swap_smoke_status": "missing_executable",
            "swap_returncode": "",
            "swap_stdout_tail": "",
            "swap_stderr_tail": "",
        }

    try:
        result = subprocess.run(
            [str(exe)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status = "completed" if result.returncode == 0 else "returned_nonzero"
        return {
            "swap_executable": str(exe),
            "swap_smoke_status": status,
            "swap_returncode": result.returncode,
            "swap_stdout_tail": result.stdout[-1000:],
            "swap_stderr_tail": result.stderr[-1000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout_tail = (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else ""
        return {
            "swap_executable": str(exe),
            "swap_smoke_status": "started_timeout_30s" if stdout_tail or stderr_tail else "timeout_30s",
            "swap_returncode": "",
            "swap_stdout_tail": stdout_tail,
            "swap_stderr_tail": stderr_tail,
        }
    except Exception as exc:
        return {
            "swap_executable": str(exe),
            "swap_smoke_status": f"error:{type(exc).__name__}",
            "swap_returncode": "",
            "swap_stdout_tail": "",
            "swap_stderr_tail": str(exc),
        }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)

    workspace_rows = []
    static_rows = []
    smoke_rows = []

    for row in manifest.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        workspace, workspace_status = copy_workspace(row_s)
        static_values = sample_static_inputs(row_s)
        write_site_config(workspace, row_s, static_values)
        smoke = run_swap_smoke(workspace)

        workspace_rows.append(
            {
                "paper_site_id": row_s["paper_site_id"],
                "code_site_id": row_s["code_site_id"],
                "workspace": str(workspace),
                "workspace_status": workspace_status,
                "site_config_written": (workspace / "site_config.json").exists(),
                "static_values_written": (workspace / "static_input_smoke_values.csv").exists(),
            }
        )
        static_rows.append(static_values)
        smoke_rows.append(
            {
                "paper_site_id": row_s["paper_site_id"],
                "code_site_id": row_s["code_site_id"],
                **smoke,
            }
        )

    workspace_df = pd.DataFrame(workspace_rows)
    static_df = pd.DataFrame(static_rows)
    smoke_df = pd.DataFrame(smoke_rows)

    workspace_out = OUT_DIR / "confirmed_5site_workspace_creation_v1.csv"
    static_out = OUT_DIR / "confirmed_5site_static_input_extraction_smoke_v1.csv"
    smoke_out = OUT_DIR / "confirmed_5site_swap_smoke_v1.csv"
    report_out = OUT_DIR / "confirmed_5site_workspace_and_swap_smoke_v1.md"

    workspace_df.to_csv(workspace_out, index=False)
    static_df.to_csv(static_out, index=False)
    smoke_df.to_csv(smoke_out, index=False)

    lines = [
        "# Confirmed 5-Site Workspace and SWAP Smoke V1",
        "",
        "## Scope",
        "- Sites: P1, P2, P3, P4, P15 only.",
        "- Created isolated Maize workspace copies under `site_general_surrogate_eval/confirmed_5site_workspaces`.",
        "- Sampled static input rasters with `rasterio` directly; no surrogate model training was run.",
        "- Attempted one SWAP executable smoke command per isolated workspace.",
        "",
        "## Workspace Creation",
        markdown_table(workspace_df),
        "",
        "## Static Input Smoke Summary",
        markdown_table(
            static_df[
                [
                    "paper_site_id",
                    "code_site_id",
                    "dem_value",
                    "dtw_value",
                    "tiledrain_value",
                    "cropat_selected_year",
                    "static_extraction_status",
                ]
            ]
        ),
        "",
        "## SWAP Smoke Summary",
        markdown_table(
            smoke_df[
                [
                    "paper_site_id",
                    "code_site_id",
                    "swap_executable",
                    "swap_smoke_status",
                    "swap_returncode",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "- Workspace isolation and static raster sampling completed for all five confirmed paper sites.",
        "- SWAP smoke status should be interpreted as an executable/workspace smoke check only, not as a calibrated multi-site simulation result.",
        "- If all SWAP smoke rows are `completed`, the next step is controlled per-site restart/decision sample generation.",
    ]
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site workspace and SWAP smoke v1")
    print(f"workspaces: {len(workspace_df)}")
    print(f"static smoke rows: {len(static_df)}")
    print("swap smoke statuses:")
    print(smoke_df[["paper_site_id", "swap_smoke_status", "swap_returncode"]].to_string(index=False))
    print(f"wrote: {workspace_out}")
    print(f"wrote: {static_out}")
    print(f"wrote: {smoke_out}")
    print(f"wrote: {report_out}")


if __name__ == "__main__":
    main()
