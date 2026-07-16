#!/usr/bin/env python3
"""Extract layer-wise soil pressure head state from saved pre-decision .end files.

SWAP .end files store final soil water pressure heads by depth. This script
extracts those layer-wise states for each decision date and creates compact
summary features for the short-term surrogate v1 dataset.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd


STATE_DIR = Path("Maize_shortterm_surrogate_v1")
TRUE_STATE_CSV = STATE_DIR / "current_state_by_date_true.csv"
OUT_CSV = STATE_DIR / "soil_pressure_state_by_date.csv"
OUT_REPORT = STATE_DIR / "soil_pressure_state_extract_report.md"


def parse_pressure_heads(path: Path) -> pd.DataFrame:
    rows = []
    in_table = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "z_h,h":
                in_table = True
                continue
            if not in_table:
                continue
            if not stripped:
                if rows:
                    break
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) != 2:
                if rows:
                    break
                continue
            try:
                z = float(parts[0])
                h = float(parts[1])
            except ValueError:
                if rows:
                    break
                continue
            rows.append({"z_cm": z, "h_cm": h})
    if not rows:
        raise RuntimeError(f"No pressure-head rows found in {path}")
    return pd.DataFrame(rows)


def summarize_depths(df: pd.DataFrame) -> dict:
    out = {
        "soil_layer_count": int(len(df)),
        "soil_depth_min_cm": float(df["z_cm"].min()),
        "soil_depth_max_cm": float(df["z_cm"].max()),
        "soil_h_profile_json": json.dumps(
            [{"z_cm": round(float(r.z_cm), 3), "h_cm": round(float(r.h_cm), 5)} for r in df.itertuples(index=False)],
            separators=(",", ":"),
        ),
    }
    bands = {
        "0_30": (-30.0, 0.0),
        "30_60": (-60.0, -30.0),
        "60_100": (-100.0, -60.0),
        "0_100": (-100.0, 0.0),
    }
    for name, (lo, hi) in bands.items():
        sub = df[(df["z_cm"] >= lo) & (df["z_cm"] < hi)]
        if sub.empty:
            out[f"soil_h_mean_{name}_cm"] = None
            out[f"soil_h_min_{name}_cm"] = None
            out[f"soil_h_max_{name}_cm"] = None
        else:
            out[f"soil_h_mean_{name}_cm"] = float(sub["h_cm"].mean())
            out[f"soil_h_min_{name}_cm"] = float(sub["h_cm"].min())
            out[f"soil_h_max_{name}_cm"] = float(sub["h_cm"].max())
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    if not TRUE_STATE_CSV.exists():
        raise FileNotFoundError(f"Missing true state table: {TRUE_STATE_CSV}")

    state = pd.read_csv(TRUE_STATE_CSV)
    rows = []
    for item in state.itertuples(index=False):
        source = str(item.state_source)
        end_path = Path(re.sub(r"\.crp$", ".end", source))
        if not end_path.exists():
            raise FileNotFoundError(f"Missing matching .end file for {source}: {end_path}")
        profile = parse_pressure_heads(end_path)
        rows.append(
            {
                "date_t": item.date_t,
                "decision_doy": int(item.decision_doy),
                "pre_end_doy": int(item.pre_end_doy),
                "soil_state_source": str(end_path),
                **summarize_depths(profile),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    display_cols = [
        "date_t",
        "decision_doy",
        "soil_layer_count",
        "soil_h_mean_0_30_cm",
        "soil_h_mean_30_60_cm",
        "soil_h_mean_60_100_cm",
        "soil_h_mean_0_100_cm",
    ]
    report = [
        "# Soil Pressure State Extract Report",
        "",
        f"- Input true state table: `{TRUE_STATE_CSV}`",
        f"- Output: `{OUT_CSV}`",
        "- Source: saved pre-decision SWAP `.end` files.",
        "- Variable: pressure head `h` by depth. This is a direct SWAP soil-water state; volumetric water content would require converting with soil hydraulic parameters.",
        "",
        "## Summary Features",
        "",
        markdown_table(out[display_cols]),
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
