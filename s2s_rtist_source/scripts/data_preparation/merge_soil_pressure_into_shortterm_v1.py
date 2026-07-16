#!/usr/bin/env python3
"""Merge soil pressure-head state features into short-term surrogate v1 table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_v1")
SAMPLES = OUT_DIR / "shortterm_surrogate_samples_v1_with_true_state.csv"
SOIL = OUT_DIR / "soil_pressure_state_by_date.csv"
MERGED = OUT_DIR / "shortterm_surrogate_samples_v1_with_true_state_soil.csv"
REPORT = OUT_DIR / "shortterm_surrogate_true_state_soil_report.md"


def main() -> None:
    if not SAMPLES.exists():
        raise FileNotFoundError(f"Missing sample table: {SAMPLES}")
    if not SOIL.exists():
        raise FileNotFoundError(f"Missing soil state table: {SOIL}")

    samples = pd.read_csv(SAMPLES)
    soil = pd.read_csv(SOIL)
    merge_cols = ["date_t", "decision_doy", "pre_end_doy"]
    missing = sorted(set(merge_cols) - set(soil.columns))
    if missing:
        raise ValueError(f"Soil table missing merge columns: {missing}")

    merged = samples.merge(soil, on=merge_cols, how="left")
    if merged["soil_state_source"].isna().any():
        raise RuntimeError("Some sample rows did not match soil pressure state rows.")

    merged["state_soil_water_layers_status"] = "filled_with_pressure_head_profile_from_predecision_end"
    merged.to_csv(MERGED, index=False)

    report = [
        "# Short-Term Surrogate V1 With True State And Soil Pressure Profile",
        "",
        f"- Input samples: `{SAMPLES}`",
        f"- Soil state table: `{SOIL}`",
        f"- Output: `{MERGED}`",
        f"- Rows: {len(merged)}",
        "- Soil state added: layer-wise SWAP pressure head profile and depth-band summaries.",
        "- Note: pressure head is not yet volumetric soil water content. Converting to water content is a later enhancement using soil hydraulic parameters.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {MERGED}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
