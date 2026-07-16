#!/usr/bin/env python3
"""Extract pre-decision current-state features for short-term surrogate v1.

Run this inside the server experiment directory, usually:

    /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/Maize_restart_dataset

The script first looks for saved pre-decision crop output files. If they do not
exist, it falls back to the earliest candidate rows in restart_decision_dataset
and writes a clear status report so the next run can save pre-decision CRP files.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


CRP_COLUMNS = [
    "Date",
    "Daynr",
    "TSum",
    "GDD",
    "MDS",
    "DVS",
    "LAIpot",
    "LAI",
    "TAGPpot",
    "TAGP",
    "WLVG",
    "WLVD",
    "WST",
    "WSO",
    "TRA",
    "RDpot",
    "RD",
    "RootdPot",
    "Rootd",
    "WRT",
    "CWDM",
    "CWSO",
]


def safe_label(date_t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", date_t)


def read_last_crp(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("2024-"):
                continue
            values = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(values) == len(CRP_COLUMNS):
                rows.append(values)
    if not rows:
        raise RuntimeError(f"No crop rows found in {path}")

    df = pd.DataFrame(rows, columns=CRP_COLUMNS)
    numeric = ["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    row = df.dropna(subset=["Daynr", "DVS", "LAI", "Rootd", "CWDM", "CWSO"]).iloc[-1]
    return {
        "state_source": str(path),
        "state_daynr": int(row["Daynr"]),
        "state_dvs": float(row["DVS"]),
        "state_lai": float(row["LAI"]),
        "state_rootd": float(row["Rootd"]),
        "state_cwdm": float(row["CWDM"]),
        "state_cwso": float(row["CWSO"]),
        "state_soil_water_layers_status": "pending_extract_from_swap_water_output",
    }


def find_pre_crp(date_t: str) -> Path | None:
    label = safe_label(date_t)
    candidates = [
        Path(f"result_pre_{label}.crp"),
        Path(f"pre_state_{label}.crp"),
        Path(f"restart_initial_{label}.crp"),
        Path("result_pre.crp"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def fallback_from_restart_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Use 0-mm 7-day result as a weak placeholder if pre-state CRP is absent."""
    base = dataset[dataset["ir"].astype(float) == 0.0].copy()
    if base.empty:
        raise RuntimeError("No pre-decision CRP files found and no 0-mm fallback rows exist.")
    out = pd.DataFrame()
    out["date_t"] = base["date_t"]
    out["decision_doy"] = base["decision_doy"].astype(int)
    out["state_source"] = "fallback_0mm_7day_result_not_pre_decision"
    out["state_daynr"] = base["end_daynr"].astype(int)
    out["state_dvs"] = base["dvs"]
    out["state_lai"] = base["lai"]
    out["state_rootd"] = base["rootd"]
    out["state_cwdm"] = base["cwdm_value"]
    out["state_cwso"] = base["cwso_value"]
    out["state_soil_water_layers_status"] = "pending_extract_from_swap_water_output"
    return out.sort_values("decision_doy").reset_index(drop=True)


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
    dataset_path = Path("restart_decision_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("Maize_restart_dataset") / "restart_decision_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("Could not find restart_decision_dataset.csv")

    dataset = pd.read_csv(dataset_path)
    dates = (
        dataset[["date_t", "decision_doy"]]
        .drop_duplicates()
        .sort_values("decision_doy")
        .reset_index(drop=True)
    )

    rows = []
    missing_pre = []
    for item in dates.itertuples(index=False):
        path = find_pre_crp(str(item.date_t))
        if path is None:
            missing_pre.append(str(item.date_t))
            continue
        row = {
            "date_t": item.date_t,
            "decision_doy": int(item.decision_doy),
            **read_last_crp(path),
        }
        rows.append(row)

    if rows and not missing_pre:
        state = pd.DataFrame(rows).sort_values("decision_doy").reset_index(drop=True)
        extraction_status = "pre_decision_crp_files"
    else:
        state = fallback_from_restart_dataset(dataset)
        extraction_status = "fallback_from_0mm_7day_result"

    output_dir = Path("Maize_shortterm_surrogate_v1")
    output_dir.mkdir(exist_ok=True)
    state_path = output_dir / "current_state_by_date.csv"
    report_path = output_dir / "current_state_extract_report.md"
    state.to_csv(state_path, index=False)

    report_lines = [
        "# Current State Extract Report",
        "",
        f"- Input dataset: `{dataset_path}`",
        f"- Decision dates: {len(dates)}",
        f"- Extraction status: `{extraction_status}`",
        f"- Output: `{state_path}`",
        "",
        "## Important Note",
        "",
    ]
    if extraction_status.startswith("fallback"):
        report_lines.extend(
            [
                "Pre-decision crop output files were not found for every date.",
                "The current output is a placeholder derived from the 0 mm 7-day result, not the true decision-day state.",
                "For the next formal run, save `result_forec.crp` after each pre-decision run as `result_pre_<date>.crp`.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "States were extracted from saved pre-decision crop output files.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Extracted State",
            "",
            markdown_table(state),
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {state_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
