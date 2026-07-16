#!/usr/bin/env python3
"""Build the first confirmed 5-site true-input surrogate candidate table.

This table builder consumes a completed confirmed 5-site restart smoke run and
creates a small candidate-level training table with explicit target-collapse
flags. It does not train a model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


RUN_ROOT = Path("site_general_surrogate_eval") / "confirmed_5site_restart_generation_smoke_v1"
OUT_DIR = Path("site_general_surrogate_eval") / "confirmed_5site_true_input_surrogate_table_v1"
DEFAULT_SITES = ["P1", "P15", "P2", "P3", "P4"]
YEAR_LENGTH = 366

SITE_META = {
    "P1": {"code_site_id": "N1", "longitude": -98.224144, "latitude": 42.015928},
    "P2": {"code_site_id": "N2", "longitude": -88.415, "latitude": 40.595},
    "P3": {"code_site_id": "N3", "longitude": -96.877, "latitude": 46.321},
    "P4": {"code_site_id": "N4", "longitude": -94.6686, "latitude": 42.6816},
    "P15": {"code_site_id": "coord_12", "longitude": -112.265, "latitude": 41.735},
}


def latest_run_dir() -> Path:
    candidates = [p for p in RUN_ROOT.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {RUN_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_site_csv(run_dir: Path, site: str) -> pd.DataFrame:
    path = run_dir / site / "site_restart_generation_smoke.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing site CSV: {path}")
    df = pd.read_csv(path)
    if "site" not in df.columns:
        df.insert(0, "site", site)
    return df


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def sample_id(site: str, date_text: str, ir: float) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    ir_token = f"{int(ir):02d}" if float(ir).is_integer() else str(ir).replace(".", "p")
    return f"{site}_{date_token}_ir{ir_token}"


def site_date_id(site: str, date_text: str) -> str:
    date_token = date_text.replace("-", "").replace("/", "").replace(" ", "")
    return f"{site}_{date_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    return json.dumps([float(ir)] + [0.0] * max(horizon_days - 1, 0), separators=(",", ":"))


def target_signature(group: pd.DataFrame, digits: int) -> str:
    ordered = group.sort_values("candidate_ir")
    return "|".join(
        f"{float(row.candidate_ir):g}:{round(float(row.target_7d), digits):g}"
        for row in ordered.itertuples(index=False)
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def build_table(run_dir: Path, sites: list[str], round_digits: int, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.concat([read_site_csv(run_dir, site) for site in sites], ignore_index=True)
    for col in ["decision_doy", "horizon_end_doy", "ir", "end_daynr", "dvs", "lai", "rootd", "cwdm_value", "cwso_value", "target_value", "best_ir_for_date", "best_target_for_date"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["is_best_ir"] = bool_series(raw["is_best_ir"])

    out = pd.DataFrame()
    out["sample_id"] = [sample_id(str(r.site), str(r.date_t), float(r.ir)) for r in raw.itertuples(index=False)]
    out["site_date_id"] = [site_date_id(str(r.site), str(r.date_t)) for r in raw.itertuples(index=False)]
    out["site_id"] = raw["site"].astype(str)
    out["paper_site_id"] = raw["site"].astype(str)
    out["code_site_id"] = out["site_id"].map(lambda s: SITE_META.get(s, {}).get("code_site_id", ""))
    out["longitude"] = out["site_id"].map(lambda s: SITE_META.get(s, {}).get("longitude", math.nan))
    out["latitude"] = out["site_id"].map(lambda s: SITE_META.get(s, {}).get("latitude", math.nan))
    out["date_t"] = raw["date_t"].astype(str)
    out["decision_doy"] = raw["decision_doy"].astype(int)
    out["horizon_end_doy"] = raw["horizon_end_doy"].astype(int)
    out["horizon_days"] = out["horizon_end_doy"] - out["decision_doy"]
    out["decision_doy_sin"] = out["decision_doy"].map(lambda x: math.sin(2 * math.pi * x / YEAR_LENGTH))
    out["decision_doy_cos"] = out["decision_doy"].map(lambda x: math.cos(2 * math.pi * x / YEAR_LENGTH))
    out["candidate_ir"] = raw["ir"].astype(float)
    out["candidate_ir_sq"] = out["candidate_ir"] ** 2
    out["is_zero_ir"] = (out["candidate_ir"] == 0.0).astype(int)
    out["candidate_ir_sequence"] = [
        candidate_sequence(float(r.candidate_ir), int(r.horizon_days)) for r in out.itertuples(index=False)
    ]
    out["end_daynr"] = raw["end_daynr"].astype(int)
    out["dvs_7d"] = raw["dvs"].astype(float)
    out["lai_7d"] = raw["lai"].astype(float)
    out["rootd_7d"] = raw["rootd"].astype(float)
    out["cwdm_7d"] = raw["cwdm_value"].astype(float)
    out["cwso_7d"] = raw["cwso_value"].astype(float)
    out["target_7d"] = raw["target_value"].astype(float)
    out["best_ir_for_date"] = raw["best_ir_for_date"].astype(float)
    out["best_target_for_date"] = raw["best_target_for_date"].astype(float)
    out["is_best_ir"] = raw["is_best_ir"].astype(bool)

    no_ir = out[out["candidate_ir"] == 0.0][["site_date_id", "target_7d"]].rename(
        columns={"target_7d": "no_irrigation_target_7d"}
    )
    out = out.merge(no_ir, on="site_date_id", how="left")
    out["net_gain_7d"] = out["target_7d"] - out["no_irrigation_target_7d"]
    out["target_regret"] = out["best_target_for_date"] - out["target_7d"]
    out["best_ir_gap"] = out["candidate_ir"] - out["best_ir_for_date"]

    site_date = (
        out.groupby(["site_id", "date_t", "site_date_id"])
        .agg(
            decision_doy=("decision_doy", "first"),
            horizon_days=("horizon_days", "first"),
            n_candidates=("candidate_ir", "count"),
            best_ir_for_date=("best_ir_for_date", "first"),
            best_target_for_date=("best_target_for_date", "first"),
            min_target_7d=("target_7d", "min"),
            max_target_7d=("target_7d", "max"),
            target_range_7d=("target_7d", lambda s: float(s.max() - s.min())),
            no_irrigation_target_7d=("no_irrigation_target_7d", "first"),
            dvs_7d_at_zero_ir=("dvs_7d", lambda s: float(out.loc[s.index[out.loc[s.index, "candidate_ir"].argmin()], "dvs_7d"])),
            lai_7d_at_zero_ir=("lai_7d", lambda s: float(out.loc[s.index[out.loc[s.index, "candidate_ir"].argmin()], "lai_7d"])),
            cwdm_7d_at_zero_ir=("cwdm_7d", lambda s: float(out.loc[s.index[out.loc[s.index, "candidate_ir"].argmin()], "cwdm_7d"])),
            cwso_7d_at_zero_ir=("cwso_7d", lambda s: float(out.loc[s.index[out.loc[s.index, "candidate_ir"].argmin()], "cwso_7d"])),
        )
        .reset_index()
    )
    site_date["target_collapse"] = (
        (site_date["best_target_for_date"] <= tolerance)
        & (site_date["min_target_7d"] <= tolerance)
        & (site_date["target_range_7d"] > tolerance)
    )

    signatures = []
    for site_date_key, group in out.groupby("site_date_id"):
        signatures.append(
            {
                "site_date_id": site_date_key,
                "target_curve_signature": target_signature(group, round_digits),
            }
        )
    signatures_df = pd.DataFrame(signatures)
    site_date = site_date.merge(signatures_df, on="site_date_id", how="left")
    same_date_counts = (
        site_date.groupby(["date_t", "target_curve_signature"])
        .size()
        .reset_index(name="same_date_target_signature_count")
    )
    site_date = site_date.merge(same_date_counts, on=["date_t", "target_curve_signature"], how="left")
    site_date["same_date_duplicate_target_curve"] = site_date["same_date_target_signature_count"] > 1

    out = out.merge(
        site_date[
            [
                "site_date_id",
                "target_collapse",
                "target_curve_signature",
                "same_date_target_signature_count",
                "same_date_duplicate_target_curve",
            ]
        ],
        on="site_date_id",
        how="left",
    )
    out["training_weight_suggested"] = out["target_collapse"].map(lambda v: 0.5 if bool(v) else 1.0)

    labels = (
        out[out["is_best_ir"]][
            [
                "site_date_id",
                "site_id",
                "date_t",
                "decision_doy",
                "horizon_days",
                "best_ir_for_date",
                "best_target_for_date",
                "target_collapse",
                "same_date_duplicate_target_curve",
            ]
        ]
        .sort_values(["site_id", "decision_doy"])
        .reset_index(drop=True)
    )
    return out, labels, site_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to confirmed multidate restart smoke run. Defaults to latest.")
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, labels, site_date = build_table(run_dir, args.sites, args.round_digits, args.tolerance)

    samples_path = out_dir / "confirmed_5site_true_input_surrogate_samples_v1.csv"
    labels_path = out_dir / "confirmed_5site_true_input_surrogate_labels_v1.csv"
    site_date_path = out_dir / "confirmed_5site_true_input_surrogate_site_date_v1.csv"
    report_path = out_dir / "confirmed_5site_true_input_surrogate_table_v1.md"

    samples.to_csv(samples_path, index=False)
    labels.to_csv(labels_path, index=False)
    site_date.to_csv(site_date_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "samples_rows": len(samples),
                "label_rows": len(labels),
                "n_sites": samples["site_id"].nunique(),
                "n_dates": samples["date_t"].nunique(),
                "n_site_dates": samples["site_date_id"].nunique(),
                "n_ir_values": samples["candidate_ir"].nunique(),
                "target_collapse_site_dates": int(site_date["target_collapse"].sum()),
                "same_date_duplicate_target_site_dates": int(site_date["same_date_duplicate_target_curve"].sum()),
            }
        ]
    )

    lines = [
        "# Confirmed 5-Site True-Input Surrogate Table V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Best Labels",
        "",
        markdown_table(labels),
        "",
        "## Site-Date Summary",
        "",
        markdown_table(site_date),
        "",
        "## Outputs",
        "",
        f"- `{samples_path}`",
        f"- `{labels_path}`",
        f"- `{site_date_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site true-input surrogate table v1")
    print(f"run_dir: {run_dir}")
    print(f"samples: {samples_path}")
    print(f"labels: {labels_path}")
    print(f"site_date: {site_date_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))
    print("")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
