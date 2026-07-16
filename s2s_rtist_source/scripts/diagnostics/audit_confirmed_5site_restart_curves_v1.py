#!/usr/bin/env python3
"""Audit candidate-curve differences for confirmed 5-site restart smoke runs.

Run this after `run_confirmed_5site_restart_generation_smoke_v1.py`.

It merges each site's `site_restart_generation_smoke.csv` into one table and
checks whether the candidate target curves differ across sites. This is still a
workflow/input audit, not surrogate model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
RUN_ROOT = OUT_DIR / "confirmed_5site_restart_generation_smoke_v1"
DEFAULT_SITES = ["P1", "P15", "P2", "P3", "P4"]


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


def curve_signature(group: pd.DataFrame, digits: int) -> str:
    ordered = group.sort_values("ir")
    pairs = []
    for row in ordered.itertuples(index=False):
        pairs.append(f"{float(row.ir):g}:{round(float(row.target_value), digits):g}")
    return "|".join(pairs)


def best_ir_for_group(group: pd.DataFrame) -> float:
    best_idx = group["target_value"].idxmax()
    return float(group.loc[best_idx, "ir"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to a restart smoke run directory. Defaults to latest.")
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    frames = [read_site_csv(run_dir, site) for site in args.sites]
    merged = pd.concat(frames, ignore_index=True)
    for col in ["ir", "target_value", "cwdm_value", "cwso_value", "dvs"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    out_prefix = run_dir / "confirmed_5site_restart_curve_audit_v1"
    merged_csv = out_prefix.with_name(out_prefix.name + "_merged.csv")
    best_csv = out_prefix.with_name(out_prefix.name + "_best_by_site.csv")
    by_ir_csv = out_prefix.with_name(out_prefix.name + "_target_by_ir.csv")
    audit_csv = out_prefix.with_name(out_prefix.name + "_curve_signatures.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_out = out_prefix.with_suffix(".md")

    merged.to_csv(merged_csv, index=False)

    best = merged.loc[merged.groupby(["site", "date_t"])["target_value"].idxmax()].copy()
    best = best[
        [
            "site",
            "date_t",
            "decision_doy",
            "ir",
            "target_value",
            "cwdm_value",
            "cwso_value",
            "dvs",
        ]
    ].rename(columns={"ir": "best_ir_for_date", "target_value": "best_target_for_date"})
    best.to_csv(best_csv, index=False)

    curve_stats = (
        merged.groupby(["site", "date_t"])
        .agg(
            n_candidates=("ir", "count"),
            best_ir_for_date=("ir", lambda s: best_ir_for_group(merged.loc[s.index])),
            best_target_for_date=("target_value", "max"),
            worst_target_for_date=("target_value", "min"),
            target_range=("target_value", lambda s: float(s.max() - s.min())),
            cwdm_range=("cwdm_value", lambda s: float(s.max() - s.min())),
            cwso_range=("cwso_value", lambda s: float(s.max() - s.min())),
            dvs_range=("dvs", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
    )

    signatures = (
        merged.groupby(["site", "date_t"])
        .apply(lambda g: curve_signature(g, args.round_digits), include_groups=False)
        .reset_index(name="curve_signature")
    )
    signature_counts = signatures.groupby("curve_signature").size().reset_index(name="n_site_dates_with_same_signature")
    date_signature_counts = (
        signatures.groupby(["date_t", "curve_signature"])
        .size()
        .reset_index(name="n_sites_with_same_date_signature")
    )
    signatures = signatures.merge(signature_counts, on="curve_signature", how="left")
    signatures = signatures.merge(date_signature_counts, on=["date_t", "curve_signature"], how="left")
    signatures.to_csv(audit_csv, index=False)

    by_ir = (
        merged.pivot_table(index=["date_t", "ir"], columns="site", values="target_value", aggfunc="first")
        .reset_index()
        .sort_values(["date_t", "ir"])
    )
    site_cols = [col for col in by_ir.columns if col not in {"date_t", "ir"}]
    by_ir["cross_site_target_range"] = by_ir[site_cols].max(axis=1) - by_ir[site_cols].min(axis=1)
    by_ir.to_csv(by_ir_csv, index=False)

    all_completed_rows = len(merged)
    expected_rows = len(args.sites) * merged["date_t"].nunique() * merged["ir"].nunique()
    missing_rows = expected_rows - all_completed_rows
    unique_curve_count = signatures["curve_signature"].nunique()
    global_duplicate_curve_groups = int((signature_counts["n_site_dates_with_same_signature"] > 1).sum())
    duplicate_curve_groups = int((date_signature_counts["n_sites_with_same_date_signature"] > 1).sum())
    cross_site_ranges = by_ir["cross_site_target_range"].dropna()
    varying_ir_points = int((cross_site_ranges > args.tolerance).sum())
    flat_ir_points = int((cross_site_ranges <= args.tolerance).sum())
    max_cross_site_target_range = float(cross_site_ranges.max()) if len(cross_site_ranges) else 0.0
    mean_cross_site_target_range = float(cross_site_ranges.mean()) if len(cross_site_ranges) else 0.0
    status = "single_site_local_check"
    if len(args.sites) > 1:
        if missing_rows != 0:
            status = "incomplete_rows"
        elif duplicate_curve_groups > 0:
            status = "duplicate_site_curves"
        elif varying_ir_points == 0:
            status = "no_cross_site_target_difference"
        else:
            status = "passed_static_site_difference_audit"

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "sites": ",".join(args.sites),
                "n_sites": len(args.sites),
                "n_dates": int(merged["date_t"].nunique()),
                "n_ir_values": int(merged["ir"].nunique()),
                "merged_rows": all_completed_rows,
                "expected_rows": expected_rows,
                "missing_rows": missing_rows,
                "unique_curve_count": unique_curve_count,
                "duplicate_curve_groups": duplicate_curve_groups,
                "global_duplicate_curve_groups": global_duplicate_curve_groups,
                "cross_site_ir_points": int(len(cross_site_ranges)),
                "varying_ir_points": varying_ir_points,
                "flat_ir_points": flat_ir_points,
                "max_cross_site_target_range": max_cross_site_target_range,
                "mean_cross_site_target_range": mean_cross_site_target_range,
                "status": status,
            }
        ]
    )
    summary.to_csv(summary_csv, index=False)

    lines = [
        "# Confirmed 5-Site Restart Curve Audit V1",
        "",
        "## Inputs",
        "",
        f"- run_dir: `{run_dir}`",
        f"- sites: `{', '.join(args.sites)}`",
        f"- merged rows: `{all_completed_rows}`",
        f"- expected rows from observed date/ir counts: `{expected_rows}`",
        f"- audit status: `{status}`",
        "",
        "## Audit Summary",
        "",
        markdown_table(summary),
        "",
        "## Best By Site",
        "",
        markdown_table(best),
        "",
        "## Curve Stats",
        "",
        markdown_table(curve_stats),
        "",
        "## Target By Irrigation",
        "",
        markdown_table(by_ir),
        "",
        "## Cross-Site Difference Check",
        "",
        f"- cross_site_ir_points: `{len(cross_site_ranges)}`",
        f"- varying_ir_points: `{varying_ir_points}`",
        f"- flat_ir_points: `{flat_ir_points}`",
        f"- max_cross_site_target_range: `{max_cross_site_target_range:g}`",
        f"- mean_cross_site_target_range: `{mean_cross_site_target_range:g}`",
        "",
        "## Signature Check",
        "",
        f"- unique_curve_count: `{unique_curve_count}`",
        f"- duplicate_curve_groups: `{duplicate_curve_groups}`",
        f"- global_duplicate_curve_groups: `{global_duplicate_curve_groups}`",
        "",
        markdown_table(
            signatures[
                [
                    "site",
                    "date_t",
                    "n_sites_with_same_date_signature",
                    "n_site_dates_with_same_signature",
                    "curve_signature",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]
    if len(args.sites) == 1:
        lines.append("Single-site local check only; this validates script wiring but cannot prove cross-site differentiation.")
    elif status == "passed_static_site_difference_audit":
        lines.append("The run has complete rows, non-duplicate site/date curve signatures, and nonzero cross-site target differences.")
    elif status == "incomplete_rows":
        lines.append("The run is missing candidate rows; inspect per-site smoke outputs before interpreting curve differences.")
    elif status == "duplicate_site_curves":
        lines.append("Some site/date target curves still match exactly at the selected rounding precision.")
    elif status == "no_cross_site_target_difference":
        lines.append("All target values are flat across sites at each irrigation value; site-specific inputs are not affecting the curve level yet.")
    elif unique_curve_count == len(signatures):
        lines.append("All site/date target curves are unique at the selected rounding precision.")
    else:
        lines.append("Some site/date target curves still match exactly at the selected rounding precision.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{merged_csv}`",
            f"- `{best_csv}`",
            f"- `{by_ir_csv}`",
            f"- `{audit_csv}`",
            f"- `{summary_csv}`",
        ]
    )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site restart curve audit v1")
    print(f"run_dir: {run_dir}")
    print(f"merged_csv: {merged_csv}")
    print(f"best_csv: {best_csv}")
    print(f"by_ir_csv: {by_ir_csv}")
    print(f"audit_csv: {audit_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"md: {md_out}")
    print(best.to_string(index=False))
    print(f"unique_curve_count: {unique_curve_count}")
    print(f"duplicate_curve_groups: {duplicate_curve_groups}")
    print(f"global_duplicate_curve_groups: {global_duplicate_curve_groups}")
    print(f"varying_ir_points: {varying_ir_points}")
    print(f"status: {status}")


if __name__ == "__main__":
    main()
