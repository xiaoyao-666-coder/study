#!/usr/bin/env python3
"""Audit training readiness for confirmed 5-site multidate restart outputs.

This is a post-smoke audit. It checks whether the multidate true-input run is
complete enough for small surrogate-training data preparation, and separates
true duplicate target curves from target-objective collapse cases where SWAP
state values differ underneath.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("site_general_surrogate_eval")
RUN_ROOT = OUT_DIR / "confirmed_5site_restart_generation_smoke_v1"
DEFAULT_SITES = ["P1", "P15", "P2", "P3", "P4"]
STATE_COLS = ["cwdm_value", "cwso_value", "dvs", "lai", "rootd"]


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


def target_signature(group: pd.DataFrame, digits: int) -> str:
    ordered = group.sort_values("ir")
    return "|".join(
        f"{float(row.ir):g}:{round(float(row.target_value), digits):g}"
        for row in ordered.itertuples(index=False)
    )


def state_signature(group: pd.DataFrame, digits: int) -> str:
    cols = [c for c in STATE_COLS if c in group.columns]
    if not cols:
        return ""
    first = group.sort_values("ir").iloc[0]
    return "|".join(f"{c}:{round(float(first[c]), digits):g}" for c in cols)


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
    parser.add_argument("--run-dir", default=None, help="Path to multidate restart smoke run. Defaults to latest.")
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    merged = pd.concat([read_site_csv(run_dir, site) for site in args.sites], ignore_index=True)
    for col in ["ir", "target_value", *STATE_COLS]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    observed_dates = sorted(merged["date_t"].dropna().unique())
    observed_ir = sorted(merged["ir"].dropna().unique())
    expected_rows = len(args.sites) * len(observed_dates) * len(observed_ir)
    missing_rows = expected_rows - len(merged)

    group_cols = ["site", "date_t"]
    site_date = (
        merged.groupby(group_cols)
        .agg(
            n_candidates=("ir", "count"),
            best_ir=("ir", lambda s: float(merged.loc[s.index, "ir"].iloc[merged.loc[s.index, "target_value"].argmax()])),
            best_target=("target_value", "max"),
            min_target=("target_value", "min"),
            target_range=("target_value", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
    )
    site_date["target_collapse"] = (
        (site_date["best_target"] <= args.tolerance)
        & (site_date["min_target"] <= args.tolerance)
        & (site_date["target_range"] > args.tolerance)
    )

    signatures = []
    for (site, date_t), group in merged.groupby(group_cols):
        row = {
            "site": site,
            "date_t": date_t,
            "target_signature": target_signature(group, args.round_digits),
            "state_signature": state_signature(group, args.round_digits),
        }
        signatures.append(row)
    signatures_df = pd.DataFrame(signatures)

    dup_groups = (
        signatures_df.groupby(["date_t", "target_signature"])
        .agg(
            n_sites=("site", "size"),
            sites=("site", lambda s: ",".join(s)),
            n_state_signatures=("state_signature", "nunique"),
            state_signatures=("state_signature", lambda s: ";".join(sorted(set(str(v) for v in s)))),
        )
        .reset_index()
    )
    dup_groups = dup_groups[dup_groups["n_sites"] > 1].copy()

    collapse_lookup = site_date.set_index(["site", "date_t"])["target_collapse"].to_dict()
    if not dup_groups.empty:
        dup_groups["all_target_collapse"] = dup_groups.apply(
            lambda row: all(
                collapse_lookup.get((site, row["date_t"]), False)
                for site in str(row["sites"]).split(",")
            ),
            axis=1,
        )
        dup_groups["state_diff_under_duplicate"] = dup_groups["n_state_signatures"] > 1
    else:
        dup_groups["all_target_collapse"] = []
        dup_groups["state_diff_under_duplicate"] = []

    same_date_duplicate_groups = int(len(dup_groups))
    duplicate_collapse_groups = int(
        (
            (dup_groups["all_target_collapse"] == True)
            & (dup_groups["state_diff_under_duplicate"] == True)
        ).sum()
    ) if not dup_groups.empty else 0

    status = "passed_multidate_training_readiness_smoke"
    if missing_rows != 0:
        status = "incomplete_rows"
    elif same_date_duplicate_groups > 0 and duplicate_collapse_groups == same_date_duplicate_groups:
        status = "passed_with_target_collapse_caveat"
    elif same_date_duplicate_groups > 0:
        status = "same_date_duplicate_target_curves"

    out_prefix = run_dir / "confirmed_5site_multidate_training_readiness_v1"
    site_date_csv = out_prefix.with_name(out_prefix.name + "_site_date.csv")
    dup_csv = out_prefix.with_name(out_prefix.name + "_duplicate_target_groups.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    md_out = out_prefix.with_suffix(".md")

    site_date.to_csv(site_date_csv, index=False)
    dup_groups.to_csv(dup_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "sites": ",".join(args.sites),
                "n_sites": len(args.sites),
                "n_dates": len(observed_dates),
                "n_ir_values": len(observed_ir),
                "merged_rows": len(merged),
                "expected_rows": expected_rows,
                "missing_rows": missing_rows,
                "target_collapse_site_dates": int(site_date["target_collapse"].sum()),
                "same_date_duplicate_groups": same_date_duplicate_groups,
                "duplicate_collapse_groups": duplicate_collapse_groups,
                "status": status,
            }
        ]
    )
    summary.to_csv(summary_csv, index=False)

    lines = [
        "# Confirmed 5-Site Multidate Training Readiness V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Site-Date Target Collapse",
        "",
        markdown_table(site_date),
        "",
        "## Same-Date Duplicate Target Groups",
        "",
        markdown_table(dup_groups),
        "",
        "## Interpretation",
        "",
    ]
    if status == "passed_multidate_training_readiness_smoke":
        lines.append("The multidate run is complete and has no same-date duplicate target curves.")
    elif status == "passed_with_target_collapse_caveat":
        lines.append(
            "The multidate run is complete. Same-date duplicate target curves are explained by target collapse with different underlying SWAP-state signatures."
        )
    elif status == "same_date_duplicate_target_curves":
        lines.append("Some same-date duplicate target curves are not fully explained by target-collapse state differences.")
    else:
        lines.append("The run is missing expected candidate rows and should not be used for training data preparation yet.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{site_date_csv}`",
            f"- `{dup_csv}`",
            f"- `{summary_csv}`",
        ]
    )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site multidate training readiness v1")
    print(f"run_dir: {run_dir}")
    print(f"site_date_csv: {site_date_csv}")
    print(f"duplicate_groups_csv: {dup_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"md: {md_out}")
    print(summary.to_string(index=False))
    if not dup_groups.empty:
        print("")
        print(dup_groups.to_string(index=False))


if __name__ == "__main__":
    main()
