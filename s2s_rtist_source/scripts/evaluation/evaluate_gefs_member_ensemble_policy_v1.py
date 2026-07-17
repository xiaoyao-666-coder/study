#!/usr/bin/env python3
"""Evaluate irrigation decisions from member-level GEFS surrogate predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from s2s_rtist.weather.gefs_ensemble_policy import (
    select_irrigation_by_mean_profit,
    summarize_member_optima,
    summarize_member_predictions,
)


DEFAULT_OUTPUT_DIR = (
    Path("site_general_surrogate_eval") / "gefs_member_ensemble_policy_v1"
)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def run_evaluation(
    *,
    input_path: Path,
    output_dir: Path,
    expected_members: Sequence[str] | None = None,
) -> dict[str, Path]:
    predictions = pd.read_csv(input_path)
    candidate_summary = summarize_member_predictions(
        predictions, expected_members=expected_members
    )
    decisions = select_irrigation_by_mean_profit(
        predictions, expected_members=expected_members
    )
    member_optima = summarize_member_optima(
        predictions, expected_members=expected_members
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary": output_dir / "gefs_member_candidate_summary_v1.csv",
        "decisions": output_dir / "gefs_member_decisions_v1.csv",
        "member_optima": output_dir / "gefs_member_optimum_baseline_v1.csv",
        "report": output_dir / "gefs_member_ensemble_policy_v1.md",
    }
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    decisions.to_csv(outputs["decisions"], index=False)
    member_optima.to_csv(outputs["member_optima"], index=False)

    lines = [
        "# GEFS Member Ensemble Irrigation Policy V1",
        "",
        "## Formal Decision Rule",
        "",
        "For every site, decision date, and candidate irrigation amount, run all "
        "available GEFS members through the same surrogate. Select the smallest "
        "irrigation amount tied for the highest ensemble-mean predicted profit.",
        "",
        "Member standard deviation and quantiles are retained as audit fields. "
        "They do not alter the V1 decision score.",
        "",
        "The official `geavg` product is retained only as a deterministic baseline; "
        "it is not used in the formal member-level decision.",
        "",
        "## Decisions",
        "",
        _markdown_table(decisions),
        "",
        "## Member-Optimum-Then-Average Baseline",
        "",
        _markdown_table(member_optima),
        "",
        "This baseline first selects an optimum for each member and then summarizes "
        "those irrigation amounts. It is reported for comparison and is not the V1 "
        "formal decision rule.",
    ]
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Member-level prediction CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--expected-members",
        nargs="+",
        default=None,
        help="Override the formal gec00 plus gep01-gep30 member set",
    )
    args = parser.parse_args()

    outputs = run_evaluation(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        expected_members=args.expected_members,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
