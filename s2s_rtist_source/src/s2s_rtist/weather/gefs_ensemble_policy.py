"""Aggregate member-level GEFS surrogate predictions into irrigation decisions."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .gefs_gridmet_bias import gefs_members


REQUIRED_COLUMNS = {
    "site_date_id",
    "candidate_ir",
    "gefs_member",
    "pred_net_gain_7d",
}


def _validated_predictions(
    frame: pd.DataFrame, expected_members: Sequence[str]
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    expected = tuple(str(member) for member in expected_members)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_members must contain unique member names")

    data = frame.copy()
    data["site_date_id"] = data["site_date_id"].astype(str)
    data["gefs_member"] = data["gefs_member"].astype(str)
    for column in ("candidate_ir", "pred_net_gain_7d"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"column {column!r} contains missing or non-numeric values")

    key_columns = ["site_date_id", "candidate_ir", "gefs_member"]
    duplicate = data.duplicated(key_columns, keep=False)
    if duplicate.any():
        row = data.loc[duplicate, key_columns].iloc[0].to_dict()
        raise ValueError(f"duplicate GEFS member prediction: {row}")

    expected_set = set(expected)
    for group_key, group in data.groupby(
        ["site_date_id", "candidate_ir"], sort=False, dropna=False
    ):
        actual_set = set(group["gefs_member"])
        if actual_set != expected_set:
            missing = sorted(expected_set.difference(actual_set))
            extra = sorted(actual_set.difference(expected_set))
            raise ValueError(
                "incomplete GEFS member set for "
                f"{group_key}: missing={missing}, extra={extra}"
            )
    return data


def summarize_member_predictions(
    frame: pd.DataFrame,
    *,
    expected_members: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize the profit distribution for each irrigation candidate."""

    members = tuple(expected_members) if expected_members is not None else gefs_members()
    data = _validated_predictions(frame, members)
    rows: list[dict[str, object]] = []
    for (site_date_id, candidate_ir), group in data.groupby(
        ["site_date_id", "candidate_ir"], sort=False
    ):
        profit = group["pred_net_gain_7d"]
        rows.append(
            {
                "site_date_id": site_date_id,
                "candidate_ir": float(candidate_ir),
                "member_count": int(len(group)),
                "mean_pred_net_gain_7d": float(profit.mean()),
                "std_pred_net_gain_7d": float(profit.std(ddof=0)),
                "min_pred_net_gain_7d": float(profit.min()),
                "p10_pred_net_gain_7d": float(profit.quantile(0.10)),
                "median_pred_net_gain_7d": float(profit.median()),
                "p90_pred_net_gain_7d": float(profit.quantile(0.90)),
                "max_pred_net_gain_7d": float(profit.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["site_date_id", "candidate_ir"], kind="stable"
    ).reset_index(drop=True)


def select_irrigation_by_mean_profit(
    frame: pd.DataFrame,
    *,
    expected_members: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Choose the smallest irrigation tied for maximum ensemble-mean profit."""

    summary = summarize_member_predictions(
        frame, expected_members=expected_members
    )
    rows: list[dict[str, object]] = []
    for site_date_id, group in summary.groupby("site_date_id", sort=False):
        chosen = group.sort_values(
            ["mean_pred_net_gain_7d", "candidate_ir"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "site_date_id": site_date_id,
                "chosen_ir": float(chosen["candidate_ir"]),
                "chosen_mean_pred_net_gain_7d": float(
                    chosen["mean_pred_net_gain_7d"]
                ),
                "chosen_std_pred_net_gain_7d": float(
                    chosen["std_pred_net_gain_7d"]
                ),
                "chosen_p10_pred_net_gain_7d": float(
                    chosen["p10_pred_net_gain_7d"]
                ),
                "chosen_p90_pred_net_gain_7d": float(
                    chosen["p90_pred_net_gain_7d"]
                ),
                "member_count": int(chosen["member_count"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_member_optima(
    frame: pd.DataFrame,
    *,
    expected_members: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize the alternative policy that optimizes each member separately."""

    members = tuple(expected_members) if expected_members is not None else gefs_members()
    data = _validated_predictions(frame, members)
    optimum_rows: list[dict[str, object]] = []
    for (site_date_id, member), group in data.groupby(
        ["site_date_id", "gefs_member"], sort=False
    ):
        chosen = group.sort_values(
            ["pred_net_gain_7d", "candidate_ir"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        optimum_rows.append(
            {
                "site_date_id": site_date_id,
                "gefs_member": member,
                "member_optimum_ir": float(chosen["candidate_ir"]),
                "member_optimum_pred_net_gain_7d": float(
                    chosen["pred_net_gain_7d"]
                ),
            }
        )

    optimum = pd.DataFrame(optimum_rows)
    rows: list[dict[str, object]] = []
    for site_date_id, group in optimum.groupby("site_date_id", sort=False):
        irrigation = group["member_optimum_ir"]
        rows.append(
            {
                "site_date_id": site_date_id,
                "member_count": int(len(group)),
                "mean_member_optimum_ir": float(irrigation.mean()),
                "median_member_optimum_ir": float(irrigation.median()),
                "mode_member_optimum_ir": float(irrigation.mode().min()),
                "min_member_optimum_ir": float(irrigation.min()),
                "max_member_optimum_ir": float(irrigation.max()),
            }
        )
    return pd.DataFrame(rows)
