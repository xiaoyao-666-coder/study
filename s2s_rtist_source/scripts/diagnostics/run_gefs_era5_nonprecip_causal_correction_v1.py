#!/usr/bin/env python3
"""Evaluate causal ERA5-based linear-scaling candidates for GEFS nonprecip weather."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


VARIABLES = (
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)
POSITIVE_VARIABLES = (
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)
MULTIPLICATIVE_VARIABLES = (
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
)
DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pairs(gefs: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    raw = gefs.copy()
    reference = era5.copy()
    for frame in (raw, reference):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.strftime("%Y-%m-%d")
        frame["target_year"] = pd.to_datetime(frame["decision_date"]).dt.year
    keys = [
        "target_year",
        "decision_date",
        "site_id",
        "local_date",
        "lead_day",
    ]
    if raw[keys].duplicated().any() or reference[keys].duplicated().any():
        raise ValueError("GEFS or ERA5 causal inputs contain duplicate keys")
    pairs = raw[keys + list(VARIABLES)].merge(
        reference[keys + list(VARIABLES)],
        on=keys,
        how="inner",
        suffixes=("_gefs", "_era5"),
        validate="one_to_one",
    )
    if len(pairs) != len(raw) or len(pairs) != len(reference):
        raise ValueError("GEFS and ERA5 causal inputs do not have exact coverage")
    pairs["temperature_center_c_gefs"] = (
        pairs["temperature_min_c_gefs"] + pairs["temperature_max_c_gefs"]
    ) / 2.0
    pairs["temperature_center_c_era5"] = (
        pairs["temperature_min_c_era5"] + pairs["temperature_max_c_era5"]
    ) / 2.0
    pairs["temperature_range_c_gefs"] = (
        pairs["temperature_max_c_gefs"] - pairs["temperature_min_c_gefs"]
    )
    pairs["temperature_range_c_era5"] = (
        pairs["temperature_max_c_era5"] - pairs["temperature_min_c_era5"]
    )
    if (
        pairs[["temperature_range_c_gefs", "temperature_range_c_era5"]] < 0.0
    ).any().any():
        raise ValueError("negative diurnal temperature range")
    return pairs.sort_values(keys).reset_index(drop=True)


def causal_fit_rows(pairs: pd.DataFrame, target_date: str) -> pd.DataFrame:
    target = pd.Timestamp(target_date)
    target_year = target.year
    decision = pd.to_datetime(pairs["decision_date"])
    if target_year == 2015:
        horizon_end = decision + pd.Timedelta(days=6)
        selected = pairs.loc[(decision.dt.year == 2015) & (horizon_end < target)].copy()
    else:
        selected = pairs.loc[decision.dt.year < target_year].copy()
    return selected


def fit_target_factors(
    pairs: pd.DataFrame,
    *,
    target_date: str,
    minimum_samples: int,
    expected_groups: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    fit = causal_fit_rows(pairs, target_date)
    group_keys = ["site_id", "lead_day"]
    if expected_groups is None:
        target = pairs.loc[pairs["decision_date"] == target_date]
        expected = target[group_keys].drop_duplicates()
    else:
        expected = expected_groups[group_keys].drop_duplicates().copy()
    if expected.empty or expected[group_keys].duplicated().any():
        raise ValueError("target factor groups must be nonempty and unique")
    if fit.empty:
        return None
    rows = []
    for keys, group in fit.groupby(group_keys, sort=False):
        site_id, lead_day = keys
        gefs_range_mean = float(group["temperature_range_c_gefs"].mean())
        positive_means = {
            variable: (
                float(group[f"{variable}_era5"].mean()),
                float(group[f"{variable}_gefs"].mean()),
            )
            for variable in MULTIPLICATIVE_VARIABLES
        }
        if gefs_range_mean <= 0.0 or any(raw_mean <= 0.0 for _, raw_mean in positive_means.values()):
            raise ValueError("nonpositive GEFS fit mean prevents multiplicative scaling")
        solar_x = group["solar_kj_m2_day_gefs"].to_numpy(dtype=float)
        solar_y = group["solar_kj_m2_day_era5"].to_numpy(dtype=float)
        solar_variance = float(np.var(solar_x))
        if solar_variance > 1e-12:
            solar_slope = max(
                0.0,
                float(np.cov(solar_x, solar_y, ddof=0)[0, 1] / solar_variance),
            )
        else:
            solar_slope = 1.0
        solar_intercept = float(solar_y.mean() - solar_slope * solar_x.mean())
        rows.append(
            {
                "target_year": pd.Timestamp(target_date).year,
                "target_date": target_date,
                "site_id": site_id,
                "lead_day": int(lead_day),
                "fit_sample_count": int(len(group)),
                "fit_first_decision_date": str(group["decision_date"].min()),
                "fit_last_decision_date": str(group["decision_date"].max()),
                "fit_first_year": int(pd.to_datetime(group["decision_date"]).dt.year.min()),
                "fit_last_year": int(pd.to_datetime(group["decision_date"]).dt.year.max()),
                "temperature_center_additive_delta_c": float(
                    (
                        group["temperature_center_c_era5"]
                        - group["temperature_center_c_gefs"]
                    ).mean()
                ),
                "temperature_range_ratio": float(
                    group["temperature_range_c_era5"].mean() / gefs_range_mean
                ),
                "solar_kj_m2_day_affine_intercept": solar_intercept,
                "solar_kj_m2_day_affine_slope": solar_slope,
                **{
                    f"{variable}_ratio": reference_mean / raw_mean
                    for variable, (reference_mean, raw_mean) in positive_means.items()
                },
            }
        )
    factors = pd.DataFrame(rows)
    factors = expected.merge(
        factors, on=group_keys, how="left", validate="one_to_one"
    )
    if factors["fit_sample_count"].isna().any():
        return None
    if int(factors["fit_sample_count"].min()) < minimum_samples:
        return None
    ratio_columns = [
        "temperature_range_ratio",
        *[f"{variable}_ratio" for variable in MULTIPLICATIVE_VARIABLES],
    ]
    if not np.isfinite(factors[ratio_columns].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite causal correction factor")
    if (factors[ratio_columns] <= 0.0).any().any():
        raise ValueError("nonpositive causal correction factor")
    if not np.isfinite(
        factors[
            [
                "solar_kj_m2_day_affine_intercept",
                "solar_kj_m2_day_affine_slope",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("nonfinite solar affine factor")
    return factors.sort_values(group_keys).reset_index(drop=True)


def apply_candidate(
    target: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    data = target.merge(
        factors,
        on=["target_year", "site_id", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    if data["fit_sample_count"].isna().any():
        raise ValueError("causal factor coverage is incomplete")
    center = data["temperature_center_c_gefs"] + alpha * data[
        "temperature_center_additive_delta_c"
    ]
    range_factor = 1.0 + alpha * (data["temperature_range_ratio"] - 1.0)
    temperature_range = data["temperature_range_c_gefs"] * range_factor
    data["temperature_min_c_corrected"] = center - temperature_range / 2.0
    data["temperature_max_c_corrected"] = center + temperature_range / 2.0
    for variable in MULTIPLICATIVE_VARIABLES:
        effective_factor = 1.0 + alpha * (data[f"{variable}_ratio"] - 1.0)
        data[f"{variable}_corrected"] = data[f"{variable}_gefs"] * effective_factor
        data[f"{variable}_effective_factor"] = effective_factor
    solar_effective_intercept = alpha * data["solar_kj_m2_day_affine_intercept"]
    solar_effective_slope = 1.0 + alpha * (
        data["solar_kj_m2_day_affine_slope"] - 1.0
    )
    data["solar_kj_m2_day_corrected"] = (
        solar_effective_intercept
        + solar_effective_slope * data["solar_kj_m2_day_gefs"]
    ).clip(lower=0.0)
    data["solar_kj_m2_day_effective_affine_intercept"] = solar_effective_intercept
    data["solar_kj_m2_day_effective_affine_slope"] = solar_effective_slope
    data["candidate_id"] = f"hybrid_affine_solar_shrink_a{alpha:g}"
    data["shrinkage_alpha"] = float(alpha)
    corrected_columns = [f"{variable}_corrected" for variable in VARIABLES]
    if not np.isfinite(data[corrected_columns].to_numpy(dtype=float)).all():
        raise ValueError("corrected causal weather contains nonfinite values")
    if (data["temperature_min_c_corrected"] > data["temperature_max_c_corrected"]).any():
        raise ValueError("corrected Tmin exceeds Tmax")
    if (data[[f"{variable}_corrected" for variable in POSITIVE_VARIABLES]] < 0.0).any().any():
        raise ValueError("corrected positive-only variable is negative")
    return data


def raw_candidate(target: pd.DataFrame) -> pd.DataFrame:
    data = target.copy()
    for variable in VARIABLES:
        data[f"{variable}_corrected"] = data[f"{variable}_gefs"]
    data["candidate_id"] = "raw_gefs"
    data["shrinkage_alpha"] = 0.0
    return data


def build_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["target_year", "candidate_id", "shrinkage_alpha"]
    for keys, group in candidates.groupby(group_keys, sort=False):
        target_year, candidate_id, alpha = keys
        for variable in VARIABLES:
            error = (
                group[f"{variable}_corrected"].to_numpy(dtype=float)
                - group[f"{variable}_era5"].to_numpy(dtype=float)
            )
            rows.append(
                {
                    "target_year": int(target_year),
                    "candidate_id": candidate_id,
                    "shrinkage_alpha": float(alpha),
                    "variable": variable,
                    "sample_count": int(len(error)),
                    "bias_corrected_minus_era5": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_year", "variable", "shrinkage_alpha"]
    ).reset_index(drop=True)


def run_evaluation(
    pairs: pd.DataFrame,
    *,
    alphas: Sequence[float],
    minimum_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    factor_parts = []
    candidate_parts = []
    eligible_cycles = []
    ineligible_cycles = []
    leakage_count = 0
    for target_date in sorted(pairs["decision_date"].unique()):
        factors = fit_target_factors(
            pairs, target_date=target_date, minimum_samples=minimum_samples
        )
        if factors is None:
            ineligible_cycles.append(target_date)
            continue
        target = pairs.loc[pairs["decision_date"] == target_date].copy()
        target_year = int(pd.Timestamp(target_date).year)
        if target_year == 2015:
            leakage_count += int(
                (
                    pd.to_datetime(factors["fit_last_decision_date"])
                    + pd.Timedelta(days=6)
                    >= pd.Timestamp(target_date)
                ).sum()
            )
        else:
            leakage_count += int((factors["fit_last_year"] >= target_year).sum())
        factor_parts.append(factors)
        candidate_parts.append(raw_candidate(target))
        for alpha in alphas:
            candidate_parts.append(apply_candidate(target, factors, alpha=float(alpha)))
        eligible_cycles.append(target_date)
    if not factor_parts or not candidate_parts:
        raise ValueError("no causal cycles were eligible for correction evaluation")
    factors = pd.concat(factor_parts, ignore_index=True)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    metrics = build_metrics(candidates)
    corrected_fields = [f"{variable}_corrected" for variable in VARIABLES]
    audit = {
        "status": "gefs_era5_nonprecip_causal_candidates_generated",
        "paired_rows": int(len(pairs)),
        "eligible_cycle_count": len(eligible_cycles),
        "ineligible_cycle_count": len(ineligible_cycles),
        "eligible_cycles_by_year": {
            str(year): int(
                sum(pd.Timestamp(cycle).year == year for cycle in eligible_cycles)
            )
            for year in sorted(pd.to_datetime(pairs["decision_date"]).dt.year.unique())
        },
        "minimum_fit_samples_per_site_lead": int(minimum_samples),
        "factor_rows": int(len(factors)),
        "candidate_rows": int(len(candidates)),
        "metric_rows": int(len(metrics)),
        "fit_leakage_rows": int(leakage_count),
        "missing_corrected_value_count": int(candidates[corrected_fields].isna().sum().sum()),
        "contains_2024": bool((candidates["target_year"] == 2024).any()),
        "candidate_selection_performed": False,
        "five_member_validation_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": "select_candidate_from_2015_2018_causal_oof_then_confirm_on_2019",
    }
    if audit["fit_leakage_rows"] or audit["missing_corrected_value_count"]:
        raise ValueError("causal candidate audit failed")
    return factors, candidates, metrics, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    pairs = prepare_pairs(pd.read_csv(args.gefs_weather), pd.read_csv(args.era5_reference))
    factors, candidates, metrics, audit = run_evaluation(
        pairs, alphas=args.alphas, minimum_samples=args.minimum_samples
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "factors": args.output_dir / "gefs_era5_nonprecip_causal_factors_v1.csv",
        "candidates": args.output_dir / "gefs_era5_nonprecip_causal_candidates_v1.csv",
        "metrics": args.output_dir / "gefs_era5_nonprecip_causal_metrics_v1.csv",
        "audit": args.output_dir / "gefs_era5_nonprecip_causal_correction_audit_v1.json",
        "manifest": args.output_dir / "gefs_era5_nonprecip_causal_correction_manifest_v1.json",
    }
    factors.to_csv(outputs["factors"], index=False)
    candidates.to_csv(outputs["candidates"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "gefs_weather_sha256": sha256_file(args.gefs_weather),
        "era5_reference_sha256": sha256_file(args.era5_reference),
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gefs-weather", type=Path, required=True)
    parser.add_argument("--era5-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--minimum-samples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))
