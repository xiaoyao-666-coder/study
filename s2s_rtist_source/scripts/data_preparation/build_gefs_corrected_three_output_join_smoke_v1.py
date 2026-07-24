#!/usr/bin/env python3
"""Join corrected GEFS weather to existing three-output SWAP smoke labels.

The existing labels were generated with gridMET, so the result validates only
schema, keys, formulas, and tensor shape. It is deliberately blocked from
surrogate training until SWAP is rerun with the same corrected GEFS weather.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_corrected_three_output_join_smoke_contract_v1.json"
)
DEFAULT_WEATHER = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "_local_gefs_corrected_surrogate_weather_smoke_v1"
    / "gefs_corrected_surrogate_weather_wide_smoke_v1.csv"
)
DEFAULT_SWAP_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "confirmed_5site_restart_generation_smoke_v1"
    / "fixed_0_100cm_npd24_5site_smoke_20260715_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "_local_gefs_corrected_three_output_join_smoke_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-corrected-three-output-join-smoke-v1":
        raise ValueError("three-output join contract id mismatch")
    bounds = contract["continuous_irrigation_constraint_mm"]
    if float(bounds["minimum"]) != 0.0 or float(bounds["maximum"]) != 60.0:
        raise ValueError("continuous irrigation constraint must remain 0-60 mm")
    provenance = contract["scenario_provenance"]
    if provenance["weather_label_scenario_consistent"] is not False:
        raise ValueError("smoke must record the known weather-label scenario mismatch")
    if provenance["training_eligible"] is not False:
        raise ValueError("scenario-mismatched smoke must not be training eligible")
    if any(contract["scope"].values()):
        raise ValueError("three-output join smoke permits a forbidden operation")
    return contract


def _canonical_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    return parsed.dt.strftime("%Y-%m-%d")


def load_swap_labels(root: Path, contract: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for site in contract["expected_sites"]:
        path = Path(root) / site / "site_restart_generation_smoke.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing SWAP smoke labels: {path}")
        frame = pd.read_csv(path)
        if set(frame["site"].astype(str)) != {site}:
            raise ValueError(f"SWAP label site mismatch in {path}")
        frame["source_label_file"] = str(path)
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    labels = labels.rename(columns={"site": "site_id", "ir": "candidate_irrigation_mm"})
    labels["decision_date"] = _canonical_date(labels["date_t"])
    return labels


def audit_label_formulas(
    labels: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, float]:
    revenue = contract["net_revenue_label"]
    labels["cwdm_value"] = pd.to_numeric(labels["cwdm_value"], errors="raise")
    labels["candidate_irrigation_mm"] = pd.to_numeric(
        labels["candidate_irrigation_mm"], errors="raise"
    )
    labels["target_value"] = pd.to_numeric(labels["target_value"], errors="raise")
    labels["net_gain_7d"] = pd.to_numeric(labels["net_gain_7d"], errors="raise")
    baseline = (
        labels.loc[labels["candidate_irrigation_mm"] == 0.0]
        .set_index(["site_id", "decision_date"])["cwdm_value"]
    )
    keys = pd.MultiIndex.from_frame(labels[["site_id", "decision_date"]])
    baseline_values = baseline.reindex(keys).to_numpy(dtype=float)
    expected_revenue = (
        (labels["cwdm_value"].to_numpy(dtype=float) - baseline_values)
        * float(revenue["yield_price_per_kg"])
        - labels["candidate_irrigation_mm"].to_numpy(dtype=float)
        * float(revenue["water_cost_per_ha_per_mm"])
        * float(revenue["weight_index"])
    )
    target = labels["target_value"].to_numpy(dtype=float)
    gain = labels["net_gain_7d"].to_numpy(dtype=float)
    daily_aet = np.column_stack(
        [labels[f"aet_day{day:02d}_mm"].to_numpy(dtype=float) for day in range(1, 8)]
    ).sum(axis=1)
    component_aet = sum(
        labels[f"{component}_7d_mm"].to_numpy(dtype=float)
        for component in ("tact", "eact", "interc")
    )
    reported_aet = labels["aet_7d_mm"].to_numpy(dtype=float)
    return {
        "maximum_absolute_net_revenue_formula_error": float(
            np.max(np.abs(target - expected_revenue))
        ),
        "maximum_absolute_target_vs_net_gain_error": float(np.max(np.abs(target - gain))),
        "maximum_absolute_aet_daily_sum_error_mm": float(
            np.max(np.abs(reported_aet - daily_aet))
        ),
        "maximum_absolute_aet_component_sum_error_mm": float(
            np.max(np.abs(reported_aet - component_aet))
        ),
    }


def validate_inputs(
    weather: pd.DataFrame, labels: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, Any]:
    weather_required = {
        "site_id",
        "decision_date",
        "horizon_days",
        "gefs_member_count",
        "weather_source",
        "artifact_sha256",
    }
    label_required = {
        "site_id",
        "decision_date",
        "candidate_irrigation_mm",
        "target_value",
        "net_gain_7d",
        "aet_7d_mm",
        "control_volume_type",
        "control_depth_cm",
        "water_balance_residual_0_100cm_7d_mm",
        *contract["physics_auxiliary_fields"],
        *[f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)],
        *[f"aet_day{day:02d}_mm" for day in range(1, 8)],
    }
    missing_weather = weather_required.difference(weather.columns)
    missing_labels = label_required.difference(labels.columns)
    if missing_weather or missing_labels:
        raise ValueError(
            f"missing fields: weather={sorted(missing_weather)}, labels={sorted(missing_labels)}"
        )
    weather["decision_date"] = _canonical_date(weather["decision_date"])
    expected_sites = set(contract["expected_sites"])
    expected_dates = set(contract["expected_decision_dates"])
    if set(weather["site_id"].astype(str)) != expected_sites:
        raise ValueError("weather site set mismatch")
    if set(labels["site_id"].astype(str)) != expected_sites:
        raise ValueError("label site set mismatch")
    if set(weather["decision_date"]) != expected_dates:
        raise ValueError("weather decision date mismatch")
    if set(labels["decision_date"]) != expected_dates:
        raise ValueError("label decision date mismatch")
    if weather.duplicated(["site_id", "decision_date"]).any():
        raise ValueError("weather join keys are not unique")
    if labels.duplicated(["site_id", "decision_date", "candidate_irrigation_mm"]).any():
        raise ValueError("label sample keys are not unique")
    if len(labels) != int(contract["expected_row_count"]):
        raise ValueError("label row count mismatch")
    expected_candidates = sorted(float(x) for x in contract["expected_irrigation_candidates_mm"])
    for key, group in labels.groupby(["site_id", "decision_date"], sort=True):
        actual = sorted(group["candidate_irrigation_mm"].astype(float).tolist())
        if actual != expected_candidates:
            raise ValueError(f"irrigation candidate set mismatch for {key}: {actual}")
    bounds = contract["continuous_irrigation_constraint_mm"]
    irrigation = labels["candidate_irrigation_mm"].astype(float)
    if not irrigation.between(float(bounds["minimum"]), float(bounds["maximum"])).all():
        raise ValueError("irrigation candidate outside 0-60 mm")
    primary_source = ["target_value", "aet_7d_mm"] + [
        f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)
    ]
    numeric = labels[primary_source].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("primary output contains missing or nonfinite values")
    if set(labels["control_volume_type"].astype(str)) != {
        contract["control_volume"]["type"]
    }:
        raise ValueError("control volume type mismatch")
    depth_error = np.max(
        np.abs(
            labels["control_depth_cm"].astype(float).to_numpy()
            - float(contract["control_volume"]["depth_cm"])
        )
    )
    residual = labels["water_balance_residual_0_100cm_7d_mm"].astype(float)
    formula = audit_label_formulas(labels, contract)
    tolerance = 1e-9
    if any(value > tolerance for value in formula.values()):
        raise ValueError(f"label formula audit failed: {formula}")
    maximum_residual = float(residual.abs().max())
    if maximum_residual > float(
        contract["control_volume"]["maximum_absolute_water_balance_residual_mm"]
    ):
        raise ValueError("water balance residual exceeds contract")
    return {
        **formula,
        "maximum_absolute_control_depth_error_cm": float(depth_error),
        "maximum_absolute_water_balance_residual_mm": maximum_residual,
    }


def build_joined_table(
    weather: pd.DataFrame, labels: pd.DataFrame, contract: dict[str, Any]
) -> pd.DataFrame:
    weather_columns = list(weather.columns)
    label_columns = [
        "site_id",
        "decision_date",
        "candidate_irrigation_mm",
        "target_value",
        "net_gain_7d",
        "aet_7d_mm",
        *[f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)],
        "end_daynr",
        "dvs",
        "lai",
        "rootd",
        "cwdm_value",
        "cwso_value",
        "predecision_soil_vwc_0_100cm",
        "predecision_soil_storage_0_100cm_mm",
        "control_volume_type",
        "control_depth_cm",
        *contract["physics_auxiliary_fields"],
        *[f"aet_day{day:02d}_mm" for day in range(1, 8)],
        "source_label_file",
    ]
    joined = labels[label_columns].merge(
        weather[weather_columns],
        on=["site_id", "decision_date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if set(joined.pop("_merge")) != {"both"}:
        raise ValueError("not every SWAP label row matched corrected GEFS weather")
    joined.insert(
        0,
        "sample_id",
        joined.apply(
            lambda row: (
                f"{row.site_id}_{str(row.decision_date).replace('-', '')}_"
                f"ir{float(row.candidate_irrigation_mm):05.1f}"
            ),
            axis=1,
        ),
    )
    joined.insert(4, "net_revenue_7d_usd_ha", joined["target_value"].astype(float))
    provenance = contract["scenario_provenance"]
    joined["weather_label_scenario_consistent"] = bool(
        provenance["weather_label_scenario_consistent"]
    )
    joined["training_eligible"] = bool(provenance["training_eligible"])
    if joined["sample_id"].duplicated().any():
        raise ValueError("joined sample ids are not unique")
    return joined.sort_values(
        ["site_id", "decision_date", "candidate_irrigation_mm"]
    ).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    weather = pd.read_csv(args.weather_wide)
    labels = load_swap_labels(args.swap_smoke_root, contract)
    audit = validate_inputs(weather, labels, contract)
    joined = build_joined_table(weather, labels, contract)
    if len(joined) != int(contract["expected_row_count"]):
        raise ValueError("joined row count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "joined": args.output_dir / "gefs_corrected_three_output_joined_smoke_v1.csv",
        "audit": args.output_dir / "gefs_corrected_three_output_join_audit_v1.json",
        "manifest": args.output_dir / "gefs_corrected_three_output_join_manifest_v1.json",
        "report": args.output_dir / "gefs_corrected_three_output_join_conclusion_v1.md",
    }
    joined.to_csv(paths["joined"], index=False, encoding="utf-8-sig")
    audit_payload = {
        "contract_id": contract["contract_id"],
        "status": "interface_join_passed_training_blocked_weather_label_scenario_mismatch",
        "joined_rows": int(len(joined)),
        "site_count": int(joined["site_id"].nunique()),
        "decision_date_count": int(joined["decision_date"].nunique()),
        "irrigation_candidate_count_per_site_date": int(
            joined.groupby(["site_id", "decision_date"]).size().min()
        ),
        "minimum_irrigation_mm": float(joined["candidate_irrigation_mm"].min()),
        "maximum_irrigation_mm": float(joined["candidate_irrigation_mm"].max()),
        "duplicate_sample_id_count": int(joined["sample_id"].duplicated().sum()),
        "primary_output_missing_value_count": int(
            joined[contract["primary_outputs"]].isna().sum().sum()
        ),
        **audit,
        "weather_label_scenario_consistent": False,
        "training_eligible": False,
        "next_required_action": contract["scenario_provenance"]["required_before_training"],
    }
    paths["audit"].write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract),
        "weather_wide_file_sha256": sha256_file(args.weather_wide),
        "swap_label_file_sha256_by_site": {
            site: sha256_file(
                args.swap_smoke_root / site / "site_restart_generation_smoke.csv"
            )
            for site in contract["expected_sites"]
        },
        "joined_file_sha256": sha256_file(paths["joined"]),
        "network_download_performed": False,
        "artifact_refit_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "training_use_allowed": False,
        "status": audit_payload["status"],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 订正 GEFS 与三输出标签联调结论

- 接口拼接：通过
- 拼接样本：`{audit_payload['joined_rows']}` 行（5 站点 x 8 灌溉候选）
- 灌溉范围：`{audit_payload['minimum_irrigation_mm']:.1f}-{audit_payload['maximum_irrigation_mm']:.1f} mm`
- 三输出缺失值：`{audit_payload['primary_output_missing_value_count']}`
- 净收益公式最大误差：`{audit_payload['maximum_absolute_net_revenue_formula_error']:.3e}`
- AET 日累计最大误差：`{audit_payload['maximum_absolute_aet_daily_sum_error_mm']:.3e} mm`
- AET 分量累计最大误差：`{audit_payload['maximum_absolute_aet_component_sum_error_mm']:.3e} mm`
- 最大绝对水量平衡残差：`{audit_payload['maximum_absolute_water_balance_residual_mm']:.6f} mm`

本次只证明订正 GEFS 天气、灌溉动作和三输出标签可以按数据契约无歧义拼接。
现有标签由 gridMET 驱动 SWAP，新特征来自订正 GEFS，天气与标签情景不一致，
因此 `training_eligible=false`，不得使用这 40 行训练或评价代理模型。

下一步：用同一订正 GEFS 天气驱动 5 站点 x 8 灌溉候选重跑 SWAP，生成情景一致的 40 行标签。

状态：`{audit_payload['status']}`
"""
    paths["report"].write_text(report, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--weather-wide", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--swap-smoke-root", type=Path, default=DEFAULT_SWAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
