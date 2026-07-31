from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import unittest
from pathlib import Path

import pandas as pd


MODULE_NAME = (
    "scripts.evaluation.freeze_gefs_p3_2022_b2_independent_protocol_v1"
)


class _PatchCompat:
    def __init__(self) -> None:
        self._saved: list[tuple[object, str, object]] = []

    def setattr(self, obj: object, name: str, value: object) -> None:
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


class _Raises:
    def __init__(self, exception: type[BaseException], match: str | None = None) -> None:
        self.exception = exception
        self.match = match

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            raise AssertionError("expected exception")
        if not issubclass(exc_type, self.exception):
            return False
        if self.match is not None and self.match not in str(exc_value):
            raise AssertionError((self.match, str(exc_value)))
        return True


def raises(exception: type[BaseException], match: str | None = None) -> _Raises:
    return _Raises(exception, match)


def load_module():
    return importlib.import_module(MODULE_NAME)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_sha256(payload: dict[str, object]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "model_sha256"}
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_feature_contract(module) -> dict[str, object]:
    return {
        "contract_id": "gefs-p3-strict-loso-gate-features-v1",
        "target_site_id": "P3",
        "candidate_representative": "zero_irrigation_row_after_invariance_check",
        "gate_feature_names": list(module.GATE_FEATURE_NAMES),
        "features": [
            {
                "gate_feature_name": name,
                "aggregation": aggregation,
                "source_columns": columns,
            }
            for name, aggregation, columns in module.GATE_FEATURE_DEFINITIONS
        ],
        "candidate_dependent_columns_forbidden": ["irrigation_mm"],
        "surrogate_predictions_used_as_gate_features": False,
        "swap_targets_used_as_gate_features": False,
        "future_observations_used_as_gate_features": False,
        "standardization_scope": "same_outer_fold_historical_P3_actionable_cycles_only",
    }


def create_inputs(tmp_path: Path, monkeypatch: _PatchCompat):
    module = load_module()
    b2_dir = tmp_path / "b2"
    final_dir = tmp_path / "final"
    dual_dir = tmp_path / "dual"
    b2_dir.mkdir()
    final_dir.mkdir()
    dual_dir.mkdir()

    gate = {
        "model": "numpy_newton_irls_regret_weighted_l2_logistic",
        "fallback": False,
        "fallback_reason": "",
        "feature_names": list(module.GATE_FEATURE_NAMES),
        "threshold": 0.5,
        "tie_route": "P15",
        "training_sites": ["P2", "P4"],
        "training_years": [2015, 2016, 2017, 2018, 2019],
        "P3_rows_used": 0,
        "year_2021_rows_used": 0,
        "year_2024_rows_used": 0,
        "policy_state": "pending_new_independent_freeze",
    }
    gate["model_sha256"] = model_sha256(gate)
    gate_path = b2_dir / module.B2_FINAL_GATE_NAME
    write_json(gate_path, gate)

    audit = {
        "status": module.EXPECTED_B2_STATUS,
        "mandatory_gate_passed": True,
        "development_gate_passed": True,
        "final_gate_written": True,
        "formal_promotion": False,
        "P3_rows_retained": 0,
        "P3_target_values_used": 0,
        "year_2020_rows_retained": 0,
        "year_2021_rows_retained": 0,
        "year_2024_rows_retained": 0,
        "source_checkpoint_retraining_performed": False,
        "source_checkpoint_reselection_performed": False,
        "gate_feature_search_performed": False,
        "gate_weight_search_performed": False,
        "gate_threshold_search_performed": False,
        "network_access_performed": False,
        "gate_threshold": 0.5,
        "gate_tie_route": "P15",
        "scope_filter": {"disallowed_rows_retained": 0},
    }
    audit_path = b2_dir / module.B2_AUDIT_NAME
    write_json(audit_path, audit)

    checkpoint_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, object]] = {}
    for source, data in (("P1", b"fixture-p1"), ("P15", b"fixture-p15")):
        checkpoint = (
            final_dir
            / "source_experts"
            / "final_2015_2019_full_history_refit"
            / source
            / "full_history_composite.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(data)
        checkpoint_paths[source] = checkpoint
        source_records[source] = {
            "checkpoint_path": checkpoint.relative_to(final_dir).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "selected_epoch": 1,
            "fit_rows": 10,
            "selection_rows": 0,
            "full_history_refit": True,
            "checkpoint_track": "composite",
            "lambda_balance": 1.0,
        }
    policy = {
        "policy_id": "gefs-p3-2021-final-full-history-two-expert-gate-v1r2",
        "fit_years": [2015, 2016, 2017, 2018, 2019],
        "target_site": "P3",
        "source_experts": source_records,
        "2021_rows_used": 0,
        "network_access_performed": False,
        "post_freeze_reselection_allowed": False,
    }
    policy_path = final_dir / module.FINAL_POLICY_NAME
    write_json(policy_path, policy)

    dual_protocol = {
        "contract_id": "gefs-p3-2021-dual-gate-evaluation-v1",
        "target_site": "P3",
        "target_year": 2021,
        "formal_primary_experiment": "B_source_only_router",
        "B_source_only_router": {
            "gate_fit_sites": ["P2", "P4"],
            "gate_development_years": [2015, 2016, 2017, 2018, 2019],
            "gate_feature_contract": expected_feature_contract(module),
            "P3_rows_allowed_in_gate_fit": 0,
            "P3_rows_allowed_in_gate_preprocessing": 0,
            "P3_rows_allowed_in_gate_label_construction": 0,
            "gate_hyperparameter_search_allowed": False,
            "gate_threshold": 0.5,
            "gate_tie_route": "P15",
            "irrigation_candidates_mm": list(module.IRRIGATION_GRID),
        },
    }
    dual_path = dual_dir / module.DUAL_PROTOCOL_NAME
    write_json(dual_path, dual_protocol)
    dual_audit_path = dual_dir / module.DUAL_AUDIT_NAME
    write_json(
        dual_audit_path,
        {
            "mandatory_gate_passed": True,
            "experiment_B_P3_rows_allowed": 0,
            "2021_GEFS_rows_read": 0,
            "2021_SWAP_candidate_labels_read": 0,
            "2021_model_inference_performed": False,
            "2021_results_read": False,
            "network_access_performed": False,
        },
    )
    dual_manifest_path = dual_dir / module.DUAL_MANIFEST_NAME
    pd.DataFrame(
        [{"role": "output_dual_gate", "path": dual_path.name, "bytes": dual_path.stat().st_size, "sha256": sha256_file(dual_path)}]
    ).to_csv(dual_manifest_path, index=False)

    manifest_path = b2_dir / module.B2_MANIFEST_NAME
    pd.DataFrame(
        [
            {
                "role": "input_final_full_history_policy",
                "path": str(policy_path),
                "bytes": policy_path.stat().st_size,
                "sha256": sha256_file(policy_path),
            },
            *[
                {
                    "role": f"input_final_source_checkpoint_{source}",
                    "path": str(checkpoint_paths[source]),
                    "bytes": checkpoint_paths[source].stat().st_size,
                    "sha256": sha256_file(checkpoint_paths[source]),
                }
                for source in ("P1", "P15")
            ],
            {
                "role": "output_audit",
                "path": audit_path.name,
                "bytes": audit_path.stat().st_size,
                "sha256": sha256_file(audit_path),
            },
            {
                "role": "output_final_gate",
                "path": gate_path.name,
                "bytes": gate_path.stat().st_size,
                "sha256": sha256_file(gate_path),
            },
            {
                "role": "input_dual_protocol_manifest",
                "path": str(dual_manifest_path),
                "bytes": dual_manifest_path.stat().st_size,
                "sha256": sha256_file(dual_manifest_path),
            },
        ]
    ).to_csv(manifest_path, index=False)

    monkeypatch.setattr(module, "EXPECTED_B2_FINAL_GATE_SHA256", sha256_file(gate_path))
    monkeypatch.setattr(module, "EXPECTED_B2_AUDIT_SHA256", sha256_file(audit_path))
    monkeypatch.setattr(module, "EXPECTED_B2_MANIFEST_SHA256", sha256_file(manifest_path))
    monkeypatch.setattr(module, "EXPECTED_B2_MODEL_SHA256", gate["model_sha256"])
    monkeypatch.setattr(module, "EXPECTED_FINAL_POLICY_SHA256", sha256_file(policy_path))
    monkeypatch.setattr(
        module,
        "EXPECTED_SOURCE_CHECKPOINT_SHA256",
        {source: sha256_file(path) for source, path in checkpoint_paths.items()},
    )
    args = argparse.Namespace(
        b2_dir=b2_dir,
        dual_protocol_dir=dual_dir,
        final_model_dir=final_dir,
        output_dir=tmp_path / "output",
    )
    return module, args, {
        "gate": gate_path,
        "audit": audit_path,
        "manifest": manifest_path,
        "policy": policy_path,
        "checkpoints": checkpoint_paths,
        "dual": dual_path,
    }


def test_contract_builders_freeze_target_weather_schedule_and_gate() -> None:
    module = load_module()
    protocol = module.build_protocol_contract()

    assert protocol["target"] == {
        "site_id": "P3",
        "site_code": "N3",
        "target_year": 2022,
        "latitude": 46.321,
        "longitude": -96.877,
        "timezone": "America/Chicago",
    }
    assert protocol["sealed_reserves"] == {
        "P3_2023_replication": True,
        "P3_2024_TTA_workflow": True,
    }
    acquisition = protocol["operational_weather"]
    assert acquisition["members"] == ["gec00", "gep01", "gep02", "gep03", "gep04"]
    assert acquisition["member_count"] == 5
    assert acquisition["cycle_hour_utc"] == 0
    assert acquisition["forecast_horizon"] == "local_decision_date_D_through_D_plus_6"
    assert protocol["evaluation"]["irrigation_candidates_mm"] == [
        0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0
    ]

    schedule = protocol["schedule_rule"]
    assert schedule["trunk_weather_source"] == "ERA5_2022_observed_daily_weather"
    assert schedule["sowing_month_day"] == "04-26"
    assert schedule["first_decision_rule"] == "first_day_after_checkpoint_DVS_reaches_0.1"
    assert schedule["interval_days"] == 7
    assert schedule["horizon_days"] == 7
    assert schedule["minimum_decision_cycles"] == 8
    assert schedule["manual_date_selection_allowed"] is False
    assert schedule["GEFS_values_used_to_select_dates"] is False
    assert schedule["SWAP_candidate_labels_used_to_select_dates"] is False

    gate = protocol["routing_contract"]
    assert gate["feature_names"] == list(module.GATE_FEATURE_NAMES)
    assert gate["threshold"] == 0.5
    assert gate["tie_route"] == "P15"
    assert gate["audited_failure_fallback"] == "always_P15"


def test_promotion_gate_has_exact_ten_frozen_conditions() -> None:
    module = load_module()
    evaluation = module.evaluation_contract()
    conditions = evaluation["independent_promotion_conditions"]

    assert evaluation["fixed_baselines"] == ["always_P1", "always_P15"]
    assert len(conditions) == 10
    assert [item["condition_order"] for item in conditions] == list(range(1, 11))
    assert conditions[0]["condition_id"] == "mean_regret_better_than_always_P1"
    assert conditions[1]["condition_id"] == "mean_regret_better_than_always_P15"
    assert conditions[-1]["condition_id"] == "all_protocol_audits_pass"
    assert all(item["required"] for item in conditions)
    assert evaluation["failure_policy"] == "freeze_negative_without_criterion_changes"


def test_stage_registry_is_ordered_and_only_protocol_freeze_is_eligible() -> None:
    module = load_module()
    stages = module.build_stage_registry()

    assert stages["stage_order"].tolist() == list(range(1, 9))
    assert stages["stage_id"].tolist()[0] == "protocol_freeze"
    assert stages["stage_id"].tolist()[-1] == "single_independent_evaluation"
    assert stages["eligible_now"].tolist() == [True] + [False] * 7
    assert not stages["model_or_threshold_selection_allowed"].any()
    assert not stages["P3_2023_access_allowed"].any()
    assert not stages["P3_2024_access_allowed"].any()


def test_run_writes_exact_outputs_and_hash_manifest(
    tmp_path: Path, monkeypatch: _PatchCompat
) -> None:
    module, args, _ = create_inputs(tmp_path, monkeypatch)

    outputs = module.run(args)

    assert set(outputs) == {"protocol", "components", "stages", "audit", "manifest"}
    assert {path.name for path in outputs.values()} == set(module.OUTPUT_NAMES.values())
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert audit["status"] == module.SUCCESS_STATUS
    assert audit["mandatory_gate_passed"] is True
    for field in module.ZERO_ACCESS_FIELDS:
        assert audit[field] == 0
    manifest = pd.read_csv(outputs["manifest"])
    output_rows = manifest.loc[manifest["role"].str.startswith("output_")]
    assert len(output_rows) == 4
    for row in output_rows.itertuples(index=False):
        path = args.output_dir / row.path
        assert path.exists()
        assert path.stat().st_size == row.bytes
        assert sha256_file(path) == row.sha256
    components = pd.read_csv(outputs["components"])
    assert not components["canonical_path"].astype(str).str.match(r"^(?:[A-Za-z]:|/)").any()


def test_run_refuses_existing_output_directory(
    tmp_path: Path, monkeypatch: _PatchCompat
) -> None:
    module, args, _ = create_inputs(tmp_path, monkeypatch)
    args.output_dir.mkdir()

    with raises(FileExistsError, match="already exists"):
        module.run(args)


def test_validation_fails_closed_on_mutation(
    tmp_path: Path,
    monkeypatch: _PatchCompat,
    mutation: str,
    match: str,
) -> None:
    module, args, paths = create_inputs(tmp_path, monkeypatch)
    if mutation == "status":
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        audit["status"] = "changed"
        write_json(paths["audit"], audit)
        monkeypatch.setattr(module, "EXPECTED_B2_AUDIT_SHA256", sha256_file(paths["audit"]))
    elif mutation == "leakage":
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        audit["year_2021_rows_retained"] = 1
        write_json(paths["audit"], audit)
        monkeypatch.setattr(module, "EXPECTED_B2_AUDIT_SHA256", sha256_file(paths["audit"]))
    elif mutation == "fallback":
        gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
        gate["fallback"] = True
        gate["model_sha256"] = model_sha256(gate)
        write_json(paths["gate"], gate)
        monkeypatch.setattr(module, "EXPECTED_B2_FINAL_GATE_SHA256", sha256_file(paths["gate"]))
        monkeypatch.setattr(module, "EXPECTED_B2_MODEL_SHA256", gate["model_sha256"])
    elif mutation == "feature_order":
        gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
        gate["feature_names"] = list(reversed(gate["feature_names"]))
        gate["model_sha256"] = model_sha256(gate)
        write_json(paths["gate"], gate)
        monkeypatch.setattr(module, "EXPECTED_B2_FINAL_GATE_SHA256", sha256_file(paths["gate"]))
        monkeypatch.setattr(module, "EXPECTED_B2_MODEL_SHA256", gate["model_sha256"])
    elif mutation == "model_hash":
        gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
        gate["model_sha256"] = "0" * 64
        write_json(paths["gate"], gate)
        monkeypatch.setattr(module, "EXPECTED_B2_FINAL_GATE_SHA256", sha256_file(paths["gate"]))
        monkeypatch.setattr(module, "EXPECTED_B2_MODEL_SHA256", "0" * 64)
    elif mutation == "policy_hash":
        paths["policy"].write_text(paths["policy"].read_text() + " ", encoding="utf-8")
    elif mutation == "checkpoint_hash":
        paths["checkpoints"]["P1"].write_bytes(b"changed")

    with raises(ValueError, match=match):
        module.run(args)


def test_dual_feature_definition_must_match_b2_contract(
    tmp_path: Path, monkeypatch: _PatchCompat
) -> None:
    module, args, paths = create_inputs(tmp_path, monkeypatch)
    dual = json.loads(paths["dual"].read_text(encoding="utf-8"))
    dual["B_source_only_router"]["gate_feature_contract"]["features"][0][
        "aggregation"
    ] = "mean"
    write_json(paths["dual"], dual)

    with raises(ValueError, match="feature contract"):
        module.run(args)


class ProtocolFreezerTests(unittest.TestCase):
    def test_contract_builders(self) -> None:
        test_contract_builders_freeze_target_weather_schedule_and_gate()

    def test_promotion_conditions(self) -> None:
        test_promotion_gate_has_exact_ten_frozen_conditions()

    def test_stage_registry(self) -> None:
        test_stage_registry_is_ordered_and_only_protocol_freeze_is_eligible()

    def test_end_to_end_outputs(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            patches = _PatchCompat()
            try:
                test_run_writes_exact_outputs_and_hash_manifest(path, patches)
            finally:
                patches.undo()

    def test_existing_output_refused(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            patches = _PatchCompat()
            try:
                test_run_refuses_existing_output_directory(path, patches)
            finally:
                patches.undo()

    def test_mutations_fail_closed(self) -> None:
        import tempfile

        cases = [
            ("status", "status"),
            ("leakage", "year_2021_rows_retained"),
            ("fallback", "fallback"),
            ("feature_order", "feature_names"),
            ("model_hash", "model payload SHA256"),
            ("policy_hash", "final source policy SHA256"),
            ("checkpoint_hash", "P1 checkpoint SHA256"),
        ]
        for mutation, match in cases:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    patches = _PatchCompat()
                    try:
                        test_validation_fails_closed_on_mutation(
                            Path(directory), patches, mutation, match
                        )
                    finally:
                        patches.undo()

    def test_dual_feature_contract(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            patches = _PatchCompat()
            try:
                test_dual_feature_definition_must_match_b2_contract(
                    Path(directory), patches
                )
            finally:
                patches.undo()


if __name__ == "__main__":
    unittest.main()
