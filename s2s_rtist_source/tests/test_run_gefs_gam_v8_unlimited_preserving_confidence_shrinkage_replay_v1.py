import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.training.run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1 import (
    CONFIDENCE_REFERENCE_LCB_MM,
    apply_unlimited_preserving_confidence_shrinkage,
    run,
    sha256_file,
    validate_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "docs/superpowers/specs/"
    "2026-08-08-teacher-guided-gam-v8-unlimited-preserving-"
    "confidence-shrinkage-replay-v1.json"
)


def equation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["P1", "P2", "P3", "P4", "P15"],
            "target_year": [2016, 2017, 2018, 2018, 2016],
            "anchor_recommendation_mm": [10.0] * 5,
            "recommendation_mm": [12.5, 12.5, 7.5, 12.5, 10.0],
            "v5_movement_from_anchor_mm": [2.5, 2.5, -2.5, 2.5, 0.0],
            "selected_lcb_mm": [-1.0, 2.5, 5.0, 10.0, 4.0],
            "trust_region_limited": [False, True, True, True, False],
            "true_peak_mm": [13.0, 11.0, 8.0, 13.0, 10.0],
            "true_oracle_is_positive": [True, True, True, True, False],
        }
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sample_v5_inventory() -> pd.DataFrame:
    counts = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
    rows = []
    for site_id, count in counts.items():
        for index in range(count):
            year = 2016 + index % 3
            date = pd.Timestamp(year, 5, 1) + pd.Timedelta(
                days=7 * (index // 3)
            )
            changed = index < 3
            if index == 0:
                anchor, v5, truth, lcb, limited = 10.0, 12.5, 12.5, 2.5, True
            elif index == 1:
                anchor, v5, truth, lcb, limited = 2.5, 0.0, 0.0, 5.0, True
            elif index == 2:
                anchor, v5, truth, lcb, limited = 10.0, 12.5, 12.5, 0.0, False
            elif index % 2:
                anchor = v5 = truth = 10.0
                lcb, limited = 0.0, False
            else:
                anchor = v5 = truth = 0.0
                lcb, limited = 0.0, False
            rows.append(
                {
                    "fold_id": f"holdout_{site_id}_rolling_to_{year}",
                    "target_site": site_id,
                    "validation_year": year,
                    "target_year": year,
                    "site_id": site_id,
                    "decision_date": date.strftime("%Y-%m-%d"),
                    "anchor_recommendation_mm": anchor,
                    "anchor_peak_distance_mm": abs(anchor - truth),
                    "recommendation_mm": v5,
                    "selected_peak_distance_mm": abs(v5 - truth),
                    "selected_lcb_mm": lcb,
                    "v5_movement_from_anchor_mm": v5 - anchor,
                    "trust_region_limited": limited,
                    "irrigation_changed": changed,
                    "true_peak_mm": truth,
                    "true_oracle_is_positive": truth > 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_upstream_fixture(root: Path) -> tuple[Path, Path, Path]:
    v5_dir, v6_dir, v7_dir = root / "v5", root / "v6", root / "v7"
    for directory in (v5_dir, v6_dir, v7_dir):
        directory.mkdir()

    v5_paths = {
        "validation": v5_dir
        / "gefs_gam_anchor_trust_region_mixture_validation_decisions_v5.csv",
        "gate": v5_dir
        / "gefs_gam_anchor_trust_region_mixture_replay_gate_v5.json",
        "audit": v5_dir
        / "gefs_gam_anchor_trust_region_mixture_replay_audit_v5.json",
    }
    sample_v5_inventory().to_csv(v5_paths["validation"], index=False)
    write_json(
        v5_paths["gate"],
        {"passed": True, "conditions": {"all_inner_source_site_holds_complete": True}},
    )
    write_json(
        v5_paths["audit"],
        {
            "status": "gam_anchor_trust_region_mixture_replay_v5_passed_pending_changed_cycle_swap_design",
            "mandatory_execution_gate_passed": True,
            "predeclared_performance_gate_passed": True,
            "2019_rows_read": 0,
            "2024_rows_read": 0,
        },
    )
    write_json(
        v5_dir / "gefs_gam_anchor_trust_region_mixture_replay_manifest_v5.json",
        {
            "outputs": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in v5_paths.items()
            }
        },
    )

    v6_paths = {
        "gate": v6_dir
        / "gefs_gam_v6_confidence_protected_anchor_replay_gate_v1.json",
        "audit": v6_dir
        / "gefs_gam_v6_confidence_protected_anchor_replay_audit_v1.json",
    }
    write_json(
        v6_paths["gate"],
        {"passed": True, "limited_candidate_required_lcb_mm": 5.0},
    )
    write_json(
        v6_paths["audit"],
        {
            "status": "gam_v6_confidence_protected_anchor_replay_passed_pending_2019_independent_confirmation_design",
            "mandatory_execution_gate_passed": True,
            "predeclared_performance_gate_passed": True,
            "2019_rows_read": 0,
            "2024_rows_read": 0,
        },
    )
    write_json(
        v6_dir
        / "gefs_gam_v6_confidence_protected_anchor_replay_manifest_v1.json",
        {
            "outputs": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in v6_paths.items()
            }
        },
    )

    v7_paths = {
        "gate": v7_dir
        / "gefs_gam_v7_continuous_confidence_shrinkage_replay_gate_v1.json",
        "audit": v7_dir
        / "gefs_gam_v7_continuous_confidence_shrinkage_replay_audit_v1.json",
    }
    write_json(v7_paths["gate"], {"passed": False})
    write_json(
        v7_paths["audit"],
        {
            "status": "gam_v7_continuous_confidence_shrinkage_replay_failed_stop_before_swap_or_2019",
            "mandatory_execution_gate_passed": True,
            "predeclared_performance_gate_passed": False,
            "2019_rows_read_by_replay": 0,
            "2019_target_rows_read_by_replay": 0,
            "2019_swap_results_read_by_replay": 0,
            "2024_rows_read": 0,
        },
    )
    write_json(
        v7_dir
        / "gefs_gam_v7_continuous_confidence_shrinkage_replay_manifest_v1.json",
        {
            "outputs": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in v7_paths.items()
            }
        },
    )
    return v5_dir, v6_dir, v7_dir


class GamV8UnlimitedPreservingConfidenceShrinkageReplayTests(unittest.TestCase):
    def test_protocol_freezes_hybrid_rule_and_boundaries(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)

        self.assertEqual(CONFIDENCE_REFERENCE_LCB_MM, 5.0)
        self.assertEqual(
            protocol["selection_rule"]["confidence_reference_lcb_mm"], 5.0
        )
        self.assertEqual(protocol["selection_rule"]["limited_scale_clip"], [0.0, 1.0])
        self.assertEqual(protocol["evaluation_boundary"]["cycle_count"], 196)
        self.assertEqual(protocol["evaluation_boundary"]["outer_fold_count"], 15)
        self.assertEqual(protocol["evaluation_boundary"]["target_site_count"], 5)
        self.assertFalse(protocol["evaluation_boundary"]["2019_access"])
        self.assertFalse(protocol["evaluation_boundary"]["2024_access"])

        protocol["selection_rule"]["confidence_reference_lcb_mm"] = 4.0
        with self.assertRaisesRegex(ValueError, "confidence reference"):
            validate_protocol(protocol)

        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["evaluation_boundary"]["2019_access"] = True
        with self.assertRaisesRegex(ValueError, "isolation"):
            validate_protocol(protocol)

    def test_unlimited_is_preserved_and_only_limited_actions_shrink(self) -> None:
        replay = apply_unlimited_preserving_confidence_shrinkage(equation_frame())

        np.testing.assert_allclose(
            replay["v8_confidence_scale"], [1.0, 0.5, 1.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            replay["v8_recommendation_mm"], [12.5, 11.25, 7.5, 12.5, 10.0]
        )
        np.testing.assert_allclose(
            replay["recommendation_mm"], replay["v8_recommendation_mm"]
        )
        self.assertEqual(
            replay["irrigation_changed"].tolist(),
            [True, True, True, True, False],
        )

    def test_identity_and_truth_fields_cannot_change_recommendation(self) -> None:
        base = equation_frame()
        changed = base.copy()
        changed["site_id"] = ["X1", "X2", "X3", "X4", "X5"]
        changed["target_year"] = [2091, 2092, 2093, 2094, 2095]
        changed["true_peak_mm"] = [60.0, 0.0, 60.0, 0.0, 60.0]
        changed["true_oracle_is_positive"] = [False, False, False, False, True]

        np.testing.assert_allclose(
            apply_unlimited_preserving_confidence_shrinkage(base)[
                "v8_recommendation_mm"
            ],
            apply_unlimited_preserving_confidence_shrinkage(changed)[
                "v8_recommendation_mm"
            ],
        )

    def test_inconsistent_frozen_v5_movement_is_rejected(self) -> None:
        frame = equation_frame().iloc[[0]].copy()
        frame["v5_movement_from_anchor_mm"] = 99.0

        with self.assertRaisesRegex(ValueError, "v5 movement"):
            apply_unlimited_preserving_confidence_shrinkage(frame)

    def test_end_to_end_writes_audited_peak_replay_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            v5_dir, v6_dir, v7_dir = write_upstream_fixture(root)
            outputs = run(
                Namespace(
                    protocol=PROTOCOL_PATH,
                    v5_dir=v5_dir,
                    v6_dir=v6_dir,
                    v7_dir=v7_dir,
                    output_dir=root / "output",
                )
            )

            decisions = pd.read_csv(outputs["decisions"])
            gate = json.loads(outputs["gate"].read_text(encoding="utf-8"))
            audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
            self.assertEqual(len(decisions), 196)
            self.assertTrue(gate["passed"])
            self.assertTrue(audit["mandatory_execution_gate_passed"])
            self.assertTrue(audit["predeclared_performance_gate_passed"])
            self.assertTrue(audit["post_v7_development_iteration"])
            self.assertEqual(audit["2019_target_rows_read_by_replay"], 0)
            self.assertEqual(audit["2019_swap_results_read_by_replay"], 0)
            self.assertEqual(audit["2024_rows_read"], 0)
            self.assertFalse(audit["swap_rerun_performed"])
            self.assertFalse(audit["automatic_model_promotion"])

    def test_changed_manifest_and_nonfailed_v7_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            v5_dir, v6_dir, v7_dir = write_upstream_fixture(root)
            validation = v5_dir / (
                "gefs_gam_anchor_trust_region_mixture_validation_decisions_v5.csv"
            )
            validation.write_text(
                validation.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                run(
                    Namespace(
                        protocol=PROTOCOL_PATH,
                        v5_dir=v5_dir,
                        v6_dir=v6_dir,
                        v7_dir=v7_dir,
                        output_dir=root / "changed-output",
                    )
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            v5_dir, v6_dir, v7_dir = write_upstream_fixture(root)
            audit_path = v7_dir / (
                "gefs_gam_v7_continuous_confidence_shrinkage_replay_audit_v1.json"
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "unexpected_v7_passed_status"
            write_json(audit_path, audit)
            manifest_path = v7_dir / (
                "gefs_gam_v7_continuous_confidence_shrinkage_replay_manifest_v1.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["audit"]["sha256"] = sha256_file(audit_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "v7 development status"):
                run(
                    Namespace(
                        protocol=PROTOCOL_PATH,
                        v5_dir=v5_dir,
                        v6_dir=v6_dir,
                        v7_dir=v7_dir,
                        output_dir=root / "status-output",
                    )
                )


if __name__ == "__main__":
    unittest.main()
