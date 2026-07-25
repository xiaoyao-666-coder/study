from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "src" / "s2s_rtist" / "pipelines" / "restart_decision_dataset.py"
)


def load_pipeline_module():
    forecast = types.ModuleType("ForecastStep")
    real_ir = types.ModuleType("real_ir_update")
    raw_audit = types.ModuleType("restart_raw_audit_v1")
    raw_audit.preserve_candidate_raw_outputs = lambda **kwargs: None
    with patch.dict(
        sys.modules,
        {
            "ForecastStep": forecast,
            "real_ir_update": real_ir,
            "restart_raw_audit_v1": raw_audit,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "restart_decision_endpoint_fallback_test_module", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class RestartDecisionEndpointFallbackTests(unittest.TestCase):
    def test_failed_positive_candidate_retries_declared_simulated_amount(self) -> None:
        module = load_pipeline_module()
        current_ir = {"value": None}
        configured: list[float | None] = []

        def configure(_date: str, irrigation: float | None) -> None:
            current_ir["value"] = irrigation
            configured.append(irrigation)

        def run_swap(_log_name: str) -> None:
            if current_ir["value"] == 20.0:
                raise RuntimeError("requested 20 mm numerical failure")

        module.configure_irrigation = configure
        module.run_pre_state = lambda *_args, **_kwargs: None
        module.set_swp_for_restart = lambda *_args, **_kwargs: None
        module.run_swap = run_swap
        module.extract_candidate_labels = lambda **_kwargs: object()
        module.flatten_candidate_labels = lambda _labels: {}
        module.preserve_candidate_raw_outputs = lambda **_kwargs: None
        module.read_last = lambda _path: {
            "end_daynr": 163,
            "dvs": 1.0,
            "lai": 2.0,
            "rootd": 100.0,
            "cwdm_value": 1000.0 + float(current_ir["value"] or 0.0),
            "cwso_value": 0.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                Path("result_forec.end").write_text("checkpoint", encoding="utf-8")
                result = module.run_one_date(
                    "06-Jun-2015",
                    157,
                    irrigation_options_mm=[0.0, 20.0],
                    numerical_endpoint_fallbacks_mm={20.0: 19.9},
                )
            finally:
                os.chdir(previous)

        endpoint = result.loc[result["ir"].eq(20.0)].iloc[0]
        self.assertEqual(configured, [None, 0.0, 20.0, 19.9])
        self.assertEqual(float(endpoint["requested_ir_mm"]), 20.0)
        self.assertEqual(float(endpoint["simulated_ir_mm"]), 19.9)
        self.assertTrue(bool(endpoint["numerical_endpoint_fallback"]))
        self.assertTrue(bool(endpoint["numerical_irrigation_fallback"]))
        self.assertAlmostEqual(
            float(endpoint["numerical_endpoint_fallback_delta_mm"]), -0.1
        )
        self.assertEqual(
            endpoint["numerical_endpoint_fallback_trigger_error_type"],
            "RuntimeError",
        )
        self.assertEqual(
            endpoint["numerical_irrigation_fallback_policy"],
            "positive_candidate_sequential_retry_minus_0p1_minus_0p2mm_v2",
        )

    def test_second_fallback_is_used_only_after_first_fallback_fails(self) -> None:
        module = load_pipeline_module()
        current_ir = {"value": None}
        configured: list[float | None] = []

        def configure(_date: str, irrigation: float | None) -> None:
            current_ir["value"] = irrigation
            configured.append(irrigation)

        def run_swap(_log_name: str) -> None:
            if current_ir["value"] in {60.0, 59.9}:
                raise RuntimeError(f"numerical failure at {current_ir['value']} mm")

        module.configure_irrigation = configure
        module.run_pre_state = lambda *_args, **_kwargs: None
        module.set_swp_for_restart = lambda *_args, **_kwargs: None
        module.run_swap = run_swap
        module.extract_candidate_labels = lambda **_kwargs: object()
        module.flatten_candidate_labels = lambda _labels: {}
        module.preserve_candidate_raw_outputs = lambda **_kwargs: None
        module.read_last = lambda _path: {
            "end_daynr": 153,
            "dvs": 1.0,
            "lai": 2.0,
            "rootd": 100.0,
            "cwdm_value": 1000.0,
            "cwso_value": 0.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                Path("result_forec.end").write_text("checkpoint", encoding="utf-8")
                result = module.run_one_date(
                    "27-May-2019",
                    147,
                    irrigation_options_mm=[0.0, 60.0],
                    numerical_endpoint_fallbacks_mm={60.0: (59.9, 59.8)},
                )
            finally:
                os.chdir(previous)

        endpoint = result.loc[result["ir"].eq(60.0)].iloc[0]
        self.assertEqual(configured, [None, 0.0, 60.0, 59.9, 59.8])
        self.assertEqual(float(endpoint["simulated_ir_mm"]), 59.8)
        self.assertEqual(
            int(endpoint["numerical_irrigation_fallback_attempt_count"]), 2
        )
        self.assertEqual(
            endpoint["numerical_irrigation_fallback_attempted_values_mm"],
            "[59.9, 59.8]",
        )
        self.assertEqual(
            endpoint["numerical_irrigation_fallback_failed_values_mm"], "[59.9]"
        )
        self.assertEqual(
            endpoint["numerical_irrigation_fallback_policy"],
            "positive_candidate_sequential_retry_minus_0p1_minus_0p2mm_v2",
        )


if __name__ == "__main__":
    unittest.main()
