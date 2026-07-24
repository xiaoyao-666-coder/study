from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from s2s_rtist.weather import gefs_quantile_mapping as qm_module


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ConfirmationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_module = load_script(
            "scripts/data_preparation/extract_gefs_qm_2019_confirmation_v1.py",
            "extract_gefs_qm_2019_confirmation_v1_test",
        )

    def test_plan_only_writes_exact_disjoint_60_task_plan_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "confirmation"
            with patch.object(
                self.data_module,
                "download_reforecast_member_points",
                side_effect=AssertionError("plan-only must not download"),
            ):
                outputs = self.data_module.run_extraction(
                    output_dir=output_dir,
                    era5_root=Path(directory) / "missing-era5",
                    workers=3,
                    timeout=1,
                    retries=1,
                    plan_only=True,
                )

            dates = json.loads(outputs["dates"].read_text(encoding="utf-8"))
            plan = pd.read_csv(outputs["plan"])
            self.assertEqual(len(plan), 60)
            self.assertEqual(plan["cycle_date"].nunique(), 12)
            self.assertEqual(plan["member"].nunique(), 5)
            self.assertEqual(set(plan["maximum_end_hour"]), {168})
            self.assertEqual(set(plan["expected_selected_message_count"]), {56})
            self.assertFalse(bool(dates["network_download_started"]))
            self.assertFalse((output_dir / "cache").exists())

    def test_confirmation_dates_are_chronological_and_disjoint(self) -> None:
        contract = self.data_module.load_contract()
        dates = self.data_module.confirmation_dates(contract)
        selection = set(contract["strategy_selection_dates_2019"])

        self.assertEqual(len(dates), 12)
        self.assertEqual(tuple(sorted(dates)), dates)
        self.assertFalse(selection.intersection(dates))


class RangeProgressTests(unittest.TestCase):
    def test_progress_callback_counts_only_successful_ranges(self) -> None:
        progress: list[tuple[int, int]] = []
        with patch.object(
            qm_module,
            "_request",
            side_effect=[(b"one", {}), (b"two", {})],
        ):
            fetch = qm_module._range_fetcher(
                timeout=1,
                retries=1,
                total_ranges=2,
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                ),
            )
            self.assertEqual(fetch("https://example.test", 0, 2), b"one")
            self.assertEqual(fetch("https://example.test", 3, 5), b"two")

        self.assertEqual(progress, [(1, 2), (2, 2)])


class ConfirmationBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        diagnostics = ROOT / "scripts" / "diagnostics"
        sys.path.insert(0, str(diagnostics))
        cls.runner = load_script(
            "scripts/diagnostics/run_gefs_qm_2019_confirmation_v1.py",
            "run_gefs_qm_2019_confirmation_v1_test",
        )

    def test_bootstrap_is_12_cycle_fixed_seed_and_reproducible(self) -> None:
        base = np.linspace(-1.0, 1.0, 12)
        inputs = {
            name: pd.DataFrame({"difference_qm_minus_raw": base})
            for name in ("crps", "mean_brier", "seven_day_mae")
        }

        first = self.runner._bootstrap_summary(inputs)
        second = self.runner._bootstrap_summary(inputs)

        self.assertEqual(len(first), 3)
        self.assertEqual(set(first["cycle_count"]), {12})
        self.assertEqual(set(first["bootstrap_replicates"]), {10000})
        self.assertEqual(set(first["random_seed"]), {20260717})
        pd.testing.assert_frame_equal(first, second)

    def test_bootstrap_rejects_non_confirmation_cycle_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 12 confirmation cycles"):
            self.runner._bootstrap_summary(
                {"crps": pd.DataFrame({"difference_qm_minus_raw": [0.0]})}
            )


if __name__ == "__main__":
    unittest.main()
