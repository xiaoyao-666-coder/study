from __future__ import annotations

import unittest

import pandas as pd

from scripts.simulation.audit_swap_season_checkpoint_equivalence_v1 import (
    checkpoint_rows,
    compare_crop_state,
    compare_profile_state,
    select_checkpoint_rows,
)


class CheckpointEquivalenceTests(unittest.TestCase):
    def test_selects_first_middle_and_last_schedule_rows(self) -> None:
        schedule = pd.DataFrame(
            {"schedule_index": range(14), "decision_date": pd.date_range("2015-05-18", periods=14, freq="7D")}
        )
        selected = select_checkpoint_rows(schedule)
        self.assertEqual(selected["schedule_index"].tolist(), [0, 7, 13])

    def test_selects_all_schedule_rows_for_generation(self) -> None:
        schedule = pd.DataFrame(
            {"schedule_index": [2, 0, 1], "decision_date": ["c", "a", "b"]}
        )
        selected = checkpoint_rows(schedule, all_checkpoints=True)
        self.assertEqual(selected["schedule_index"].tolist(), [0, 1, 2])

    def test_compares_crop_state(self) -> None:
        full = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2015-05-17")],
                "DVS": [0.1],
                "LAI": [0.4],
                "Rootd": [20.0],
                "CWDM": [50.0],
                "CWSO": [0.0],
            }
        )
        prefix = full.copy()
        prefix.loc[0, "CWDM"] = 50.000001
        result = compare_crop_state(full, prefix, "2015-05-17")
        self.assertAlmostEqual(result["maximum_absolute_crop_state_error"], 1e-6)

    def test_compares_profile_state_by_layer(self) -> None:
        full = pd.DataFrame(
            {
                "date": [pd.Timestamp("2015-05-17")] * 2,
                "depth": [-5.0, -15.0],
                "top": [0.0, -10.0],
                "bottom": [-10.0, -20.0],
                "wcontent": [0.2, 0.3],
                "phead": [-100.0, -120.0],
                "rootext": [0.01, 0.02],
                "waterflux": [0.0, -0.1],
            }
        )
        prefix = full.copy()
        prefix.loc[1, "wcontent"] += 1e-7
        result = compare_profile_state(full, prefix, "2015-05-17")
        self.assertEqual(result["profile_layer_rows"], 2)
        self.assertAlmostEqual(result["maximum_absolute_profile_state_error"], 1e-7)

    def test_accepts_matching_missing_profile_values(self) -> None:
        full = pd.DataFrame(
            {
                "date": [pd.Timestamp("2015-05-17")] * 2,
                "depth": [-5.0, -15.0],
                "top": [0.0, -10.0],
                "bottom": [-10.0, -20.0],
                "wcontent": [0.2, float("nan")],
                "phead": [-100.0, -120.0],
                "rootext": [0.01, 0.02],
                "waterflux": [0.0, -0.1],
            }
        )
        result = compare_profile_state(full, full.copy(), "2015-05-17")
        self.assertEqual(result["profile_layer_rows"], 2)
        self.assertEqual(result["maximum_absolute_profile_state_error"], 0.0)

    def test_rejects_different_missing_profile_values(self) -> None:
        full = pd.DataFrame(
            {
                "date": [pd.Timestamp("2015-05-17")],
                "depth": [-5.0],
                "top": [0.0],
                "bottom": [-10.0],
                "wcontent": [float("nan")],
                "phead": [-100.0],
                "rootext": [0.01],
                "waterflux": [0.0],
            }
        )
        prefix = full.copy()
        prefix.loc[0, "wcontent"] = 0.2
        with self.assertRaisesRegex(ValueError, "missing-value pattern differs"):
            compare_profile_state(full, prefix, "2015-05-17")


if __name__ == "__main__":
    unittest.main()
