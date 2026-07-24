from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_preparation.build_swap_season_decision_schedule_v1 import (
    build_from_manifest,
)
from s2s_rtist.labels.swap_three_output_labels import CRP_COLUMNS
from s2s_rtist.pipelines.season_decision_schedule import (
    build_decision_schedule,
    read_crop_trajectory,
)


def crop_frame(start: str, periods: int, threshold_index: int = 2) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    if threshold_index <= 0:
        raise ValueError("threshold_index must be positive")
    dvs = [0.1 * index / threshold_index for index in range(periods)]
    dvs[threshold_index] = 0.1
    for index in range(threshold_index + 1, periods):
        dvs[index] = 0.1 + 0.04 * (index - threshold_index)
    return pd.DataFrame({"Date": dates, "DVS": dvs})


def write_crop(path: Path, frame: pd.DataFrame) -> None:
    lines = ["SWAP crop output"]
    for row in frame.itertuples(index=False):
        values = ["0"] * len(CRP_COLUMNS)
        values[CRP_COLUMNS.index("Date")] = pd.Timestamp(row.Date).strftime("%Y-%m-%d")
        values[CRP_COLUMNS.index("DVS")] = str(float(row.DVS))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SeasonDecisionScheduleTests(unittest.TestCase):
    def test_starts_after_first_checkpoint_with_dvs_point_one(self) -> None:
        result = build_decision_schedule(
            crop_frame("2015-04-01", 30),
            site_id="P1",
            target_year=2015,
            split="training",
        )
        self.assertEqual(result["decision_date"].tolist(), [
            "2015-04-04",
            "2015-04-11",
            "2015-04-18",
        ])
        self.assertEqual(result.iloc[0]["state_checkpoint_date"], "2015-04-03")
        self.assertAlmostEqual(float(result.iloc[0]["state_dvs"]), 0.1)
        self.assertEqual(result.iloc[-1]["horizon_end_date"], "2015-04-24")
        self.assertTrue(
            (
                pd.to_datetime(result["horizon_end_date"])
                <= pd.to_datetime(result["harvest_date"])
            ).all()
        )

    def test_never_reaching_dvs_threshold_is_rejected(self) -> None:
        crop = pd.DataFrame(
            {
                "Date": pd.date_range("2015-04-01", periods=20, freq="D"),
                "DVS": [0.05] * 20,
            }
        )
        with self.assertRaisesRegex(ValueError, "never reaches DVS"):
            build_decision_schedule(
                crop,
                site_id="P1",
                target_year=2015,
                split="training",
            )

    def test_crop_reader_rejects_missing_daily_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.crp"
            frame = crop_frame("2015-04-01", 10).drop(index=4)
            write_crop(path, frame)
            with self.assertRaisesRegex(ValueError, "not daily-contiguous"):
                read_crop_trajectory(path)

    def test_manifest_builds_independent_site_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p1 = root / "p1.crp"
            p2 = root / "p2.crp"
            write_crop(p1, crop_frame("2015-04-01", 30, threshold_index=2))
            write_crop(p2, crop_frame("2015-04-01", 30, threshold_index=5))
            manifest = pd.DataFrame(
                [
                    {"site_id": "P1", "target_year": 2015, "crop_output_path": p1},
                    {"site_id": "P2", "target_year": 2015, "crop_output_path": p2},
                ]
            )
            schedule, sources = build_from_manifest(
                manifest,
                dvs_threshold=0.1,
                interval_days=7,
                horizon_days=7,
            )
        first = schedule.groupby("site_id")["decision_date"].first().to_dict()
        self.assertEqual(first, {"P1": "2015-04-04", "P2": "2015-04-07"})
        self.assertEqual(len(sources), 2)
        self.assertEqual(set(schedule["split"]), {"training"})


if __name__ == "__main__":
    unittest.main()
