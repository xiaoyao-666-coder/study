from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.simulation.run_swap_season_trunk_smoke_v1 import (
    patch_trunk_swp_text,
    weather_record_years,
)


SWP = """  TSTART  = 26-apr-2019 ! Start date of simulation run
  TEND    = 10-oct-2019 ! End date of simulation run
  OUTFIL   = 'result_forec' ! Generic output
 SWINCO = 1 ! initial soil moisture
  METFIL = 'oldweather' ! weather
  INITCRP    CROPSTART      CROPEND       CROPNAME   CROPFIL     CROPTYPE
     2       26-apr-2019    10-oct-2019   'mais'    'GMaizeD'      2
"""


class SeasonTrunkSmokeTests(unittest.TestCase):
    def test_patches_full_season_dates_weather_and_output(self) -> None:
        result = patch_trunk_swp_text(
            SWP,
            year=2015,
            sowing_month_day="04-26",
            harvest_month_day="10-10",
            output_prefix="trunk2015",
        )
        self.assertIn("TSTART  = 26-apr-2015", result)
        self.assertIn("TEND    = 10-oct-2015", result)
        self.assertIn("OUTFIL   = 'trunk2015'", result)
        self.assertIn("METFIL = 'weather'", result)
        self.assertIn("2       26-apr-2015    10-oct-2015", result)

    def test_rejects_restart_initial_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot start from a restart"):
            patch_trunk_swp_text(
                SWP.replace("SWINCO = 1", "SWINCO = 3"),
                year=2015,
                sowing_month_day="04-26",
                harvest_month_day="10-10",
                output_prefix="trunk2015",
            )

    def test_reads_weather_record_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.015"
            path.write_text(
                "header\n"
                " 'Weather' 1 1 2015 1000 1 2 0.5 3 0 1\n"
                " 'Weather' 2 1 2015 1000 1 2 0.5 3 0 1\n",
                encoding="utf-8",
            )
            years, rows = weather_record_years(path)
        self.assertEqual(years, {2015})
        self.assertEqual(rows, 2)


if __name__ == "__main__":
    unittest.main()
