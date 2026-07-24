# GEFS Exact-Schedule Freeze And Package Status (2026-07-24)

## Completed gates

- All 61 formal exact-schedule GEFS batches passed their structural audits.
- Annual raw-weather assemblies passed for 2015-2019.
- Frozen causal six-variable weather passed for every target year from 2015 through 2019.
- The unified pre-packaging review passed with status
  `exact_schedule_2015_2019_corrected_weather_packaging_review_passed`.

| Target year | Batches | Cycles | Member rows | Site-cycles | Minimum fit samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015 | 12 | 46 | 2,555 | 73 | 8 |
| 2016 | 13 | 52 | 2,240 | 64 | 14 |
| 2017 | 11 | 43 | 2,345 | 67 | 26 |
| 2018 | 14 | 54 | 2,275 | 65 | 39 |
| 2019 | 11 | 44 | 2,415 | 69 | 52 |
| **Total** | **61** | **239** | **11,830** | **338** | - |

All final years retain the six SWAP weather variables, five GEFS members
(`c00,p01,p02,p03,p04`), and complete seven-day horizons. Missing values,
nonfinite values, duplicate sample keys, target/future ERA5 fit leakage,
temperature-order errors, and member-structure inversions are zero.

## Review and server archive

The unified review source manifest covers 75 frozen-weather and audit artifacts.
An independent rehash found zero missing files, size mismatches, or SHA256
mismatches.

The locally generated server archive is intentionally excluded from Git:

- Filename: `gefs_exact_schedule_2015_2019_frozen_weather_server_v1.tar.gz`
- Size: `3,212,600` bytes
- SHA256: `c1d3a1cc5e5c4ce46c653048d67f92b30c5c5b42f972a27e4c71a4d157c62cdf`
- Archive members: 79 files
- Internal SHA256 entries: 78
- GRIB files and raw-download caches: 0

The archive is locally ready but has not yet been recorded as uploaded,
extracted, or verified on the server.

## Next gate

After user-managed upload and server-side SHA256 verification, run the bounded
one-date, eight-irrigation checkpoint branch smoke. This stage did not run new
SWAP label generation, surrogate training, or TTA.
