# Story number review

Recomputed from `data/interim/requests_story.parquet` with Python and DuckDB. Compared with `data/interim/story_stats.json`. Every query succeeded. A row is a mismatch when the relative difference is greater than 1%. Shares below agree with the stored JSON at its printed precision (relative difference under 10^-16).

| metric | your value | json value | match or mismatch |
| --- | --- | --- | --- |
| row_count | 3963959 | 3963959 | match |
| distinct_unique_key | 3963959 | 3963959 | match |
| min_created | 2025-11-01 21:09:28 | 2025-11-01 21:09:28 | match |
| max_created | 2026-10-31 21:09:02 | 2026-10-31 21:09:02 | match |
| full_days_2025-11-02_to_2026-10-30 | 363 | 363 | match |
| mean_daily_count | 10890.440771349862 | 10890.440771349862 | match |
| busiest_day | 2026-02-24 | 2026-02-24 | match |
| busiest_count | 22805 | 22805 | match |
| quietest_day | 2025-12-25 | 2025-12-25 | match |
| quietest_count | 6216 | 6216 | match |
| illegal_parking_count | 618157 | 618157 | match |
| illegal_parking_share | 0.1559443475575807 | 0.1559443475575807 | match |
| nypd_count | 1721099 | 1721099 | match |
| nypd_share | 0.43418688235675496 | 0.43418688235675496 | match |
| fireworks_2026-07-04 | 2257 | 2257 | match |
| fireworks_mean_full_day | 29.983471074380165 | 29.983471074380166 | match |
| fireworks_july4_over_mean | 75.27480705622933 | 75.27480705622932 | match |
| snow_2026-02-23 | 6524 | 6524 | match |
| snow_2026-02-24 | 11370 | 11370 | match |
| snow_total | 63941 | 63941 | match |
| snow_two_days_over_all | 0.2798517383212649 | 0.2798517383212649 (from 6524+11370 over 63941) | match |
| top_address_borough | BRONX | BRONX | match |
| top_address_zip | 10466 | not stored | no json field |
| top_address_count | 47105 | 47105 | match |
| top_address_noise_residential | 47013 | 47013 | match |
| address_top_1pct_address_count | 5929 | 5929 | match |
| address_top_1pct_share | 0.30118000660638145 | 0.30118000660638145 | match |
| nypd_rows_with_resolution | 1720638 | 1720638 | match |
| nypd_issued_a_summons_count | 224738 | 224738 | match |
| nypd_issued_a_summons_share | 0.13061317952991855 | 0.13061317952991855 (224738/1720638) | match |
| nypd_arrest_action_phrases | 1169 | 423243 | MISMATCH (99.7% lower) |
| encampment_before_NYPD | 30974 | 30974 | match |
| encampment_before_DHS | 11248 | 11248 | match |
| encampment_after_DHS | 4988 | 4988 | match |
| encampment_after_NYPD | 159 | 159 | match |
| zips_11102_11103_11105_11106_rows | 80648 | 80648 | match |
| those_zips_heat_share | 0.055388850312469 | 0.055388850312469 | match |
| those_zips_parking_share | 0.24770608074595774 | 0.24770608074595774 | match |
| those_zips_noise_residential_share | 0.07607132228945541 | 0.07607132228945541 | match |

## Flag

`nypd_resolution.arrest_word` is 423243. The four action phrases total 1169, all of them `made an arrest` (1147 + 22). None matched `effected an arrest`, `placed under arrest`, or `an arrest was made`.

The stored 423243 equals `resolution ILIKE '%arrest%'`. Of those rows, 422074 say there was no criminal violation and the condition was corrected without the need to issue a summons or effect an arrest. That boilerplate contains `effect an arrest`, which is not one of the four action phrases.

## Checks that do not change the table

- No `created` or `unique_key` nulls. Distinct keys equal the row count.
- The 363-day mean zero-fills 2025-11-02 through 2026-10-30. None of those days is missing, so the fill does not change the mean. Excluded edge days: 2025-11-01 has 1734 rows; 2026-10-31 has 8995.
- Fireworks mean uses that same calendar, including 2026-07-04. The calendar sum is 10884. The mean last digit differs by one unit in the last place.
- Snow ratio and summons share are not stored as their own fields. Both are the ratios of stored counts that matched.
- Top address is one borough and one zip (10466), all 47105 rows. Street address omitted. Noise - Residential is 47013 of those rows.
- Top 1% share is 1149761 / 3817521 non-empty-address rows. Cutoff is `round(0.01 * 592940) = 5929`. Addresses on the cutoff boundary all have count 80, so the tie does not change the share. The same numerator over all rows is 0.290054, which is not the stored share.
- Summons uses `ILIKE '%issued a summons%'` over NYPD rows with a non-null resolution. 461 NYPD rows have a null resolution. No resolution is an empty string.
- Encampment before is `created < 2026-09-22`; after is `created >= 2026-09-22`. No row is exactly midnight. Day counts also match the JSON: before NYPD 313, before DHS 311, after DHS 40, after NYPD 1 (that one day is 2026-09-22).
- Zip shares are the four zips combined, compared with `zip_astoria`. Zip counts: 11102 = 16411, 11103 = 29339, 11105 = 16484, 11106 = 18414.
