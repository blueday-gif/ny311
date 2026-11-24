"""Recompute the six new questions. Numbers only, no prose."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

ROOT = Path(r"D:\nyc311")
PQ = STORY_PQ_SQL
WEATHER = WEATHER
OUT = INTERIM / "book_new_stats.json"

con = duckdb.connect()
con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{PQ}')")
print(con.execute("DESCRIBE t").fetchdf().to_string(index=False))

weather = json.loads(WEATHER.read_text(encoding="utf-8"))["daily"]
wdf = pd.DataFrame(
    {
        "day": pd.to_datetime(weather["time"]),
        "temp": weather["temperature_2m_mean"],
        "snow": weather["snowfall_sum"],
    }
)
con.register("weather", wdf)

# Q20 building vs apartment
print("\n== heat detail")
print(
    con.execute(
        """
        SELECT coalesce(detail,'(none)') d, count(*) n
        FROM t WHERE problem = 'HEAT/HOT WATER'
        GROUP BY 1 ORDER BY n DESC LIMIT 12
        """
    ).fetchdf().to_string(index=False)
)

# 1 July 4 without fireworks vs other Saturdays
print("\n== july4")
print(
    con.execute(
        """
        WITH days AS (
          SELECT CAST(created AS DATE) d,
                 dayofweek(created) dow,
                 count(*) n,
                 count(*) FILTER (WHERE problem = 'Illegal Fireworks') fw,
                 count(*) FILTER (WHERE problem = 'Illegal Parking') park,
                 count(*) FILTER (WHERE problem = 'Noise - Street/Sidewalk') street,
                 count(*) FILTER (WHERE problem = 'Noise - Residential') res
          FROM t
          GROUP BY 1, 2
        )
        SELECT
          (SELECT n FROM days WHERE d = DATE '2026-07-04') july4,
          (SELECT fw FROM days WHERE d = DATE '2026-07-04') fw,
          (SELECT n - fw FROM days WHERE d = DATE '2026-07-04') rest,
          (SELECT park FROM days WHERE d = DATE '2026-07-04') park,
          (SELECT street FROM days WHERE d = DATE '2026-07-04') street,
          (SELECT res FROM days WHERE d = DATE '2026-07-04') res,
          (SELECT avg(n) FROM days WHERE dow = 6 AND d NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-07-04')) sat_mean,
          (SELECT count(*) FROM days WHERE dow = 6 AND d NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-07-04')) sat_n,
          (SELECT avg(park) FROM days WHERE dow = 6 AND d NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-07-04')) sat_park,
          (SELECT avg(street) FROM days WHERE dow = 6 AND d NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-07-04')) sat_street,
          (SELECT avg(res) FROM days WHERE dow = 6 AND d NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-07-04')) sat_res
        """
    ).fetchdf().to_string(index=False)
)

# 2 fireworks hours July 4 18:00 through July 5 08:00
print("\n== fw hours")
print(
    con.execute(
        """
        SELECT CAST(created AS DATE) d, hour(created) hr, count(*) n
        FROM t
        WHERE problem = 'Illegal Fireworks'
          AND created >= TIMESTAMP '2026-07-04 18:00:00'
          AND created < TIMESTAMP '2026-07-05 09:00:00'
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf().to_string(index=False)
)

# 3 thanksgiving
print("\n== thanksgiving")
print(
    con.execute(
        """
        WITH topaddr AS (
          SELECT address FROM t
          WHERE problem = 'Noise - Residential' AND address IS NOT NULL AND address <> ''
          GROUP BY 1 ORDER BY count(*) DESC LIMIT 1
        ),
        thurs AS (
          SELECT CAST(created AS DATE) d,
                 count(*) FILTER (WHERE problem = 'Noise - Residential' AND address NOT IN (SELECT address FROM topaddr)) noise,
                 count(*) FILTER (WHERE problem = 'Illegal Parking') park
          FROM t
          WHERE dayofweek(created) = 4
            AND created >= TIMESTAMP '2025-11-06'
            AND created < TIMESTAMP '2026-01-30'
            AND CAST(created AS DATE) NOT IN (DATE '2025-11-27', DATE '2025-12-25', DATE '2026-01-01')
          GROUP BY 1
        )
        SELECT
          (SELECT count(*) FROM t WHERE problem = 'Noise - Residential' AND CAST(created AS DATE) = DATE '2025-11-27') noise_all,
          (SELECT count(*) FROM t WHERE problem = 'Noise - Residential' AND CAST(created AS DATE) = DATE '2025-11-27'
             AND address IN (SELECT address FROM topaddr)) noise_top,
          (SELECT avg(noise) FROM thurs) thu_noise,
          (SELECT count(*) FROM thurs) thu_n,
          (SELECT count(*) FROM t WHERE problem = 'Illegal Parking' AND CAST(created AS DATE) = DATE '2025-11-27') park,
          (SELECT avg(park) FROM thurs) thu_park
        """
    ).fetchdf().to_string(index=False)
)

# 4 coldest day heat
print("\n== coldest")
print(
    con.execute(
        """
        WITH heat AS (
          SELECT CAST(created AS DATE) d, count(*) n
          FROM t WHERE problem = 'HEAT/HOT WATER' GROUP BY 1
        ),
        allc AS (
          SELECT CAST(created AS DATE) d, count(*) n FROM t GROUP BY 1
        )
        SELECT w.day, w.temp, coalesce(h.n, 0) heat, coalesce(a.n, 0) alln
        FROM weather w
        LEFT JOIN heat h ON h.d = CAST(w.day AS DATE)
        LEFT JOIN allc a ON a.d = CAST(w.day AS DATE)
        WHERE w.temp = (SELECT min(temp) FROM weather)
        """
    ).fetchdf().to_string(index=False)
)
print(
    con.execute(
        """
        WITH heat AS (
          SELECT CAST(created AS DATE) d, count(*) n
          FROM t WHERE problem = 'HEAT/HOT WATER' GROUP BY 1
        )
        SELECT
          avg(coalesce(h.n, 0)) FILTER (
            WHERE w.temp < 0
              AND CAST(w.day AS DATE) NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-02-08')
          ) cold_mean,
          count(*) FILTER (
            WHERE w.temp < 0
              AND CAST(w.day AS DATE) NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-02-08')
          ) cold_days,
          avg(coalesce(h.n, 0)) FILTER (WHERE w.temp >= 20) warm_mean,
          count(*) FILTER (WHERE w.temp >= 20) warm_days
        FROM weather w
        LEFT JOIN heat h ON h.d = CAST(w.day AS DATE)
        """
    ).fetchdf().to_string(index=False)
)

# 5 snow days parking
print("\n== snow parking")
print(
    con.execute(
        """
        WITH days AS (
          SELECT CAST(created AS DATE) d, dayofweek(created) dow,
                 count(*) FILTER (WHERE problem = 'Illegal Parking') park,
                 count(*) FILTER (WHERE problem = 'Snow or Ice') snow
          FROM t GROUP BY 1, 2
        )
        SELECT
          (SELECT park FROM days WHERE d = DATE '2026-02-23') p23,
          (SELECT snow FROM days WHERE d = DATE '2026-02-23') s23,
          (SELECT park FROM days WHERE d = DATE '2026-02-24') p24,
          (SELECT snow FROM days WHERE d = DATE '2026-02-24') s24,
          (SELECT avg(park) FROM days WHERE dow = 1 AND d NOT IN (DATE '2026-01-26', DATE '2026-02-23', DATE '2026-05-25', DATE '2026-09-07')) mon_park,
          (SELECT count(*) FROM days WHERE dow = 1 AND d NOT IN (DATE '2026-01-26', DATE '2026-02-23', DATE '2026-05-25', DATE '2026-09-07')) mon_n,
          (SELECT avg(park) FROM days WHERE dow = 2 AND d NOT IN (DATE '2026-01-27', DATE '2026-02-24')) tue_park,
          (SELECT count(*) FROM days WHERE dow = 2 AND d NOT IN (DATE '2026-01-27', DATE '2026-02-24')) tue_n
        """
    ).fetchdf().to_string(index=False)
)

# 6 helicopter hours
print("\n== heli")
print(
    con.execute(
        """
        SELECT
          count(*) n,
          count(*) FILTER (WHERE hour(created) BETWEEN 15 AND 19) dusk,
          count(*) FILTER (WHERE hour(created) BETWEEN 0 AND 5) night
        FROM t WHERE problem = 'Noise - Helicopter'
        """
    ).fetchdf().to_string(index=False)
)
print(
    con.execute(
        """
        SELECT hour(created) hr, count(*) n
        FROM t WHERE problem = 'Noise - Helicopter'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf().to_string(index=False)
)
print(
    "july5",
    con.execute(
        """
        SELECT count(*) FROM t
        WHERE problem = 'Illegal Fireworks' AND CAST(created AS DATE) = DATE '2026-07-05'
        """
    ).fetchone()[0],
)
