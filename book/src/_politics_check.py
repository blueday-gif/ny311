"""Verify the politics-chapter cuts. Print tables, write JSON."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

OUT = INTERIM / "book_politics_stats.json"
con = duckdb.connect()
con.execute(
    "CREATE VIEW t AS SELECT * FROM read_parquet('" + STORY_PQ_SQL + "')"
)

AG = ("NYPD", "HPD", "DSNY", "DOT", "DPR", "DEP")
ASTORIA = ("11102", "11103", "11105", "11106")
BAYSIDE = ("11360", "11361", "11364")
AOC = (
    "11102", "11103", "11105", "11106",
    "11369", "11370", "11372", "11368", "11354", "11356",
)
AOC_REST = tuple(z for z in AOC if z not in ASTORIA)


def q(sql: str):
    return con.execute(sql).fetchdf()


def shares(where: str) -> dict:
    row = q(
        f"""
        SELECT count(*) n,
               count(*) FILTER (WHERE agency = 'NYPD') nypd,
               count(*) FILTER (WHERE agency = 'HPD') hpd,
               count(*) FILTER (WHERE agency = 'DSNY') dsny,
               count(*) FILTER (WHERE agency = 'DOT') dot,
               count(*) FILTER (WHERE agency = 'DPR') dpr,
               count(*) FILTER (WHERE agency = 'DEP') dep,
               count(*) FILTER (WHERE problem = 'HEAT/HOT WATER') heat,
               count(*) FILTER (WHERE problem = 'Illegal Parking') parking,
               count(*) FILTER (WHERE problem = 'Blocked Driveway') driveway,
               count(*) FILTER (WHERE problem = 'Street Condition') street,
               count(*) FILTER (WHERE problem = 'Damaged Tree') tree,
               count(*) FILTER (WHERE problem = 'Abandoned Vehicle') car,
               count(*) FILTER (WHERE problem = 'Noise - Residential') noise
        FROM t WHERE {where}
        """
    ).iloc[0]
    n = int(row["n"])
    out = {"n": n}
    for k in row.index:
        if k == "n":
            continue
        out[k] = int(row[k])
        out[k + "_share"] = (int(row[k]) / n) if n else None
    return out


boro = q(
    """
    SELECT borough, count(*) n,
           count(*) FILTER (WHERE agency = 'NYPD') nypd,
           count(*) FILTER (WHERE agency = 'HPD') hpd,
           count(*) FILTER (WHERE agency = 'DSNY') dsny,
           count(*) FILTER (WHERE agency = 'DOT') dot,
           count(*) FILTER (WHERE agency = 'DPR') dpr,
           count(*) FILTER (WHERE agency = 'DEP') dep
    FROM t
    WHERE borough IN ('BRONX','MANHATTAN','BROOKLYN','QUEENS','STATEN ISLAND')
    GROUP BY 1 ORDER BY n DESC
    """
)
print("== borough agency")
print(boro.to_string(index=False))

zlist = ",".join(f"'{z}'" for z in AOC)
zips = q(
    f"""
    SELECT zip, count(*) n,
           count(*) FILTER (WHERE problem = 'HEAT/HOT WATER') heat,
           count(*) FILTER (WHERE problem = 'Illegal Parking') parking,
           count(*) FILTER (WHERE problem = 'Blocked Driveway') driveway,
           count(*) FILTER (WHERE agency = 'HPD') hpd
    FROM t WHERE zip IN ({zlist})
    GROUP BY 1 ORDER BY n DESC
    """
)
print("\n== aoc zips")
print(zips.to_string(index=False))

enc = q(
    """
    SELECT borough,
           count(*) FILTER (WHERE created < TIMESTAMP '2026-09-22' AND agency = 'NYPD') before_nypd,
           count(*) FILTER (WHERE created < TIMESTAMP '2026-09-22') before_n,
           count(*) FILTER (WHERE created >= TIMESTAMP '2026-09-22' AND agency = 'NYPD') after_nypd,
           count(*) FILTER (WHERE created >= TIMESTAMP '2026-09-22') after_n
    FROM t
    WHERE problem = 'Encampment'
      AND borough IN ('BRONX','MANHATTAN','BROOKLYN','QUEENS','STATEN ISLAND')
    GROUP BY 1 ORDER BY before_n DESC
    """
)
print("\n== encampment")
print(enc.to_string(index=False))

unk = q(
    """
    SELECT agency, count(*) n
    FROM t WHERE channel = 'UNKNOWN' OR channel IS NULL OR channel = ''
    GROUP BY 1 ORDER BY n DESC LIMIT 8
    """
)
print("\n== blank channel")
print(unk.to_string(index=False))
print("unknown total", q("SELECT count(*) n FROM t WHERE channel = 'UNKNOWN'").iloc[0]["n"])

phone = q(
    """
    SELECT borough,
           count(*) n,
           count(*) FILTER (WHERE channel = 'PHONE') phone,
           count(*) FILTER (WHERE channel IS NOT NULL AND channel <> '' AND channel <> 'UNKNOWN') known,
           count(*) FILTER (WHERE channel = 'PHONE' AND channel <> 'UNKNOWN') phone_known
    FROM t
    WHERE borough IN ('BRONX','MANHATTAN','BROOKLYN','QUEENS','STATEN ISLAND')
    GROUP BY 1
    """
)
print("\n== phone")
print(phone.to_string(index=False))

dur = q(
    """
    WITH c AS (
      SELECT agency, date_diff('second', created, closed) / 86400.0 AS days
      FROM t
      WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
        AND agency IN ('NYPD','DEP','DSNY','DOT','HPD','DPR')
    )
    SELECT agency, count(*) n, median(days) med, avg(days) mean
    FROM c GROUP BY 1 ORDER BY med
    """
)
print("\n== duration")
print(dur.to_string(index=False))

midnight = q(
    """
    WITH c AS (
      SELECT date_diff('second', created, closed) / 86400.0 AS days,
             (hour(closed) = 0 AND minute(closed) = 0 AND second(closed) = 0) AS mid
      FROM t
      WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
    )
    SELECT count(*) n,
           count(*) FILTER (WHERE mid) mid_n,
           median(days) med,
           avg(days) mean,
           median(days) FILTER (WHERE NOT mid) med_ex,
           avg(days) FILTER (WHERE NOT mid) mean_ex
    FROM c
    """
)
print("\n== midnight")
print(midnight.to_string(index=False))

season = q(
    """
    SELECT borough,
           count(*) FILTER (WHERE month(created) IN (12, 1, 2)) winter_n,
           count(*) FILTER (WHERE month(created) IN (12, 1, 2) AND problem = 'HEAT/HOT WATER') winter_heat,
           count(*) FILTER (WHERE month(created) IN (6, 7, 8)) summer_n,
           count(*) FILTER (WHERE month(created) IN (6, 7, 8) AND problem = 'Illegal Parking') summer_park
    FROM t
    WHERE borough IN ('BRONX','MANHATTAN','BROOKLYN','QUEENS','STATEN ISLAND')
    GROUP BY 1
    """
)
print("\n== season")
print(season.to_string(index=False))

problems = q(
    """
    SELECT problem, count(*) n FROM t
    WHERE problem ILIKE '%tree%' OR problem ILIKE '%street%' OR problem ILIKE '%abandon%'
    GROUP BY 1 ORDER BY n DESC LIMIT 20
    """
)
print("\n== street-like names")
print(problems.to_string(index=False))

payload = {
    "borough_agency": boro.to_dict(orient="records"),
    "aoc_zips": zips.to_dict(orient="records"),
    "astoria": shares("zip IN (" + ",".join(f"'{z}'" for z in ASTORIA) + ")"),
    "bayside": shares("zip IN (" + ",".join(f"'{z}'" for z in BAYSIDE) + ")"),
    "aoc_rest": shares("zip IN (" + ",".join(f"'{z}'" for z in AOC_REST) + ")"),
    "encampment": enc.to_dict(orient="records"),
    "unknown_channel_top": unk.to_dict(orient="records"),
    "phone": phone.to_dict(orient="records"),
    "duration": dur.to_dict(orient="records"),
    "midnight": midnight.to_dict(orient="records"),
    "season": season.to_dict(orient="records"),
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("wrote", OUT)
