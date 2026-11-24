import duckdb
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

con = duckdb.connect()
con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('" + STORY_PQ_SQL + "')")
sql = """
SELECT
  CASE
    WHEN zip IN ('11102','11103','11105','11106') THEN 'astoria'
    WHEN zip IN ('11360','11361','11364') THEN 'bayside'
    WHEN zip IN ('11369','11370','11372','11368','11354','11356') THEN 'rest'
  END AS g,
  count(*) FILTER (WHERE month(created) IN (12,1,2)) winter_n,
  count(*) FILTER (WHERE month(created) IN (12,1,2) AND problem = 'HEAT/HOT WATER') winter_heat,
  count(*) FILTER (WHERE month(created) IN (6,7,8)) summer_n,
  count(*) FILTER (WHERE month(created) IN (6,7,8) AND problem = 'Illegal Parking') summer_park
FROM t
WHERE zip IN ('11102','11103','11105','11106','11360','11361','11364','11369','11370','11372','11368','11354','11356')
GROUP BY 1
"""
print(con.execute(sql).fetchdf().to_string(index=False))
print(
    con.execute(
        """
        SELECT median(date_diff('second', created, closed) / 86400.0) med,
               avg(date_diff('second', created, closed) / 86400.0) mean
        FROM t
        WHERE problem = 'HEAT/HOT WATER' AND status = 'Closed' AND closed >= created
        """
    ).fetchdf().to_string(index=False)
)
print(
    "nypd over 7d",
    con.execute(
        """
        SELECT count(*) FROM t
        WHERE agency = 'NYPD' AND status = 'Closed' AND closed >= created
          AND date_diff('second', created, closed) > 7 * 86400
        """
    ).fetchone()[0],
)
