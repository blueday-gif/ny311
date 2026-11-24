"""Headline numbers for the public question book. Every figure is recomputable."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

ROOT = Path(r"D:\nyc311")
PQ = STORY_PQ
WEATHER = WEATHER
OUT = INTERIM / "story_stats.json"

# 2020 Census, borough populations. Used only for per-100,000 rates.
POP = {
    "MANHATTAN": 1694251,
    "BROOKLYN": 2736074,
    "QUEENS": 2405464,
    "BRONX": 1472654,
    "STATEN ISLAND": 495747,
}

ASTORIA = ("11102", "11103", "11105", "11106")
BAYSIDE = ("11360", "11361", "11364")
# NY-14 Queens side used in the politics chapter. Bronx side is not reduced to a few zips.
AOC_QUEENS = ("11102", "11103", "11105", "11106", "11369", "11370", "11372", "11368", "11354", "11356")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    pq = PQ.as_posix()
    weather = json.loads(WEATHER.read_text(encoding="utf-8"))["daily"]
    wdf = pd.DataFrame(
        {
            "day": pd.to_datetime(weather["time"]),
            "temp": weather["temperature_2m_mean"],
            "snow": weather["snowfall_sum"],
        }
    )
    con.register("weather", wdf)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW t AS
        SELECT
          *,
          CAST(created AS DATE) AS day,
          dayofweek(created) AS dow,
          hour(created) AS hr,
          date_diff('second', created, closed) / 86400.0 AS duration_days
        FROM read_parquet('{pq}')
        WHERE created IS NOT NULL AND unique_key IS NOT NULL
        """
    )

    base = con.execute(
        """
        SELECT
          count(*) AS n,
          count(DISTINCT unique_key) AS n_keys,
          min(created) AS min_created,
          max(created) AS max_created
        FROM t
        """
    ).fetchdf().iloc[0]

    daily = con.execute(
        """
        SELECT day, count(*) AS n
        FROM t
        WHERE day NOT IN (DATE '2025-11-01', DATE '2026-10-31')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    # Days inside the window with zero complaints still count in the mean.
    full_days = pd.date_range("2025-11-02", "2026-10-30", freq="D")
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.set_index("day").reindex(full_days, fill_value=0)
    daily.index.name = "day"
    mean_day = float(daily["n"].mean())
    busiest = daily["n"].idxmax()
    quietest = daily["n"].idxmin()

    span_seconds = (pd.Timestamp(base["max_created"]) - pd.Timestamp(base["min_created"])).total_seconds()
    seconds_per = span_seconds / int(base["n"])

    top = con.execute(
        """
        SELECT problem, count(*) n FROM t GROUP BY 1 ORDER BY n DESC
        """
    ).fetchdf()
    top10_share = float(top.head(10)["n"].sum() / top["n"].sum())
    nypd = con.execute("SELECT count(*) FROM t WHERE agency = 'NYPD'").fetchone()[0]

    weekday = con.execute(
        """
        SELECT dow, count(*) n, count(DISTINCT day) AS n_days
        FROM t
        WHERE day NOT IN (DATE '2025-11-01', DATE '2026-10-31')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    # DuckDB dayofweek: Sunday=0 ... Saturday=6
    weekday["per_day"] = weekday["n"] / weekday["n_days"]

    hour = con.execute(
        """
        SELECT hr, count(*) n FROM t GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()

    def day_count(problem: str | None, day: str) -> int:
        if problem is None:
            return int(con.execute("SELECT count(*) FROM t WHERE day = CAST(? AS DATE)", [day]).fetchone()[0])
        return int(
            con.execute(
                "SELECT count(*) FROM t WHERE problem = ? AND day = CAST(? AS DATE)",
                [problem, day],
            ).fetchone()[0]
        )

    def mean_problem(problem: str) -> float:
        s = con.execute(
            """
            SELECT day, count(*) n FROM t
            WHERE problem = ? AND day NOT IN (DATE '2025-11-01', DATE '2026-10-31')
            GROUP BY 1
            """,
            [problem],
        ).fetchdf()
        s["day"] = pd.to_datetime(s["day"])
        s = s.set_index("day").reindex(full_days, fill_value=0)
        return float(s["n"].mean())

    fireworks_july4 = day_count("Illegal Fireworks", "2026-07-04")
    fireworks_mean = mean_problem("Illegal Fireworks")
    fireworks_hour = con.execute(
        """
        SELECT hr, count(*) n FROM t
        WHERE problem = 'Illegal Fireworks' AND day = DATE '2026-07-04'
        GROUP BY 1 ORDER BY n DESC LIMIT 1
        """
    ).fetchone()
    fireworks_newyear = con.execute(
        """
        SELECT count(*) FROM t
        WHERE problem = 'Illegal Fireworks'
          AND created >= TIMESTAMP '2026-01-01 00:00:00'
          AND created < TIMESTAMP '2026-01-01 01:00:00'
        """
    ).fetchone()[0]
    fireworks_hour0_mean = con.execute(
        """
        SELECT count(*) * 1.0 / count(DISTINCT day)
        FROM t
        WHERE problem = 'Illegal Fireworks' AND hr = 0
          AND day NOT IN (DATE '2025-11-01', DATE '2026-10-31', DATE '2026-01-01')
        """
    ).fetchone()[0]

    christmas_all = day_count(None, "2025-12-25")
    noise_xmas = day_count("Noise - Residential", "2025-12-25")
    noise_mean = mean_problem("Noise - Residential")
    parking_xmas = day_count("Illegal Parking", "2025-12-25")
    parking_mean = mean_problem("Illegal Parking")

    joined = con.execute(
        """
        WITH d AS (
          SELECT day, problem, count(*) n
          FROM t
          WHERE day NOT IN (DATE '2025-11-01', DATE '2026-10-31')
          GROUP BY 1, 2
        )
        SELECT w.day, w.temp, w.snow, d.problem, coalesce(d.n, 0) AS n
        FROM weather w
        LEFT JOIN d ON CAST(w.day AS DATE) = d.day
        WHERE w.day NOT IN (DATE '2025-11-01', DATE '2026-10-31')
        """
    ).fetchdf()

    def temp_ratio(problem: str) -> dict:
        sub = joined[joined["problem"] == problem]
        # days with no rows are missing; add zero days from weather
        days = wdf[(wdf["day"] >= "2025-11-02") & (wdf["day"] <= "2026-10-30")].copy()
        counts = sub.groupby("day", as_index=False)["n"].sum() if len(sub) else pd.DataFrame({"day": [], "n": []})
        days = days.merge(counts, on="day", how="left")
        days["n"] = days["n"].fillna(0)
        cold = days[days["temp"] < 0]["n"].mean()
        hot = days[days["temp"] >= 25]["n"].mean()
        return {
            "cold_days": int((days["temp"] < 0).sum()),
            "hot_days": int((days["temp"] >= 25).sum()),
            "cold_per_day": float(cold),
            "hot_per_day": float(hot),
            "ratio_hot_over_cold": float(hot / cold) if cold else None,
        }

    snow_days = wdf[wdf["snow"] >= 2].copy()
    snow_complaint = con.execute(
        """
        SELECT day, count(*) n FROM t WHERE problem = 'Snow or Ice' GROUP BY 1
        """
    ).fetchdf()
    snow_complaint["day"] = pd.to_datetime(snow_complaint["day"])
    snow_total = int(snow_complaint["n"].sum())
    top_snow_days = snow_complaint.sort_values("n", ascending=False).head(5)

    borough = con.execute(
        """
        SELECT borough,
               count(*) n,
               count(*) FILTER (WHERE problem = 'HEAT/HOT WATER') heat,
               count(*) FILTER (WHERE problem = 'Illegal Parking') parking,
               count(*) FILTER (WHERE problem = 'Noise - Residential') noise,
               count(*) FILTER (WHERE problem = 'Blocked Driveway') driveway,
               count(*) FILTER (WHERE problem = 'Rodent') rodent,
               count(*) FILTER (WHERE problem = 'Noise - Helicopter') heli,
               count(*) FILTER (WHERE channel = 'PHONE') phone,
               count(*) FILTER (WHERE channel = 'ONLINE') online,
               count(*) FILTER (WHERE channel = 'MOBILE') mobile
        FROM t
        GROUP BY 1
        """
    ).fetchdf()

    def per100k(row, col):
        pop = POP.get(row["borough"])
        if not pop:
            return None
        return row[col] / pop * 100000

    for col in ["n", "heat", "parking", "noise", "driveway", "rodent", "heli", "phone"]:
        borough[f"{col}_per100k"] = borough.apply(lambda r, c=col: per100k(r, c), axis=1)

    addresses = con.execute(
        """
        SELECT address, borough, count(*) n,
               count(*) FILTER (WHERE problem = 'Noise - Residential') noise
        FROM t
        WHERE address IS NOT NULL AND address <> ''
        GROUP BY 1, 2
        ORDER BY n DESC
        LIMIT 15
        """
    ).fetchdf()
    top_addr = addresses.iloc[0]
    noise_total = int(con.execute("SELECT count(*) FROM t WHERE problem = 'Noise - Residential'").fetchone()[0])
    addr_counts = con.execute(
        """
        SELECT count(*) n FROM t
        WHERE address IS NOT NULL AND address <> ''
        GROUP BY address
        """
    ).fetchdf()
    addr_counts = addr_counts.sort_values("n", ascending=False)
    cutoff = max(1, int(round(len(addr_counts) * 0.01)))
    top1_share = float(addr_counts.head(cutoff)["n"].sum() / addr_counts["n"].sum())

    nypd_res = con.execute(
        """
        SELECT
          count(*) AS n,
          count(*) FILTER (WHERE resolution ILIKE '%issued a summons%') summons,
          count(*) FILTER (WHERE resolution ILIKE '%unable to gain entry%') no_entry,
          count(*) FILTER (WHERE resolution ILIKE '%no evidence%' OR resolution ILIKE '%observed no criminal%' OR resolution ILIKE '%no criminal violation%') no_evidence,
          count(*) FILTER (WHERE resolution ILIKE '%arrest%') arrest_word
        FROM t
        WHERE agency = 'NYPD' AND resolution IS NOT NULL
        """
    ).fetchdf().iloc[0]

    enc = con.execute(
        """
        SELECT
          CASE WHEN created < TIMESTAMP '2026-09-22' THEN 'before' ELSE 'after' END AS period,
          agency,
          count(*) n,
          count(DISTINCT CAST(created AS DATE)) AS n_days
        FROM t
        WHERE problem = 'Encampment'
        GROUP BY 1, 2
        ORDER BY 1, n DESC
        """
    ).fetchdf()

    def zip_mix(zips: tuple[str, ...]) -> dict:
        row = con.execute(
            f"""
            SELECT count(*) n,
                   count(*) FILTER (WHERE problem = 'HEAT/HOT WATER') heat,
                   count(*) FILTER (WHERE problem = 'Illegal Parking') parking,
                   count(*) FILTER (WHERE problem = 'Noise - Residential') noise
            FROM t
            WHERE zip IN ({",".join("'" + z + "'" for z in zips)})
            """
        ).fetchdf().iloc[0]
        n = int(row["n"])
        return {
            "n": n,
            "heat_share": float(row["heat"] / n) if n else None,
            "parking_share": float(row["parking"] / n) if n else None,
            "noise_share": float(row["noise"] / n) if n else None,
        }

    city_mix = {
        "heat_share": float(top.loc[top["problem"] == "HEAT/HOT WATER", "n"].iloc[0] / int(base["n"])),
        "parking_share": float(top.loc[top["problem"] == "Illegal Parking", "n"].iloc[0] / int(base["n"])),
        "noise_share": float(top.loc[top["problem"] == "Noise - Residential", "n"].iloc[0] / int(base["n"])),
    }

    taxi_complaint = int(con.execute("SELECT count(*) FROM t WHERE problem = 'Taxi Complaint'").fetchone()[0])
    taxi_compliment = int(con.execute("SELECT count(*) FROM t WHERE problem = 'Taxi Compliment'").fetchone()[0])
    rare = int((top["n"] <= 10).sum())

    parking_hour = con.execute(
        """
        SELECT hr, count(*) n FROM t WHERE problem = 'Illegal Parking' GROUP BY 1 ORDER BY n DESC LIMIT 1
        """
    ).fetchone()
    parking_hour_low = con.execute(
        """
        SELECT hr, count(*) n FROM t WHERE problem = 'Illegal Parking' GROUP BY 1 ORDER BY n ASC LIMIT 1
        """
    ).fetchone()

    rodent_month = con.execute(
        """
        SELECT date_trunc('month', created) m, count(*) n
        FROM t WHERE problem = 'Rodent'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()

    result = {
        "n": int(base["n"]),
        "n_keys": int(base["n_keys"]),
        "min_created": str(base["min_created"]),
        "max_created": str(base["max_created"]),
        "full_days": int(len(full_days)),
        "mean_per_full_day": mean_day,
        "seconds_between": seconds_per,
        "per_minute": 60 / seconds_per,
        "busiest_day": str(busiest.date()),
        "busiest_n": int(daily.loc[busiest, "n"]),
        "busiest_vs_mean": float(daily.loc[busiest, "n"] / mean_day),
        "quietest_day": str(quietest.date()),
        "quietest_n": int(daily.loc[quietest, "n"]),
        "quietest_vs_mean": float(daily.loc[quietest, "n"] / mean_day),
        "problem_types": int(len(top)),
        "top_problem": str(top.iloc[0]["problem"]),
        "top_problem_n": int(top.iloc[0]["n"]),
        "top_problem_share": float(top.iloc[0]["n"] / int(base["n"])),
        "top10_share": top10_share,
        "nypd_n": int(nypd),
        "nypd_share": float(nypd / int(base["n"])),
        "weekday": weekday.to_dict(orient="records"),
        "hour_peak": {"hr": int(hour.loc[hour["n"].idxmax(), "hr"]), "n": int(hour["n"].max())},
        "fireworks_july4": fireworks_july4,
        "fireworks_mean_day": fireworks_mean,
        "fireworks_july4_ratio": fireworks_july4 / fireworks_mean if fireworks_mean else None,
        "fireworks_peak_hour_july4": {"hr": int(fireworks_hour[0]), "n": int(fireworks_hour[1])},
        "fireworks_newyear_first_hour": int(fireworks_newyear),
        "fireworks_midnight_other_days": float(fireworks_hour0_mean or 0),
        "christmas_n": christmas_all,
        "christmas_vs_mean": christmas_all / mean_day,
        "noise_christmas": noise_xmas,
        "noise_mean_day": noise_mean,
        "noise_christmas_ratio": noise_xmas / noise_mean if noise_mean else None,
        "parking_christmas": parking_xmas,
        "parking_mean_day": parking_mean,
        "parking_christmas_ratio": parking_xmas / parking_mean if parking_mean else None,
        "weather_ratios": {
            p: temp_ratio(p)
            for p in [
                "Noise - Street/Sidewalk",
                "Noise - Residential",
                "Illegal Fireworks",
                "Drinking",
                "Rodent",
            ]
        },
        "snow_total": snow_total,
        "top_snow_days": [
            {"day": str(pd.Timestamp(r.day).date()), "n": int(r.n)}
            for r in top_snow_days.itertuples(index=False)
        ],
        "borough": borough.fillna(0).to_dict(orient="records"),
        "top_address": {
            "address_redacted": "Bronx address, neighborhood only in the book",
            "borough": str(top_addr["borough"]),
            "n": int(top_addr["n"]),
            "noise": int(top_addr["noise"]),
            "noise_share_of_city": float(top_addr["noise"] / noise_total),
        },
        "address_top1pct_n": cutoff,
        "address_top1pct_share": top1_share,
        "nypd_resolution": {k: int(v) if k != "n" else int(v) for k, v in nypd_res.items()},
        "encampment_by_period": enc.to_dict(orient="records"),
        "zip_astoria": zip_mix(ASTORIA),
        "zip_bayside": zip_mix(BAYSIDE),
        "zip_aoc_queens": zip_mix(AOC_QUEENS),
        "city_mix": city_mix,
        "taxi_complaint": taxi_complaint,
        "taxi_compliment": taxi_compliment,
        "rare_types_le_10": rare,
        "parking_peak_hour": {"hr": int(parking_hour[0]), "n": int(parking_hour[1])},
        "parking_low_hour": {"hr": int(parking_hour_low[0]), "n": int(parking_hour_low[1])},
        "rodent_by_month": [
            {"month": str(pd.Timestamp(r.m).date()), "n": int(r.n)}
            for r in rodent_month.itertuples(index=False)
        ],
        "populations_2020": POP,
    }
    # JSON-safe weekday
    result["weekday"] = [
        {"dow": int(r["dow"]), "n": int(r["n"]), "days": int(r["n_days"]), "per_day": float(r["per_day"])}
        for r in result["weekday"]
    ]
    result["borough"] = json.loads(borough.to_json(orient="records"))
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT)
    print("busiest", result["busiest_day"], result["busiest_n"], "quietest", result["quietest_day"], result["quietest_n"])
    print("july4 fireworks", result["fireworks_july4"], "ratio", round(result["fireworks_july4_ratio"], 2))
    print("top address n", result["top_address"]["n"], "noise share", round(result["top_address"]["noise_share_of_city"], 3))
    print("top1", round(result["address_top1pct_share"], 3))


if __name__ == "__main__":
    main()
