"""Stages 0-8: beginner EDA of NYC 311 requests, following All of Statistics.

The raw CSV is read once into a slim parquet file. Later stages aggregate that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

ROOT = Path(r"D:\nyc311")
RAW = RAW_CSV
INTERIM = INTERIM
FIG = FIGURES
PARQUET = INTERIM / "requests_slim.parquet"

# NYC five-borough bounding box (including edges of Staten Island and the Bronx).
LAT_MIN, LAT_MAX = 40.4774, 40.9176
LON_MIN, LON_MAX = -74.2591, -73.7004

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
BLUE = "#1f4e79"
ACCENT = "#c47b2b"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def build_parquet(con: duckdb.DuckDBPyConnection) -> None:
    if PARQUET.exists():
        print("parquet exists, reusing", PARQUET)
        return
    print("building slim parquet...")
    src = RAW.as_posix()
    dst = PARQUET.as_posix()
    con.execute(
        f"""
        COPY (
          SELECT
            NULLIF(trim("Unique Key"), '') AS unique_key,
            try_strptime(NULLIF(trim("Created Date"), ''), '%m/%d/%Y %I:%M:%S %p') AS created,
            (
              NULLIF(trim("Created Date"), '') IS NOT NULL
              AND try_strptime(NULLIF(trim("Created Date"), ''), '%m/%d/%Y %I:%M:%S %p') IS NULL
            ) AS created_unparsed,
            try_strptime(NULLIF(trim("Closed Date"), ''), '%m/%d/%Y %I:%M:%S %p') AS closed,
            (
              NULLIF(trim("Closed Date"), '') IS NOT NULL
              AND try_strptime(NULLIF(trim("Closed Date"), ''), '%m/%d/%Y %I:%M:%S %p') IS NULL
            ) AS closed_unparsed,
            NULLIF(trim(Agency), '') AS agency,
            NULLIF(trim("Problem (formerly Complaint Type)"), '') AS problem,
            NULLIF(trim(Status), '') AS status,
            NULLIF(trim(Borough), '') AS borough,
            NULLIF(trim("Open Data Channel Type"), '') AS channel,
            try_cast(NULLIF(trim(Latitude), '') AS DOUBLE) AS latitude,
            try_cast(NULLIF(trim(Longitude), '') AS DOUBLE) AS longitude
          FROM read_csv('{src}', header=true, all_varchar=true, strict_mode=false, parallel=true)
        ) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    print("parquet written")


def save_bar(path: Path, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=BLUE)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.18 if values else 1)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    con = connect()
    build_parquet(con)
    pq = PARQUET.as_posix()

    print("stage 1: sample definition")
    quality = con.execute(
        f"""
        SELECT
          count(*) AS n_rows,
          count(DISTINCT unique_key) AS n_keys,
          count(*) FILTER (WHERE unique_key IS NULL) AS n_null_key,
          count(*) FILTER (WHERE created_unparsed) AS n_created_unparsed,
          count(*) FILTER (WHERE created IS NULL AND NOT created_unparsed) AS n_created_blank,
          min(created) AS min_created,
          max(created) AS max_created,
          count(*) FILTER (WHERE closed_unparsed) AS n_closed_unparsed,
          count(*) FILTER (WHERE closed IS NULL AND NOT closed_unparsed) AS n_closed_blank,
          count(*) FILTER (WHERE created IS NOT NULL AND closed IS NOT NULL AND closed < created) AS n_closed_before_created,
          count(*) FILTER (WHERE agency IS NULL) AS n_agency_null,
          count(*) FILTER (WHERE problem IS NULL) AS n_problem_null,
          count(*) FILTER (WHERE status IS NULL) AS n_status_null,
          count(*) FILTER (WHERE borough IS NULL) AS n_borough_null,
          count(*) FILTER (WHERE channel IS NULL) AS n_channel_null,
          count(*) FILTER (WHERE latitude IS NULL OR longitude IS NULL) AS n_coord_null,
          count(*) FILTER (
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND (latitude NOT BETWEEN {LAT_MIN} AND {LAT_MAX}
                   OR longitude NOT BETWEEN {LON_MIN} AND {LON_MAX})
          ) AS n_coord_outside_nyc,
          count(*) FILTER (WHERE status = 'Closed' AND closed IS NULL) AS n_closed_status_no_date,
          count(*) FILTER (WHERE status IS DISTINCT FROM 'Closed' AND closed IS NOT NULL) AS n_other_status_with_closed_date
        FROM read_parquet('{pq}')
        """
    ).fetchdf()
    quality.to_csv(INTERIM / "01_quality.csv", index=False)

    n_dup_keys = con.execute(
        f"""
        SELECT count(*) FROM (
          SELECT unique_key
          FROM read_parquet('{pq}')
          WHERE unique_key IS NOT NULL
          GROUP BY 1
          HAVING count(*) > 1
        )
        """
    ).fetchone()[0]

    # One row per key. Duplicate keys, if any, keep the earliest parsed timestamp.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW analysis AS
        SELECT * EXCLUDE (rn)
        FROM (
          SELECT
            *,
            row_number() OVER (PARTITION BY unique_key ORDER BY created) AS rn
          FROM read_parquet('{pq}')
          WHERE unique_key IS NOT NULL AND created IS NOT NULL
        )
        WHERE rn = 1
        """
    )
    n_analysis = con.execute("SELECT count(*) FROM analysis").fetchone()[0]

    print("stage 2: one-way distributions")
    frames = {}
    for name, col in [
        ("borough", "borough"),
        ("status", "status"),
        ("agency", "agency"),
        ("channel", "channel"),
        ("problem", "problem"),
    ]:
        df = con.execute(
            f"""
            SELECT coalesce({col}, '(빈칸)') AS level, count(*) AS n
            FROM analysis
            GROUP BY 1
            ORDER BY n DESC
            """
        ).fetchdf()
        df["share"] = df["n"] / df["n"].sum()
        df.to_csv(INTERIM / f"02_{name}.csv", index=False)
        frames[name] = df

    def top_plus_other(df: pd.DataFrame, k: int) -> pd.DataFrame:
        head = df.head(k).copy()
        if len(df) > k:
            rest_n = int(df.iloc[k:]["n"].sum())
            rest_share = float(df.iloc[k:]["share"].sum())
            head = pd.concat(
                [head, pd.DataFrame([{"level": f"그 밖 ({len(df) - k}종)", "n": rest_n, "share": rest_share}])],
                ignore_index=True,
            )
        return head

    save_bar(
        FIG / "02_borough.png",
        frames["borough"]["level"].tolist(),
        frames["borough"]["share"].tolist(),
        "자치구 상대도수",
        "상대도수",
    )
    save_bar(
        FIG / "02_status.png",
        frames["status"]["level"].tolist(),
        frames["status"]["share"].tolist(),
        "상태 상대도수",
        "상대도수",
    )
    agency_plot = top_plus_other(frames["agency"], 12)
    save_bar(
        FIG / "02_agency.png",
        agency_plot["level"].tolist(),
        agency_plot["share"].tolist(),
        "기관 상대도수 (상위 12개와 그 밖)",
        "상대도수",
    )
    save_bar(
        FIG / "02_channel.png",
        frames["channel"]["level"].tolist(),
        frames["channel"]["share"].tolist(),
        "접수 채널 상대도수",
        "상대도수",
    )
    problem_plot = top_plus_other(frames["problem"], 15)
    save_bar(
        FIG / "02_problem.png",
        problem_plot["level"].tolist(),
        problem_plot["share"].tolist(),
        "민원 종류 상대도수 (상위 15개와 그 밖)",
        "상대도수",
    )

    print("stage 3: resolution time")
    duration = con.execute(
        """
        WITH d AS (
          SELECT date_diff('second', created, closed) / 86400.0 AS days
          FROM analysis
          WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
        )
        SELECT
          count(*) AS n,
          min(days) AS min_days,
          max(days) AS max_days,
          avg(days) AS mean_days,
          median(days) AS median_days,
          quantile_cont(days, 0.25) AS p25,
          quantile_cont(days, 0.75) AS p75,
          quantile_cont(days, 0.90) AS p90,
          quantile_cont(days, 0.99) AS p99,
          avg((days <= 1)::INT) AS cdf_1,
          avg((days <= 3)::INT) AS cdf_3,
          avg((days <= 7)::INT) AS cdf_7,
          avg((days <= 14)::INT) AS cdf_14,
          avg((days <= 30)::INT) AS cdf_30,
          avg((days <= 90)::INT) AS cdf_90,
          avg((days <= 365)::INT) AS cdf_365
        FROM d
        """
    ).fetchdf()
    duration.to_csv(INTERIM / "03_duration_summary.csv", index=False)

    hist = con.execute(
        """
        WITH d AS (
          SELECT date_diff('second', created, closed) / 86400.0 AS days
          FROM analysis
          WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
        )
        SELECT
          CASE WHEN days >= 30 THEN 30 ELSE CAST(floor(days) AS INTEGER) END AS day_bin,
          count(*) AS n
        FROM d
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    hist.to_csv(INTERIM / "03_duration_hist.csv", index=False)
    n_dur = int(duration.loc[0, "n"])
    hist["share"] = hist["n"] / n_dur

    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [str(int(b)) if int(b) < 30 else "30+" for b in hist["day_bin"]]
    ax.bar(labels, hist["share"], color=BLUE, width=0.9)
    ax.set_xlabel("처리시간 (일, 막대 하나는 1일 너비)")
    ax.set_ylabel("상대도수")
    ax.set_title("닫힌 민원의 처리시간 히스토그램")
    fig.tight_layout()
    fig.savefig(FIG / "03_duration_hist.png", dpi=120)
    plt.close(fig)

    cdf_ts = [1, 3, 7, 14, 30, 90, 365]
    cdf_ys = [float(duration.loc[0, f"cdf_{t}"]) for t in cdf_ts]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.step(cdf_ts, cdf_ys, where="post", color=BLUE, linewidth=2)
    ax.scatter(cdf_ts, cdf_ys, color=ACCENT, zorder=3)
    for t, y in zip(cdf_ts, cdf_ys):
        ax.text(t, y + 0.03, f"{y:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("t (일)")
    ax.set_ylabel("t일 안에 닫힌 비율")
    ax.set_title("처리시간의 경험적 분포함수 (고른 t에서만 표시)")
    fig.tight_layout()
    fig.savefig(FIG / "03_duration_ecdf.png", dpi=120)
    plt.close(fig)

    print("stage 4: monthly counts")
    monthly = con.execute(
        """
        SELECT date_trunc('month', created) AS month, count(*) AS n
        FROM analysis
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    monthly.to_csv(INTERIM / "04_monthly.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(monthly["month"], monthly["n"], color=BLUE, linewidth=1.4)
    ax.set_title("월별 접수 건수")
    ax.set_ylabel("건수")
    ax.set_xlabel("접수 월")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "04_monthly.png", dpi=120)
    plt.close(fig)

    print("stage 5: conditional distributions")
    top_problems = frames["problem"].loc[frames["problem"]["level"] != "(빈칸)", "level"].head(5).tolist()
    problem_list = ", ".join("'" + p.replace("'", "''") + "'" for p in top_problems)
    by_problem = con.execute(
        f"""
        WITH base AS (
          SELECT coalesce(borough, '(빈칸)') AS borough, problem
          FROM analysis
          WHERE problem IN ({problem_list})
        ),
        counts AS (
          SELECT borough, problem, count(*) AS n FROM base GROUP BY 1, 2
        ),
        totals AS (
          SELECT coalesce(borough, '(빈칸)') AS borough, count(*) AS n_borough
          FROM analysis
          GROUP BY 1
        )
        SELECT c.borough, c.problem, c.n, t.n_borough, c.n * 1.0 / t.n_borough AS share
        FROM counts c
        JOIN totals t USING (borough)
        ORDER BY c.borough, c.n DESC
        """
    ).fetchdf()
    by_problem.to_csv(INTERIM / "05_borough_problem.csv", index=False)

    city_n = int(frames["problem"]["n"].sum())
    city_share = {
        p: float(frames["problem"].loc[frames["problem"]["level"] == p, "share"].iloc[0]) for p in top_problems
    }

    borough_order = frames["borough"]["level"].tolist()
    pivot = by_problem.pivot(index="borough", columns="problem", values="share").reindex(borough_order)
    pivot = pivot[top_problems]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(top_problems)), top_problems, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index.tolist())
    ax.set_title("자치구 안에서 시 전체 상위 5개 민원이 차지하는 비율")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "05_borough_problem.png", dpi=120)
    plt.close(fig)

    def mix(col: str, filename: str) -> pd.DataFrame:
        df = con.execute(
            f"""
            SELECT coalesce({col}, '(빈칸)') AS row_level, coalesce(status, '(빈칸)') AS status, count(*) AS n
            FROM analysis
            GROUP BY 1, 2
            """
        ).fetchdf()
        df["share"] = df["n"] / df.groupby("row_level")["n"].transform("sum")
        df.to_csv(INTERIM / filename, index=False)
        return df

    borough_status = mix("borough", "05_borough_status.csv")
    agency_status = mix("agency", "05_agency_status.csv")
    channel_status = mix("channel", "05_channel_status.csv")

    def stacked(df: pd.DataFrame, title: str, path: Path, row_order: list[str]) -> None:
        statuses = df.groupby("status")["n"].sum().sort_values(ascending=False).index.tolist()
        wide = df.pivot(index="row_level", columns="status", values="share").reindex(row_order)
        wide = wide.reindex(columns=statuses).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        left = np.zeros(len(wide))
        colors = ["#1f4e79", "#4f86c6", "#c47b2b", "#d6a15e", "#6b7280", "#94a3b8", "#111827"]
        y = np.arange(len(wide))
        for i, status in enumerate(wide.columns):
            vals = wide[status].to_numpy()
            ax.barh(y, vals, left=left, color=colors[i % len(colors)], label=status)
            left = left + vals
        ax.set_yticks(y, wide.index.tolist())
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("상태 비율")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    stacked(borough_status, "자치구별 상태 비율", FIG / "05_borough_status.png", borough_order)
    stacked(
        channel_status,
        "접수 채널별 상태 비율",
        FIG / "05_channel_status.png",
        frames["channel"]["level"].tolist(),
    )

    duration_by_borough = con.execute(
        """
        SELECT
          coalesce(borough, '(빈칸)') AS borough,
          count(*) AS n,
          median(date_diff('second', created, closed) / 86400.0) AS median_days,
          quantile_cont(date_diff('second', created, closed) / 86400.0, 0.75) AS p75_days
        FROM analysis
        WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
        GROUP BY 1
        ORDER BY n DESC
        """
    ).fetchdf()
    duration_by_borough.to_csv(INTERIM / "05_borough_duration.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(duration_by_borough))
    ax.barh(y, duration_by_borough["median_days"], color=BLUE, label="중앙값")
    ax.plot(duration_by_borough["p75_days"], y, "o", color=ACCENT, label="75% 분위수")
    ax.set_yticks(y, duration_by_borough["borough"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("일")
    ax.set_title("자치구별 처리시간")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "05_borough_duration.png", dpi=120)
    plt.close(fig)

    print("stage 7: bootstrap median")
    top_borough = str(duration_by_borough.iloc[0]["borough"])
    durations = con.execute(
        """
        SELECT date_diff('second', created, closed) / 86400.0 AS days
        FROM analysis
        WHERE status = 'Closed' AND closed IS NOT NULL AND closed >= created
          AND borough = ?
        """,
        [top_borough],
    ).fetchnumpy()["days"]
    population_median = float(np.median(durations))
    rng = np.random.default_rng(20261112)
    boot_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, n in zip(axes, [200, 2000, 20000]):
        study = rng.choice(durations, size=n, replace=False)
        replicates = np.empty(1000)
        for b in range(1000):
            replicates[b] = float(np.median(rng.choice(study, size=n, replace=True)))
        lo, hi = np.quantile(replicates, [0.025, 0.975])
        boot_rows.append(
            {
                "borough": top_borough,
                "sample_size": n,
                "study_median": float(np.median(study)),
                "boot_mean": float(replicates.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "ci95_width": float(hi - lo),
            }
        )
        ax.hist(replicates, bins=30, color=BLUE)
        ax.axvline(lo, color=ACCENT, linewidth=1)
        ax.axvline(hi, color=ACCENT, linewidth=1)
        ax.set_title(f"n={n:,}\n너비 {hi - lo:.3f}일")
        ax.set_xlabel("부트스트랩 중앙값 (일)")
    axes[0].set_ylabel("반복 횟수")
    fig.suptitle(f"{top_borough} 닫힌 민원에서 뽑은 표본의 처리시간 중앙값", y=1.05)
    fig.tight_layout()
    fig.savefig(FIG / "07_bootstrap_median.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    boot_df = pd.DataFrame(boot_rows)
    width_20k = float(boot_df.loc[boot_df["sample_size"] == 20000, "ci95_width"].iloc[0])
    n_borough = int(durations.shape[0])
    extrapolated = width_20k * np.sqrt(20000 / n_borough)
    boot_df.to_csv(INTERIM / "07_bootstrap.csv", index=False)

    print("stage 8: independence, channel x status")
    ct = channel_status.pivot(index="row_level", columns="status", values="n").fillna(0).astype(int)
    ct.to_csv(INTERIM / "08_channel_status_counts.csv")
    chi2, pvalue, dof, expected = chi2_contingency(ct.to_numpy())
    n_ct = int(ct.to_numpy().sum())
    r, k = ct.shape
    cramer = float(np.sqrt(chi2 / (n_ct * (min(r - 1, k - 1)))))
    resid = (ct.to_numpy() - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(resid, index=ct.index, columns=ct.columns)
    resid_df.to_csv(INTERIM / "08_pearson_residuals.csv")
    flat = resid_df.stack()
    extreme = flat.abs().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(resid, aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30)
    ax.set_xticks(range(k), ct.columns.tolist(), rotation=25, ha="right")
    ax.set_yticks(range(r), ct.index.tolist())
    ax.set_title("채널 × 상태, 피어슨 잔차 (독립일 때 기대와의 차이)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "08_channel_status_residuals.png", dpi=120)
    plt.close(fig)

    def records(df: pd.DataFrame, n: int = 20) -> list[dict]:
        out = df.head(n).copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        return json.loads(out.to_json(orient="records", double_precision=6))

    summary = {
        "n_rows": int(quality.loc[0, "n_rows"]),
        "n_keys": int(quality.loc[0, "n_keys"]),
        "n_null_key": int(quality.loc[0, "n_null_key"]),
        "n_duplicate_keys": int(n_dup_keys),
        "n_analysis": int(n_analysis),
        "min_created": str(quality.loc[0, "min_created"]),
        "max_created": str(quality.loc[0, "max_created"]),
        "n_created_unparsed": int(quality.loc[0, "n_created_unparsed"]),
        "n_created_blank": int(quality.loc[0, "n_created_blank"]),
        "n_closed_unparsed": int(quality.loc[0, "n_closed_unparsed"]),
        "n_closed_blank": int(quality.loc[0, "n_closed_blank"]),
        "n_closed_before_created": int(quality.loc[0, "n_closed_before_created"]),
        "n_agency_null": int(quality.loc[0, "n_agency_null"]),
        "n_problem_null": int(quality.loc[0, "n_problem_null"]),
        "n_status_null": int(quality.loc[0, "n_status_null"]),
        "n_borough_null": int(quality.loc[0, "n_borough_null"]),
        "n_channel_null": int(quality.loc[0, "n_channel_null"]),
        "n_coord_null": int(quality.loc[0, "n_coord_null"]),
        "n_coord_outside_nyc": int(quality.loc[0, "n_coord_outside_nyc"]),
        "n_closed_status_no_date": int(quality.loc[0, "n_closed_status_no_date"]),
        "n_other_status_with_closed_date": int(quality.loc[0, "n_other_status_with_closed_date"]),
        "duration": records(duration, 1)[0],
        "monthly_head": records(monthly, 3),
        "monthly_tail": records(monthly.tail(3), 3),
        "top_borough_for_bootstrap": top_borough,
        "borough_closed_n": n_borough,
        "borough_population_median_days": population_median,
        "bootstrap": boot_rows,
        "bootstrap_width_extrapolated_to_borough_days": float(extrapolated),
        "chi2": float(chi2),
        "chi2_pvalue": float(pvalue),
        "chi2_dof": int(dof),
        "cramers_v": cramer,
        "chi2_n": n_ct,
        "residual_extremes": [
            {"channel": str(idx[0]), "status": str(idx[1]), "residual": float(val)}
            for idx, val in extreme.items()
        ],
        "citywide_top5_problem_share": city_share,
        "top5_problems": top_problems,
    }
    (INTERIM / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")
    print(json.dumps({k: summary[k] for k in ["n_rows", "n_analysis", "min_created", "max_created", "chi2_pvalue", "cramers_v"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
