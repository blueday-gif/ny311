"""1단계: 데이터 정제 및 기술통계.

입력:  data/raw/nyc311_all_2024_apr_sep.csv           (수정하지 않음, stat_00_download.py 출력)
       data/raw/nyc_daily_weather_2024_apr_sep.csv    (있으면 일별 표에 기온을 붙인다)
출력:  data/processed/nyc311_all_clean.parquet        정제된 민원 단위 데이터 + 파생변수
       data/processed/nyc311_all_daily.csv            일별 접수 건수 (+ 요일, 주말, 기온)
       data/processed/nyc311_all_agg_daily_borough_type.csv   대시보드 필터용 집계
       data/processed/nyc311_all_map_sample.csv       대시보드 지도용 무작위 표본
       outputs/tables/stat1_*.csv, outputs/figures/stat1_*.png

정제 기준 (순서대로 적용, 건수는 stat1_exclusion_log.csv):
  1. unique_key 중복 -> 첫 행만 유지
  2. created_date 파싱 실패 -> 제외
  3. closed_date 결측(미종결) 또는 파싱 실패 -> 제외 (처리 시간을 계산할 수 없음)
  4. Resolution_Hours < 0 (종결이 접수보다 빠른 기록 오류) -> 제외
  5. agency / complaint_type 결측 -> 제외
  6. borough 결측 또는 'Unspecified' -> 제외
  7. 위경도 결측·뉴욕 범위 밖 -> 제외하지 않고 has_coord=False 로 표시 (지도에서만 빠짐)
     좌표 결측은 민원 유형에 따라 몰려 있어, 행을 지우면 통계가 치우치기 때문이다.

일별 접수 건수는 '접수' 기준이므로 1~2번만 적용한 전체 접수 건으로 센다 (미종결 포함).

실행: 프로젝트 루트에서  python analysis/stat_01_clean_descriptive.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from stat_common import (
    AGG, BOROUGH_KO, BOROUGHS, CLEAN, DAILY, END, MAP_SAMPLE, PROC, RAW_311, RAW_WEATHER, ROOT, SEED,
    SINGLE, ACCENT, GRID, MUTED, OTHER_GRAY, START,
    save_fig, save_table, setup_style, time_band, top_n_other,
)

MAP_N = 40_000
NYC_BBOX = dict(lat=(40.45, 40.95), lon=(-74.30, -73.65))


def load_and_clean(keep_open=False):
    """원본을 읽어 정제한다.

    keep_open=True 이면 미종결 민원(closed_date 결측)을 지우지 않고 남긴다 (resolution_hours 는 NaN).
    생존분석(stat_05)이 미종결을 '중도절단' 자료로 쓰기 위한 옵션이며, 나머지 기준은 똑같이 적용된다.
    """
    if not RAW_311.exists():
        raise SystemExit(f"{RAW_311.relative_to(ROOT)} 가 없습니다. 먼저 analysis/stat_00_download.py 를 실행하세요.")
    df = pd.read_csv(RAW_311, dtype=str)
    log = [("원본 행 수", len(df), "")]

    dup = df["unique_key"].duplicated()
    log.append(("unique_key 중복", int(dup.sum()), "제외 (첫 행만 유지)"))
    df = df[~dup]

    df["created_dt"] = pd.to_datetime(df["created_date"], errors="coerce", format="ISO8601")
    bad = df["created_dt"].isna()
    log.append(("created_date 파싱 실패", int(bad.sum()), "제외"))
    df = df[~bad]
    out_range = (df["created_dt"] < START) | (df["created_dt"] >= END)
    log.append(("created_date 가 분석 기간 밖", int(out_range.sum()), "제외"))
    df = df[~out_range]
    intake = df[["created_dt"]].copy()           # 일별 접수 건수용 (미종결 포함)
    log.append(("접수 건수 (일별 건수 집계 기준)", len(df), "미종결 포함"))

    df["closed_dt"] = pd.to_datetime(df["closed_date"], errors="coerce", format="ISO8601")
    bad = df["closed_dt"].isna()
    log.append(("closed_date 결측·파싱 실패 (미종결)", int(bad.sum()), "유지 (중도절단 자료)" if keep_open else "제외"))
    if not keep_open:
        df = df[~bad]

    df["resolution_hours"] = (df["closed_dt"] - df["created_dt"]).dt.total_seconds() / 3600
    bad = df["resolution_hours"] < 0
    log.append(("Resolution_Hours < 0", int(bad.sum()), "제외"))
    df = df[~bad]

    for col in ["agency", "complaint_type", "borough"]:
        df[col] = df[col].str.strip()
    df["borough"] = df["borough"].str.upper()
    bad = df["agency"].isna() | df["complaint_type"].isna() | (df["agency"] == "") | (df["complaint_type"] == "")
    log.append(("agency / complaint_type 결측", int(bad.sum()), "제외"))
    df = df[~bad]
    bad = ~df["borough"].isin(BOROUGHS)
    log.append(("borough 결측 또는 Unspecified", int(bad.sum()), "제외"))
    df = df[~bad]

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["has_coord"] = (
        df["latitude"].between(*NYC_BBOX["lat"]) & df["longitude"].between(*NYC_BBOX["lon"])
    )
    log.append(("위경도 결측·뉴욕 범위 밖", int((~df["has_coord"]).sum()), "제외 아님 (has_coord=False, 지도에서만 제외)"))
    log.append(("Resolution_Hours = 0 (접수 즉시 종결)", int((df["resolution_hours"] == 0).sum()), "제외 아님 (확인용)"))
    log.append(("최종 분석 대상 행 수", len(df), ""))

    # ---- 파생변수: 모두 접수 시점에 알 수 있는 정보 (종결 시각은 목표변수에만 쓴다) ----
    df["date"] = df["created_dt"].dt.normalize()
    df["hour"] = df["created_dt"].dt.hour
    df["weekday"] = df["created_dt"].dt.day_of_week         # 0=월 ... 6=일
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["month"] = df["created_dt"].dt.month
    df["time_band"] = time_band(df["hour"])
    df["log_resolution"] = np.log1p(df["resolution_hours"])
    df["within_24h"] = (df["resolution_hours"] <= 24).astype(int)

    cols = ["unique_key", "created_dt", "closed_dt", "agency", "complaint_type", "borough", "latitude", "longitude",
            "has_coord", "resolution_hours", "log_resolution", "within_24h", "date", "hour", "weekday",
            "is_weekend", "month", "time_band"]
    return df[cols].reset_index(drop=True), intake, pd.DataFrame(log, columns=["항목", "건수", "처리"])


def numeric_summary(s, name):
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return {
        "변수": name, "n": len(s), "평균": s.mean(), "중앙값": med, "표준편차": s.std(ddof=1),
        "최솟값": s.min(), "Q1": q1, "Q3": q3, "IQR": q3 - q1, "최댓값": s.max(),
        "왜도": stats.skew(s), "첨도(초과)": stats.kurtosis(s),
    }


def freq_table(s, label, top=None):
    vc = s.value_counts()
    if top and len(vc) > top:
        vc = pd.concat([vc.iloc[:top], pd.Series({f"기타 ({len(vc) - top}개 범주)": vc.iloc[top:].sum()})])
    out = vc.rename_axis(label).reset_index(name="건수")
    out["비율(%)"] = (out["건수"] / out["건수"].sum() * 100).round(2)
    out["누적비율(%)"] = out["비율(%)"].cumsum().round(2)
    return out


def build_daily(intake):
    days = pd.date_range(START, END, inclusive="left")
    daily = intake["created_dt"].dt.normalize().value_counts().reindex(days, fill_value=0)
    daily = daily.rename_axis("date").reset_index(name="n_complaints")
    daily["weekday"] = daily["date"].dt.day_of_week
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
    if RAW_WEATHER.exists():
        w = pd.read_csv(RAW_WEATHER, parse_dates=["date"])
        daily = daily.merge(w, on="date", how="left")
        print(f"  기온 결합: {daily['temp_mean_c'].notna().sum()}일 / {len(daily)}일")
    else:
        print("  기온 파일 없음 -> 일별 표에 기온 열을 넣지 않음")
    return daily


def figures(df, daily, plt):
    # 자치구별 건수
    vc = df["borough"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([BOROUGH_KO[b] for b in vc.index][::-1], vc.values[::-1], color=SINGLE, height=0.62)
    for y, v in enumerate(vc.values[::-1]):
        ax.text(v, y, f" {v:,} ({v / vc.sum():.1%})", va="center", fontsize=9, color=MUTED)
    ax.set_title("자치구별 민원 건수")
    ax.set_xlim(0, vc.max() * 1.22)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save_fig(fig, "stat1_01_borough_counts.png")

    # 민원 유형 상위 15
    vc = df["complaint_type"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(vc.index[::-1], vc.values[::-1], color=SINGLE, height=0.66)
    for y, v in enumerate(vc.values[::-1]):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8.5, color=MUTED)
    ax.set_title("민원 유형 상위 15개")
    ax.set_xlim(0, vc.max() * 1.15)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save_fig(fig, "stat1_02_type_top15.png")

    # 일별 접수 건수
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(daily["date"], daily["n_complaints"], color=SINGLE, lw=1.3)
    wk = daily[daily["is_weekend"] == 1]
    ax.scatter(wk["date"], wk["n_complaints"], s=12, color=ACCENT, zorder=3, label="주말")
    ax.axhline(daily["n_complaints"].mean(), color=OTHER_GRAY, lw=1, ls="--")
    ax.text(daily["date"].iloc[-1], daily["n_complaints"].mean(), " 평균", va="center", fontsize=8.5, color=MUTED)
    ax.set_title("일별 민원 접수 건수")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(loc="upper left")
    save_fig(fig, "stat1_03_daily_counts.png")


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    print("[1] 정제")
    df, intake, log = load_and_clean()
    save_table(log, "stat1_exclusion_log.csv")
    print(log.to_string(index=False))
    df.to_parquet(CLEAN, index=False)
    print(f"  저장: {CLEAN.relative_to(ROOT)}  ({len(df):,}행)")

    print("[2] 일별 접수 건수")
    daily = build_daily(intake)
    daily.to_csv(DAILY, index=False)
    print(f"  저장: {DAILY.relative_to(ROOT)}  ({len(daily)}일)")

    print("[3] 수치형 기술통계")
    summ = pd.DataFrame([
        numeric_summary(df["resolution_hours"], "Resolution_Hours (시간)"),
        numeric_summary(daily["n_complaints"].astype(float), "일별 민원 접수 건수"),
    ]).round(3)
    save_table(summ, "stat1_numeric_summary.csv")
    print(summ.T.to_string(header=False))

    pct = df["resolution_hours"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    pct_t = pd.DataFrame({"백분위": [f"P{int(q * 100)}" for q in pct.index], "Resolution_Hours": pct.values.round(3)})
    save_table(pct_t, "stat1_resolution_percentiles.csv")

    print("[4] 범주형 빈도표")
    save_table(freq_table(df["borough"], "borough"), "stat1_freq_borough.csv")
    save_table(freq_table(df["agency"], "agency", top=15), "stat1_freq_agency.csv")
    save_table(freq_table(df["complaint_type"], "complaint_type", top=20), "stat1_freq_complaint_type.csv")
    print(f"  범주 수: agency {df['agency'].nunique()}, complaint_type {df['complaint_type'].nunique()}")

    print("[5] 대시보드용 집계·표본")
    df["type_group"] = top_n_other(df["complaint_type"], 15)
    agg = (df.groupby(["date", "borough", "type_group"], observed=True)
             .agg(n=("resolution_hours", "size"), sum_hours=("resolution_hours", "sum"),
                  n_within_24h=("within_24h", "sum"))
             .reset_index())
    agg.to_csv(AGG, index=False)
    print(f"  저장: {AGG.relative_to(ROOT)}  ({len(agg):,}행)")
    coord = df[df["has_coord"]]
    sample = coord.sample(min(MAP_N, len(coord)), random_state=SEED)
    sample = sample[["date", "borough", "agency", "complaint_type", "type_group", "latitude", "longitude",
                     "resolution_hours"]].round({"latitude": 5, "longitude": 5, "resolution_hours": 2})
    sample.to_csv(MAP_SAMPLE, index=False)
    print(f"  저장: {MAP_SAMPLE.relative_to(ROOT)}  (좌표 있는 {len(coord):,}행 중 무작위 {len(sample):,}행, seed={SEED})")

    print("[6] 그래프")
    figures(df, daily, setup_style())


if __name__ == "__main__":
    main()
