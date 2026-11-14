"""3단계: 정규화 및 표준화 (자치구 지표).

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력)
출력:  outputs/tables/stat3_borough_indicators.csv     자치구별 원지표
       outputs/tables/stat3_borough_scaled.csv         같은 지표의 Z-score / Min-Max 값
       outputs/tables/stat3_efficiency_index.csv       민원 처리 효율성 종합 지수 (0~100)
       outputs/tables/stat3_outlier_borough_days.csv   |Z| > 3 인 자치구-날짜
       outputs/figures/stat3_*.png

두 스케일링의 차이:
  - 표준화(Z-score, StandardScaler): (x - 평균) / 표준편차. 평균 0, 표준편차 1. 범위가 정해져 있지 않고
    "평균에서 표준편차 몇 배 떨어졌는가"를 뜻하므로 이상치 탐지와 거리 기반 군집화에 쓴다.
  - 정규화(Min-Max, MinMaxScaler): (x - 최솟값) / (최댓값 - 최솟값). 항상 0~1. 최고·최저가 1·0 으로
    고정되므로 점수·지수로 읽기 쉽지만, 극단값 하나가 나머지를 한쪽으로 몰아넣는다.

이상치 탐지 단위에 대한 주의:
  자치구는 5개뿐이라 자치구 수준에서는 |Z| > 3 이 수학적으로 나올 수 없다
  (StandardScaler 는 모집단 표준편차(n 으로 나눔)를 쓰며, 이때 n개 값의 |Z| 최댓값은 sqrt(n-1) 이다.
   n=5 이면 2.00. 표본 표준편차(n-1 로 나눔)를 쓰면 (n-1)/sqrt(n) = 1.79 다.)
  그래서 자치구 수준 Z 는 비교용으로만 쓰고,
  이상치 탐지는 '자치구 x 날짜' 단위(자치구 안에서 표준화한 일별 건수, 일별 평균 로그 처리시간)로 한다.

효율성 지수 정의 (동일 가중 평균 x 100, 높을수록 빠르게 처리):
  - 중앙 처리시간 (낮을수록 좋음 -> 1 - MinMax)
  - P90 처리시간  (낮을수록 좋음 -> 1 - MinMax)   오래 걸리는 민원의 꼬리
  - 24시간 내 처리율 (높을수록 좋음 -> MinMax)
  평균 처리시간은 극단값에 끌려가므로 지수에는 넣지 않았다. 민원 건수와 인구 대비 비율은 '수요'이지
  '효율'이 아니므로 표에는 싣되 지수에는 넣지 않았다.
  한계: 자치구마다 민원 유형 구성이 다르다. 이 지수는 유형 구성을 보정하지 않은 단순 비교다.

실행: 프로젝트 루트에서  python analysis/stat_03_scaling.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from stat_common import (
    BOROUGH_KO, BOROUGH_POP, BOROUGHS, INK, MUTED, SINGLE, load_clean, save_fig, save_table, setup_style,
)

INDICATORS = {
    "n_complaints": "민원 건수",
    "per_1000_pop": "인구 1천 명당 건수",
    "mean_hours": "평균 처리시간(h)",
    "median_hours": "중앙 처리시간(h)",
    "p90_hours": "P90 처리시간(h)",
    "within_24h_rate": "24시간 내 처리율(%)",
}
# 지수 구성요소: (지표, 높을수록 좋은가)
INDEX_PARTS = [("median_hours", False), ("p90_hours", False), ("within_24h_rate", True)]


def borough_indicators(df):
    g = df.groupby("borough")["resolution_hours"]
    ind = pd.DataFrame({
        "n_complaints": g.size(),
        "mean_hours": g.mean(),
        "median_hours": g.median(),
        "p90_hours": g.quantile(0.90),
        "within_24h_rate": df.groupby("borough")["within_24h"].mean() * 100,
    }).reindex(BOROUGHS)
    ind["population_2020"] = pd.Series(BOROUGH_POP)
    ind["per_1000_pop"] = ind["n_complaints"] / ind["population_2020"] * 1000
    return ind[["population_2020", *INDICATORS]]


def outlier_days(df):
    d = (df.groupby(["borough", "date"])
           .agg(n_complaints=("resolution_hours", "size"), mean_log_hours=("log_resolution", "mean"))
           .reset_index())
    for col in ["n_complaints", "mean_log_hours"]:
        # 자치구마다 규모가 다르므로 자치구 안에서 표준화한다
        d[f"z_{col}"] = d.groupby("borough")[col].transform(
            lambda s: StandardScaler().fit_transform(s.to_numpy(dtype=float).reshape(-1, 1)).ravel())
    out = d[(d["z_n_complaints"].abs() > 3) | (d["z_mean_log_hours"].abs() > 3)].copy()
    out["weekday"] = out["date"].dt.day_name()
    out["mean_hours_geometric"] = np.expm1(out["mean_log_hours"])
    return d, out.sort_values(["date", "borough"])


def heatmap(ax, values, rows, cols, title, fmt, cmap, vmin, vmax):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    mid = (vmin + vmax) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            # 발산형(RdBu)은 양 끝이 어둡고, 순차형(Blues)은 값이 클수록 어둡다
            dark = abs(v - mid) > (vmax - vmin) * 0.3 if cmap == "RdBu_r" else (v - vmin) > (vmax - vmin) * 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9, color="white" if dark else INK)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def main():
    df = load_clean()

    print("[1] 자치구 지표")
    ind = borough_indicators(df)
    save_table(ind.round(3).rename_axis("borough").reset_index(), "stat3_borough_indicators.csv")
    print(ind.round(2).to_string())

    print("[2] 표준화 / 정규화")
    X = ind[list(INDICATORS)].to_numpy(dtype=float)
    z = pd.DataFrame(StandardScaler().fit_transform(X), index=ind.index, columns=[f"z_{c}" for c in INDICATORS])
    mm = pd.DataFrame(MinMaxScaler().fit_transform(X), index=ind.index, columns=[f"mm_{c}" for c in INDICATORS])
    scaled = pd.concat([z, mm], axis=1)
    save_table(scaled.round(4).rename_axis("borough").reset_index(), "stat3_borough_scaled.csv")
    n = len(ind)
    print(f"  자치구 수준 |Z| 최댓값: {np.abs(z.to_numpy()).max():.2f}  (n={n} 의 이론상 한계 sqrt(n-1) = {np.sqrt(n - 1):.2f} -> |Z|>3 불가)")

    print("[3] 이상치 탐지: 자치구 x 날짜 단위 |Z| > 3")
    units, out = outlier_days(df)
    cols = ["date", "weekday", "borough", "n_complaints", "z_n_complaints", "mean_hours_geometric", "z_mean_log_hours"]
    num = ["z_n_complaints", "mean_hours_geometric", "z_mean_log_hours"]
    save_table(out[cols].round(dict.fromkeys(num, 3)), "stat3_outlier_borough_days.csv")
    print(f"  단위 수 {len(units):,} 중 이상치 {len(out)}건 "
          f"(건수 기준 {(out['z_n_complaints'].abs() > 3).sum()}, 처리시간 기준 {(out['z_mean_log_hours'].abs() > 3).sum()})")
    print("  참고: 정규분포라면 |Z|>3 은 약 0.27%, 즉 이 단위 수에서 우연히 "
          f"{len(units) * 2 * 0.0027:.1f}건 정도가 기대된다 (두 지표 합).")

    print("[4] 민원 처리 효율성 종합 지수")
    parts = pd.DataFrame(index=ind.index)
    for col, higher_better in INDEX_PARTS:
        s = mm[f"mm_{col}"]
        parts[f"점수_{INDICATORS[col]}"] = s if higher_better else 1 - s
    parts["효율성 지수(0-100)"] = parts.mean(axis=1) * 100
    parts["순위"] = parts["효율성 지수(0-100)"].rank(ascending=False).astype(int)
    index_t = pd.concat([ind[[c for c, _ in INDEX_PARTS]], parts], axis=1).sort_values("순위")
    save_table(index_t.round(3).rename_axis("borough").reset_index(), "stat3_efficiency_index.csv")
    print(index_t.round(2).to_string())

    print("[5] 그래프")
    plt = setup_style()
    rows = [BOROUGH_KO[b] for b in ind.index]
    labels = list(INDICATORS.values())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    heatmap(axes[0], z.to_numpy(), rows, labels, "표준화 (Z-score): 평균 0, 표준편차 1", "{:+.2f}", "RdBu_r", -2, 2)
    heatmap(axes[1], mm.to_numpy(), rows, labels, "정규화 (Min-Max): 0 ~ 1", "{:.2f}", "Blues", 0, 1)
    axes[1].set_yticklabels([])
    fig.suptitle("같은 자치구 지표, 두 가지 스케일링", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "stat3_01_scaling_compare.png")

    s = index_t["효율성 지수(0-100)"][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([BOROUGH_KO[b] for b in s.index], s.values, color=SINGLE, height=0.62)
    for y, (b, v) in enumerate(s.items()):
        ax.text(v, y, f" {v:.1f}  (중앙 {ind.loc[b, 'median_hours']:.1f}h · 24h내 {ind.loc[b, 'within_24h_rate']:.0f}%)",
                va="center", fontsize=9, color=MUTED)
    ax.set_title("자치구별 민원 처리 효율성 종합 지수 (0-100, 높을수록 빠름)")
    ax.set_xlim(0, 150)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save_fig(fig, "stat3_02_efficiency_index.png")


if __name__ == "__main__":
    main()
