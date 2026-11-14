"""2단계: 분포 진단 및 데이터 변환 (Resolution_Hours).

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력)
출력:  outputs/tables/stat2_normality_tests.csv    변환별 왜도·첨도, Shapiro-Wilk, Kolmogorov-Smirnov
       outputs/tables/stat2_hist_bins.csv          히스토그램 구간 (대시보드용)
       outputs/tables/stat2_qq_points.csv          Q-Q 좌표 (대시보드용)
       outputs/figures/stat2_01_hist_qq.png

방법 메모:
  - Shapiro-Wilk 는 n > 5000 에서 p 값 근사가 부정확하므로 무작위 5,000건 표본에 적용한다.
  - K-S 는 전체 자료에 적용하되, 평균·표준편차를 자료에서 추정해 넣었으므로 p 값은 참고용이다
    (엄밀하게는 Lilliefors 보정이 필요하며 보정 없는 p 값은 실제보다 크게 나온다).
  - 표본이 매우 크면 사소한 이탈도 유의하게 나오므로, 검정 결과보다 왜도·첨도와 Q-Q 그림을 우선해 판단한다.
  - Box-Cox 는 양수만 받으므로 (시간 + 1) 에 적용한다. lambda=0 이면 log1p 와 같다.

실행: 프로젝트 루트에서  python analysis/stat_02_distribution.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from stat_common import ACCENT, GRID, MUTED, SEED, SINGLE, load_clean, p_text, save_fig, save_table, setup_style

SHAPIRO_N = 5_000
LAMBDA_FIT_N = 200_000
QQ_POINTS = 199
BINS = 60


def transforms(x):
    rng = np.random.default_rng(SEED)
    fit = rng.choice(x, size=min(LAMBDA_FIT_N, len(x)), replace=False)
    _, lam = stats.boxcox(fit + 1)                       # lambda 는 표본으로 추정 (최대우도)
    return {
        "원자료 (시간)": x,
        "로그 변환 log1p": np.log1p(x),
        f"Box-Cox (λ={lam:.3f})": stats.boxcox(x + 1, lmbda=lam),
    }, lam


def diagnose(name, v, rng):
    sample = rng.choice(v, size=min(SHAPIRO_N, len(v)), replace=False)
    w, p_sw = stats.shapiro(sample)
    d, p_ks = stats.kstest(v, stats.norm(loc=v.mean(), scale=v.std(ddof=1)).cdf)
    return {
        "변환": name, "n": len(v), "평균": v.mean(), "중앙값": np.median(v), "표준편차": v.std(ddof=1),
        "왜도": stats.skew(v), "첨도(초과)": stats.kurtosis(v),
        f"Shapiro-Wilk W (표본 {len(sample):,})": w, "Shapiro-Wilk p": p_sw,
        "K-S D (전체)": d, "K-S p": p_ks,
        "정규성 (α=0.05)": "기각" if max(p_sw, p_ks) < 0.05 else "기각 못함",
    }


def qq_points(v):
    probs = np.linspace(0.005, 0.995, QQ_POINTS)
    z = (np.quantile(v, probs) - v.mean()) / v.std(ddof=1)
    return stats.norm.ppf(probs), z


def main():
    df = load_clean()
    x = df["resolution_hours"].to_numpy(dtype=float)
    data, lam = transforms(x)
    rng = np.random.default_rng(SEED)

    print("[1] 정규성 검정")
    tests = pd.DataFrame([diagnose(k, v, rng) for k, v in data.items()])
    base = abs(tests.loc[0, "왜도"])
    tests["왜도 개선(|왜도| 감소율 %)"] = ((base - tests["왜도"].abs()) / base * 100).round(1)
    save_table(tests.round(4), "stat2_normality_tests.csv")
    for _, r in tests.iterrows():
        print(f"  {r['변환']:<22} 왜도 {r['왜도']:>8.3f}  첨도 {r['첨도(초과)']:>9.3f}  "
              f"SW p {p_text(r['Shapiro-Wilk p'])}  KS D {r['K-S D (전체)']:.3f}  -> 정규성 {r['정규성 (α=0.05)']}")

    print("[2] 히스토그램·Q-Q 좌표")
    hist_rows, qq_rows = [], []
    for name, v in data.items():
        # 원자료는 꼬리가 너무 길어 P99 까지만 구간을 나눈다 (그 위는 마지막 구간에 넣지 않고 따로 표기)
        hi = np.quantile(v, 0.99) if name.startswith("원자료") else v.max()
        counts, edges = np.histogram(v[v <= hi], bins=BINS)
        hist_rows += [{"변환": name, "구간시작": a, "구간끝": b, "건수": c}
                      for a, b, c in zip(edges[:-1], edges[1:], counts)]
        tq, sq = qq_points(v)
        qq_rows += [{"변환": name, "이론분위수": a, "표본분위수(표준화)": b} for a, b in zip(tq, sq)]
    save_table(pd.DataFrame(hist_rows).round(4), "stat2_hist_bins.csv")
    save_table(pd.DataFrame(qq_rows).round(4), "stat2_qq_points.csv")

    print("[3] 그래프")
    plt = setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for (name, v), (ax_h, ax_q), (_, r) in zip(data.items(), axes, tests.iterrows()):
        raw = name.startswith("원자료")
        hi = np.quantile(v, 0.99) if raw else v.max()
        ax_h.hist(v[v <= hi], bins=BINS, color=SINGLE, edgecolor="white", linewidth=0.3)
        ax_h.set_title(f"{name}   왜도 {r['왜도']:.2f} · 첨도 {r['첨도(초과)']:.2f}")
        ax_h.set_ylabel("건수")
        ax_h.yaxis.set_major_formatter(lambda t, _: f"{t:,.0f}")
        ax_h.yaxis.grid(True, color=GRID)
        ax_h.set_axisbelow(True)
        if raw:
            ax_h.set_xlabel(f"처리 시간 (시간) · P99={hi:,.0f}시간까지 표시, 최댓값 {v.max():,.0f}시간", color=MUTED)
        tq, sq = qq_points(v)
        lim = [min(tq.min(), sq.min()) - 0.2, max(tq.max(), sq.max()) + 0.2]
        ax_q.plot(lim, lim, color=ACCENT, lw=1.2, label="정규분포일 때")
        ax_q.scatter(tq, sq, s=9, color=SINGLE, zorder=3)
        ax_q.set_title("Q-Q Plot")
        ax_q.set_xlabel("이론 분위수 (표준정규)")
        ax_q.set_ylabel("표본 분위수 (표준화)")
        ax_q.set_xlim(tq.min() - 0.2, tq.max() + 0.2)
        ax_q.set_ylim(lim)
    axes[0, 1].legend(loc="upper left")
    fig.suptitle("Resolution_Hours 분포: 변환 전후 비교", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "stat2_01_hist_qq.png")

    best = tests.loc[tests["왜도"].abs().idxmin()]
    print(f"[요약] 원자료 왜도 {tests.loc[0, '왜도']:.2f} -> log1p {tests.loc[1, '왜도']:.2f}, "
          f"Box-Cox(λ={lam:.3f}) {tests.loc[2, '왜도']:.2f}.  |왜도| 최소: {best['변환']}")


if __name__ == "__main__":
    main()
