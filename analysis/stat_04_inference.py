"""4단계: 추론통계 및 가설검정.

입력:  data/processed/nyc311_all_clean.parquet, data/processed/nyc311_all_daily.csv
       (stat_01_clean_descriptive.py 출력)
출력:  outputs/tables/stat4_*.csv, outputs/figures/stat4_*.png

(a) 집단 간 차이: 자치구별 처리 시간
    자치구별 정규성(Shapiro-Wilk, 로그 변환값, 표본 5,000건)과 등분산(Levene)을 먼저 확인하고,
    하나라도 어긋나면 Kruskal-Wallis + Dunn 사후검정(Bonferroni 보정)을 쓴다. 모두 만족하면 일원 ANOVA.
    선택되지 않은 쪽 검정도 참고용으로 표에 남긴다.
    표본이 매우 커서 p 값은 거의 항상 유의하게 나오므로 효과크기(ε², η²)와 중앙값 차이를 함께 본다.
(b) 범주형 독립성: 자치구 x 민원 유형 상위 10개, 카이제곱 검정 + Cramér's V + 수정 표준화 잔차.
(c) 상관: 일별 접수 건수와 기온·강수·주말 여부의 Pearson / Spearman 상관.
(d) 회귀: 처리 시간 예측.
    - 다중선형회귀(OLS): 목표 log1p(Resolution_Hours), 이분산에 강건한 표준오차(HC1)
    - 로지스틱 회귀: 목표 24시간 내 처리 여부
    설명변수는 모두 접수 시점에 알 수 있는 정보만 쓴다: 자치구, 기관(상위 8 + 기타), 접수 시간대, 주말 여부.
    민원 유형은 기관에 거의 종속(유형마다 담당 기관이 정해짐)되어 다중공선성이 심하므로 넣지 않았다.
    시간 순 분할: 8월 31일까지 접수분으로 적합하고 9월 접수분으로 평가한다 (미래 정보 누출 방지).
    적합은 학습 구간에서 무작위 300,000건 이하 표본으로 한다.
    한계: 종결된 민원만 포함한다 (미종결은 1단계에서 제외).

실행: 프로젝트 루트에서  python analysis/stat_04_inference.py
"""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

from stat_common import (
    ACCENT, BOROUGH_COLOR, BOROUGH_KO, BOROUGHS, DAILY, GRID, INK, MUTED, OTHER_GRAY, ROOT, SEED, SINGLE,
    TIME_BANDS, load_clean, p_text, save_fig, save_table, setup_style,
)

ALPHA = 0.05
SHAPIRO_N = 5_000
TOP_TYPES = 10
TOP_AGENCIES = 8
SPLIT_DATE = "2024-09-01"
FIT_N = 300_000


# ---------------------------------------------------------------- (a) 집단 간 차이
def dunn_posthoc(values, groups, order):
    """Dunn 사후검정 (동순위 보정, Bonferroni). 전체 순위의 평균 순위 차이를 z 검정한다."""
    ranks = stats.rankdata(values)
    n_total = len(values)
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = (tie_counts ** 3 - tie_counts).sum() / (12 * (n_total - 1))
    base_var = n_total * (n_total + 1) / 12 - tie_term
    r = pd.Series(ranks).groupby(groups.to_numpy()).agg(["mean", "size"])
    med = pd.Series(values).groupby(groups.to_numpy()).median()
    pairs = list(combinations(order, 2))
    rows = []
    for a, b in pairs:
        zval = (r.loc[a, "mean"] - r.loc[b, "mean"]) / np.sqrt(base_var * (1 / r.loc[a, "size"] + 1 / r.loc[b, "size"]))
        p = 2 * stats.norm.sf(abs(zval))
        p_adj = min(p * len(pairs), 1.0)
        rows.append({"집단 A": a, "집단 B": b, "중앙값 A(h)": med[a], "중앙값 B(h)": med[b],
                     "중앙값 차이 A-B(h)": med[a] - med[b], "평균순위 차이": r.loc[a, "mean"] - r.loc[b, "mean"],
                     "z": zval, "p": p, "p (Bonferroni)": p_adj, "유의 (α=0.05)": "예" if p_adj < ALPHA else "아니오"})
    return pd.DataFrame(rows)


def group_difference(df):
    rng = np.random.default_rng(SEED)
    samples = {b: df.loc[df["borough"] == b, "resolution_hours"].to_numpy() for b in BOROUGHS}
    logs = {b: np.log1p(v) for b, v in samples.items()}
    rows, normal_ok = [], True
    for b in BOROUGHS:
        s = rng.choice(logs[b], size=min(SHAPIRO_N, len(logs[b])), replace=False)
        w, p = stats.shapiro(s)
        normal_ok &= p >= ALPHA
        rows.append({"검정": f"Shapiro-Wilk (log1p, {b}, 표본 {len(s):,})", "통계량": w, "자유도": "", "p": p,
                     "효과크기": "", "판정": "정규성 기각" if p < ALPHA else "정규성 기각 못함"})
    lev, p_lev = stats.levene(*logs.values(), center="median")
    equal_var = p_lev >= ALPHA
    rows.append({"검정": "Levene 등분산 (log1p, 중앙값 기준)", "통계량": lev, "자유도": "", "p": p_lev,
                 "효과크기": "", "판정": "등분산 기각" if not equal_var else "등분산 기각 못함"})

    n, k = len(df), len(BOROUGHS)
    h, p_kw = stats.kruskal(*samples.values())
    eps2 = (h - k + 1) / (n - k)
    f, p_f = stats.f_oneway(*logs.values())
    eta2 = f * (k - 1) / (f * (k - 1) + (n - k))
    use_anova = normal_ok and equal_var
    rows.append({"검정": "Kruskal-Wallis H (원자료)", "통계량": h, "자유도": k - 1, "p": p_kw,
                 "효과크기": f"ε² = {eps2:.4f}", "판정": ("참고용" if use_anova else "채택 검정") +
                 (": 자치구 간 차이 유의" if p_kw < ALPHA else ": 차이 유의하지 않음")})
    rows.append({"검정": "일원 ANOVA F (log1p)", "통계량": f, "자유도": f"{k - 1}, {n - k}", "p": p_f,
                 "효과크기": f"η² = {eta2:.4f}", "판정": ("채택 검정" if use_anova else "참고용 (가정 불충족)") +
                 (": 자치구 간 차이 유의" if p_f < ALPHA else ": 차이 유의하지 않음")})
    tests = pd.DataFrame(rows)
    save_table(tests, "stat4_group_tests.csv")
    print(f"  정규성 {'충족' if normal_ok else '불충족'}, 등분산 {'충족' if equal_var else '불충족'} "
          f"-> {'일원 ANOVA' if use_anova else 'Kruskal-Wallis'} 채택")
    print(f"  Kruskal-Wallis H={h:,.1f}, p {p_text(p_kw)}, ε²={eps2:.4f} | ANOVA(log) F={f:,.1f}, η²={eta2:.4f}")

    if use_anova:
        res = stats.tukey_hsd(*logs.values())
        post = pd.DataFrame([{"집단 A": BOROUGHS[i], "집단 B": BOROUGHS[j],
                              "평균 차이 A-B (log1p)": res.statistic[i, j], "p (Tukey HSD)": res.pvalue[i, j],
                              "유의 (α=0.05)": "예" if res.pvalue[i, j] < ALPHA else "아니오"}
                             for i, j in combinations(range(k), 2)])
        save_table(post.round(5), "stat4_posthoc_tukey.csv")
    else:
        post = dunn_posthoc(df["resolution_hours"].to_numpy(), df["borough"], BOROUGHS)
        save_table(post.round(5), "stat4_posthoc_dunn.csv")
        print(f"  Dunn 사후검정: {len(post)}쌍 중 {(post['유의 (α=0.05)'] == '예').sum()}쌍 유의")

    q = (df.groupby("borough")["resolution_hours"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
           .reindex(BOROUGHS))
    q.columns = ["P5", "Q1", "중앙값", "Q3", "P95"]
    q.insert(0, "n", df["borough"].value_counts().reindex(BOROUGHS))
    q["평균"] = df.groupby("borough")["resolution_hours"].mean()
    save_table(q.round(3).rename_axis("borough").reset_index(), "stat4_borough_quantiles.csv")
    return q


# ---------------------------------------------------------------- (b) 독립성
def independence(df):
    top = df["complaint_type"].value_counts().index[:TOP_TYPES]
    sub = df[df["complaint_type"].isin(top)]
    ct = pd.crosstab(sub["borough"], sub["complaint_type"]).reindex(index=BOROUGHS, columns=top)
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    row_p = ct.sum(axis=1).to_numpy()[:, None] / n
    col_p = ct.sum(axis=0).to_numpy()[None, :] / n
    resid = (ct - expected) / np.sqrt(expected * (1 - row_p) * (1 - col_p))
    summary = pd.DataFrame([{
        "검정": f"카이제곱 독립성 (자치구 x 민원 유형 상위 {TOP_TYPES})", "n": n, "카이제곱": chi2, "자유도": dof, "p": p,
        "Cramér's V": v, "기대빈도 최솟값": expected.min(),
        "분석 대상 비율(%)": len(sub) / len(df) * 100,
        "판정": "독립 아님 (연관 있음)" if p < ALPHA else "독립 가정 기각 못함",
    }])
    save_table(summary.round(4), "stat4_chi2_summary.csv")
    save_table(ct.rename_axis("borough").reset_index(), "stat4_chi2_crosstab.csv")
    save_table((ct.div(ct.sum(axis=1), axis=0) * 100).round(2).rename_axis("borough").reset_index(),
               "stat4_chi2_row_pct.csv")
    save_table(resid.round(2).rename_axis("borough").reset_index(), "stat4_chi2_residuals.csv")
    print(f"  χ²={chi2:,.1f}, df={dof}, p {p_text(p)}, Cramér's V={v:.3f} "
          f"(상위 {TOP_TYPES}개 유형이 전체의 {len(sub) / len(df):.1%})")
    return resid


# ---------------------------------------------------------------- (c) 상관
def correlations(daily):
    labels = {"temp_mean_c": "일평균 기온(°C)", "precip_mm": "강수량(mm)", "is_weekend": "주말 여부(0/1)"}
    rows = []
    for col, label in labels.items():
        if col not in daily.columns:
            continue
        d = daily[[col, "n_complaints"]].dropna()
        r, p_r = stats.pearsonr(d[col], d["n_complaints"])
        rho, p_s = stats.spearmanr(d[col], d["n_complaints"])
        rows.append({"변수": label, "n(일)": len(d), "Pearson r": r, "Pearson p": p_r, "Spearman ρ": rho,
                     "Spearman p": p_s, "판정 (α=0.05)": "유의" if max(p_r, p_s) < ALPHA else
                     ("일부만 유의" if min(p_r, p_s) < ALPHA else "유의하지 않음")})
    out = pd.DataFrame(rows)
    save_table(out.round(4), "stat4_correlation.csv")
    print(out.round(3).to_string(index=False))
    if "temp_mean_c" not in daily.columns:
        print("  기온 파일이 없어 주말 여부만 분석했다 (주말 여부와의 Pearson r 은 점이연 상관계수).")
    return out


# ---------------------------------------------------------------- (d) 회귀
def design_matrix(d, agency_levels):
    """기준 범주: 자치구=건수 최다, 기관=건수 최다, 시간대=오전(06-11). 주말은 0/1."""
    parts = []
    for col, levels in [("borough", None), ("agency_group", agency_levels), ("time_band", None)]:
        lv = levels if levels is not None else REF_ORDER[col]
        cat = pd.Categorical(d[col], categories=lv)
        parts.append(pd.get_dummies(cat, prefix=col, prefix_sep="=", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X.index = d.index
    X["is_weekend"] = d["is_weekend"].astype(float)
    return sm.add_constant(X, has_constant="add")


REF_ORDER = {}


def regression(df):
    df = df.copy()
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("시간 순 분할 결과 학습 또는 평가 구간이 비었습니다.")
    # 범주 묶음과 기준 범주는 학습 구간만 보고 정한다
    top_ag = train["agency"].value_counts().index[:TOP_AGENCIES]
    df["agency_group"] = df["agency"].where(df["agency"].isin(top_ag), "기타")
    train, test = df[df["date"] < SPLIT_DATE], df[df["date"] >= SPLIT_DATE]
    REF_ORDER["borough"] = list(train["borough"].value_counts().index)
    REF_ORDER["time_band"] = [TIME_BANDS[1]] + [t for t in TIME_BANDS if t != TIME_BANDS[1]]
    agency_levels = list(train["agency_group"].value_counts().index)

    fit = train.sample(min(FIT_N, len(train)), random_state=SEED)
    ev = test.sample(min(FIT_N, len(test)), random_state=SEED)
    X, Xt = design_matrix(fit, agency_levels), design_matrix(ev, agency_levels)
    print(f"  학습 {len(train):,}건 중 표본 {len(fit):,}건 적합, 평가(9월 접수) {len(test):,}건 중 {len(ev):,}건")
    print(f"  기준 범주: 자치구={REF_ORDER['borough'][0]}, 기관={agency_levels[0]}, 시간대={REF_ORDER['time_band'][0]}")

    # OLS: log1p(시간)
    ols = sm.OLS(fit["log_resolution"], X).fit(cov_type="HC1")
    pred = ols.predict(Xt)
    sst = ((ev["log_resolution"] - fit["log_resolution"].mean()) ** 2).sum()
    r2_test = 1 - ((ev["log_resolution"] - pred) ** 2).sum() / sst
    ci = ols.conf_int()
    ols_t = pd.DataFrame({
        "변수": ols.params.index, "계수": ols.params.values, "표준오차(HC1)": ols.bse.values, "z": ols.tvalues.values,
        "p": ols.pvalues.values, "CI 2.5%": ci[0].values, "CI 97.5%": ci[1].values,
        "exp(계수): (시간+1) 배율": np.exp(ols.params.values),
        "유의 (α=0.05)": np.where(ols.pvalues.values < ALPHA, "예", "아니오"),
    })
    save_table(ols_t.round(5), "stat4_ols_coef.csv")

    # 로지스틱: 24시간 내 처리 여부
    logit = sm.Logit(fit["within_24h"], X).fit(disp=0, maxiter=200)
    prob = logit.predict(Xt)
    auc = roc_auc_score(ev["within_24h"], prob) if ev["within_24h"].nunique() == 2 else np.nan
    acc = accuracy_score(ev["within_24h"], (prob >= 0.5).astype(int))
    base = max(ev["within_24h"].mean(), 1 - ev["within_24h"].mean())
    ci = logit.conf_int()
    logit_t = pd.DataFrame({
        "변수": logit.params.index, "계수(log-odds)": logit.params.values, "표준오차": logit.bse.values,
        "z": logit.tvalues.values, "p": logit.pvalues.values, "오즈비": np.exp(logit.params.values),
        "오즈비 CI 2.5%": np.exp(ci[0].values), "오즈비 CI 97.5%": np.exp(ci[1].values),
        "유의 (α=0.05)": np.where(logit.pvalues.values < ALPHA, "예", "아니오"),
    })
    save_table(logit_t.round(5), "stat4_logit_coef.csv")

    summary = pd.DataFrame([
        {"모델": "다중선형회귀 (OLS, HC1)", "목표변수": "log1p(Resolution_Hours)", "학습 n": len(fit), "평가 n": len(ev),
         "설명변수 수": X.shape[1] - 1, "R² (학습)": ols.rsquared, "수정 R²": ols.rsquared_adj,
         "모형 유의성 p": float(ols.f_pvalue), "R² (평가, 9월)": r2_test, "AUC (평가)": np.nan,
         "정확도 (평가)": np.nan, "다수 범주 비율 (평가)": np.nan},
        {"모델": "로지스틱 회귀", "목표변수": "24시간 내 처리 여부", "학습 n": len(fit), "평가 n": len(ev),
         "설명변수 수": X.shape[1] - 1, "R² (학습)": logit.prsquared, "수정 R²": np.nan,
         "모형 유의성 p": float(logit.llr_pvalue), "R² (평가, 9월)": np.nan, "AUC (평가)": auc,
         "정확도 (평가)": acc, "다수 범주 비율 (평가)": base},
    ])
    save_table(summary.round(4), "stat4_model_summary.csv")
    print(f"  OLS: R²={ols.rsquared:.3f} (수정 {ols.rsquared_adj:.3f}), 평가 R²={r2_test:.3f}, F검정 p {p_text(ols.f_pvalue)}")
    print(f"  Logit: McFadden pseudo R²={logit.prsquared:.3f}, LLR p {p_text(logit.llr_pvalue)}, "
          f"평가 AUC={auc:.3f}, 정확도={acc:.3f} (다수 범주만 찍으면 {base:.3f})")
    return logit_t


# ---------------------------------------------------------------- 그래프
def figures(plt, q, resid, daily, logit_t):
    # 자치구별 처리시간 상자 (P5-P95 수염, 로그 축)
    fig, ax = plt.subplots(figsize=(8, 4))
    boxes = [{"label": BOROUGH_KO[b], "whislo": np.log1p(r["P5"]), "q1": np.log1p(r["Q1"]), "med": np.log1p(r["중앙값"]),
              "q3": np.log1p(r["Q3"]), "whishi": np.log1p(r["P95"]), "fliers": []} for b, r in q.iterrows()]
    bp = ax.bxp(boxes, orientation="horizontal", patch_artist=True, widths=0.55, showfliers=False,
                medianprops={"color": INK, "lw": 1.6}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, b in zip(bp["boxes"], q.index):
        patch.set(facecolor=BOROUGH_COLOR[b], edgecolor="none", alpha=0.85)
    ticks = [0, 1, 6, 24, 72, 168, 720, 2160]
    ax.set_xticks(np.log1p(ticks), ["0", "1h", "6h", "1일", "3일", "1주", "30일", "90일"])
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_title("자치구별 처리 시간 분포 (로그 축, 상자 Q1-Q3, 수염 P5-P95)")
    save_fig(fig, "stat4_01_box_borough.png")

    # 카이제곱 잔차
    fig, ax = plt.subplots(figsize=(10, 3.8))
    vals = resid.to_numpy()
    lim = np.abs(vals).max()
    ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(resid.shape[1]), resid.columns, rotation=30, ha="right")
    ax.set_yticks(range(resid.shape[0]), [BOROUGH_KO[b] for b in resid.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:+.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(vals[i, j]) > lim * 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("자치구 x 민원 유형: 수정 표준화 잔차 (빨강=기대보다 많음, 파랑=적음)")
    save_fig(fig, "stat4_02_chi2_residuals.png")

    # 일별 건수와 기온 / 주말
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if "temp_mean_c" in daily.columns:
        for flag, label, color in [(0, "평일", SINGLE), (1, "주말", ACCENT)]:
            d = daily[daily["is_weekend"] == flag]
            ax.scatter(d["temp_mean_c"], d["n_complaints"], s=22, color=color, alpha=0.8, label=label)
        ax.set_xlabel("일평균 기온 (°C)")
        ax.legend(loc="upper left")
        ax.set_title("일평균 기온과 일별 민원 접수 건수")
    else:
        groups = [daily.loc[daily["is_weekend"] == f, "n_complaints"] for f in (0, 1)]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5, medianprops={"color": INK})
        for patch, color in zip(bp["boxes"], [SINGLE, ACCENT]):
            patch.set(facecolor=color, edgecolor="none", alpha=0.85)
        ax.set_xticks([1, 2], ["평일", "주말"])
        ax.set_title("평일·주말 일별 민원 접수 건수")
    ax.set_ylabel("일별 접수 건수")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save_fig(fig, "stat4_03_daily_corr.png")

    # 로지스틱 오즈비
    t = logit_t[logit_t["변수"] != "const"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.4))
    y = np.arange(len(t))
    colors = [SINGLE if s == "예" else OTHER_GRAY for s in t["유의 (α=0.05)"]]
    ax.hlines(y, t["오즈비 CI 2.5%"], t["오즈비 CI 97.5%"], color=colors, lw=1.6)
    ax.scatter(t["오즈비"], y, color=colors, s=26, zorder=3)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")      # 기본 로그 눈금의 지수 표기(유니코드 음수 기호) 회피
    ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.set_yticks(y, t["변수"])
    ax.set_xlabel("오즈비 (로그 축) · 1보다 크면 24시간 내 처리 가능성 높음 · 회색은 유의하지 않음")
    ax.set_title("로지스틱 회귀: 24시간 내 처리 오즈비와 95% 신뢰구간")
    save_fig(fig, "stat4_04_logit_odds_ratio.png")


def main():
    df = load_clean()
    if not DAILY.exists():
        raise SystemExit(f"{DAILY.relative_to(ROOT)} 가 없습니다. 먼저 analysis/stat_01_clean_descriptive.py 를 실행하세요.")
    daily = pd.read_csv(DAILY, parse_dates=["date"])

    print("[a] 자치구별 처리 시간 차이")
    q = group_difference(df)
    print("[b] 자치구 x 민원 유형 독립성")
    resid = independence(df)
    print("[c] 일별 접수 건수 상관")
    correlations(daily)
    print("[d] 회귀")
    logit_t = regression(df)
    print("[e] 그래프")
    figures(setup_style(), q, resid, daily, logit_t)


if __name__ == "__main__":
    main()
