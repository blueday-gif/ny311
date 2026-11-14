"""7단계(추가 분석 3): 기관 간 비교 - 이원배치 분산분석, 단순 주효과, 혼합효과 모형.

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력, 종결된 민원)
출력:  outputs/tables/stat7_*.csv, outputs/figures/stat7_*.png
실행:  프로젝트 루트에서  python analysis/stat_07_agency_comparison.py

======================================================================================
[질문] 4단계에서 남은 의문
======================================================================================
4단계의 Kruskal-Wallis 검정은 "자치구에 따라 처리 시간이 다르다"고 했다. 하지만 이것만으로는
  (가) 같은 종류의 민원을 자치구마다 다른 속도로 처리하는 것인지,
  (나) 자치구마다 들어오는 민원의 종류가 달라서(오래 걸리는 유형이 많아서) 그렇게 보이는 것인지
구별할 수 없다. 4단계 카이제곱 검정에서 자치구와 민원 유형이 독립이 아니었으므로 (나)의 가능성이 실제로 있다.
이렇게 두 변수의 관계를 제3의 변수가 만들어 내거나 왜곡하는 것을 교란(confounding)이라고 한다.

이 스크립트는 요인을 하나 더 넣어 이 둘을 가려낸다.
  (1) 이원배치 분산분석: 자치구와 기관을 함께 넣고, 둘의 상호작용이 있는지 본다
  (2) 단순 주효과: 기관을 하나씩 고정하고 그 안에서 자치구 차이를 본다
  (3) 혼합효과 모형: 민원 유형(수백 개)을 임의효과로 통제했을 때 자치구 차이가 얼마나 남는지 본다

목표변수는 log1p(Resolution_Hours) 이다. 2단계에서 원자료는 심하게 치우쳐 있었고 로그 변환 후 훨씬
대칭에 가까워졌다. 로그 척도의 평균을 되돌린 값 exp(평균) - 1 은 기하평균에 해당하며 중앙값과 비슷하게 움직인다.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from stat_common import (
    BOROUGH_KO, BOROUGHS, CATEGORICAL, GRID, INK, MUTED, OTHER_GRAY, SEED, SINGLE, ACCENT,
    load_clean, p_text, save_fig, save_table, setup_style,
)

TOP_AGENCIES = 6
CELL_N = 2_000          # 이원배치 ANOVA: 칸(자치구 x 기관)마다 뽑는 표본 수
SHAPIRO_N = 5_000
MIXED_N = 40_000        # 혼합효과 모형 적합 표본 수
MIN_TYPE_N = 30         # 혼합효과 모형: 표본 안에서 이보다 적게 나온 민원 유형은 '기타'로 묶는다


# ============================================================================ (1) 이원배치 분산분석
def two_way_anova(df, agencies):
    """이원배치 분산분석 (two-way ANOVA).

    [복습] 일원배치 ANOVA 는 요인 하나(자치구)로 집단을 나눠 평균을 비교한다. 전체 변동(제곱합 SS)을
           '집단 간 변동'과 '집단 내 변동'으로 나누고, 그 비율(F)이 우연으로 보기엔 너무 큰지 본다.
    [확장] 이원배치는 요인이 두 개(A=자치구, B=기관)다. 전체 변동을 네 조각으로 나눈다.
             SS_전체 = SS_A + SS_B + SS_AxB + SS_오차
    [세 가지 가설]
      주효과 A:     기관을 평균했을 때 자치구 간 평균이 같은가
      주효과 B:     자치구를 평균했을 때 기관 간 평균이 같은가
      상호작용 AxB: 자치구의 효과가 기관에 따라 달라지는가
                    (예: A 기관은 맨해튼에서 빠른데 B 기관은 맨해튼에서 느리다면 상호작용이 있는 것)
    [해석 순서] 상호작용부터 본다. 상호작용이 크면 "자치구 효과가 얼마다"라고 한마디로 말할 수 없으므로
                주효과 대신 기관별로 나눠서(단순 주효과) 해석해야 한다.

    [효과크기]
      η² (에타제곱)       = SS_효과 / SS_전체              전체 변동 가운데 이 효과가 설명하는 비율
      편 η² (partial)    = SS_효과 / (SS_효과 + SS_오차)   다른 효과를 걷어낸 뒤의 비율
      관례적 기준(Cohen): 0.01 작음, 0.06 중간, 0.14 큼

    [균형 설계] 칸마다 표본 수가 같은 설계를 균형 설계(balanced design)라고 한다. 실제 자료는 칸마다 건수가
      수백 배씩 다르다(불균형). 불균형이면 요인들이 서로 얽혀 제곱합을 나누는 방법(Type I, II, III)에 따라
      결과가 달라진다. 여기서는 칸마다 같은 수를 무작위로 뽑아 균형 설계를 만든다. 그러면
        - 세 가지 제곱합이 모두 같아져 해석이 단순해지고
        - 등분산 가정이 어긋나도 F 검정이 잘 버티며
        - 주효과가 "칸 평균의 단순 평균"을 비교하게 되어, 건수가 많은 기관이 결과를 지배하지 않는다.
      대신 "모든 기관을 같은 비중으로 본" 비교가 된다는 점을 기억해야 한다. 실제 민원 구성을 반영한
      비교가 아니다.

    [가정] 오차의 정규성, 등분산성, 독립성. 아래에서 점검하고, 어긋날 때를 대비해 이분산에 강건한
           공분산(HC3)으로 계산한 F 검정도 함께 낸다.
    """
    sub = df[df["agency"].isin(agencies)]
    sizes = sub.groupby(["borough", "agency"]).size().unstack().reindex(index=BOROUGHS, columns=agencies)
    m = int(min(CELL_N, sizes.min().min()))
    print(f"  칸 {sizes.size}개, 칸별 건수 최소 {sizes.min().min():,.0f} ~ 최대 {sizes.max().max():,.0f} -> 칸마다 {m:,}건씩 무작위 추출")
    bal = (sub.groupby(["borough", "agency"]).sample(n=m, random_state=SEED)      # 칸마다 m건씩 층화 추출
              [["borough", "agency", "log_resolution"]].reset_index(drop=True))

    # C(...) 는 "범주형 변수로 다뤄라"는 뜻, * 는 "두 주효과와 상호작용을 모두 넣어라"는 뜻이다.
    model = smf.ols("log_resolution ~ C(borough) * C(agency)", data=bal).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_total = table["sum_sq"].sum()
    ss_err = table.loc["Residual", "sum_sq"]
    robust = sm.stats.anova_lm(model, typ=2, robust="hc3")
    names = {"C(borough)": "주효과: 자치구", "C(agency)": "주효과: 기관", "C(borough):C(agency)": "상호작용: 자치구 x 기관",
             "Residual": "오차"}
    out = pd.DataFrame({
        "변동 요인": [names[i] for i in table.index], "제곱합(SS)": table["sum_sq"].to_numpy(),
        "자유도": table["df"].to_numpy(), "평균제곱(MS)": (table["sum_sq"] / table["df"]).to_numpy(),
        "F": table["F"].to_numpy(), "p": table["PR(>F)"].to_numpy(),
        "η²": (table["sum_sq"] / ss_total).to_numpy(),
        "편 η²": (table["sum_sq"] / (table["sum_sq"] + ss_err)).to_numpy(),
        "F (HC3 강건)": robust["F"].to_numpy(), "p (HC3 강건)": robust["PR(>F)"].to_numpy(),
    })
    out.loc[out["변동 요인"] == "오차", ["편 η²"]] = np.nan

    # 가정 점검: 잔차의 정규성(Shapiro-Wilk, 표본)과 칸 간 등분산(Levene)
    rng = np.random.default_rng(SEED)
    w, p_sw = stats.shapiro(rng.choice(model.resid.to_numpy(), size=min(SHAPIRO_N, len(bal)), replace=False))
    groups = [g["log_resolution"].to_numpy() for _, g in bal.groupby(["borough", "agency"])]
    lev, p_lev = stats.levene(*groups, center="median")
    sd = bal.groupby(["borough", "agency"])["log_resolution"].std()
    assume = pd.DataFrame([
        {"점검": "잔차 정규성: Shapiro-Wilk", "통계량": w, "p": p_sw, "판정": "기각" if p_sw < 0.05 else "기각 못함",
         "비고": f"잔차 왜도 {stats.skew(model.resid):.2f}, 초과첨도 {stats.kurtosis(model.resid):.2f}"},
        {"점검": "등분산: Levene (중앙값 기준)", "통계량": lev, "p": p_lev, "판정": "기각" if p_lev < 0.05 else "기각 못함",
         "비고": f"칸별 표준편차 최대/최소 = {sd.max() / sd.min():.2f} (2 이하면 대체로 무난)"},
    ])
    return out, assume, bal, m


def cell_summary(df, agencies):
    """칸(자치구 x 기관)별 기술통계. 전체 자료 기준이며, 그래프와 해석의 바탕이 된다."""
    sub = df[df["agency"].isin(agencies)]
    g = sub.groupby(["agency", "borough"])
    out = pd.DataFrame({
        "n": g.size(), "중앙값(h)": g["resolution_hours"].median(),
        "기하평균(h)": np.expm1(g["log_resolution"].mean()),       # 로그 평균을 원래 단위로 되돌린 값
        "24시간 내 처리율(%)": g["within_24h"].mean() * 100,
    }).reset_index()
    return out


# ============================================================================ (2) 단순 주효과
def simple_effects(df, agencies):
    """단순 주효과 (simple main effects): 기관을 하나로 고정하고 그 안에서 자치구 차이를 검정한다.

    상호작용이 있으면 "자치구 효과"는 기관마다 다르므로 기관별로 따로 봐야 한다.
    여기서는 전체 자료를 쓰고, 분포 가정이 필요 없는 Kruskal-Wallis 검정을 기관마다 한 번씩 한다.

    [다중비교 문제] 검정을 여러 번 하면 실제로는 차이가 없어도 우연히 유의한 결과가 하나쯤 나올 확률이 커진다.
      유의수준 0.05 로 6번 검정하면 적어도 한 번 잘못 기각할 확률은 1 - 0.95^6 ≈ 26% 다.
    [Bonferroni 보정] p 값에 검정 횟수를 곱해(최대 1) 전체 오류율을 0.05 이하로 묶는다. 단순하고 보수적이다.

    [효과크기] ε² (엡실론제곱) = (H - k + 1) / (n - k).  Kruskal-Wallis 판 η² 로, 순위 변동 가운데 집단이
      설명하는 비율이다. 표본이 수십만이면 p 는 거의 항상 0 에 가까우므로 이 값과 중앙값 격차로 판단한다.
    """
    rows = []
    for a in agencies:
        sub = df[df["agency"] == a]
        groups = [sub.loc[sub["borough"] == b, "resolution_hours"].to_numpy() for b in BOROUGHS]
        h, p = stats.kruskal(*groups)
        n, k = len(sub), len(BOROUGHS)
        med = sub.groupby("borough")["resolution_hours"].median().reindex(BOROUGHS)
        rows.append({
            "기관": a, "n": n, "Kruskal-Wallis H": h, "p": p, "p (Bonferroni)": min(p * len(agencies), 1.0),
            "ε²": (h - k + 1) / (n - k),
            "가장 빠른 자치구": BOROUGH_KO[med.idxmin()], "중앙값 최소(h)": med.min(),
            "가장 느린 자치구": BOROUGH_KO[med.idxmax()], "중앙값 최대(h)": med.max(),
            "최대/최소 배율": med.max() / med.min() if med.min() > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out["유의 (α=0.05, 보정 후)"] = np.where(out["p (Bonferroni)"] < 0.05, "예", "아니오")
    return out


# ============================================================================ (3) 혼합효과 모형
def mixed_model(df):
    """혼합효과 모형 (mixed-effects model, 다층모형·위계적 선형모형이라고도 한다).

    [자료 구조] 민원은 '민원 유형'이라는 묶음 안에 들어 있다. 같은 유형의 민원은 처리 절차가 같아서 처리 시간이
      서로 비슷하다. 즉 관측값들이 독립이 아니다. 이런 구조를 군집 자료(clustered data)라고 한다.

    [고정효과와 임의효과]
      고정효과(fixed effect):  수준 하나하나에 관심이 있고 수준의 수가 적은 요인. 여기서는 자치구(5개).
                              "맨해튼은 브루클린보다 얼마나 빠른가"처럼 수준별 계수를 추정한다.
      임의효과(random effect): 수준이 많고, 각 수준이 더 큰 모집단에서 뽑힌 표본처럼 취급되는 요인.
                              여기서는 민원 유형(수백 개). 유형마다 계수를 하나씩 추정하는 대신
                              "유형별 평균이 얼마나 퍼져 있는가(분산)" 하나만 추정한다.
      민원 유형을 가변수 수백 개로 회귀에 넣을 수도 있지만, 건수가 적은 유형의 계수는 매우 불안정해진다.
      임의효과는 그런 유형의 추정값을 전체 평균 쪽으로 끌어당겨(축소, shrinkage) 안정시킨다.

    [모형: 임의절편 모형]
        log 처리시간_ij = b0 + b·자치구_ij + u_j + e_ij
        u_j ~ N(0, σ_u²): 유형 j 의 고유한 높낮이 (유형 간 분산)
        e_ij ~ N(0, σ_e²): 같은 유형 안에서의 개별 민원 차이 (유형 내 분산)

    [급내상관계수 ICC] = σ_u² / (σ_u² + σ_e²)
      전체 분산 가운데 '유형이 다르기 때문에' 생기는 부분의 비율. 같은 유형에 속한 두 민원의 상관계수이기도 하다.
      0 에 가까우면 유형을 고려할 필요가 없고, 클수록 유형을 무시한 분석(독립 가정)이 위험하다.

    [이 분석의 핵심 비교]
      모형 A (OLS, 유형 무시):      자치구 계수 = 민원 구성 차이까지 섞인 '겉보기' 격차
      모형 B (혼합모형, 유형 통제): 자치구 계수 = 같은 유형끼리 비교한 격차
      B 에서 계수가 0 쪽으로 줄면, 겉보기 격차의 상당 부분이 민원 구성 차이(교란)였다는 뜻이다.
      반대로 B 에서 격차가 더 커지거나 부호가 뒤집힐 수도 있다. 민원 구성 차이가 실제 격차를 가리거나
      반대 방향으로 보이게 만들고 있었다는 뜻이며, 전체로 본 방향과 집단 안에서 본 방향이 반대가 되는
      이 현상을 심슨의 역설(Simpson's paradox)이라고 한다.

    [추정] REML(제한최대가능도). 분산 성분을 추정할 때 일반 최대가능도보다 치우침이 작다.
    """
    s = df.sample(min(MIXED_N, len(df)), random_state=SEED)[["borough", "complaint_type", "log_resolution"]].copy()
    counts = s["complaint_type"].value_counts()
    s["type_group"] = s["complaint_type"].where(s["complaint_type"].map(counts) >= MIN_TYPE_N, "기타(희소 유형)")
    ref = s["borough"].value_counts().index[0]                   # 기준 범주: 건수가 가장 많은 자치구
    formula = f"log_resolution ~ C(borough, Treatment(reference='{ref}'))"

    ols = smf.ols(formula, data=s).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null = smf.mixedlm("log_resolution ~ 1", data=s, groups=s["type_group"]).fit(reml=True)
        mixed = smf.mixedlm(formula, data=s, groups=s["type_group"]).fit(reml=True)

    def icc(res):
        var_u = float(res.cov_re.iloc[0, 0])
        return var_u, float(res.scale), var_u / (var_u + res.scale)

    v0, e0, icc0 = icc(null)
    v1, e1, icc1 = icc(mixed)
    variance = pd.DataFrame([
        {"모형": "영모형 (절편 + 유형 임의효과)", "유형 간 분산 σ_u²": v0, "유형 내 분산 σ_e²": e0, "ICC": icc0},
        {"모형": "자치구 고정효과 추가", "유형 간 분산 σ_u²": v1, "유형 내 분산 σ_e²": e1, "ICC": icc1},
    ])

    rows = []
    for name in ols.params.index:
        if name == "Intercept":
            continue
        b = name.split("[T.")[1].rstrip("]")
        ci_o, ci_m = ols.conf_int().loc[name], mixed.conf_int().loc[name]
        rows.append({
            "자치구": BOROUGH_KO[b], "기준": BOROUGH_KO[ref],
            "A. OLS 계수 (유형 무시)": ols.params[name], "A CI 2.5%": ci_o[0], "A CI 97.5%": ci_o[1], "A p": ols.pvalues[name],
            "B. 혼합모형 계수 (유형 통제)": mixed.params[name], "B CI 2.5%": ci_m[0], "B CI 97.5%": ci_m[1], "B p": mixed.pvalues[name],
        })
    fixed = pd.DataFrame(rows)
    # 로그 척도 계수를 배율로: exp(b) 는 (시간+1) 이 기준 자치구의 몇 배인지
    fixed["A 배율 exp(b)"] = np.exp(fixed["A. OLS 계수 (유형 무시)"])
    fixed["B 배율 exp(b)"] = np.exp(fixed["B. 혼합모형 계수 (유형 통제)"])
    a, b = fixed["A. OLS 계수 (유형 무시)"], fixed["B. 혼합모형 계수 (유형 통제)"]
    # 통제 전후 비교는 '줄었다' 한 가지로 끝나지 않는다. 세 가지 경우를 구분해 적는다.
    fixed["유형 통제 후 변화"] = np.select(
        [np.sign(a) != np.sign(b), b.abs() < a.abs()],
        ["부호가 바뀜 (구성 차이가 방향을 뒤집고 있었음)", "격차 축소 (구성 차이가 격차를 부풀리고 있었음)"],
        default="격차 확대 (구성 차이가 격차를 가리고 있었음)")
    info = {"n": len(s), "n_types": s["type_group"].nunique(), "ref": BOROUGH_KO[ref], "icc": icc0}
    return fixed, variance, info


# ============================================================================ 그래프
def figures(plt, cells, agencies, fixed, info):
    # 상호작용 그림: 선들이 평행하면 상호작용 없음, 교차하거나 벌어지면 상호작용 있음
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(BOROUGHS))
    for i, a in enumerate(agencies):
        c = cells[cells["agency"] == a].set_index("borough").reindex(BOROUGHS)
        ax.plot(x, np.log1p(c["기하평균(h)"]), marker="o", ms=5, lw=1.6, color=CATEGORICAL[i % len(CATEGORICAL)], label=a)
    ticks = [0, 1, 6, 24, 72, 168, 720]
    ax.set_yticks(np.log1p(ticks), ["0", "1h", "6h", "1일", "3일", "1주", "30일"])
    ax.set_xticks(x, [BOROUGH_KO[b] for b in BOROUGHS])
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_ylabel("처리 시간 기하평균 (로그 축)")
    ax.set_title("상호작용 그림: 선이 평행하면 자치구 효과가 기관과 무관")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save_fig(fig, "stat7_01_interaction.png")

    # 칸별 기하평균 히트맵: 기관 안에서 자치구끼리 비교하도록 행(기관)별 평균 대비 배율로 색칠
    piv = cells.pivot(index="agency", columns="borough", values="기하평균(h)").reindex(index=agencies, columns=BOROUGHS)
    rel = np.log2(piv.div(np.exp(np.log(piv).mean(axis=1)), axis=0))      # 기관 내 기하평균 대비 log2 배율
    lim = max(0.3, float(np.abs(rel.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(agencies) + 1.4))
    ax.imshow(rel.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:,.1f}h", ha="center", va="center", fontsize=9,
                    color="white" if abs(rel.iloc[i, j]) > lim * 0.6 else INK)
    ax.set_xticks(range(len(BOROUGHS)), [BOROUGH_KO[b] for b in BOROUGHS])
    ax.set_yticks(range(len(agencies)), agencies)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("기관 x 자치구 처리 시간 기하평균 (색: 같은 기관 안에서 빨강=느림, 파랑=빠름)")
    save_fig(fig, "stat7_02_cell_heatmap.png")

    # 자치구 계수: 유형 무시(OLS) 대 유형 통제(혼합모형)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(fixed))[::-1]
    for off, col, lo, hi, color, label in [
        (0.16, "A. OLS 계수 (유형 무시)", "A CI 2.5%", "A CI 97.5%", ACCENT, "A. 유형 무시 (OLS)"),
        (-0.16, "B. 혼합모형 계수 (유형 통제)", "B CI 2.5%", "B CI 97.5%", SINGLE, "B. 유형 통제 (혼합모형)"),
    ]:
        ax.hlines(y + off, fixed[lo], fixed[hi], color=color, lw=1.8)
        ax.scatter(fixed[col], y + off, color=color, s=30, zorder=3, label=label)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(y, fixed["자치구"])
    ax.set_xlabel(f"로그 처리시간 차이 (기준: {info['ref']}) · 0보다 크면 기준보다 느림")
    ax.set_title("민원 유형을 통제하면 자치구 격차가 얼마나 남는가")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    save_fig(fig, "stat7_03_borough_effect_ols_vs_mixed.png")


# ============================================================================ 실행
def main():
    df = load_clean()
    agencies = list(df["agency"].value_counts().index[:TOP_AGENCIES])
    share = df["agency"].isin(agencies).mean()
    print(f"기관 상위 {TOP_AGENCIES}개: {', '.join(agencies)} (전체 민원의 {share:.1%})")

    print("[1] 칸별 기술통계")
    cells = cell_summary(df, agencies)
    save_table(cells.round(3), "stat7_cell_summary.csv")

    print("[2] 이원배치 분산분석 (균형 표본)")
    anova, assume, _, m = two_way_anova(df, agencies)
    save_table(anova.round(5), "stat7_anova.csv")
    save_table(assume.round(5), "stat7_anova_assumptions.csv")
    for _, r in anova.iloc[:3].iterrows():
        print(f"  {r['변동 요인']:<18} F={r['F']:>10,.1f}  p {p_text(r['p'])}  η²={r['η²']:.4f}  편η²={r['편 η²']:.4f}  "
              f"(HC3 p {p_text(r['p (HC3 강건)'])})")
    print(assume[["점검", "판정", "비고"]].to_string(index=False))

    print("[3] 단순 주효과: 기관별 자치구 차이")
    se = simple_effects(df, agencies)
    save_table(se.round(5), "stat7_simple_effects.csv")
    print(se[["기관", "n", "ε²", "가장 빠른 자치구", "중앙값 최소(h)", "가장 느린 자치구", "중앙값 최대(h)", "최대/최소 배율"]]
          .round(3).to_string(index=False))

    print("[4] 혼합효과 모형: 민원 유형 통제")
    fixed, variance, info = mixed_model(df)
    save_table(fixed.round(5), "stat7_mixed_fixed.csv")
    save_table(variance.round(5), "stat7_mixed_variance.csv")
    print(f"  표본 {info['n']:,}건, 유형 묶음 {info['n_types']}개, ICC = {info['icc']:.3f} "
          f"(처리 시간 분산의 {info['icc']:.0%}가 민원 유형 차이)")
    print(fixed[["자치구", "A. OLS 계수 (유형 무시)", "B. 혼합모형 계수 (유형 통제)", "B p", "유형 통제 후 변화"]].round(3).to_string(index=False))

    print("[5] 그래프")
    figures(setup_style(), cells, agencies, fixed, info)


if __name__ == "__main__":
    main()
