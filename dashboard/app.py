"""NYC 311 민원 통계 학습 대시보드 (Streamlit).

analysis/stat_01 ~ stat_12 가 만든 표와 집계만 읽는다 (대시보드 안에서 무거운 통계를 다시 계산하지 않는다).
탭은 공부 순서다 (4부 13단계). 스크립트 번호는 분석을 만든 순서라서 탭 순서와 다르다.
  1부 데이터와 친해지기  0 시작하기(stat_10, 12)  1 둘러보기(stat_01)  2 대푯값(stat_10)
                         3 퍼짐·기술통계(stat_01, 10)  4 분포·변환·표준화(stat_02, 03)
  2부 불확실성과 추론    5 불확실성(stat_10)  6 검정·회귀(stat_04)  7 현대 방법(stat_11)
  3부 비교와 시간        8 기관 비교(stat_07)  9 시계열(stat_06)  10 생존분석(stat_05)
  4부 위치와 유형        11 공간 분석(stat_08)  12 군집·PCA(stat_09)
각 단계: 머리말(핵심 질문, 개념, 쓰는 열) -> 본문 -> 꼬리말(개념 ↔ 이 데이터 정리표, stat_12 가 만듦)

실행: 프로젝트 루트에서  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from stat_common import (  # noqa: E402
    AGG, BOROUGH_COLOR, BOROUGH_KO, BOROUGHS, CATEGORICAL, DAILY, MAP_SAMPLE, OTHER_GRAY, TABLES,
)

st.set_page_config(page_title="NYC 311 민원 기초통계", page_icon=":material/monitoring:", layout="wide")

BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
KO = list(BOROUGH_KO.values())
BOROUGH_SCALE = alt.Scale(domain=KO, range=[BOROUGH_COLOR[b] for b in BOROUGHS])
REQUIRED = [
    "stat1_numeric_summary.csv", "stat2_normality_tests.csv", "stat3_efficiency_index.csv", "stat4_model_summary.csv",
]


@st.cache_data(max_entries=64)
def table(name, mtime):
    """outputs/tables 의 CSV. mtime 을 키에 넣어 분석을 다시 돌리면 캐시가 갱신되게 한다."""
    return pd.read_csv(TABLES / name)


def t(name):
    path = TABLES / name
    return table(name, path.stat().st_mtime) if path.exists() else None


@st.cache_data(max_entries=4)
def load_processed(path, mtime):
    df = pd.read_csv(path, parse_dates=["date"])
    if "borough" in df.columns:
        df["자치구"] = df["borough"].map(BOROUGH_KO)
    return df


def processed(path):
    return load_processed(str(path), path.stat().st_mtime)


def ko(df, col="borough"):
    out = df.copy()
    out["자치구"] = out[col].map(BOROUGH_KO)
    return out


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def hbar(df, cat, val, title=None, color=BLUE, label_fmt=",.0f", height=None):
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{val}:Q", title=title, axis=alt.Axis(grid=False, labels=False, ticks=False, domain=False)),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=label_fmt)],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=2)
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{val}:Q", format=label_fmt))
    return (bars + text).properties(height=height or max(140, 26 * len(df)))


def heat(df, x, y, val, fmt, diverging, x_sort=None, y_sort=None, height=230):
    lim = float(np.abs(df[val]).max()) or 1.0
    scale = (alt.Scale(scheme="redblue", reverse=True, domain=[-lim, lim]) if diverging
             else alt.Scale(scheme="blues", domain=[0, 1]))
    dark = (f"abs(datum['{val}']) > {lim * 0.6}" if diverging else f"datum['{val}'] > 0.55")
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", sort=x_sort, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=180, labelOverlap=False)),
        y=alt.Y(f"{y}:N", sort=y_sort, title=None),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{val}:Q", format=fmt)],
    )
    rect = base.mark_rect().encode(color=alt.Color(f"{val}:Q", scale=scale, legend=None))
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{val}:Q", format=fmt),
        color=alt.condition(dark, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (rect + text).properties(height=height)


# ============================================================ 학습 안내: 단계 머리말과 꼬리말
# 탭 순서는 공부 순서(0~12단계)다. 스크립트 번호(stat_01~12)는 분석을 만든 순서라서 탭 순서와 다르다.
# 머리말과 꼬리말의 내용은 analysis/stat_12_concept_map.py 가 만든 표에서 읽는다.
def stage_header(key):
    lp = t("stat12_learning_path.csv")
    if lp is None or not (lp["key"] == key).any():
        return
    r = lp[lp["key"] == key].iloc[0]
    with st.container(border=True):
        st.caption(f"{r['파트']} · {r['단계']}단계")
        st.markdown(f"### {r['제목']}")
        c1, c2, c3 = st.columns([2, 3, 2])
        c1.markdown(f"**핵심 질문**  \n{r['핵심 질문']}")
        c2.markdown(f"**이 단계에서 배우는 개념**  \n{r['배우는 개념']}")
        c3.markdown(f"**쓰는 열**  \n{r['쓰는 열']}  \n**먼저 볼 단계**: {r['먼저 볼 단계']}")


def concept_footer(key):
    cm, lp = t("stat12_concept_map.csv"), t("stat12_learning_path.csv")
    if cm is None or not (cm["key"] == key).any():
        return
    st.divider()
    st.subheader("이 단계의 개념 ↔ 이 데이터")
    st.caption("이 단계에서 나온 개념을 이 데이터의 어느 열에 적용했고, 지금 값이 무엇을 뜻하는지 정리한 표입니다. 값은 분석 결과 표에서 자동으로 읽어 옵니다.")
    st.table(cm[cm["key"] == key].drop(columns=["key", "단계"]).set_index("개념"), alt=f"{key} 단계 개념과 데이터 연결표")
    if lp is not None:
        idx = lp.index[lp["key"] == key]
        if len(idx) and idx[0] + 1 < len(lp):
            nx = lp.iloc[idx[0] + 1]
            st.info(f"**{nx['단계']}단계 · {nx['제목']}**: {nx['핵심 질문']}", title="다음 단계", icon=":material/arrow_forward:")


st.title("NYC 311 민원 기초통계 분석")
st.caption("2024년 4월 ~ 9월 접수분 · 5개 자치구 전 유형 · 출처 NYC Open Data (311 Service Requests)")
st.caption("탭은 공부 순서입니다. 처음이라면 **0 시작하기**부터 차례대로 읽으세요. 각 단계 끝에 그 단계의 개념을 이 데이터에 연결한 정리표가 있습니다.")

missing = [n for n in REQUIRED if not (TABLES / n).exists()] + [p.name for p in (AGG, DAILY, MAP_SAMPLE) if not p.exists()]
if missing:
    st.error(
        "분석 스크립트를 먼저 실행하세요: `analysis/stat_00_download.py` → `stat_01_clean_descriptive.py` → "
        "`stat_02_distribution.py` → `stat_03_scaling.py` → `stat_04_inference.py`\n\n없는 파일: " + ", ".join(missing),
        title="분석 결과 파일이 없습니다",
    )
    st.stop()

# 탭 = 공부 순서. 기존 탭 변수(tab1~tab9)는 그대로 두고 보여 주는 순서만 바꿨다.
#   tab4 = 1단계 둘러보기, tab1 = 3단계 퍼짐·기술통계, tab2 = 4단계 분포, tab3 = 6단계 검정·회귀,
#   tab7 = 8단계 기관, tab6 = 9단계 시계열, tab5 = 10단계 생존, tab8 = 11단계 공간, tab9 = 12단계 군집
(tab_s0, tab4, tab_s2, tab1, tab2, tab_s5, tab3, tab_s7, tab7, tab6, tab5, tab8, tab9) = st.tabs([
    "0 시작하기", "1 둘러보기", "2 대푯값", "3 퍼짐·기술통계", "4 분포·변환·표준화", "5 불확실성", "6 검정·회귀",
    "7 현대 방법", "8 기관 비교", "9 시계열", "10 생존분석", "11 공간 분석", "12 군집·PCA",
])

# ============================================================ 탭 1: 기술통계 요약
with tab1:
    stage_header("s3")
    summ = t("stat1_numeric_summary.csv").set_index("변수")
    res, day = summ.iloc[0], summ.iloc[1]
    ind = t("stat3_borough_indicators.csv")
    rate_24h = (ind["within_24h_rate"] * ind["n_complaints"]).sum() / ind["n_complaints"].sum()
    daily = processed(DAILY)

    with st.container(horizontal=True):
        st.metric("분석 대상 민원", f"{res['n']:,.0f}건", border=True)
        st.metric("처리시간 중앙값", f"{res['중앙값']:,.1f}시간", border=True)
        st.metric("처리시간 평균", f"{res['평균']:,.1f}시간", border=True,
                  help="극단적으로 오래 걸린 민원이 평균을 끌어올립니다. 대표값은 중앙값으로 읽으세요.")
        st.metric("24시간 내 처리율", f"{rate_24h:.1f}%", border=True)
        st.metric("일평균 접수", f"{day['평균']:,.0f}건", border=True)

    c1, c2 = st.columns([3, 2])
    with c1, st.container(border=True):
        st.subheader("수치형 변수 기술통계")
        show = summ.T.rename(columns=lambda c: c.replace(" (시간)", ""))
        st.dataframe(show.style.format("{:,.2f}"), alt="수치형 변수 기술통계")
        st.caption(f"왜도 {res['왜도']:.1f}, 초과첨도 {res['첨도(초과)']:.0f}: 처리시간은 오른쪽 꼬리가 매우 길어 "
                   "평균이 중앙값보다 훨씬 큽니다. 분포의 모양은 4단계(분포·변환·표준화)에서 다룹니다.")
    with c2, st.container(border=True):
        st.subheader("처리시간 백분위")
        pct = t("stat1_resolution_percentiles.csv")
        st.dataframe(pct, hide_index=True, alt="처리시간 백분위",
                     column_config={"Resolution_Hours": st.column_config.NumberColumn("처리시간(시간)", format="%.2f")})

    with st.container(border=True):
        st.subheader("일별 민원 접수 건수")
        d = daily.assign(구분=np.where(daily["is_weekend"] == 1, "주말", "평일"))
        line = alt.Chart(d).mark_line(color=BLUE).encode(
            x=alt.X("date:T", title=None), y=alt.Y("n_complaints:Q", title="접수 건수", scale=alt.Scale(zero=False)))
        pts = alt.Chart(d).mark_circle(size=38).encode(
            x="date:T", y="n_complaints:Q",
            color=alt.Color("구분:N", scale=alt.Scale(domain=["평일", "주말"], range=[BLUE, ORANGE]), title=None),
            tooltip=[alt.Tooltip("date:T", title="날짜"), "구분:N", alt.Tooltip("n_complaints:Q", title="접수 건수", format=",")])
        st.altair_chart((line + pts).properties(height=260), alt="일별 민원 접수 건수 추이")

    st.subheader("범주형 변수 빈도")
    f1, f2, f3 = st.columns(3)
    for col, name, label, head in [(f1, "stat1_freq_borough.csv", "borough", "자치구"),
                                   (f2, "stat1_freq_agency.csv", "agency", "담당 기관 상위 10"),
                                   (f3, "stat1_freq_complaint_type.csv", "complaint_type", "민원 유형 상위 10")]:
        with col, st.container(border=True):
            freq = t(name)
            st.markdown(f"**{head}**")
            top = freq[~freq[label].str.startswith("기타")].head(10)
            if label == "borough":
                top = top.assign(borough=top["borough"].map(BOROUGH_KO))
            st.altair_chart(hbar(top, label, "비율(%)", label_fmt=".1f", height=300), alt=f"{label} 빈도 비율")
            with st.expander("빈도표 전체"):
                st.dataframe(freq, hide_index=True, alt=f"{label} 빈도표")

    with st.expander("데이터 정제 기준과 제외 건수"):
        st.dataframe(t("stat1_exclusion_log.csv"), hide_index=True, alt="정제 기준과 제외 건수")

# ============================================================ 탭 2: 분포 변환 및 표준화
with tab2:
    stage_header("s4")
    tests = t("stat2_normality_tests.csv")
    names = tests["변환"].tolist()
    st.subheader("처리시간 분포와 변환 효과")
    with st.container(horizontal=True):
        for _, r in tests.iterrows():
            st.metric(f"왜도 · {r['변환']}", f"{r['왜도']:.2f}", border=True,
                      help=f"초과첨도 {r['첨도(초과)']:.2f}, |왜도| 감소율 {r['왜도 개선(|왜도| 감소율 %)']:.1f}%")
    pick = st.segmented_control("변환 선택", names, default=names[0], required=True)
    row = tests.set_index("변환").loc[pick]
    h1, h2 = st.columns([3, 2])
    with h1, st.container(border=True):
        st.markdown("**히스토그램**")
        bins = t("stat2_hist_bins.csv")
        bins = bins[bins["변환"] == pick]
        st.altair_chart(alt.Chart(bins).mark_bar(color=BLUE).encode(
            x=alt.X("구간시작:Q", bin="binned", title="값"), x2="구간끝:Q", y=alt.Y("건수:Q", title="건수"),
            tooltip=[alt.Tooltip("구간시작:Q", format=",.2f"), alt.Tooltip("구간끝:Q", format=",.2f"),
                     alt.Tooltip("건수:Q", format=",")]).properties(height=300), alt=f"{pick} 히스토그램")
        if pick == names[0]:
            st.caption("원자료는 꼬리가 길어 99번째 백분위까지만 표시했습니다.")
    with h2, st.container(border=True):
        st.markdown("**Q-Q plot**")
        qq = t("stat2_qq_points.csv")
        qq = qq[qq["변환"] == pick]
        lo, hi = float(qq["이론분위수"].min()), float(qq["이론분위수"].max())
        ref = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color=ORANGE).encode(x="x:Q", y="y:Q")
        dots = alt.Chart(qq).mark_circle(size=22, color=BLUE).encode(
            x=alt.X("이론분위수:Q", title="이론 분위수 (표준정규)"), y=alt.Y("표본분위수(표준화):Q", title="표본 분위수 (표준화)"))
        st.altair_chart((ref + dots).properties(height=300), alt=f"{pick} Q-Q plot")
        st.caption("점이 주황색 선 위에 놓일수록 정규분포에 가깝습니다.")

    sw_col = next(c for c in tests.columns if c.startswith("Shapiro-Wilk W"))
    st.info(
        f"Shapiro-Wilk W = {row[sw_col]:.4f} (p {p_fmt(row['Shapiro-Wilk p'])}), "
        f"Kolmogorov-Smirnov D = {row['K-S D (전체)']:.4f} (p {p_fmt(row['K-S p'])}) → 정규성 **{row['정규성 (α=0.05)']}**.  \n"
        "표본이 매우 크면 사소한 이탈도 유의하게 나오므로, p 값보다 왜도·첨도와 Q-Q 그림을 우선해 읽습니다. "
        "Shapiro-Wilk 는 무작위 5,000건 표본에, K-S 는 전체 자료에 적용했습니다.",
        title=f"정규성 검정: {pick}", icon=":material/rule:")
    with st.expander("변환별 검정 결과 전체"):
        st.dataframe(tests, hide_index=True, alt="변환별 정규성 검정 결과")

    st.divider()
    st.subheader("자치구 지표의 표준화와 정규화")
    scaled = ko(t("stat3_borough_scaled.csv"))
    labels = {"n_complaints": "민원 건수", "per_1000_pop": "인구 1천 명당 건수", "mean_hours": "평균 처리시간",
              "median_hours": "중앙 처리시간", "p90_hours": "P90 처리시간", "within_24h_rate": "24시간 내 처리율"}
    s1, s2 = st.columns(2)
    for col, prefix, title, fmt, div in [(s1, "z_", "표준화 (Z-score)", "+.2f", True), (s2, "mm_", "정규화 (Min-Max)", ".2f", False)]:
        long = scaled.melt(id_vars="자치구", value_vars=[prefix + k for k in labels], var_name="지표", value_name="값")
        long["지표"] = long["지표"].str.removeprefix(prefix).map(labels)
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.altair_chart(heat(long, "지표", "자치구", "값", fmt, div, list(labels.values()), KO), alt=f"자치구 지표 {title}")
    st.markdown(
        "- **표준화**는 평균에서 표준편차 몇 배 떨어졌는지를 나타냅니다. 범위가 정해져 있지 않아 이상치 탐지와 군집화에 씁니다.\n"
        "- **정규화**는 최저를 0, 최고를 1로 고정합니다. 점수로 읽기 쉽지만 5개 자치구의 작은 차이도 0과 1로 벌어집니다.")
    with st.expander("자치구 원지표"):
        st.dataframe(ko(ind).drop(columns="borough").set_index("자치구").style.format("{:,.2f}"), alt="자치구 원지표")

    e1, e2 = st.columns([2, 3])
    with e1, st.container(border=True):
        st.markdown("**민원 처리 효율성 종합 지수 (0-100)**")
        eff = ko(t("stat3_efficiency_index.csv"))
        st.altair_chart(hbar(eff, "자치구", "효율성 지수(0-100)", label_fmt=".1f"), alt="자치구별 효율성 지수")
        st.caption("중앙 처리시간, P90 처리시간, 24시간 내 처리율을 Min-Max 로 0~1 점수화해 같은 가중치로 평균했습니다. "
                   "민원 유형 구성은 보정하지 않은 단순 비교입니다.")
    with e2, st.container(border=True):
        st.markdown("**이상치: 자치구 × 날짜 단위 |Z| > 3**")
        out = t("stat3_outlier_borough_days.csv")
        st.dataframe(ko(out).drop(columns="borough"), hide_index=True, alt="Z 점수 3 초과 자치구-날짜",
                     column_order=["date", "weekday", "자치구", "n_complaints", "z_n_complaints", "mean_hours_geometric", "z_mean_log_hours"],
                     column_config={"date": "날짜", "weekday": "요일", "n_complaints": "건수", "z_n_complaints": "Z(건수)",
                                    "mean_hours_geometric": "처리시간 기하평균(h)", "z_mean_log_hours": "Z(로그 처리시간)"})
        st.caption("자치구는 5개뿐이라 자치구 수준에서는 |Z|가 2.0을 넘을 수 없습니다. "
                   "그래서 자치구 안에서 표준화한 일별 건수와 일별 평균 로그 처리시간으로 이상치를 찾았습니다.")

# ============================================================ 탭 3: 가설검정 리포트
with tab3:
    stage_header("s6")
    st.subheader("① 자치구별 처리 시간에 차이가 있는가")
    g = t("stat4_group_tests.csv")
    chosen = g[g["판정"].str.startswith("채택")].iloc[0]
    kw = g[g["검정"].str.startswith("Kruskal")].iloc[0]
    verdict = st.success if chosen["p"] < 0.05 else st.info
    verdict(f"{chosen['검정']} = {chosen['통계량']:,.1f}, p {p_fmt(chosen['p'])}, {chosen['효과크기']}  \n"
            f"{chosen['판정'].split(': ')[-1]}. 효과크기가 작으면 통계적으로 유의해도 실질 차이는 작다는 뜻입니다.",
            title="집단 간 차이 검정", icon=":material/rule:")
    b1, b2 = st.columns([3, 2])
    with b1, st.container(border=True):
        st.markdown("**자치구별 처리 시간 분포** (상자 Q1-Q3, 선 P5-P95, 로그 축)")
        q = ko(t("stat4_borough_quantiles.csv"))
        # 0시간도 그릴 수 있게 log1p 로 바꾼 좌표에 그리고, 눈금 라벨만 원래 시간으로 되돌린다
        for c in ["P5", "Q1", "중앙값", "Q3", "P95"]:
            q[f"l_{c}"] = np.log1p(q[c])
        ticks = [0, 1, 6, 24, 72, 168, 720, 2160]
        axis = alt.Axis(values=list(np.log1p(ticks)), labelExpr="format(exp(datum.value) - 1, ',.0f') + 'h'", title="처리 시간 (시간)")
        base = alt.Chart(q).encode(y=alt.Y("자치구:N", sort=KO, title=None))
        rule = base.mark_rule(color=OTHER_GRAY).encode(x=alt.X("l_P5:Q", axis=axis), x2="l_P95:Q")
        box = base.mark_bar(size=22).encode(x="l_Q1:Q", x2="l_Q3:Q",
                                             color=alt.Color("자치구:N", scale=BOROUGH_SCALE, legend=None),
                                             tooltip=["자치구:N", alt.Tooltip("중앙값:Q", format=",.1f"),
                                                      alt.Tooltip("Q1:Q", format=",.1f"), alt.Tooltip("Q3:Q", format=",.1f"),
                                                      alt.Tooltip("P95:Q", format=",.0f"), alt.Tooltip("n:Q", format=",")])
        med = base.mark_tick(color="#0b0b0b", thickness=2, size=22).encode(x="l_중앙값:Q")
        st.altair_chart((rule + box + med).properties(height=240), alt="자치구별 처리 시간 상자그림")
    with b2, st.container(border=True):
        st.markdown("**가정 점검과 검정**")
        st.dataframe(g, hide_index=True, alt="정규성, 등분산, 집단 차이 검정 결과",
                     column_config={"p": st.column_config.NumberColumn(format="%.2e"),
                                    "통계량": st.column_config.NumberColumn(format="%.3f")})
    post = t("stat4_posthoc_dunn.csv")
    post_name = "Dunn 사후검정 (Bonferroni 보정)"
    if post is None:
        post, post_name = t("stat4_posthoc_tukey.csv"), "Tukey HSD 사후검정"
    with st.expander(post_name, expanded=True):
        for c in ("집단 A", "집단 B"):
            post[c] = post[c].map(BOROUGH_KO)
        st.dataframe(post, hide_index=True, alt=post_name)

    st.divider()
    st.subheader("② 자치구와 민원 유형은 독립인가")
    chi = t("stat4_chi2_summary.csv").iloc[0]
    v = chi["Cramér's V"]
    strength = "약한" if v < 0.1 else ("중간 정도의" if v < 0.3 else "강한")
    verdict = st.success if chi["p"] < 0.05 else st.info
    verdict(f"χ² = {chi['카이제곱']:,.1f}, 자유도 {chi['자유도']:.0f}, p {p_fmt(chi['p'])}, Cramér's V = {v:.3f} ({strength} 연관)  \n"
            f"{chi['판정']}. 상위 10개 유형이 전체 민원의 {chi['분석 대상 비율(%)']:.1f}%를 차지합니다.",
            title="카이제곱 독립성 검정", icon=":material/rule:")
    with st.container(border=True):
        view = st.segmented_control("표시 값", ["수정 표준화 잔차", "자치구 내 비율(%)"], default="수정 표준화 잔차", required=True)
        src = ko(t("stat4_chi2_residuals.csv" if view.startswith("수정") else "stat4_chi2_row_pct.csv"))
        long = src.drop(columns="borough").melt(id_vars="자치구", var_name="민원 유형", value_name="값")
        if view.startswith("수정"):
            st.altair_chart(heat(long, "민원 유형", "자치구", "값", "+.0f", True, None, KO, 260), alt="자치구와 민원 유형의 표준화 잔차")
            st.caption("빨강은 독립일 때 기대보다 많고 파랑은 적다는 뜻입니다. 절댓값이 2를 넘으면 그 칸이 연관에 기여합니다.")
        else:
            long["값"] = long["값"] / 100
            st.altair_chart(heat(long, "민원 유형", "자치구", "값", ".1%", False, None, KO, 260), alt="자치구 내 민원 유형 비율")

    st.divider()
    st.subheader("③ 일별 접수 건수는 기온·주말과 관련이 있는가")
    corr = t("stat4_correlation.csv")
    k1, k2 = st.columns([2, 3])
    with k1, st.container(border=True):
        st.markdown("**상관계수**")
        st.dataframe(corr, hide_index=True, alt="일별 접수 건수 상관계수",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Pearson r", "Spearman ρ"]}
                     | {c: st.column_config.NumberColumn(format="%.2e") for c in ["Pearson p", "Spearman p"]})
        st.caption("주말 여부처럼 0/1 변수와의 Pearson r 은 점이연 상관계수입니다. 상관은 인과를 뜻하지 않습니다.")
    with k2, st.container(border=True):
        d = daily.assign(구분=np.where(daily["is_weekend"] == 1, "주말", "평일"))
        color = alt.Color("구분:N", scale=alt.Scale(domain=["평일", "주말"], range=[BLUE, ORANGE]), title=None)
        if "temp_mean_c" in d.columns:
            st.markdown("**일평균 기온과 접수 건수**")
            st.altair_chart(alt.Chart(d).mark_circle(size=46, opacity=0.8).encode(
                x=alt.X("temp_mean_c:Q", title="일평균 기온 (°C)", scale=alt.Scale(zero=False)),
                y=alt.Y("n_complaints:Q", title="접수 건수", scale=alt.Scale(zero=False)), color=color,
                tooltip=[alt.Tooltip("date:T", title="날짜"), alt.Tooltip("temp_mean_c:Q", title="기온"),
                         alt.Tooltip("n_complaints:Q", title="건수", format=",")]).properties(height=280),
                alt="일평균 기온과 접수 건수 산점도")
        else:
            st.markdown("**평일·주말 접수 건수**")
            st.altair_chart(alt.Chart(d).mark_boxplot(size=40).encode(
                x=alt.X("구분:N", title=None), y=alt.Y("n_complaints:Q", title="접수 건수", scale=alt.Scale(zero=False)),
                color=color).properties(height=280), alt="평일과 주말의 접수 건수 상자그림")

    st.divider()
    st.subheader("④ 접수 시점 정보로 처리 시간을 예측할 수 있는가")
    ms = t("stat4_model_summary.csv")
    ols_m, logit_m = ms.iloc[0], ms.iloc[1]
    with st.container(horizontal=True):
        st.metric("선형회귀 R² (학습)", f"{ols_m['R² (학습)']:.3f}", border=True, help="목표변수: log1p(처리시간)")
        st.metric("선형회귀 R² (9월 평가)", f"{ols_m['R² (평가, 9월)']:.3f}", border=True)
        st.metric("로지스틱 pseudo R²", f"{logit_m['R² (학습)']:.3f}", border=True, help="McFadden. 목표변수: 24시간 내 처리 여부")
        st.metric("로지스틱 AUC (9월 평가)", f"{logit_m['AUC (평가)']:.3f}", border=True)
        st.metric("정확도 (9월 평가)", f"{logit_m['정확도 (평가)']:.3f}", border=True,
                  help=f"다수 범주만 예측할 때의 정확도는 {logit_m['다수 범주 비율 (평가)']:.3f} 입니다.")
    st.caption(f"두 모형 모두 모형 유의성 p {p_fmt(ols_m['모형 유의성 p'])}. 8월까지 접수분 {ols_m['학습 n']:,.0f}건으로 적합하고 "
               f"9월 접수분 {ols_m['평가 n']:,.0f}건으로 평가했습니다. 설명변수는 접수 시점에 알 수 있는 자치구, 담당 기관, 시간대, 주말 여부입니다.")
    model = st.segmented_control("모형", ["로지스틱 회귀 (24시간 내 처리)", "다중선형회귀 (로그 처리시간)"],
                                 default="로지스틱 회귀 (24시간 내 처리)", required=True)
    is_logit = model.startswith("로지스틱")
    coef = t("stat4_logit_coef.csv" if is_logit else "stat4_ols_coef.csv")
    est, lo_c, hi_c = (("오즈비", "오즈비 CI 2.5%", "오즈비 CI 97.5%") if is_logit else ("계수", "CI 2.5%", "CI 97.5%"))
    # Vega-Lite 는 열 이름의 점(.)을 중첩 필드로 해석하므로 그래프용 열은 단순한 이름으로 바꾼다
    plot = coef[coef["변수"] != "const"].rename(columns={est: "est", lo_c: "lo", hi_c: "hi", "유의 (α=0.05)": "sig"})
    m1, m2 = st.columns([3, 2])
    with m1, st.container(border=True):
        st.markdown(f"**{est}와 95% 신뢰구간** (회색은 α=0.05 에서 유의하지 않음)")
        xs = alt.Scale(type="log") if is_logit else alt.Scale(zero=True)
        base = alt.Chart(plot).encode(
            y=alt.Y("변수:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
            color=alt.Color("sig:N", scale=alt.Scale(domain=["예", "아니오"], range=[BLUE, OTHER_GRAY]), legend=None))
        ci = base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", scale=xs, title=est), x2="hi:Q")
        pt = base.mark_circle(size=70, opacity=1).encode(
            x=alt.X("est:Q", scale=xs),
            tooltip=["변수:N", alt.Tooltip("est:Q", title=est, format=".3f"), alt.Tooltip("lo:Q", title="CI 2.5%", format=".3f"),
                     alt.Tooltip("hi:Q", title="CI 97.5%", format=".3f"), alt.Tooltip("p:Q", format=".2e")])
        zero = alt.Chart(pd.DataFrame({"x": [1 if is_logit else 0]})).mark_rule(strokeDash=[4, 3], color="#898781").encode(x="x:Q")
        st.altair_chart((zero + ci + pt).properties(height=max(260, 24 * len(plot))), alt=f"{model} 계수와 신뢰구간")
        st.caption("오즈비가 1보다 크면 기준 범주보다 24시간 내 처리될 가능성이 높습니다." if is_logit else
                   "계수가 0보다 크면 기준 범주보다 처리 시간이 깁니다. exp(계수)는 (시간+1)의 배율입니다.")
    with m2, st.container(border=True):
        st.markdown("**계수표**")
        st.dataframe(coef, hide_index=True, height=max(300, 24 * len(plot) + 40), alt=f"{model} 계수표",
                     column_config={"p": st.column_config.NumberColumn(format="%.2e")})

# ============================================================ 탭 4: 지도 및 필터링
with tab4:
    stage_header("s1")
    agg, pts = processed(AGG), processed(MAP_SAMPLE)
    types = agg.groupby("type_group")["n"].sum().sort_values(ascending=False).index.tolist()
    d_min, d_max = agg["date"].min().date(), agg["date"].max().date()
    with st.container(border=True):
        fc1, fc2 = st.columns([2, 3])
        sel_b = fc1.pills("자치구", KO, selection_mode="multi", default=KO)
        sel_range = fc1.slider("접수일", d_min, d_max, (d_min, d_max), format="YYYY-MM-DD")
        sel_t = fc2.multiselect("민원 유형 (상위 15개 + 기타, 비우면 전체)", types, placeholder="전체 유형")
    start, end = pd.Timestamp(sel_range[0]), pd.Timestamp(sel_range[1])

    def apply(df):
        m = df["자치구"].isin(sel_b) & df["date"].between(start, end)
        return df[m & df["type_group"].isin(sel_t)] if sel_t else df[m]

    fa, fp = apply(agg), apply(pts)
    if fa.empty:
        st.warning("선택한 조건에 해당하는 민원이 없습니다. 자치구를 하나 이상 선택하세요.", title="결과 없음")
    else:
        n = fa["n"].sum()
        with st.container(horizontal=True):
            st.metric("민원 건수", f"{n:,.0f}건", border=True)
            st.metric("전체 대비", f"{n / agg['n'].sum():.1%}", border=True)
            st.metric("평균 처리시간", f"{fa['sum_hours'].sum() / n:,.1f}시간", border=True)
            st.metric("24시간 내 처리율", f"{fa['n_within_24h'].sum() / n:.1%}", border=True)
        g1, g2 = st.columns([3, 2])
        with g1, st.container(border=True):
            st.markdown(f"**민원 위치** (좌표가 있는 민원의 무작위 표본 중 {len(fp):,}건)")
            m = fp.assign(color=fp["borough"].map(BOROUGH_COLOR))
            st.map(m, latitude="latitude", longitude="longitude", color="color", size=25, height=520,
                   alt="민원 위치 지도, 자치구별 색")
            st.altair_chart(alt.Chart(pd.DataFrame({"자치구": KO})).mark_point(filled=True, size=140, opacity=1).encode(
                x=alt.X("자치구:N", sort=KO, title=None, axis=alt.Axis(labelAngle=0, domain=False, ticks=False)),
                color=alt.Color("자치구:N", scale=BOROUGH_SCALE, legend=None)).properties(height=54),
                alt="지도 색 범례: 자치구별 색")
        with g2:
            by = (fa.groupby("자치구")[["n", "sum_hours", "n_within_24h"]].sum()
                    .assign(평균처리시간=lambda x: x["sum_hours"] / x["n"], 처리율=lambda x: x["n_within_24h"] / x["n"] * 100)
                    .reindex([k for k in KO if k in sel_b]).reset_index())
            with st.container(border=True):
                st.markdown("**자치구별 요약**")
                st.dataframe(by, hide_index=True, alt="자치구별 요약", column_order=["자치구", "n", "평균처리시간", "처리율"],
                             column_config={"n": st.column_config.NumberColumn("건수", format="localized"),
                                            "평균처리시간": st.column_config.NumberColumn("평균 처리시간(h)", format="%.1f"),
                                            "처리율": st.column_config.ProgressColumn("24시간 내 처리율", format="%.0f%%", min_value=0, max_value=100)})
            with st.container(border=True):
                st.markdown("**유형별 건수**")
                bt = fa.groupby("type_group")["n"].sum().sort_values(ascending=False).head(10).rename_axis("민원 유형").reset_index(name="건수")
                st.altair_chart(hbar(bt, "민원 유형", "건수", height=270), alt="유형별 건수")
        with st.container(border=True):
            st.markdown("**주별 접수 추이**")
            fw = fa.assign(주=fa["date"].dt.to_period("W-SUN").dt.start_time)
            full = fw.groupby("주")["date"].nunique()
            fw = fw[fw["주"].isin(full[full == 7].index)]      # 7일이 다 차지 않은 주는 낮게 보이므로 뺀다
            wk = fw.groupby(["주", "자치구"])["n"].sum().reset_index()
            st.altair_chart(alt.Chart(wk).mark_line().encode(
                x=alt.X("주:T", title=None), y=alt.Y("n:Q", title="주간 접수 건수"),
                color=alt.Color("자치구:N", scale=BOROUGH_SCALE, sort=KO, title=None),
                tooltip=[alt.Tooltip("주:T", title="주 시작일"), "자치구:N", alt.Tooltip("n:Q", title="건수", format=",")]).properties(height=260),
                alt="자치구별 주간 접수 추이")
            st.caption("월요일 시작 주 단위입니다. 선택한 기간 안에서 7일이 다 차지 않은 주는 표시하지 않습니다.")


# ============================================================ 추가 분석 탭 공통
HOUR_AXIS = alt.Axis(values=list(np.log1p([0, 1, 6, 24, 72, 168, 720, 2160])),
                     labelExpr="format(exp(datum.value) - 1, ',.0f') + 'h'", title="접수 후 경과 시간 (로그 축)")


def need(script, *names):
    """추가 분석 결과 파일이 없으면 안내만 띄우고 False 를 돌려준다."""
    missing = [n for n in names if not (TABLES / n).exists()]
    if missing:
        st.info(f"`analysis/{script}` 를 실행하면 이 탭이 채워집니다.", title="아직 분석 결과가 없습니다", icon=":material/info:")
    return not missing


# ============================================================ 탭 5: 생존분석
with tab5:
    stage_header("s10")
    if need("stat_05_survival.py", "stat5_km_curves.csv", "stat5_km_summary.csv", "stat5_cox_coef.csv"):
        log5 = t("stat5_data_log.csv").set_index("항목")["값"]
        bias = t("stat5_naive_vs_km.csv").set_index("지표")
        km_sum = t("stat5_km_summary.csv")
        st.subheader("민원이 종결되기까지: 미종결 민원을 버리지 않는 분석")
        with st.container(horizontal=True):
            st.metric("분석 대상", f"{log5['생존분석 대상 행 수']:,.0f}건", border=True)
            st.metric("중도절단 비율", f"{log5['중도절단 비율(%)']:.1f}%", border=True,
                      help="관측 종료 시점(2024-10-01)까지 종결되지 않은 민원의 비율입니다.")
            st.metric("중앙값 · Kaplan-Meier", f"{bias.loc['처리시간 중앙값(h)', 'Kaplan-Meier (미종결 포함)']:,.1f}시간", border=True,
                      help="미종결 민원을 중도절단 자료로 포함해 추정한 값입니다.")
            st.metric("중앙값 · 종결 건만", f"{bias.loc['처리시간 중앙값(h)', '종결 건만 사용']:,.1f}시간", border=True,
                      help="미종결 민원을 빼고 계산한 값입니다. 오래 걸리는 민원이 빠져 실제보다 짧게 나옵니다.")
        with st.expander("용어 설명: 생존함수, 중도절단, 위험비"):
            st.markdown(
                "- **사건**: 여기서는 민원 종결. **생존**: 아직 열려 있는 상태.\n"
                "- **생존함수 S(t)**: 접수 후 t시간이 지나도 아직 열려 있을 확률. 곡선이 빨리 내려갈수록 빨리 처리된다는 뜻입니다.\n"
                "- **중도절단**: 관측이 끝날 때까지 종결되지 않은 민원. 처리 시간은 모르지만 '적어도 그만큼은 걸렸다'는 정보를 계산에 씁니다.\n"
                "- **Kaplan-Meier 추정**: 사건이 일어난 시점마다 '그 순간을 넘길 확률'을 구해 차례로 곱합니다.\n"
                "- **위험비(HR)**: 기준 범주에 비해 종결되는 순간 속도가 몇 배인지. 사건이 종결이므로 1보다 크면 더 빨리 처리된다는 뜻입니다.")
        with st.container(border=True):
            curves = t("stat5_km_curves.csv").rename(columns={"시간(h)": "hours", "CI 하한": "lo", "CI 상한": "hi"})
            kind = st.segmented_control("비교 기준", ["전체", "자치구", "기관"], default="자치구", required=True)
            c = curves[curves["구분"] == kind].assign(lt=lambda d: np.log1p(d["hours"]))
            groups = c["집단"].unique().tolist()
            scale = BOROUGH_SCALE if kind == "자치구" else alt.Scale(domain=groups, range=CATEGORICAL[:len(groups)])
            base = alt.Chart(c).encode(x=alt.X("lt:Q", axis=HOUR_AXIS), color=alt.Color("집단:N", scale=scale, title=None))
            line = base.mark_line(interpolate="step-after", strokeWidth=2).encode(
                y=alt.Y("생존확률:Q", title="아직 열려 있는 비율 S(t)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["집단:N", alt.Tooltip("hours:Q", title="경과 시간(h)", format=",.1f"), alt.Tooltip("생존확률:Q", format=".1%")])
            layers = [alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 3], color="#898781").encode(y="y:Q"), line]
            if kind == "전체":
                layers.insert(1, base.mark_area(interpolate="step-after", opacity=0.2).encode(y="lo:Q", y2="hi:Q"))
            st.altair_chart(alt.layer(*layers).properties(height=340), alt=f"Kaplan-Meier 생존곡선 ({kind})")
            st.caption("점선(50%)과 곡선이 만나는 시점이 중앙 처리시간입니다. 가로축은 로그 축입니다.")
        k1, k2 = st.columns([3, 2])
        with k1, st.container(border=True):
            st.markdown("**집단별 요약**")
            st.dataframe(km_sum, hide_index=True, alt="집단별 Kaplan-Meier 요약",
                         column_config={"n": st.column_config.NumberColumn(format="localized"),
                                        "사건 수": st.column_config.NumberColumn(format="localized"),
                                        "KM 중앙값(h)": st.column_config.NumberColumn(format="%.1f")})
        with k2, st.container(border=True):
            st.markdown("**미종결을 버리면 생기는 치우침**")
            st.dataframe(bias.reset_index(), hide_index=True, alt="종결 건만 쓴 값과 Kaplan-Meier 값 비교",
                         column_config={col: st.column_config.NumberColumn(format="%.2f") for col in bias.columns})
            st.markdown("**로그순위 검정** (생존곡선이 같은가)")
            st.dataframe(t("stat5_logrank.csv"), hide_index=True, alt="로그순위 검정 결과",
                         column_order=["검정", "카이제곱", "자유도", "p", "판정 (α=0.05)"],
                         column_config={"p": st.column_config.NumberColumn(format="%.2e"),
                                        "카이제곱": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("Cox 비례위험 모형")
        cs = t("stat5_cox_summary.csv").iloc[0]
        st.caption(f"적합 {cs['적합 n']:,.0f}건 · C-index 학습 {cs['C-index (학습 표본)']:.3f}, 9월 접수분 평가 {cs['C-index (평가, 9월 접수)']:.3f} "
                   f"(0.5는 무작위, 1은 순서를 완벽히 맞힘) · 기준 범주: {cs['기준 범주']}")
        cox = t("stat5_cox_coef.csv")
        cplot = cox.rename(columns={"위험비(HR)": "hr", "HR CI 2.5%": "lo", "HR CI 97.5%": "hi", "유의 (α=0.05)": "sig"})
        x1, x2 = st.columns([3, 2])
        with x1, st.container(border=True):
            st.markdown("**위험비와 95% 신뢰구간** (1보다 크면 기준보다 빨리 종결, 회색은 유의하지 않음)")
            xs = alt.Scale(type="log")
            base = alt.Chart(cplot).encode(
                y=alt.Y("변수:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
                color=alt.Color("sig:N", scale=alt.Scale(domain=["예", "아니오"], range=[BLUE, OTHER_GRAY]), legend=None))
            st.altair_chart((alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 3], color="#898781").encode(x="x:Q")
                             + base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", scale=xs, title="위험비 (로그 축)"), x2="hi:Q")
                             + base.mark_circle(size=70, opacity=1).encode(
                                 x=alt.X("hr:Q", scale=xs),
                                 tooltip=["변수:N", alt.Tooltip("hr:Q", title="위험비", format=".3f"), "해석:N"])
                             ).properties(height=max(260, 24 * len(cplot))), alt="Cox 모형 위험비와 신뢰구간")
        with x2, st.container(border=True):
            st.markdown("**계수표**")
            st.dataframe(cox, hide_index=True, height=max(300, 24 * len(cplot) + 40), alt="Cox 모형 계수표",
                         column_order=["변수", "위험비(HR)", "HR CI 2.5%", "HR CI 97.5%", "p", "해석"],
                         column_config={"p": st.column_config.NumberColumn(format="%.2e")})

# ============================================================ 탭 6: 시계열
with tab6:
    stage_header("s9")
    if need("stat_06_timeseries.py", "stat6_stl_components.csv", "stat6_forecast.csv", "stat6_forecast_metrics.csv"):
        comp = t("stat6_stl_components.csv")
        comp["date"] = pd.to_datetime(comp["date"])
        strength = t("stat6_stl_strength.csv").set_index("지표")["값"]
        st.subheader("일별 접수 건수 분해: 추세 + 주간 계절성 + 나머지")
        with st.container(horizontal=True):
            st.metric("추세 세기", f"{strength['추세 세기 F_T']:.2f}", border=True, help="0~1. 1에 가까울수록 추세가 뚜렷합니다.")
            st.metric("주간 계절성 세기", f"{strength['계절성(주간) 세기 F_S']:.2f}", border=True, help="0~1. 1에 가까울수록 요일 무늬가 뚜렷합니다.")
            st.metric("계절성 진폭", f"{strength['계절성 진폭 (최대 - 최소, 건)']:,.0f}건", border=True, help="요일에 따라 벌어지는 최대 폭입니다.")
            st.metric("나머지 표준편차", f"{strength['나머지 표준편차 (건)']:,.0f}건", border=True, help="추세와 요일로 설명되지 않는 변동의 크기입니다.")
        with st.expander("용어 설명: 분해, 정상성, 자기상관"):
            st.markdown(
                "- **STL 분해**: 시계열을 길게 변하는 추세, 7일마다 반복되는 계절성, 둘로 설명되지 않는 나머지로 나눕니다.\n"
                "- **자기상관(ACF)**: 오늘 값과 k일 전 값의 상관. 시차 7, 14에서 높으면 주간 반복이 있다는 뜻입니다.\n"
                "- **정상성**: 평균과 분산이 시간에 따라 변하지 않는 성질. ARIMA 모형의 전제이며 차분으로 만듭니다.\n"
                "- **ADF 와 KPSS**: 귀무가설이 반대입니다. ADF는 '비정상', KPSS는 '정상'이 귀무가설입니다.\n"
                "- **계절 단순 예측**: '다음 주 월요일은 지난주 월요일과 같다'는 기준 모형. 다른 모형은 이것을 이겨야 의미가 있습니다.")
        s1, s2 = st.columns([3, 2])
        with s1, st.container(border=True):
            part = st.segmented_control("성분", ["관측값", "추세", "계절성", "나머지"], default="관측값", required=True)
            ch = alt.Chart(comp).encode(x=alt.X("date:T", title=None),
                                        tooltip=[alt.Tooltip("date:T", title="날짜"), alt.Tooltip(f"{part}:Q", format=",.0f")])
            ch = (ch.mark_bar(color=BLUE, size=2).encode(y=alt.Y(f"{part}:Q", title="건")) if part == "나머지"
                  else ch.mark_line(color=BLUE).encode(y=alt.Y(f"{part}:Q", title="건", scale=alt.Scale(zero=False))))
            st.altair_chart(ch.properties(height=300), alt=f"STL 분해 성분: {part}")
        with s2, st.container(border=True):
            st.markdown("**요일 효과** (평균적인 날 대비 건수)")
            wd = t("stat6_weekday_effect.csv").rename(columns={"요일 효과(건)": "effect"})
            st.altair_chart(alt.Chart(wd).mark_bar().encode(
                x=alt.X("요일:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("effect:Q", title="요일 효과 (건)"),
                color=alt.condition("datum.effect < 0", alt.value(ORANGE), alt.value(BLUE)),
                tooltip=["요일:N", alt.Tooltip("effect:Q", title="요일 효과", format="+,.0f"),
                         alt.Tooltip("평균 접수 건수:Q", format=",.0f")]).properties(height=335), alt="요일 효과 막대그래프")
        d1, d2 = st.columns(2)
        with d1, st.container(border=True):
            st.markdown("**정상성 검정**")
            st.dataframe(t("stat6_stationarity.csv"), hide_index=True, alt="ADF와 KPSS 정상성 검정 결과",
                         column_config={col: st.column_config.NumberColumn(format="%.3f") for col in ["ADF 통계량", "ADF p", "KPSS 통계량", "KPSS p"]})
            unusual = t("stat6_unusual_days.csv")
            st.markdown(f"**특이한 날** (나머지의 robust |z| > 3.5, {len(unusual)}일, 큰 순서)")
            st.dataframe(unusual, hide_index=True, height=220, alt="나머지 성분이 유난히 큰 날",
                         column_order=["date", "요일", "관측값", "나머지", "robust_z"],
                         column_config={"date": "날짜", "관측값": st.column_config.NumberColumn(format="localized"),
                                        "robust_z": st.column_config.NumberColumn("robust z", format="%.1f")})
        with d2, st.container(border=True):
            ac = t("stat6_acf_pacf.csv").rename(columns={"시차(일)": "lag", "유의 한계(±)": "bound"})
            which = st.segmented_control("자기상관", ["ACF", "PACF"], default="ACF", required=True)
            b = float(ac["bound"].iloc[0])
            band = alt.Chart(pd.DataFrame({"lo": [-b], "hi": [b]})).mark_rect(opacity=0.25, color=OTHER_GRAY).encode(y="lo:Q", y2="hi:Q")
            bars = alt.Chart(ac[ac["lag"] > 0]).mark_bar(color=BLUE, size=6).encode(
                x=alt.X("lag:Q", title="시차 (일)", axis=alt.Axis(values=list(range(0, 29, 7)))), y=alt.Y(f"{which}:Q", title=which),
                tooltip=[alt.Tooltip("lag:Q", title="시차"), alt.Tooltip(f"{which}:Q", format=".2f")])
            st.altair_chart((band + bars).properties(height=330), alt=f"{which} 그래프")
            st.caption("회색 띠 안의 막대는 0과 구별되지 않습니다.")
        temp = t("stat6_temperature_effect.csv")
        if temp is not None:
            with st.container(border=True):
                st.markdown("**기온 효과 재추정**: 오차의 자기상관을 허용하면 결론이 달라지는가")
                st.dataframe(temp, hide_index=True, alt="기온 효과: 단순 회귀와 ARMA 오차 회귀 비교",
                             column_order=["모형", "기온 계수 (건/°C)", "표준오차", "p", "CI 2.5%", "CI 97.5%", "잔차 자기상관", "비고"],
                             column_config={"p": st.column_config.NumberColumn(format="%.2e")})
                st.caption("단순 회귀의 잔차에 자기상관이 남아 있으면 그 표준오차와 p 값은 믿기 어렵습니다. "
                           "기온은 계절 추세와 함께 움직이므로 이 계수를 순수한 인과 효과로 읽으면 안 됩니다.")
        st.subheader("9월 접수 건수 예측")
        fc = t("stat6_forecast.csv").rename(columns={"SARIMA 95% 하한": "lo", "SARIMA 95% 상한": "hi", "계절 단순 예측": "naive"})
        fc["date"] = pd.to_datetime(fc["date"])
        met = t("stat6_forecast_metrics.csv")
        p1, p2 = st.columns([3, 2])
        with p1, st.container(border=True):
            long = fc.melt(id_vars="date", value_vars=["실제", "SARIMA", "naive"], var_name="계열", value_name="건수")
            long["계열"] = long["계열"].replace({"naive": "계절 단순 예측"})
            band = alt.Chart(fc).mark_area(opacity=0.18, color=BLUE).encode(x="date:T", y="lo:Q", y2="hi:Q")
            lines = alt.Chart(long).mark_line().encode(
                x=alt.X("date:T", title=None), y=alt.Y("건수:Q", title="접수 건수", scale=alt.Scale(zero=False)),
                color=alt.Color("계열:N", scale=alt.Scale(domain=["실제", "SARIMA", "계절 단순 예측"], range=["#52514e", BLUE, ORANGE]), title=None),
                strokeDash=alt.condition("datum['계열'] == '계절 단순 예측'", alt.value([5, 3]), alt.value([0])),
                tooltip=[alt.Tooltip("date:T", title="날짜"), "계열:N", alt.Tooltip("건수:Q", format=",.0f")])
            st.altair_chart((band + lines).properties(height=320), alt="9월 접수 건수 실제값과 예측값")
            st.caption("8월 31일까지의 자료로 30일을 한 번에 예측했습니다. 파란 띠는 SARIMA 95% 예측구간입니다. 기온은 예측에 쓰지 않았습니다.")
        with p2, st.container(border=True):
            st.markdown("**예측 오차 비교**")
            st.dataframe(met, hide_index=True, alt="예측 모형별 오차 지표",
                         column_order=["모형", "MAE", "RMSE", "MAPE(%)", "기준 대비 MAE 변화(%)"],
                         column_config={col: st.column_config.NumberColumn(format="%.1f") for col in ["MAE", "RMSE", "MAPE(%)", "기준 대비 MAE 변화(%)"]})
            cov = met["95% 예측구간 포함률(%)"].dropna()
            if len(cov):
                st.caption(f"SARIMA 95% 예측구간이 실제값을 포함한 비율: {cov.iloc[0]:.0f}%")
            with st.expander("SARIMA 후보 모형 (AIC 순)"):
                st.dataframe(t("stat6_model_selection.csv"), hide_index=True, alt="SARIMA 후보 모형과 AIC")

# ============================================================ 탭 7: 기관 비교
with tab7:
    stage_header("s8")
    if need("stat_07_agency_comparison.py", "stat7_anova.csv", "stat7_cell_summary.csv", "stat7_mixed_fixed.csv"):
        st.subheader("자치구 간 차이는 처리 속도 차이인가, 민원 구성 차이인가")
        with st.expander("용어 설명: 상호작용, 편 에타제곱, 임의효과, ICC"):
            st.markdown(
                "- **이원배치 분산분석**: 요인 두 개(자치구, 기관)가 평균에 주는 영향을 동시에 봅니다.\n"
                "- **상호작용**: 한 요인의 효과가 다른 요인의 수준에 따라 달라지는 것. 상호작용 그림에서 선이 평행하지 않으면 상호작용이 있습니다.\n"
                "- **편 η²**: 다른 효과를 걷어낸 뒤 그 효과가 설명하는 변동의 비율. 0.01 작음, 0.06 중간, 0.14 큼.\n"
                "- **임의효과**: 수준이 아주 많은 요인(민원 유형)을 계수 수백 개 대신 분산 하나로 다루는 방법입니다.\n"
                "- **ICC**: 전체 분산 중 민원 유형 차이로 생기는 비율. 클수록 유형을 무시한 비교가 위험합니다.")
        an = t("stat7_anova.csv")
        inter = an[an["변동 요인"].str.startswith("상호작용")].iloc[0]
        verdict = st.success if inter["p"] < 0.05 else st.info
        verdict(f"상호작용 자치구 × 기관: F = {inter['F']:,.1f}, p {p_fmt(inter['p'])}, 편 η² = {inter['편 η²']:.4f}  \n"
                + ("자치구의 효과가 기관에 따라 다릅니다. 주효과 하나로 요약하지 말고 기관별로 나눠 읽어야 합니다."
                   if inter["p"] < 0.05 else "자치구의 효과가 기관에 따라 달라진다는 증거가 약합니다. 주효과로 해석할 수 있습니다."),
                title="이원배치 분산분석 (칸마다 같은 수를 뽑은 균형 표본)", icon=":material/rule:")
        cells = ko(t("stat7_cell_summary.csv")).rename(columns={"기하평균(h)": "gm", "중앙값(h)": "med"})
        cells["lg"] = np.log1p(cells["gm"])
        a1, a2 = st.columns([3, 2])
        with a1, st.container(border=True):
            st.markdown("**상호작용 그림** (처리 시간 기하평균, 로그 축)")
            ags = cells["agency"].unique().tolist()
            yaxis = alt.Axis(values=list(np.log1p([0, 1, 6, 24, 72, 168, 720])), labelExpr="format(exp(datum.value) - 1, ',.0f') + 'h'", title="처리 시간")
            st.altair_chart(alt.Chart(cells).mark_line(point=True).encode(
                x=alt.X("자치구:N", sort=KO, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("lg:Q", axis=yaxis, scale=alt.Scale(zero=False)),
                color=alt.Color("agency:N", scale=alt.Scale(domain=ags, range=CATEGORICAL[:len(ags)]), title="기관"),
                tooltip=[alt.Tooltip("agency:N", title="기관"), "자치구:N", alt.Tooltip("gm:Q", title="기하평균(h)", format=",.1f"),
                         alt.Tooltip("med:Q", title="중앙값(h)", format=",.1f"), alt.Tooltip("n:Q", format=",")]).properties(height=340),
                alt="자치구와 기관의 상호작용 그림")
            st.caption("선들이 평행하면 자치구 효과가 기관과 무관하고, 교차하거나 벌어지면 상호작용이 있습니다.")
        with a2, st.container(border=True):
            st.markdown("**분산분석표**")
            st.dataframe(an, hide_index=True, alt="이원배치 분산분석표", column_order=["변동 요인", "자유도", "F", "p", "η²", "편 η²", "p (HC3 강건)"],
                         column_config={"F": st.column_config.NumberColumn(format="%.1f"), "p": st.column_config.NumberColumn(format="%.2e"),
                                        "p (HC3 강건)": st.column_config.NumberColumn(format="%.2e"),
                                        "η²": st.column_config.NumberColumn(format="%.4f"), "편 η²": st.column_config.NumberColumn(format="%.4f")})
            st.markdown("**가정 점검**")
            st.dataframe(t("stat7_anova_assumptions.csv"), hide_index=True, alt="분산분석 가정 점검", column_order=["점검", "판정", "비고"])
        with st.container(border=True):
            st.markdown("**기관별 자치구 차이** (단순 주효과, Kruskal-Wallis, Bonferroni 보정)")
            st.dataframe(t("stat7_simple_effects.csv"), hide_index=True, alt="기관별 자치구 차이 검정",
                         column_config={"n": st.column_config.NumberColumn(format="localized"), "p": st.column_config.NumberColumn(format="%.2e"),
                                        "p (Bonferroni)": st.column_config.NumberColumn(format="%.2e"),
                                        "ε²": st.column_config.NumberColumn(format="%.4f"),
                                        "Kruskal-Wallis H": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("민원 유형을 통제한 자치구 비교 (혼합효과 모형)")
        var = t("stat7_mixed_variance.csv")
        fx = t("stat7_mixed_fixed.csv")
        m1, m2 = st.columns([2, 3])
        with m1:
            st.metric("ICC (민원 유형)", f"{var['ICC'].iloc[0]:.2f}", border=True,
                      help="로그 처리시간 분산 가운데 민원 유형 차이로 생기는 비율입니다.")
            with st.container(border=True):
                st.markdown("**분산 성분**")
                st.dataframe(var, hide_index=True, alt="혼합효과 모형 분산 성분",
                             column_config={col: st.column_config.NumberColumn(format="%.3f") for col in var.columns[1:]})
        with m2, st.container(border=True):
            st.markdown(f"**자치구 계수** (기준: {fx['기준'].iloc[0]}, 0보다 크면 기준보다 느림)")
            parts = []
            for tag, col, lo, hi in [("A. 유형 무시 (OLS)", "A. OLS 계수 (유형 무시)", "A CI 2.5%", "A CI 97.5%"),
                                     ("B. 유형 통제 (혼합모형)", "B. 혼합모형 계수 (유형 통제)", "B CI 2.5%", "B CI 97.5%")]:
                parts.append(pd.DataFrame({"자치구": fx["자치구"], "모형": tag, "est": fx[col], "lo": fx[lo], "hi": fx[hi]}))
            fl = pd.concat(parts)
            base = alt.Chart(fl).encode(
                y=alt.Y("자치구:N", sort=None, title=None), yOffset=alt.YOffset("모형:N"),
                color=alt.Color("모형:N", scale=alt.Scale(range=[ORANGE, BLUE]), title=None, legend=alt.Legend(orient="bottom")))
            st.altair_chart((alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 3], color="#898781").encode(x="x:Q")
                             + base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="로그 처리시간 차이"), x2="hi:Q")
                             + base.mark_circle(size=70, opacity=1).encode(
                                 x="est:Q", tooltip=["자치구:N", "모형:N", alt.Tooltip("est:Q", title="계수", format="+.3f")])
                             ).properties(height=260), alt="자치구 계수: 유형 무시와 유형 통제 비교")
            st.caption("B가 A보다 0에 가까우면 민원 구성 차이가 격차를 부풀리고 있었고, 더 멀어지거나 부호가 바뀌면 "
                       "구성 차이가 실제 격차를 가리고 있었다는 뜻입니다(심슨의 역설).")
            st.dataframe(fx, hide_index=True, alt="자치구별 유형 통제 전후 변화", column_order=["자치구", "A 배율 exp(b)", "B 배율 exp(b)", "유형 통제 후 변화"],
                         column_config={"A 배율 exp(b)": st.column_config.NumberColumn("A 배율", format="%.2f"),
                                        "B 배율 exp(b)": st.column_config.NumberColumn("B 배율", format="%.2f")})


# ============================================================ 지도 공통 (탭 8, 9)
SEQ_STOPS = ["#fcfdbf", "#fec287", "#f1605d", "#b63679", "#641a80", "#1d1147"]      # 연함 -> 진함
DIV_STOPS = ["#2166ac", "#92c5de", "#f2f1ec", "#f4a582", "#b2182b"]                 # 낮음 - 중간 - 높음
LISA_COLORS = {"HH (핫스팟)": "#e34948", "LL (콜드스팟)": "#2a78d6", "HL (낮은 곳 속 높은 섬)": "#eda100",
               "LH (높은 곳 속 낮은 구멍)": "#1baf7a", "유의하지 않음": "#c9c8bf"}
FIGS = TABLES.parent / "figures"


def hex_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def ramp(values01, stops, alpha=200):
    """0~1 값을 색 띠 위의 색으로 바꾼다 (지도 점 색칠용)."""
    v = np.clip(np.asarray(values01, dtype=float), 0, 1)
    pts = np.array([hex_rgb(s) for s in stops], dtype=float)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.column_stack([np.interp(v, pos, pts[:, k]) for k in range(3)]).astype(int)
    return [[int(r), int(g), int(b), alpha] for r, g, b in rgb]


def deck_map(df, radius, tooltip, height=520, zoom=9.5, lat=40.71, lon=-73.94):
    """위경도 점을 실제 지도 위에 그린다. df 에는 lat, lon, color(RGBA 목록) 열이 있어야 한다."""
    import pydeck as pdk

    layer = pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="color", get_radius=radius,
                      radius_min_pixels=1.5, pickable=True, stroked=False)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_style=None, tooltip={"text": tooltip},
                             initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)), height=height)


def legend_strip(stops, left, right):
    cells = "".join(f"<span style='display:inline-block;width:26px;height:10px;background:{c}'></span>" for c in stops)
    st.html(f"<div style='font-size:0.8rem;opacity:0.8'>{left} {cells} {right}</div>")


def legend_dots(mapping):
    st.html("<div style='font-size:0.8rem;opacity:0.85'>" + " &nbsp; ".join(
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c}'></span> {k}"
        for k, c in mapping.items()) + "</div>")


# ============================================================ 탭 8: 공간 분석
with tab8:
    stage_header("s11")
    if need("stat_08_spatial.py", "stat8_kde_map.csv", "stat8_dbscan_points.csv", "stat8_moran_lisa.csv"):
        st.subheader("민원은 어디에 몰려 있는가: 세 가지 방법, 세 가지 질문")
        st.markdown(
            "| 방법 | 묻는 것 | 넣는 자료 | 얻는 것 | p 값 |\n|---|---|---|---|---|\n"
            "| **KDE** | 어디가 진한가 | 점(좌표) | 매끄러운 밀도 지도 | 없음 |\n"
            "| **DBSCAN** | 밀집된 덩어리가 몇 개, 어디인가 | 점(좌표) | 점마다 군집 번호 또는 잡음 | 없음 |\n"
            "| **Moran's I** | 가까운 곳끼리 값이 비슷한가 | 면(격자 칸) | 상관계수 하나와 국지 지도 | 있음 |\n")
        st.caption("KDE는 그림을, DBSCAN은 경계를, Moran's I는 '우연이 아니다'라는 증거를 줍니다. "
                   "그래서 보통 KDE로 훑어보고, DBSCAN으로 덩어리를 확정하고, Moran's I로 검정합니다.")
        rep = t("stat8_repeat_locations.csv")
        if rep is not None:
            r0 = rep.iloc[0]
            st.warning(
                f"이 자료의 점은 '문제가 있는 곳'이 아니라 '신고가 들어온 곳'입니다. 좌표는 건물 주소 단위라서 같은 곳의 신고가 한 점에 겹칩니다. "
                f"가장 심한 예로, {r0['자치구']}의 한 주소에서만 {r0['민원 건수']:,.0f}건이 들어왔습니다 "
                f"({r0['가장 많은 유형']} {r0['그 유형 비중(%)']:.0f}%, 전체 민원의 {r0['전체 민원 중 비중(%)']:.1f}%). "
                "이런 주소 하나가 지도에 큰 봉우리를 만들기 때문에, 봉우리를 '시끄러운 동네'로 바로 읽으면 안 됩니다.",
                title="지도를 읽기 전에", icon=":material/warning:")
            with st.expander("반복 신고가 가장 많은 주소"):
                st.dataframe(rep, hide_index=True, alt="반복 신고가 많은 좌표",
                             column_config={"민원 건수": st.column_config.NumberColumn(format="localized")})

        # ---------------- KDE
        st.divider()
        st.subheader("① KDE 핫스팟: 밀도 지도")
        with st.expander("처음 배우는 사람을 위한 설명: KDE란", expanded=True):
            st.markdown(
                "**비유.** 민원 하나하나가 지도 위에 모래를 한 줌씩 떨어뜨린다고 생각해 보세요. 민원이 많은 곳은 모래가 쌓여 언덕이 됩니다. "
                "그 언덕의 높이가 밀도입니다. 히스토그램을 매끄럽게 만든 것과 같습니다.\n\n"
                "**대역폭(bandwidth)** 은 모래 한 줌이 퍼지는 너비입니다. KDE에서 가장 중요한 선택입니다.\n"
                "- 너무 좁으면 건물 하나하나가 뾰족하게 솟아 큰 흐름이 안 보입니다.\n"
                "- 너무 넓으면 도시 전체가 한 덩어리로 뭉개집니다.\n"
                "- 정답은 없고, 보고 싶은 현상의 크기로 정합니다. 여기서는 동네 수준을 보려고 0.5km를 썼습니다.\n\n"
                "**입지계수(LQ)** 는 '이 동네 민원 중 이 유형의 비중이 시 전체 평균의 몇 배인가'입니다. "
                "밀도 지도는 사람이 많은 곳이 진하게 나오는 경향이 있지만, LQ 지도는 신고량의 차이를 걷어내고 그 동네 민원의 성격만 보여 줍니다. "
                "LQ가 1이면 시 평균과 같고, 2면 두 배로 두드러집니다.\n\n"
                "| 장점 | 단점 |\n|---|---|\n"
                "| 행정구역 경계에 묶이지 않는다 | 대역폭에 따라 결과가 크게 달라진다 |\n"
                "| 연속적인 지도로 한눈에 보인다 | 어디까지가 핫스팟인지 경계를 정해 주지 않는다 |\n"
                "| 계산이 단순하다 | 우연인지 아닌지 검정해 주지 않는다 |\n"
                "| | 밀도가 높다고 문제가 심하다는 뜻은 아니다 (신고량, 인구의 영향) |")
        kde = t("stat8_kde_map.csv")
        conc = t("stat8_kde_concentration.csv")
        c1, c2 = st.columns([3, 2])
        with c1, st.container(border=True):
            layers = kde["분류"].unique().tolist()
            f1, f2 = st.columns(2)
            layer = f1.segmented_control("민원 분류", layers, default=layers[0], required=True, key="kde_layer")
            opts = ["밀도 (건/km²)"] if layer == "전체" else ["밀도 (건/km²)", "입지계수 (LQ)"]
            measure = f2.segmented_control("표시 값", opts, default=opts[0], required=True, key=f"kde_measure_{layer == '전체'}")
            d = kde[kde["분류"] == layer].copy()
            if measure.startswith("밀도"):
                lo, hi = np.log10(max(d["밀도(건/km²)"].quantile(0.05), 1)), np.log10(d["밀도(건/km²)"].quantile(0.995))
                d["color"] = ramp((np.log10(d["밀도(건/km²)"].clip(lower=1)) - lo) / (hi - lo), SEQ_STOPS)
            else:
                d["color"] = ramp((np.log2(d["입지계수(LQ)"].clip(0.25, 4)) + 2) / 4, DIV_STOPS)
            d["밀도"] = d["밀도(건/km²)"].map("{:,.0f}".format)
            d["LQ"] = d["입지계수(LQ)"].map("{:.2f}".format)
            deck_map(d, 260, "밀도 {밀도} 건/km²\n입지계수 {LQ}")
            if measure.startswith("밀도"):
                legend_strip(SEQ_STOPS, "낮음", "높음 (로그 눈금)")
            else:
                legend_strip(DIV_STOPS, "시 평균의 1/4", "4배 (가운데 = 시 평균)")
        with c2:
            row = conc[conc["분류"] == layer].iloc[0]
            with st.container(border=True):
                st.markdown(f"**얼마나 몰려 있는가: {layer}**")
                with st.container(horizontal=True):
                    st.metric("상위 5% 면적의 민원", f"{row['상위 5% 면적의 민원 점유율(%)']:.0f}%", border=True)
                    st.metric("지니계수", f"{row['지니계수']:.2f}", border=True)
                st.caption("지니계수는 0이면 모든 곳에 고르게 퍼진 것이고, 1에 가까울수록 좁은 곳에 몰린 것입니다. "
                           "민원이 고르게 퍼져 있다면 상위 5% 면적은 민원의 5%만 차지해야 합니다.")
                st.dataframe(conc, hide_index=True, alt="분류별 집중도",
                             column_order=["분류", "상위 5% 면적의 민원 점유율(%)", "상위 20% 면적의 민원 점유율(%)", "지니계수"],
                             column_config={"상위 5% 면적의 민원 점유율(%)": st.column_config.NumberColumn("상위 5% 면적", format="%.0f%%"),
                                            "상위 20% 면적의 민원 점유율(%)": st.column_config.NumberColumn("상위 20% 면적", format="%.0f%%"),
                                            "지니계수": st.column_config.NumberColumn(format="%.2f")})
            with st.container(border=True):
                st.markdown("**밀도가 가장 높은 지점**")
                pk = t("stat8_kde_peaks.csv")
                st.dataframe(pk[pk["분류"] == layer], hide_index=True, alt="밀도 봉우리",
                             column_order=["순위", "자치구", "lat", "lon", "시 평균 대비 배율"],
                             column_config={"lat": st.column_config.NumberColumn(format="%.4f"), "lon": st.column_config.NumberColumn(format="%.4f"),
                                            "시 평균 대비 배율": st.column_config.NumberColumn(format="%.1f배")})
        most = conc[conc["분류"] != "전체"].sort_values("지니계수", ascending=False).iloc[0]
        least = conc[conc["분류"] != "전체"].sort_values("지니계수").iloc[0]
        st.info(
            f"가장 좁은 곳에 몰린 분류는 **{most['분류']}** (지니 {most['지니계수']:.2f}, 상위 5% 면적에 {most['상위 5% 면적의 민원 점유율(%)']:.0f}%)이고, "
            f"가장 고르게 퍼진 분류는 **{least['분류']}** (지니 {least['지니계수']:.2f})입니다. "
            "몰려 있는 유형은 특정 장소를 겨냥한 대응이 효과적이고, 퍼져 있는 유형은 도시 전체에 걸친 대응이 필요하다는 뜻으로 읽을 수 있습니다. "
            "다만 봉우리 가운데 일부는 위에서 본 반복 신고 주소 한 곳이 만든 것입니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")
        if (FIGS / "stat8_01_kde_bandwidth.png").exists():
            with st.expander("대역폭을 바꾸면 지도가 얼마나 달라지는가"):
                st.image(str(FIGS / "stat8_01_kde_bandwidth.png"), alt="대역폭 0.2, 0.5, 2km로 그린 같은 자료의 KDE 지도")

        # ---------------- DBSCAN
        st.divider()
        info = t("stat8_dbscan_info.csv").iloc[0]
        st.subheader(f"② DBSCAN: 밀집된 덩어리 찾기 (대상: {info['대상 유형']})")
        with st.expander("처음 배우는 사람을 위한 설명: DBSCAN과 K-means의 차이", expanded=True):
            st.markdown(
                "**DBSCAN의 생각.** '주변에 이웃이 충분히 많은 점들이 이어진 것'을 하나의 군집으로 봅니다. 정할 것은 두 가지입니다.\n"
                "- **eps**: 이웃으로 칠 거리(반경)\n"
                "- **min_samples**: 그 반경 안에 점이 최소 몇 개 있어야 '밀집'으로 볼 것인가\n\n"
                "점은 세 종류로 나뉩니다. 이웃이 충분한 **핵심점**, 핵심점 곁에 붙은 **경계점**, 어디에도 속하지 않는 **잡음점**입니다.\n\n"
                "**같은 군집분석이지만 K-means와는 생각이 정반대입니다.**\n\n"
                "| | K-means | DBSCAN | HDBSCAN |\n|---|---|---|---|\n"
                "| 군집이란 | 중심에 가까운 점들의 모임 | 밀집된 점들이 이어진 덩어리 | 여러 밀도 수준에서 오래 유지되는 덩어리 |\n"
                "| 군집 수 | 미리 정한다 (k) | 결과로 나온다 | 결과로 나온다 |\n"
                "| 군집 모양 | 둥근 덩어리 | 자유 (길쭉해도 됨) | 자유 |\n"
                "| 외딴 점 | 가장 가까운 군집에 억지로 넣는다 | 잡음으로 남긴다 | 잡음으로 남긴다 |\n"
                "| 정할 것 | k | eps, min_samples | 최소 군집 크기 |\n"
                "| 약점 | 이상값에 중심이 끌려간다 | 밀도가 다른 곳이 섞이면 eps 하나로 못 맞춘다 | 계산이 무겁고 결과 설명이 어렵다 |\n\n"
                "**이 자료에서의 설계.** 좌표가 건물 단위라 같은 업소에 대한 신고가 한 점에 겹칩니다. 그대로 돌리면 '신고가 반복된 건물 하나'가 군집이 됩니다. "
                "그래서 민원 건이 아니라 **서로 다른 신고 위치**를 대상으로 돌렸습니다. "
                f"군집의 뜻은 '소음 신고 대상 업소가 반경 {info['선택한 eps(m)']:.0f}m 안에 {info['min_samples']:.0f}곳 이상 모인 구역', 곧 유흥가입니다.")
        cmp8 = t("stat8_cluster_method_compare.csv")
        dbp = t("stat8_dbscan_points.csv")
        db_row = cmp8[cmp8["방법"].str.startswith("DBSCAN")].iloc[0]
        with st.container(horizontal=True):
            st.metric("신고 위치", f"{info['신고 위치 수']:,.0f}곳", border=True, help=f"민원 {info['민원 건수']:,.0f}건이 들어온 서로 다른 좌표의 수입니다.")
            st.metric("DBSCAN 군집", f"{db_row['군집 수']:.0f}개", border=True, help="밀집 구역(유흥가)의 수입니다.")
            st.metric("잡음 위치", f"{db_row['잡음 위치 비율(%)']:.0f}%", border=True, help="어느 밀집 구역에도 속하지 않는, 흩어진 신고 위치의 비율입니다.")
            st.metric("군집 반경 중앙값", f"{db_row['군집 반경 중앙값(m)']:.0f}m", border=True, help="군집 하나의 전형적인 크기입니다.")
        g1, g2 = st.columns([3, 2])
        with g1, st.container(border=True):
            which = st.segmented_control("같은 점에 적용한 방법", ["DBSCAN", "HDBSCAN", "K-means"], default="DBSCAN", required=True, key="db_method")
            col = {"DBSCAN": "dbscan", "HDBSCAN": "hdbscan", "K-means": "kmeans"}[which]
            p = dbp.copy()
            pal = [hex_rgb(c) for c in CATEGORICAL]
            shuffled = (p[col] * 7919) % len(pal)                 # 이웃한 번호가 같은 색이 되지 않게 섞는다
            p["color"] = [[201, 200, 191, 110] if lab < 0 else pal[int(s)] + [230] for lab, s in zip(p[col], shuffled)]
            p["군집"] = np.where(p[col] < 0, "잡음", "군집 " + p[col].astype(str))
            deck_map(p, 45, "{군집}\n민원 {민원건수}건", zoom=10.6, lat=40.735, lon=-73.975)
            st.caption("회색 점은 잡음입니다. K-means로 바꿔 보면 모든 점이 색을 받고(잡음이 없음) 도시가 조각조각 나뉘는 것을 볼 수 있습니다. "
                       "색은 군집을 구별하기 위한 것이며 같은 색이 여러 군집에 쓰입니다.")
        with g2:
            with st.container(border=True):
                st.markdown("**세 방법의 결과 비교**")
                st.dataframe(cmp8, hide_index=True, alt="K-means, DBSCAN, HDBSCAN 결과 비교",
                             column_order=["방법", "군집 수", "잡음 위치 비율(%)", "군집 반경 중앙값(m)", "실루엣 (잡음 제외)"],
                             column_config={"잡음 위치 비율(%)": st.column_config.NumberColumn("잡음", format="%.0f%%"),
                                            "군집 반경 중앙값(m)": st.column_config.NumberColumn("반경 중앙값", format="%.0f m"),
                                            "실루엣 (잡음 제외)": st.column_config.NumberColumn("실루엣", format="%.2f")})
            with st.container(border=True):
                st.markdown("**민원이 가장 많은 DBSCAN 군집**")
                st.dataframe(t("stat8_dbscan_clusters.csv").head(8), hide_index=True, alt="민원이 많은 DBSCAN 군집",
                             column_order=["자치구", "신고 위치 수", "민원 건수", "민원 점유율(%)", "중심 lat", "중심 lon"],
                             column_config={"민원 점유율(%)": st.column_config.NumberColumn("점유율", format="%.1f%%"),
                                            "중심 lat": st.column_config.NumberColumn(format="%.4f"),
                                            "중심 lon": st.column_config.NumberColumn(format="%.4f")})
        km_row = cmp8[cmp8["방법"].str.startswith("K-means")].iloc[0]
        st.info(
            f"DBSCAN은 신고 위치의 {db_row['잡음 위치 비율(%)']:.0f}%를 잡음으로 남기고 나머지를 {db_row['군집 수']:.0f}개의 좁은 구역"
            f"(반경 중앙값 {db_row['군집 반경 중앙값(m)']:.0f}m)으로 묶었습니다. 상업 소음 신고의 절반 이상은 특정 유흥가가 아니라 흩어진 개별 업소에서 나온다는 뜻입니다. "
            f"K-means는 군집 수를 같게 맞춰도 반경이 {km_row['군집 반경 중앙값(m)']:.0f}m로 훨씬 넓습니다. 외딴 점까지 모두 떠안기 때문입니다. "
            "'유흥가 찾기'라는 목적에는 DBSCAN이, '도시를 빠짐없이 구역으로 나누기'에는 K-means가 맞습니다. 어느 쪽이 옳다기보다 질문이 다릅니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")
        e1, e2 = st.columns(2)
        with e1, st.container(border=True):
            st.markdown("**eps 고르기: k-거리 그래프**")
            kd = t("stat8_dbscan_k_distance.csv").rename(columns={"순위 비율": "rank", "k-거리(m)": "dist"})
            kd["pct"] = kd["rank"] * 100
            line = alt.Chart(kd).mark_line(color=BLUE).encode(
                x=alt.X("pct:Q", title="신고 위치 (k-거리가 짧은 순, %)"),
                y=alt.Y("dist:Q", title="10번째 이웃까지의 거리 (m)", scale=alt.Scale(type="log")),
                tooltip=[alt.Tooltip("pct:Q", format=".0f"), alt.Tooltip("dist:Q", format=",.0f")])
            rules = alt.Chart(pd.DataFrame({"y": [info["선택한 eps(m)"], info["k-거리 무릎(m)"]], "뜻": ["선택한 eps", "자동 규칙의 무릎"]})).mark_rule(
                strokeDash=[5, 3]).encode(y="y:Q", color=alt.Color("뜻:N", scale=alt.Scale(range=["#898781", ORANGE]), title=None,
                                                                 legend=alt.Legend(orient="bottom")))
            st.altair_chart((line + rules).properties(height=300), alt="k-거리 그래프")
            st.caption(f"각 점에서 10번째로 가까운 이웃까지의 거리를 짧은 순으로 늘어놓은 것입니다. 급히 솟기 시작하는 '무릎'이 eps 후보입니다. "
                       f"자동으로 찾은 무릎은 {info['k-거리 무릎(m)']:.0f}m였지만, 그 값을 쓰면 촘촘한 맨해튼에서 여러 유흥가가 한 덩어리로 이어집니다. "
                       f"그래서 걸어서 한두 블록인 {info['선택한 eps(m)']:.0f}m로 정했습니다.")
        with e2, st.container(border=True):
            st.markdown("**민감도: eps와 min_samples를 바꾸면 군집 수가 어떻게 변하나**")
            sw = t("stat8_dbscan_sweep.csv").rename(columns={"eps(m)": "eps", "군집 수": "n"})
            sw["eps"] = sw["eps"].astype(int).astype(str) + "m"
            sw["ms"] = "min_samples " + sw["min_samples"].astype(str)
            base = alt.Chart(sw).encode(x=alt.X("eps:N", sort=None, title="eps"), y=alt.Y("ms:N", title=None))
            st.altair_chart((base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                                     tooltip=["eps:N", "ms:N", alt.Tooltip("n:Q", title="군집 수")])
                             + base.mark_text().encode(text="n:Q", color=alt.condition("datum.n > 200", alt.value("white"), alt.value("#0b0b0b")))
                             ).properties(height=300), alt="eps와 min_samples에 따른 군집 수")
            st.caption("같은 자료인데 설정에 따라 군집이 4개에서 393개까지 나옵니다. DBSCAN의 가장 큰 약점입니다. "
                       "eps가 작으면 군집이 부서져 사라지고, 크면 서로 합쳐져 수가 줄어듭니다.")

        # ---------------- Moran
        st.divider()
        st.subheader("③ Moran's I: 가까운 곳끼리 비슷한가")
        with st.expander("처음 배우는 사람을 위한 설명: 공간 자기상관", expanded=True):
            st.markdown(
                "**질문.** 지도를 1km 격자로 나누고 칸마다 값을 구했을 때, 붙어 있는 칸끼리 값이 비슷할까요? "
                "비슷하다면 그 값은 '동네의 성질'이고, 아니라면 위치와 무관하게 흩어져 있는 것입니다.\n\n"
                "**Moran's I** 는 '내 값'과 '내 이웃들의 평균값'의 상관계수입니다. 보통의 상관계수가 두 변수 사이의 관계를 본다면, "
                "이것은 한 변수가 공간에서 자기 자신과 맺는 관계를 봅니다. 그래서 공간 자기상관이라고 부릅니다.\n"
                "- **I > 0**: 비슷한 값끼리 모여 있다 (0.3 이상이면 뚜렷, 0.5 이상이면 강함)\n"
                "- **I ≈ 0**: 공간적으로 무작위\n"
                "- **I < 0**: 높은 값과 낮은 값이 체스판처럼 번갈아 놓인다 (드묾)\n\n"
                "**순열 검정.** 값들을 칸에 무작위로 다시 뿌려 I를 계산하는 일을 999번 반복합니다. "
                "실제 I가 그 999개보다 훨씬 크면 '우연히 이렇게 뭉칠 수는 없다'고 판단합니다. 그 비율이 p 값입니다.\n\n"
                "**국지 Moran(LISA)** 은 '어디가' 뭉쳐 있는지를 칸마다 판정합니다.\n"
                "- **HH**: 나도 높고 이웃도 높다 (핫스팟) · **LL**: 나도 낮고 이웃도 낮다 (콜드스팟)\n"
                "- **HL / LH**: 나만 주변과 다르다 (공간 이상점)\n\n"
                "| 장점 | 단점 |\n|---|---|\n"
                "| 뭉침이 우연인지 p 값으로 판단한다 (KDE, DBSCAN에는 없음) | '이웃'을 어떻게 정의하느냐에 따라 결과가 달라진다 |\n"
                "| 숫자 하나로 요약되어 변수끼리 비교하기 쉽다 | 격자 크기를 바꾸면 값이 달라진다 (MAUP) |\n"
                "| LISA로 위치까지 짚어 준다 | 칸이 수백 개라 LISA는 우연히 유의한 칸이 섞인다 |\n\n"
                "**KDE 핫스팟과 LISA의 HH는 다릅니다.** KDE는 '여기가 진하다'는 그림이고, HH는 '여기와 주변이 함께 높으며 우연으로 보기 어렵다'는 검정 결과입니다.")
        glob = t("stat8_moran_global.csv")
        MI = "Moran's I"
        lisa = t("stat8_moran_lisa.csv")
        top_g, low_g = glob.iloc[0], glob.iloc[-1]
        adj = glob[glob["변수"] == "유형 보정 로그 처리시간"].iloc[0]
        m1, m2 = st.columns([2, 3])
        with m1, st.container(border=True):
            st.markdown("**전역 Moran's I** (1km 격자, 퀸 인접, 순열 999번)")
            g = glob.rename(columns={"Moran's I": "I"})
            st.altair_chart(hbar(g, "변수", "I", label_fmt=".2f", height=250), alt="변수별 Moran's I")
            st.dataframe(glob, hide_index=True, alt="전역 Moran's I 결과", column_order=["변수", "Moran's I", "z 점수", "유사 p 값", "세기"],
                         column_config={"Moran's I": st.column_config.NumberColumn(format="%.3f"), "z 점수": st.column_config.NumberColumn(format="%.1f"),
                                        "유사 p 값": st.column_config.NumberColumn(format="%.3f")})
        with m2, st.container(border=True):
            var = st.selectbox("LISA 지도에 표시할 변수", glob["변수"].tolist(), index=int(glob["변수"].tolist().index("유형 보정 로그 처리시간")))
            d = lisa[lisa["변수"] == var].copy()
            d["color"] = [hex_rgb(LISA_COLORS[k]) + [215 if k != "유의하지 않음" else 120] for k in d["유형"]]
            d["값표시"] = d["값"].map("{:.3f}".format)
            deck_map(d, 430, "{유형}\n값 {값표시}\n{자치구}", height=430)
            legend_dots(LISA_COLORS)
        s1, s2 = st.columns([3, 2])
        with s1, st.container(border=True):
            st.markdown(f"**Moran 산점도: {var}**")
            sd = d.rename(columns={"표준화 값": "z", "이웃 평균(표준화)": "lag"})
            slope = float(np.polyfit(sd["z"], sd["lag"], 1)[0])
            pts_c = alt.Chart(sd).mark_circle(size=28, opacity=0.8).encode(
                x=alt.X("z:Q", title="내 값 (표준화)"), y=alt.Y("lag:Q", title="이웃 평균 (표준화)"),
                color=alt.Color("유형:N", scale=alt.Scale(domain=list(LISA_COLORS), range=list(LISA_COLORS.values())), legend=None),
                tooltip=["유형:N", "자치구:N", alt.Tooltip("z:Q", format=".2f"), alt.Tooltip("lag:Q", format=".2f")])
            xr = [float(sd["z"].min()), float(sd["z"].max())]
            fit = alt.Chart(pd.DataFrame({"z": xr, "lag": [slope * v for v in xr]})).mark_line(color="#52514e").encode(x="z:Q", y="lag:Q")
            st.altair_chart((pts_c + fit).properties(height=300), alt=f"{var} Moran 산점도")
            st.caption(f"직선의 기울기가 곧 Moran's I입니다 (여기서는 {slope:.2f}). 오른쪽 위(HH)와 왼쪽 아래(LL)에 점이 많을수록 I가 커집니다.")
        with s2, st.container(border=True):
            st.markdown("**격자 크기를 바꿔도 결론이 같은가 (MAUP)**")
            st.dataframe(t("stat8_moran_maup.csv"), hide_index=True, alt="격자 크기별 Moran's I",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in t("stat8_moran_maup.csv").columns if c.startswith("I:")})
            st.caption("같은 점 자료라도 어떤 크기의 구역으로 묶느냐에 따라 통계가 달라집니다(공간 단위 임의성 문제). "
                       "크기를 바꿔도 I가 계속 양수이고 크다면 결론은 믿을 만합니다.")
        st.info(
            f"모든 변수에서 공간 군집이 유의합니다(p = {glob['유사 p 값'].max():.3f}). 가장 강한 것은 **{top_g['변수']}** (I = {top_g[MI]:.2f}), "
            f"가장 약한 것은 **{low_g['변수']}** (I = {low_g[MI]:.2f})입니다. "
            f"특히 **유형 보정 로그 처리시간**의 I가 {adj[MI]:.2f}로 높습니다. "
            "민원 유형의 차이를 걷어낸 뒤에도 '같은 유형의 민원이 빨리 처리되는 동네'와 '늦게 처리되는 동네'가 지도에서 덩어리를 이룬다는 뜻입니다. "
            "8단계(기관 비교)에서 처리 시간의 대부분을 유형이 결정한다고 했지만, 남은 차이에는 뚜렷한 지역성이 있습니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

# ============================================================ 탭 9: 군집·PCA
with tab9:
    stage_header("s12")
    if need("stat_09_cluster_pca.py", "stat9_units.csv", "stat9_pca_variance.csv", "stat9_k_selection.csv"):
        st.subheader("어떤 동네인가: 민원 구성으로 동네를 요약하고 묶기")
        units = t("stat9_units.csv")
        ksel = t("stat9_k_selection.csv")
        pvar = t("stat9_pca_variance.csv")
        k_best = int(ksel.loc[ksel["선택"] == "선택", "k"].iloc[0])
        krow = ksel[ksel["k"] == k_best].iloc[0]
        st.markdown(
            "11단계(공간 분석)는 지도 위의 **위치**를 분석했습니다. 여기서는 위치를 잠시 잊고, 1km 격자 칸 하나하나를 "
            "'민원 구성이 이러한 동네'라는 **프로필**(변수 11개)로 봅니다.\n\n"
            "| 방법 | 묻는 것 | 줄이는 대상 | 얻는 것 |\n|---|---|---|---|\n"
            "| **PCA** | 변수 11개를 몇 개의 축으로 줄여 말할 수 있는가 | 변수(열) | 새 축과 각 칸의 좌표 |\n"
            "| **K-means** | 프로필이 닮은 칸끼리 묶으면 몇 가지 유형인가 | 관측값(행) | 칸마다 군집 번호 |\n")
        st.caption("둘은 경쟁하는 방법이 아니라 함께 쓰는 방법입니다. PCA로 축을 줄여 그림을 그리고, 그 위에 K-means 군집을 색칠합니다. "
                   "둘 다 정답이 없는 자료에서 구조를 찾는 비지도 학습이라, '맞혔는가'가 아니라 '안정적이고 해석할 수 있는가'로 평가합니다.")
        with st.container(horizontal=True):
            st.metric("분석 단위", f"{len(units):,}칸", border=True, help="민원 300건 이상이고 한 주소가 절반을 넘지 않는 1km 격자 칸입니다.")
            st.metric("PC1 + PC2 설명 분산", f"{pvar['누적(%)'].iloc[1]:.0f}%", border=True, help="축 두 개로 요약되는 정보의 비율입니다.")
            st.metric("선택한 군집 수", f"{k_best}개", border=True, help="안정성과 실루엣으로 고른 k입니다.")
            st.metric("실루엣", f"{krow['실루엣']:.2f}", border=True, help="0.5 이상이면 뚜렷, 0.25~0.5는 약함, 그 아래는 군집 구조가 희미합니다.")
        with st.expander("분석 단위와 제외 기준"):
            st.dataframe(t("stat9_input_log.csv"), hide_index=True, alt="군집분석 입력 단위와 제외 건수",
                         column_config={"민원 건수": st.column_config.NumberColumn(format="localized")})
            st.caption("변수는 민원 분류 9개의 비중, 야간(22~05시) 비중, 주말 비중입니다. 모두 평균 0, 표준편차 1로 표준화한 뒤 분석했습니다. "
                       "거리에 바탕을 둔 방법이라 표준화하지 않으면 값의 범위가 큰 변수가 결과를 좌우합니다.")

        # ---------------- PCA
        st.divider()
        st.subheader("① PCA: 변수 11개를 몇 개의 축으로")
        with st.expander("처음 배우는 사람을 위한 설명: PCA란", expanded=True):
            st.markdown(
                "**왜 필요한가.** 변수 11개는 서로 얽혀 있습니다. 어떤 민원이 많은 동네는 다른 민원이 적은 식으로 같이 움직입니다. "
                "같이 움직인다면 11개를 따로 볼 필요 없이 '같이 움직이는 방향' 몇 개만 보면 됩니다. 11차원은 그릴 수 없지만 2차원은 산점도로 그릴 수 있습니다.\n\n"
                "**하는 일.** 길쭉한 구름 모양의 점들을 가장 길게 뻗은 방향으로 돌려 보는 것입니다. "
                "가장 길게 뻗은 방향이 첫 번째 주성분(PC1), 그와 직각이면서 그다음으로 길게 뻗은 방향이 PC2입니다.\n\n"
                "**용어.**\n"
                "- **주성분(PC)**: 새로 만든 축. 원래 변수들의 가중합입니다.\n"
                "- **적재값(loading)**: 그 가중치. 축의 뜻을 읽는 열쇠입니다. 부호가 반대인 변수들은 그 축에서 서로 맞서는 관계입니다.\n"
                "- **설명 분산 비율**: 전체 정보(퍼짐) 가운데 그 축이 담아낸 비율입니다.\n"
                "- **고윳값**: 그 축의 분산. 1보다 크면 원래 변수 하나보다 많은 정보를 담았다는 뜻입니다(카이저 기준).\n\n"
                "| 장점 | 단점 |\n|---|---|\n"
                "| 여러 변수를 두세 개 축으로 줄여 그림으로 볼 수 있다 | 직선적인 관계만 잡는다 |\n"
                "| 서로 상관된 변수의 중복을 없앤다 | 축의 뜻은 사람이 해석해야 하고, 애매한 축도 많다 |\n"
                "| 군집분석 전에 쓰면 잡음을 줄일 수 있다 | 표준화하지 않으면 단위가 큰 변수가 축을 독차지한다 |\n"
                "| | 합이 100%인 비중 자료에서는 기계적인 음의 상관이 섞인다 |")
        p1, p2 = st.columns([2, 3])
        with p1, st.container(border=True):
            st.markdown("**스크리 그림** (막대: 축별 설명 분산, 선: 누적)")
            pv = pvar.rename(columns={"설명 분산 비율(%)": "ratio", "누적(%)": "cum"})
            bars = alt.Chart(pv).mark_bar(color=BLUE).encode(
                x=alt.X("주성분:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("ratio:Q", title="설명 분산 (%)"),
                tooltip=["주성분:N", alt.Tooltip("ratio:Q", title="설명 분산(%)", format=".1f"), alt.Tooltip("cum:Q", title="누적(%)", format=".1f"),
                         alt.Tooltip("고윳값:Q", format=".2f")])
            cum = alt.Chart(pv).mark_line(color=ORANGE, point=True).encode(x=alt.X("주성분:N", sort=None), y=alt.Y("cum:Q", title="누적 (%)"))
            st.altair_chart(alt.layer(bars, cum).resolve_scale(y="independent").properties(height=300), alt="주성분별 설명 분산과 누적")
            n80 = int((pvar["누적(%)"] < 80).sum()) + 1
            nk = int((pvar["고윳값"] > 1).sum())
            st.caption(f"누적 80%에 이르려면 축 {n80}개, 고윳값이 1을 넘는 축은 {nk}개입니다. "
                       f"마지막 고윳값은 {pvar['고윳값'].iloc[-1]:.3f}으로 사실상 0입니다. 비중 9개의 합이 항상 1이라 하나는 나머지로 결정되기 때문입니다.")
        with p2, st.container(border=True):
            st.markdown("**적재값: 각 축과 원래 변수의 상관**")
            ld = t("stat9_pca_loadings.csv")
            long = ld.melt(id_vars="변수", var_name="주성분", value_name="적재값")
            st.altair_chart(heat(long, "주성분", "변수", "적재값", "+.2f", True, None, ld["변수"].tolist(), 330), alt="주성분 적재값")
            st.caption("빨강은 그 축의 +쪽, 파랑은 -쪽으로 끄는 변수입니다. 절댓값이 0.5 이상인 변수를 중심으로 축의 뜻을 읽습니다.")
        ax = t("stat9_pca_axis_names.csv")
        st.dataframe(ax, hide_index=True, alt="주성분별 해석",
                     column_config={"설명 분산 비율(%)": st.column_config.NumberColumn("설명 분산", format="%.0f%%")})
        st.info(
            f"PC1은 전체 정보의 {pvar['설명 분산 비율(%)'].iloc[0]:.0f}%를 담습니다. {ax['축의 양 끝 (적재값)'].iloc[0]}  \n"
            f"PC2는 {pvar['설명 분산 비율(%)'].iloc[1]:.0f}%입니다. {ax['축의 양 끝 (적재값)'].iloc[1]}  \n"
            f"두 축을 합쳐도 {pvar['누적(%)'].iloc[1]:.0f}%뿐이라, 뉴욕 동네의 민원 구성은 한두 개의 축으로 깔끔하게 요약되지 않습니다. "
            "PC1을 '밤 생활 소음이 많은 번화한 동네 ↔ 도로·공원 민원이 많은 한적한 동네'로 읽을 수 있지만, 이런 이름은 적재값을 보고 사람이 붙인 해석입니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        # ---------------- K-means
        st.divider()
        st.subheader("② K-means: 닮은 동네끼리 묶기")
        with st.expander("처음 배우는 사람을 위한 설명: K-means와 군집 수 고르기", expanded=True):
            st.markdown(
                "**알고리즘.** ① 중심점 k개를 놓는다 → ② 각 칸을 가장 가까운 중심점에 배정한다 → ③ 각 군집의 평균으로 중심점을 옮긴다 → "
                "④ 배정이 바뀌지 않을 때까지 ②③을 반복한다.\n\n"
                "**k는 자료가 정해 주지 않습니다.** k를 늘리면 '군집 내 제곱합'은 무조건 줄어듭니다. 그래서 여러 지표를 함께 봅니다.\n"
                "- **엘보우**: 제곱합이 줄어드는 폭이 갑자기 완만해지는 k. 그림으로 판단하므로 주관적입니다.\n"
                "- **실루엣**(-1에서 1 사이): 자기 군집과는 가깝고 남의 군집과는 멀수록 큽니다. 0.5 이상 뚜렷, 0.25에서 0.5 사이는 약함, 그 아래는 희미.\n"
                "- **부트스트랩 안정성**(0에서 1 사이): 자료를 다시 뽑아도 같은 군집이 나오는가. 낮으면 그 군집은 표본에 따라 달라지는 우연입니다.\n\n"
                "| 장점 | 단점 |\n|---|---|\n"
                "| 빠르고 단순하다 | k를 미리 정해야 하고 정답이 없다 |\n"
                "| 군집 중심으로 특징을 설명하기 쉽다 | 둥글고 크기가 비슷한 군집을 가정한다 |\n"
                "| 모든 관측값이 군집을 받는다 | 외딴 값도 억지로 배정하며, 이상값에 중심이 끌려간다 |\n"
                "| | 시작 위치에 따라 결과가 달라질 수 있다 |\n\n"
                "**11단계의 DBSCAN과 무엇이 다른가.** 앞에서는 지도 좌표에 군집화를 적용해 '붙어 있는 곳'을 찾았습니다. "
                "여기서는 프로필 변수에 적용해 '닮은 곳'을 찾습니다. 닮은 동네는 서로 멀리 떨어져 있을 수 있습니다.")
        k1, k2 = st.columns([3, 2])
        with k1, st.container(border=True):
            metric_k = st.segmented_control("지표", ["실루엣", "부트스트랩 안정성(ARI)", "군집 내 제곱합"], default="실루엣", required=True, key="k_metric")
            kk = ksel.rename(columns={metric_k: "v"})
            kk["구분"] = np.where(kk["k"] == k_best, "선택한 k", "후보")
            ln = alt.Chart(kk).mark_line(color=BLUE).encode(x=alt.X("k:O", title="군집 수 k", axis=alt.Axis(labelAngle=0)),
                                                           y=alt.Y("v:Q", title=metric_k, scale=alt.Scale(zero=False)))
            pt = alt.Chart(kk).mark_circle(size=110, opacity=1).encode(
                x="k:O", y="v:Q", color=alt.Color("구분:N", scale=alt.Scale(domain=["후보", "선택한 k"], range=[BLUE, ORANGE]), title=None),
                tooltip=["k:O", alt.Tooltip("v:Q", title=metric_k, format=",.3f")])
            st.altair_chart((ln + pt).properties(height=290), alt=f"군집 수별 {metric_k}")
        with k2, st.container(border=True):
            st.markdown("**군집 수별 지표**")
            st.dataframe(ksel, hide_index=True, alt="군집 수 선택 지표", column_order=["k", "실루엣", "부트스트랩 안정성(ARI)", "최소 군집 비율(%)", "선택"],
                         column_config={"실루엣": st.column_config.NumberColumn(format="%.3f"),
                                        "부트스트랩 안정성(ARI)": st.column_config.NumberColumn("안정성", format="%.2f"),
                                        "최소 군집 비율(%)": st.column_config.NumberColumn("최소 군집", format="%.0f%%")})
            st.caption("선택 규칙: k = 3~7 가운데 안정성 0.80 이상이고 가장 작은 군집이 3% 이상인 것 중 실루엣이 가장 큰 k. "
                       "k = 2는 너무 거칠어 후보에서 뺐습니다(분석자의 판단).")
        unstable = ksel[(ksel["k"] > k_best)]["부트스트랩 안정성(ARI)"].max()
        st.info(
            f"k = {k_best}를 골랐습니다(실루엣 {krow['실루엣']:.2f}, 안정성 {krow['부트스트랩 안정성(ARI)']:.2f}). "
            f"**실루엣이 {krow['실루엣']:.2f}으로 낮습니다.** 동네들이 뚜렷이 갈린 몇 개의 무리로 나뉘어 있지 않고, 연속적으로 이어진 스펙트럼에 가깝다는 뜻입니다. "
            f"군집은 그 스펙트럼을 {k_best}구간으로 자른 '쓸모 있는 요약'으로 읽어야 하며, 자연에 존재하는 {k_best}종류의 동네를 '발견'한 것이 아닙니다. "
            f"k를 {k_best}보다 늘리면 안정성이 {unstable:.2f} 이하로 떨어져, 자료가 조금만 달라져도 군집이 바뀝니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        prof = t("stat9_cluster_profiles.csv")
        names = dict(zip(prof["군집"], prof["이름"]))
        cl_colors = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(sorted(names))}
        units["군집명"] = units["cluster"].astype(str) + ": " + units["cluster_name"]
        dom = [f"{c}: {names[c]}" for c in sorted(names)]
        v1, v2 = st.columns(2)
        with v1, st.container(border=True):
            st.markdown("**PCA 평면에서 본 군집** (점 하나 = 1km 칸)")
            st.altair_chart(alt.Chart(units).mark_circle(size=34, opacity=0.8).encode(
                x=alt.X("pc1:Q", title=f"PC1 ({pvar['설명 분산 비율(%)'].iloc[0]:.0f}%)"),
                y=alt.Y("pc2:Q", title=f"PC2 ({pvar['설명 분산 비율(%)'].iloc[1]:.0f}%)"),
                color=alt.Color("군집명:N", scale=alt.Scale(domain=dom, range=[cl_colors[c] for c in sorted(names)]), title=None,
                                legend=alt.Legend(orient="bottom", columns=1, labelLimit=420)),
                tooltip=["군집명:N", alt.Tooltip("n:Q", title="민원 건수", format=","), "borough:N"]).properties(height=430),
                alt="PCA 평면 위의 군집")
            st.caption("군집끼리 경계가 맞닿아 있고 빈틈이 없습니다. 실루엣이 낮게 나온 이유를 눈으로 확인할 수 있습니다.")
        with v2, st.container(border=True):
            st.markdown("**같은 군집을 지도에 표시** (위치는 군집화에 쓰지 않았음)")
            m = units.copy()
            m["color"] = [hex_rgb(cl_colors[c]) + [205] for c in m["cluster"]]
            m["건수"] = m["n"].map("{:,}".format)
            deck_map(m, 430, "{군집명}\n민원 {건수}건", height=455)
        coh = t("stat9_spatial_coherence.csv").iloc[0]
        zt = t("stat9_cluster_profiles_z.csv")
        with st.container(border=True):
            st.markdown("**군집 프로필** (전체 평균에서 표준편차 몇 배 떨어져 있는가)")
            zl = zt.assign(군집명=zt["군집"].astype(str) + ": " + zt["이름"]).drop(columns=["군집", "이름"]).melt(
                id_vars="군집명", var_name="변수", value_name="z")
            st.altair_chart(heat(zl, "변수", "군집명", "z", "+.1f", True, None, dom, 60 + 46 * len(dom)), alt="군집별 변수 프로필")
            st.dataframe(prof, hide_index=True, alt="군집 요약",
                         column_order=["군집", "이름", "칸 수", "민원 건수", "칸당 민원 중앙값", "24시간 내 처리율(%)", "유형 보정 로그 처리시간"],
                         column_config={"민원 건수": st.column_config.NumberColumn(format="localized"),
                                        "칸당 민원 중앙값": st.column_config.NumberColumn(format="localized"),
                                        "24시간 내 처리율(%)": st.column_config.NumberColumn("24시간 내 처리율", format="%.0f%%"),
                                        "유형 보정 로그 처리시간": st.column_config.NumberColumn("유형 보정 처리시간", format="%+.2f")})
            st.caption("유형 보정 처리시간이 0보다 작으면 같은 유형의 민원이 시 평균보다 빨리 처리되는 군집입니다.")
        big = prof.sort_values("민원 건수", ascending=False).iloc[0]
        fast = prof.sort_values("유형 보정 로그 처리시간").iloc[0]
        st.info(
            f"민원이 가장 많은 군집은 **{big['군집']}: {big['이름']}** ({big['칸 수']:.0f}칸, {big['민원 건수']:,.0f}건)입니다. "
            f"같은 유형의 민원이 가장 빨리 처리되는 군집은 **{fast['군집']}: {fast['이름']}** (유형 보정 {fast['유형 보정 로그 처리시간']:+.2f})입니다.  \n"
            f"군집화에 위치를 넣지 않았는데도, 인접한 칸 쌍의 {coh['같은 군집인 쌍 비율(%)']:.0f}%가 같은 군집입니다. "
            f"무작위였다면 {coh['무작위일 때 기대 비율(%)']:.0f}%였을 것이므로 {coh['배율']:.1f}배입니다(p = {coh['유사 p 값']:.3f}). "
            "닮은 동네는 지도에서도 붙어 있다는 뜻이고, 11단계의 Moran's I가 높았던 것과 같은 이야기입니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        # ---------------- 방법 비교
        st.divider()
        st.subheader("③ 방법을 바꿔도 같은 군집이 나오는가")
        cmp9 = t("stat9_method_compare.csv")
        c1, c2 = st.columns([3, 2])
        with c1, st.container(border=True):
            st.dataframe(cmp9, hide_index=True, alt="군집화 방법별 결과와 일치도",
                         column_config={"잡음 칸 비율(%)": st.column_config.NumberColumn("잡음", format="%.0f%%"),
                                        "실루엣 (표준화 변수 기준)": st.column_config.NumberColumn("실루엣", format="%.2f"),
                                        "기준 K-means 와의 일치도(ARI)": st.column_config.ProgressColumn("K-means와의 일치도(ARI)", format="%.2f", min_value=0, max_value=1)})
            st.caption("ARI는 두 군집 결과가 얼마나 같은지 재는 값입니다. 1이면 완전히 같고 0이면 무작위 수준입니다.")
        with c2, st.container(border=True):
            st.markdown(
                "**각 방법의 생각**\n"
                "- **PCA 후 K-means**: 분산이 작은 축을 버리고 군집화\n"
                "- **CLR 변환 후 K-means**: 비중의 '합이 1' 문제를 풀고 군집화\n"
                "- **Ward 계층적 군집**: 가장 닮은 둘을 차례로 합쳐 나무를 만든다\n"
                "- **가우시안 혼합(GMM)**: 군집마다 모양과 크기가 달라도 된다\n"
                "- **DBSCAN**: 밀집된 덩어리와 잡음으로 나눈다")
        pca_r = cmp9[cmp9["방법"].str.startswith("PCA")].iloc[0]
        ward_r = cmp9[cmp9["방법"].str.startswith("Ward")].iloc[0]
        clr_r = cmp9[cmp9["방법"].str.startswith("CLR")].iloc[0]
        db_r = cmp9[cmp9["방법"].str.startswith("DBSCAN")].iloc[0]
        ari = "기준 K-means 와의 일치도(ARI)"
        st.info(
            f"PCA로 축을 줄인 뒤 군집화해도 결과가 거의 같습니다(ARI {pca_r[ari]:.2f}). 버린 축에는 군집을 가르는 정보가 거의 없었다는 뜻입니다.  \n"
            f"반면 Ward(ARI {ward_r[ari]:.2f})와 CLR 변환(ARI {clr_r[ari]:.2f})은 절반 남짓만 일치합니다. "
            "군집의 큰 줄기는 같지만 경계에 있는 칸들은 방법에 따라 다른 군집으로 갑니다. 실루엣이 낮았던 것과 같은 이야기입니다.  \n"
            f"DBSCAN은 군집 {db_r['군집 수']:.0f}개와 잡음 {db_r['잡음 칸 비율(%)']:.0f}%로 끝났습니다. 프로필 공간에서는 동네들이 끊김 없이 이어져 있어 "
            "'밀집된 덩어리'로 가를 수 없기 때문입니다. 지도 좌표에서는 유흥가를 잘 찾아낸 같은 방법이 여기서는 쓸모가 없습니다. "
            "**방법의 좋고 나쁨은 자료의 모양에 달려 있습니다.**",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        st.divider()
        st.subheader("정리: 다섯 방법을 한 표로")
        st.markdown(
            "| 방법 | 묻는 것 | 단위 | 잘하는 것 | 조심할 것 | 이 자료에서 본 것 |\n|---|---|---|---|---|---|\n"
            "| KDE | 어디가 진한가 | 점 | 한눈에 보이는 지도 | 대역폭, 신고량의 영향 | 소음은 좁은 곳에, 주차는 넓게 퍼짐 |\n"
            "| DBSCAN | 덩어리가 어디인가 | 점 | 모양이 자유로운 군집과 잡음 분리 | eps에 매우 민감 | 상업 소음 유흥가와 흩어진 개별 업소 |\n"
            "| Moran's I | 가까우면 비슷한가 | 면 | 우연이 아님을 검정 | 이웃 정의, 격자 크기 | 유형을 보정해도 처리 속도에 지역성 |\n"
            "| PCA | 몇 개의 축인가 | 변수 | 차원 축소, 시각화 | 선형만, 해석은 사람 몫 | 두 축으로 절반 미만만 요약됨 |\n"
            "| K-means | 닮은 것끼리 몇 유형인가 | 관측값 | 빠르고 설명하기 쉬움 | k 선택, 둥근 군집 가정 | 군집 경계가 흐린 연속 스펙트럼 |\n")


# ============================================================ 0단계: 시작하기 (데이터 사전, 학습 지도, 용어집)
with tab_s0:
    stage_header("s0")
    st.markdown(
        "이 대시보드는 **뉴욕시 311 민원 171만 건**으로 통계를 처음부터 차례대로 익히도록 만든 것입니다. "
        "탭은 공부 순서이며, 앞 단계에서 배운 개념을 다음 단계가 이어서 씁니다. 처음이라면 0단계부터 순서대로 읽으세요.\n\n"
        "각 단계는 같은 틀로 되어 있습니다.\n"
        "1. **머리말**: 이 단계의 핵심 질문, 배우는 개념, 쓰는 열\n"
        "2. **본문**: 개념 설명(펼치기 상자) → 이 데이터에 적용한 결과 → '현재 값이 뜻하는 것' 상자\n"
        "3. **꼬리말**: 개념 ↔ 이 데이터 정리표, 다음 단계 안내")
    lp = t("stat12_learning_path.csv")
    if lp is not None:
        with st.container(border=True):
            st.markdown("**학습 지도: 4부 13단계**")
            st.dataframe(lp.drop(columns=["key"]), hide_index=True, alt="13단계 학습 지도",
                         column_config={"단계": st.column_config.NumberColumn(width="small"),
                                        "핵심 질문": st.column_config.TextColumn(width="large"),
                                        "배우는 개념": st.column_config.TextColumn(width="large")})
    if need("stat_10_concepts.py", "stat10_data_dictionary.csv"):
        dd = t("stat10_data_dictionary.csv")
        log0 = t("stat1_exclusion_log.csv").set_index("항목")["건수"]
        with st.container(horizontal=True):
            st.metric("원본 행", f"{log0['원본 행 수']:,.0f}건", border=True, help="NYC Open Data 에서 내려받은 2024년 4~9월 민원")
            st.metric("분석 대상 행", f"{log0['최종 분석 대상 행 수']:,.0f}건", border=True, help="미종결, 음수 처리시간, 자치구 미상을 뺀 행")
            st.metric("열", f"{len(dd)}개", border=True, help="원본 8개 + 정제 과정에서 만든 파생 열")
            st.metric("기간", "183일", border=True, help="2024-04-01 ~ 2024-09-30")
        st.subheader("측정 척도: 값의 종류가 할 수 있는 계산을 정한다")
        with st.expander("처음 배우는 사람을 위한 설명: 척도 4가지", expanded=True):
            st.markdown(
                "| 척도 | 뜻 | 이 데이터의 예 | 뜻이 있는 계산 |\n|---|---|---|---|\n"
                "| **명목** | 이름표일 뿐 순서가 없다 | 자치구, 기관, 민원 유형, `unique_key` | 개수, 비율, 최빈값 |\n"
                "| **순서** | 순서는 있지만 간격이 같다는 보장이 없다 | 시간대(새벽·오전·오후·저녁), 월 | + 중앙값, 순위 |\n"
                "| **구간** | 간격은 같지만 0 이 '없음'을 뜻하지 않는다 | 날짜, 위도·경도 | + 평균, 차이 |\n"
                "| **비율** | 진짜 0 이 있어 '몇 배'를 말할 수 있다 | 처리 시간, 일별 건수 | + 비, 기하평균, 조화평균, 로그 |\n\n"
                "아래로 갈수록 할 수 있는 계산이 늘어납니다. 섭씨 20도가 10도의 '두 배 덥다'고 할 수 없는 것처럼(구간), "
                "'몇 배'를 말하려면 비율 척도여야 합니다. 2단계에서 배울 기하평균과 조화평균이 처리 시간에만 쓰이는 이유입니다.")
        st.warning(
            "- `unique_key` 는 숫자로 되어 있지만 민원 번호(이름표)입니다. 평균을 내면 아무 뜻이 없습니다.\n"
            "- `weekday` 의 0부터 6까지는 월요일부터 일요일까지를 가리키는 번호입니다. '평균 요일 2.7' 같은 값은 뜻이 없습니다.\n"
            "- `hour` 는 숫자지만 23시 다음이 0시인 원형 값입니다. 그냥 평균을 내면 틀립니다 (2단계에서 직접 확인).",
            title="숫자처럼 보이지만 숫자로 다루면 안 되는 열", icon=":material/warning:")
        with st.container(border=True):
            st.markdown("**데이터 사전**")
            st.dataframe(dd, hide_index=True, alt="열별 데이터 사전",
                         column_config={"원본 결측 건수": st.column_config.NumberColumn(format="localized"),
                                        "서로 다른 값": st.column_config.NumberColumn(format="localized"),
                                        "주의": st.column_config.TextColumn(width="large")})
            st.caption("원본 결측 건수가 빈 칸인 열은 정제 과정에서 새로 만든 파생 열입니다. 종결일 결측은 '아직 처리 중'이라는 뜻이라 10단계 생존분석이 살려 씁니다.")
        with st.expander("정제 기준과 제외 건수"):
            st.dataframe(t("stat1_exclusion_log.csv"), hide_index=True, alt="정제 기준과 제외 건수")
    gl = t("stat12_glossary.csv")
    if gl is not None:
        st.subheader("용어집")
        q = st.text_input("용어 찾기", placeholder="예: 표준오차, 중도절단, 실루엣", key="glossary_q")
        g = gl
        if q:
            m = gl["용어"].str.contains(q, case=False, regex=False) | gl["쉬운 설명"].str.contains(q, case=False, regex=False) \
                | gl["영어"].str.contains(q, case=False, regex=False)
            g = gl[m]
        st.dataframe(g, hide_index=True, alt="통계 용어집", height=420,
                     column_config={"처음 나오는 단계": st.column_config.NumberColumn("단계", width="small"),
                                    "쉬운 설명": st.column_config.TextColumn(width="large")})
        st.caption(f"{len(g)}개 / 전체 {len(gl)}개 용어")
    concept_footer("s0")

# ============================================================ 2단계: 대푯값
with tab_s2:
    stage_header("s2")
    if need("stat_10_concepts.py", "stat10_means.csv", "stat10_zero_sensitivity.csv", "stat10_speed_means.csv"):
        with st.expander("처음 배우는 사람을 위한 설명: 평균에도 종류가 있다", expanded=True):
            st.markdown(
                "평균은 '자료 전체를 숫자 하나로 바꾸면 무엇이 되는가'에 대한 답입니다. 무엇을 **유지**하고 싶으냐에 따라 답이 달라집니다.\n\n"
                "| 대푯값 | 계산 | 유지하는 것 | 어울리는 자료 | 약점 |\n|---|---|---|---|---|\n"
                "| **산술평균** | 모두 더해 n 으로 나눔 | 합 | 건수, 금액처럼 더하는 양 | 극단값 하나에 크게 끌려간다 |\n"
                "| **기하평균** | 모두 곱해 n제곱근 = 로그 평균을 되돌림 | 곱(배율) | 성장률, 배율, 오른쪽 꼬리가 긴 양수 | 0 이나 음수가 있으면 계산 불가 |\n"
                "| **조화평균** | 역수의 평균의 역수 | 역수의 합 | 속도, 단가처럼 분자가 고정된 비율 | 0 근처 값 하나가 결과를 지배 |\n"
                "| **중앙값** | 줄 세운 가운데 값 | 순서 | 치우친 자료 전반 | 값의 크기 정보를 버린다 |\n"
                "| **절사평균** | 양 끝 10% 를 잘라내고 산술평균 | 대부분의 합 | 극단값이 섞인 자료 | 자를 비율을 정해야 한다 |\n"
                "| **최빈값** | 가장 많이 나온 값 | 가장 흔한 경우 | 범주형 자료 | 연속형에서는 거의 쓰지 않는다 |\n\n"
                "**항상 성립하는 순서**: 모든 값이 양수이면 **산술평균 ≥ 기하평균 ≥ 조화평균**이고, 세 값이 같아지는 것은 모든 값이 똑같을 때뿐입니다. "
                "그래서 값들이 고르지 않을수록 세 평균의 간격이 벌어집니다.")
            c1, c2, c3 = st.columns(3)
            c1.latex(r"\bar{x}_{\text{산술}} = \frac{1}{n}\sum_{i=1}^{n} x_i")
            c2.latex(r"\bar{x}_{\text{기하}} = \Big(\prod_{i=1}^{n} x_i\Big)^{1/n} = \exp\Big(\frac{1}{n}\sum \ln x_i\Big)")
            c3.latex(r"\bar{x}_{\text{조화}} = \frac{n}{\sum_{i=1}^{n} 1/x_i}")

        # ---- 계산기
        with st.container(border=True):
            st.markdown("**직접 계산해 보기**")
            txt = st.text_input("숫자를 쉼표로 구분해 입력하세요", "1, 2, 4, 8, 100", key="calc_numbers",
                                help="예시를 바꿔 보세요. 100 을 1000 으로 바꾸면 산술평균만 크게 움직입니다. 모든 값을 같게 하면 세 평균이 같아집니다.")
            vals, bad = [], []
            for tok in (txt or "").replace("，", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    bad.append(tok)
            if bad:
                st.warning(f"숫자로 읽을 수 없는 값은 뺐습니다: {', '.join(bad)}", title="입력 확인")
            if len(vals) < 2:
                st.info("숫자를 두 개 이상 입력하세요.", title="입력이 부족합니다")
            else:
                x = np.array(vals)
                n = len(x)
                srt = np.sort(x)
                k = int(n * 0.1)
                trimmed = srt[k:n - k].mean() if n - 2 * k > 0 else x.mean()
                positive = bool((x > 0).all())
                gm = float(np.exp(np.log(x).mean())) if positive else None
                hm = float(n / (1 / x).sum()) if positive else None
                with st.container(horizontal=True):
                    st.metric("산술평균", f"{x.mean():,.3f}", border=True)
                    st.metric("기하평균", f"{gm:,.3f}" if gm is not None else "계산 불가", border=True)
                    st.metric("조화평균", f"{hm:,.3f}" if hm is not None else "계산 불가", border=True)
                    st.metric("중앙값", f"{np.median(x):,.3f}", border=True)
                    st.metric("절사평균 (10%)", f"{trimmed:,.3f}", border=True, help=f"양 끝에서 {k}개씩 잘라냈습니다. 값이 10개 미만이면 잘라낼 것이 없어 산술평균과 같습니다.")
                steps = [f"- 산술평균 = ({' + '.join(f'{v:g}' for v in x[:8])}{' + …' if n > 8 else ''}) / {n} = {x.sum():,.4g} / {n} = **{x.mean():,.4g}**"]
                if positive:
                    steps.append(f"- 기하평균 = exp( (ln {' + ln '.join(f'{v:g}' for v in x[:6])}{' + …' if n > 6 else ''}) / {n} ) = exp({np.log(x).mean():.4f}) = **{gm:,.4g}**")
                    steps.append(f"- 조화평균 = {n} / ({' + '.join(f'1/{v:g}' for v in x[:6])}{' + …' if n > 6 else ''}) = {n} / {(1 / x).sum():.4f} = **{hm:,.4g}**")
                    steps.append(f"- 순서 확인: {x.mean():,.4g} ≥ {gm:,.4g} ≥ {hm:,.4g}")
                else:
                    steps.append("- 0 이나 음수가 있어 기하·조화평균은 계산할 수 없습니다. 로그와 역수가 정의되지 않기 때문입니다. 아래 '0 처리' 절을 보세요.")
                st.markdown("\n".join(steps))
                pts = pd.DataFrame({"값": x})
                marks = pd.DataFrame({"대푯값": ["산술평균", "중앙값"] + (["기하평균", "조화평균"] if positive else []),
                                      "값": [x.mean(), float(np.median(x))] + ([gm, hm] if positive else [])})
                use_log = positive and x.max() / x.min() > 50
                xs = alt.Scale(type="log") if use_log else alt.Scale(zero=False)
                dots = alt.Chart(pts).mark_circle(size=120, color="#52514e", opacity=0.6).encode(
                    x=alt.X("값:Q", scale=xs, title="값" + (" (로그 축)" if use_log else "")), tooltip=[alt.Tooltip("값:Q", format=",.3f")])
                rules = alt.Chart(marks).mark_rule(strokeWidth=2.5).encode(
                    x="값:Q", color=alt.Color("대푯값:N", scale=alt.Scale(domain=["산술평균", "기하평균", "조화평균", "중앙값"],
                                                                   range=[ORANGE, BLUE, CATEGORICAL[2], "#0b0b0b"]), title=None,
                                             legend=alt.Legend(orient="bottom")),
                    tooltip=["대푯값:N", alt.Tooltip("값:Q", format=",.3f")])
                st.altair_chart((rules + dots).properties(height=110), alt="입력한 값과 대푯값의 위치")

        # ---- 실제 데이터
        st.subheader("이 데이터에서는: 같은 자료, 다른 답")
        means = t("stat10_means.csv")
        means["선택"] = means["변수"] + " · " + means["0 처리"]
        m1, m2 = st.columns([3, 2])
        with m1, st.container(border=True):
            pick = st.selectbox("변수", means["선택"].tolist(), key="means_pick")
            r = means[means["선택"] == pick].iloc[0]
            bars = pd.DataFrame({"대푯값": ["산술평균", "절사평균(10%)", "중앙값", "기하평균", "조화평균"],
                                 "값": [r["산술평균"], r["절사평균(10%)"], r["중앙값"], r["기하평균"], r["조화평균"]]})
            use_log = bars["값"].max() / max(bars["값"].min(), 1e-9) > 20
            st.altair_chart(hbar(bars, "대푯값", "값", title=None, label_fmt=",.2f") if not use_log else
                            alt.Chart(bars).mark_bar(color=BLUE).encode(
                                y=alt.Y("대푯값:N", sort=None, title=None), x=alt.X("값:Q", scale=alt.Scale(type="log"), title="값 (로그 축)"),
                                tooltip=["대푯값:N", alt.Tooltip("값:Q", format=",.3f")]).properties(height=200),
                            alt=f"{pick}의 대푯값 비교")
            st.caption(f"n = {r['n']:,.0f} · 산술/기하 비 = {r['산술/기하 비']:.2f} (1 이면 완전히 고르고, 클수록 고르지 않음)" +
                       (" · 값 차이가 커서 로그 축으로 그렸습니다." if use_log else "") + (f" · {r['비고']}" if isinstance(r["비고"], str) else ""))
        with m2, st.container(border=True):
            st.markdown("**모든 변수의 대푯값**")
            st.dataframe(means.drop(columns=["선택", "열"]), hide_index=True, height=330, alt="변수별 대푯값 비교",
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                        ["산술평균", "기하평균", "조화평균", "중앙값", "절사평균(10%)", "산술/기하 비"]})
        hrow = means[(means["열"] == "resolution_hours") & means["0 처리"].str.startswith("0 포함")].iloc[0]
        drow = means[means["변수"] == "일별 접수 건수"].iloc[0]
        st.info(
            f"처리 시간의 산술평균은 **{hrow['산술평균']:,.0f}시간**, 기하평균은 **{hrow['기하평균']:.1f}시간**, 중앙값은 **{hrow['중앙값']:.1f}시간**입니다. "
            f"같은 자료인데 산술평균이 중앙값의 {hrow['산술평균'] / hrow['중앙값']:.0f}배입니다. 몇 달씩 걸린 소수의 민원이 합을 키웠기 때문입니다. "
            f"'평균 처리 시간이 {hrow['산술평균'] / 24:.0f}일'이라고 말하면 대부분의 민원 경험과 동떨어진 숫자가 됩니다.  \n"
            f"반대로 일별 접수 건수는 세 평균이 {drow['산술평균']:,.0f}, {drow['기하평균']:,.0f}, {drow['조화평균']:,.0f}건으로 거의 같습니다. "
            "하루하루의 건수가 고르게 퍼져 있어서 어떤 평균을 써도 답이 같습니다. **어떤 평균을 쓸지는 자료가 얼마나 고르지 않은가에 달려 있습니다.**",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        st.subheader("0 이 있을 때: 분석자의 선택이 답을 바꾼다")
        sens = t("stat10_zero_sensitivity.csv")
        z1, z2 = st.columns([3, 2])
        with z1, st.container(border=True):
            sl = sens[sens["더한 상수(시간)"] > 0].melt(id_vars="더한 상수(시간)", value_vars=["기하평균", "조화평균"], var_name="평균", value_name="값")
            st.altair_chart(alt.Chart(sl).mark_line(point=True).encode(
                x=alt.X("더한 상수(시간):Q", scale=alt.Scale(type="log"), title="0 을 피하려고 더한 상수 (시간, 로그 축)"),
                y=alt.Y("값:Q", scale=alt.Scale(type="log"), title="평균 (시간, 로그 축)"),
                color=alt.Color("평균:N", scale=alt.Scale(domain=["기하평균", "조화평균"], range=[BLUE, CATEGORICAL[2]]), title=None),
                tooltip=["평균:N", alt.Tooltip("더한 상수(시간):Q", format="g"), alt.Tooltip("값:Q", format=",.3f")]).properties(height=280),
                alt="더한 상수에 따른 기하평균과 조화평균")
        with z2, st.container(border=True):
            st.dataframe(sens, hide_index=True, alt="0 처리 방법별 평균",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["산술평균", "기하평균", "조화평균"]}
                         | {"n": st.column_config.NumberColumn(format="localized")})
            st.caption("산술평균은 상수를 더했다 빼면 그대로지만, 기하·조화평균은 상수에 따라 크게 달라집니다. "
                       "그래서 이런 평균을 보고할 때는 0 을 어떻게 처리했는지 꼭 함께 적어야 합니다. 이 대시보드는 '1시간을 더해 계산 후 뺌'을 기본으로 씁니다.")

        st.subheader("조화평균이 진짜 필요한 곳: 처리 속도")
        with st.container(border=True):
            st.markdown(
                "같은 거리를 시속 60km 로 가고 시속 40km 로 돌아오면 평균 속력은 50 이 아니라 **48(조화평균)** 입니다. "
                "느린 구간에서 시간을 더 오래 쓰기 때문입니다. 민원 처리 속도(= 1 / 처리 시간, 건/시간)에도 같은 일이 일어납니다. "
                "'평균 처리 속도'라는 같은 말이 질문에 따라 전혀 다른 숫자가 됩니다.")
            st.dataframe(t("stat10_speed_means.csv"), hide_index=True, alt="처리 속도의 세 가지 평균",
                         column_config={"값 (건/시간)": st.column_config.NumberColumn(format="%.4f")})
            st.caption("A 는 몇 분 만에 끝난 민원들이 값을 끌어올리고, B 는 몇 달씩 걸린 민원들이 시간을 다 차지합니다. "
                       "담당자 한 명의 실제 처리량을 묻는다면 B(조화평균)가, 민원 하나의 전형적인 경험을 묻는다면 C 가 맞는 답입니다.")

        st.subheader("가중평균, 최빈값, 원형 평균")
        w1, w2, w3 = st.columns(3)
        with w1, st.container(border=True):
            st.markdown("**가중평균: 자치구별 비율 합치기**")
            ws = t("stat10_weighted_summary.csv")
            with st.container(horizontal=True):
                st.metric("단순평균", f"{ws['값(%)'].iloc[0]:.2f}%", border=True, help=ws["뜻"].iloc[0])
                st.metric("가중평균", f"{ws['값(%)'].iloc[1]:.2f}%", border=True, help=ws["뜻"].iloc[1])
            st.dataframe(t("stat10_weighted_groups.csv"), hide_index=True, alt="자치구별 24시간 내 처리율과 가중치",
                         column_config={"민원 건수": st.column_config.NumberColumn(format="localized"),
                                        "24시간 내 처리율(%)": st.column_config.NumberColumn(format="%.1f"),
                                        "가중치(건수 비중 %)": st.column_config.NumberColumn(format="%.1f")})
            st.caption("규모가 작은 스태튼아일랜드(처리율 최저)가 단순평균에서는 1/5 의 무게를 갖지만 실제 민원의 4% 뿐입니다. "
                       "집단 평균을 다시 평균할 때는 무엇을 한 표로 칠지 먼저 정해야 합니다.")
        with w2, st.container(border=True):
            st.markdown("**최빈값: 범주형 변수의 대푯값**")
            st.dataframe(t("stat10_modes.csv"), hide_index=True, alt="범주형 변수의 최빈값",
                         column_config={"최빈값의 비중(%)": st.column_config.NumberColumn(format="%.1f")})
            st.caption("자치구나 기관에는 평균이 없습니다. 가장 흔한 값과 그 비중이 대푯값입니다.")
        with w3, st.container(border=True):
            st.markdown("**원형 평균: 시각의 평균**")
            hmt = t("stat10_hour_means.csv")
            st.dataframe(hmt, hide_index=True, alt="접수 시각의 산술평균과 원형 평균",
                         column_config={"값(시)": st.column_config.NumberColumn(format="%.1f"), "설명": st.column_config.TextColumn(width="large")})
            st.caption("23시와 1시의 산술평균은 12시지만 상식적인 답은 0시입니다. 시각을 시계 위의 점으로 보고 평균해야 합니다.")
        with st.container(border=True):
            st.markdown("**어떤 대푯값을 쓸까**")
            st.markdown(
                "| 상황 | 권하는 대푯값 | 이 데이터의 예 |\n|---|---|---|\n"
                "| 대칭이고 극단값이 없다 | 산술평균 | 일별 접수 건수 |\n"
                "| 오른쪽 꼬리가 길다 | 중앙값, 기하평균 | 처리 시간 |\n"
                "| 합계가 중요하다 (예산, 총 업무량) | 산술평균 | 총 처리 시간 = 산술평균 × 건수 |\n"
                "| 속도·단가처럼 비율을 평균한다 | 조화평균 | 담당자의 실제 처리량 |\n"
                "| 크기가 다른 집단의 비율을 합친다 | 가중평균 | 시 전체 24시간 내 처리율 |\n"
                "| 범주형이다 | 최빈값 | 가장 흔한 민원 유형 |\n"
                "| 원형이다 (시각, 방향) | 원형 평균 | 접수 시각 |")
    concept_footer("s2")

# ============================================================ 3단계에 덧붙이는 절: 퍼짐
with tab1:
    if need("stat_10_concepts.py", "stat10_spread.csv", "stat10_spread_rules.csv"):
        st.divider()
        st.subheader("퍼짐: 대푯값만 보면 절반만 본 것이다")
        with st.expander("처음 배우는 사람을 위한 설명: 퍼짐의 측도", expanded=True):
            st.markdown(
                "평균이 같아도 모든 값이 평균 근처에 있는 자료와 넓게 흩어진 자료는 전혀 다릅니다. 퍼짐(산포)은 그 차이를 숫자로 잽니다.\n\n"
                "| 측도 | 계산 | 단위 | 극단값에 강한가 | 짝이 되는 대푯값 |\n|---|---|---|---|---|\n"
                "| 범위 | 최댓값 - 최솟값 | 원래 단위 | 매우 약함 (두 값으로만 정해짐) | |\n"
                "| 분산 | (값 - 평균)² 의 평균 (n-1 로 나눔) | 단위의 제곱 | 약함 | 평균 |\n"
                "| **표준편차** | 분산의 제곱근 | 원래 단위 | 약함 | **평균** |\n"
                "| **IQR** | Q3 - Q1 (가운데 50% 의 폭) | 원래 단위 | 강함 | **중앙값** |\n"
                "| MAD | |값 - 중앙값| 의 중앙값 | 원래 단위 | 가장 강함 | 중앙값 |\n"
                "| **변동계수 CV** | 표준편차 / 평균 | 없음 (%) | 약함 | 규모가 다른 변수끼리 비교할 때 |\n\n"
                "**분산을 n 이 아니라 n-1 로 나누는 이유**: 평균을 자료에서 계산하느라 정보를 하나 썼기 때문입니다(자유도). "
                "n 으로 나누면 모집단 분산을 조금 작게 추정합니다.\n\n"
                "**경험 법칙 68-95-99.7**: 정규분포라면 평균 ±1, ±2, ±3 표준편차 안에 각각 68%, 95%, 99.7% 가 들어옵니다. "
                "실제로 세어 보면 자료가 정규분포와 얼마나 다른지 알 수 있습니다.")
        sp = t("stat10_spread.csv")
        st.dataframe(sp, hide_index=True, alt="변수별 퍼짐 측도",
                     column_order=["변수", "평균", "중앙값", "표준편차", "IQR", "MAD (x1.4826)", "변동계수 CV(%)", "범위", "평균 - 표준편차", "왜도", "어울리는 요약"],
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                    ["평균", "중앙값", "표준편차", "IQR", "MAD (x1.4826)", "변동계수 CV(%)", "범위", "평균 - 표준편차", "왜도"]})
        ru = t("stat10_spread_rules.csv")
        r1, r2 = st.columns(2)
        with r1, st.container(border=True):
            st.markdown("**경험 법칙 확인: ±k 표준편차 안에 든 비율**")
            el = ru.melt(id_vars="변수", value_vars=["±1 표준편차 안(%)", "±2 표준편차 안(%)", "±3 표준편차 안(%)"], var_name="범위", value_name="비율")
            el["범위"] = el["범위"].str.replace(" 표준편차 안(%)", "", regex=False)
            ref = pd.DataFrame({"범위": ["±1", "±2", "±3"], "정규분포": [68.3, 95.4, 99.7]})
            bars = alt.Chart(el).mark_bar().encode(
                x=alt.X("범위:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="변수:N", y=alt.Y("비율:Q", title="비율 (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("변수:N", scale=alt.Scale(range=[ORANGE, BLUE, CATEGORICAL[2]]), title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["변수:N", "범위:N", alt.Tooltip("비율:Q", format=".1f")])
            ticks = alt.Chart(ref).mark_tick(color="#0b0b0b", thickness=2, size=60).encode(x="범위:N", y="정규분포:Q")
            st.altair_chart((bars + ticks).properties(height=290), alt="경험 법칙과 실제 비율 비교")
            st.caption("검은 선이 정규분포의 기대값입니다. 처리 시간은 ±1 표준편차 안에 95% 가 몰려 있습니다. 극단값 때문에 표준편차가 너무 커졌기 때문입니다.")
        with r2, st.container(border=True):
            st.markdown("**이상치 규칙 세 가지: 같은 자료, 다른 답**")
            st.dataframe(ru, hide_index=True, alt="이상치 규칙별 개수",
                         column_order=["변수", "이상치: Z 규칙", "이상치: IQR 규칙", "이상치: MAD 규칙", "IQR 규칙 이상치 비율(%)"],
                         column_config={c: st.column_config.NumberColumn(format="localized") for c in ["이상치: Z 규칙", "이상치: IQR 규칙", "이상치: MAD 규칙"]}
                         | {"IQR 규칙 이상치 비율(%)": st.column_config.NumberColumn(format="%.1f")})
            st.markdown(
                "- **Z 규칙** (|Z| > 3): 이상치 자신이 평균과 표준편차를 부풀려 스스로를 가립니다(마스킹).\n"
                "- **IQR 규칙** (상자그림의 수염 밖): 치우친 자료에서는 꼬리 전체가 이상치로 잡힙니다.\n"
                "- **MAD 규칙** (robust |z| > 3.5): 가장 튼튼하지만, 꼬리가 긴 자료에서는 역시 많이 잡힙니다.\n\n"
                "처리 시간에서 수십만 건이 '이상치'로 나온다는 것은 이상치가 많은 것이 아니라 **분포 자체가 긴 꼬리**라는 뜻입니다. "
                "지우지 말고 4단계처럼 변환하거나 중앙값·분위수로 다뤄야 합니다.")
        cv = t("stat10_cv_by_agency.csv")
        if cv is not None:
            with st.container(border=True):
                st.markdown("**변동계수로 기관 비교: 평균이 수백 배 달라도 '상대적인 고르지 않음'은 비교할 수 있다**")
                st.altair_chart(hbar(cv.sort_values("변동계수 CV(%)", ascending=False), "기관", "변동계수 CV(%)", label_fmt=".0f", height=210),
                                alt="기관별 처리 시간의 변동계수")
                st.caption("NYPD 는 평균이 2시간대로 짧지만 상대적인 들쭉날쭉함은 결코 작지 않습니다. 변동계수는 단위와 규모를 지우고 퍼짐만 비교합니다.")
        hsp = sp.set_index("변수").loc["처리 시간 (시간)"]
        st.info(
            f"처리 시간의 표준편차는 **{hsp['표준편차']:,.0f}시간**인데 IQR 은 **{hsp['IQR']:,.0f}시간**입니다. "
            f"'평균 {hsp['평균']:,.0f}시간 ± 표준편차 {hsp['표준편차']:,.0f}시간'이라고 쓰면 아래쪽 끝이 {hsp['평균 - 표준편차']:,.0f}시간, 즉 음수가 됩니다. "
            "처리 시간은 음수일 수 없으므로, 이 자료에는 평균·표준편차보다 **중앙값과 IQR** 로 요약하는 것이 맞습니다. "
            "같은 이유로 4단계에서 로그 변환을 배웁니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

# ============================================================ 5단계: 표본과 불확실성
with tab_s5:
    stage_header("s5")
    if need("stat_10_concepts.py", "stat10_clt_summary.csv", "stat10_clt_samples.csv", "stat10_bootstrap_ci.csv"):
        with st.expander("처음 배우는 사람을 위한 설명: 표본, 표준오차, 중심극한정리", expanded=True):
            st.markdown(
                "**모집단과 표본.** 알고 싶은 것은 전체(모집단)의 값이지만, 보통은 일부(표본)만 관찰합니다. "
                "표본에서 구한 평균은 표본을 다시 뽑을 때마다 조금씩 달라집니다. 이 흔들림이 '불확실성'입니다.\n\n"
                "**이 단계의 실험 방법.** 171만 건 전체를 '모집단'으로 삼고, 거기서 작은 표본을 수천 번 뽑아 봅니다. "
                "평소에는 표본이 하나뿐이라 볼 수 없는 '표본을 다시 뽑으면 답이 얼마나 달라지는가'를 직접 눈으로 확인할 수 있습니다.\n\n"
                "| | 표준편차 | 표준오차 |\n|---|---|---|\n"
                "| 무엇의 퍼짐인가 | 자료 하나하나 | 표본평균이라는 **추정값** |\n"
                "| 표본이 커지면 | 거의 그대로 | 줄어든다 (σ/√n) |\n"
                "| 질문 | 민원 하나는 보통 얼마나 다른가 | 평균을 얼마나 정확히 알았나 |\n\n"
                "**중심극한정리.** 원자료가 어떤 모양이든 표본이 충분히 크면 표본평균의 분포는 정규분포에 가까워집니다. "
                "t 검정과 신뢰구간이 이 정리에 기댑니다. 다만 '충분히 크다'의 기준은 자료마다 다릅니다. 아래에서 직접 확인합니다.")
        clt = t("stat10_clt_summary.csv")
        smp = t("stat10_clt_samples.csv")
        c1, c2 = st.columns([3, 2])
        with c1, st.container(border=True):
            f1, f2 = st.columns(2)
            var = f1.segmented_control("변수", clt["변수"].unique().tolist(), default=clt["변수"].iloc[0], required=True, key="clt_var")
            nsel = f2.segmented_control("표본 크기 n", sorted(clt["표본 크기 n"].unique().tolist()), default=30, required=True, key="clt_n")
            d = smp[(smp["변수"] == var) & (smp["표본 크기 n"] == nsel)]
            row = clt[(clt["변수"] == var) & (clt["표본 크기 n"] == nsel)].iloc[0]
            hist = alt.Chart(d).mark_bar(color=BLUE).encode(
                x=alt.X("표본평균:Q", bin=alt.Bin(maxbins=45), title=f"표본평균 (n = {nsel}인 표본 3,000개)"), y=alt.Y("count():Q", title="표본 수"))
            mu = alt.Chart(pd.DataFrame({"x": [row["모집단 평균"]]})).mark_rule(color=ORANGE, strokeWidth=2.5).encode(x="x:Q")
            st.altair_chart((hist + mu).properties(height=300), alt=f"{var}, n={nsel} 표본평균의 분포")
            st.caption("주황 선이 모집단 평균(171만 건 전체의 평균)입니다. n 을 바꿔 보세요. 막대가 좁아지고(표준오차 감소) 좌우 대칭이 되어 갑니다(중심극한정리).")
        with c2:
            with st.container(horizontal=True):
                st.metric("표준오차 (이론 σ/√n)", f"{row['표준오차 (이론 σ/√n)']:,.3f}", border=True)
                st.metric("표준오차 (실험)", f"{row['표준오차 (시뮬레이션)']:,.3f}", border=True)
            with st.container(horizontal=True):
                st.metric("표본평균의 왜도", f"{row['표본평균의 왜도']:.2f}", border=True, help="0 이면 좌우 대칭. 0.3 보다 작으면 정규분포에 가깝다고 봤습니다.")
                st.metric("모집단 평균 ±10% 안", f"{row['모집단 평균의 ±10% 안에 든 비율(%)']:.0f}%", border=True)
            with st.container(border=True):
                st.markdown("**표본평균의 왜도: n 이 커지면 0 으로**")
                st.altair_chart(alt.Chart(clt).mark_line(point=True).encode(
                    x=alt.X("표본 크기 n:Q", scale=alt.Scale(type="log"), title="표본 크기 n (로그 축)"), y=alt.Y("표본평균의 왜도:Q", title="왜도"),
                    color=alt.Color("변수:N", scale=alt.Scale(range=[ORANGE, BLUE]), title=None, legend=alt.Legend(orient="bottom")),
                    tooltip=["변수:N", "표본 크기 n:Q", alt.Tooltip("표본평균의 왜도:Q", format=".2f")]).properties(height=200),
                    alt="표본 크기에 따른 표본평균의 왜도")
        raw30 = clt[(clt["표본 크기 n"] == 30)].set_index("변수")["표본평균의 왜도"]
        st.info(
            f"교과서는 흔히 'n ≥ 30 이면 정규분포로 근사해도 된다'고 합니다. 그런데 처리 시간 원자료에서는 n = 30 일 때 표본평균의 왜도가 "
            f"**{raw30.iloc[0]:.2f}** 로 아직 크게 치우쳐 있고, n = 2000 은 되어야 대칭에 가까워집니다. 로그 변환한 값은 n = 30 에서 이미 {raw30.iloc[1]:.2f} 입니다. "
            "**꼬리가 긴 자료에서는 '30' 이 마법의 숫자가 아니며**, 4단계의 로그 변환이 뒤의 추론을 쉽게 만드는 이유가 여기에 있습니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        st.subheader("부트스트랩: 공식 없이 불확실성 재기")
        with st.expander("처음 배우는 사람을 위한 설명: 부트스트랩과 신뢰구간", expanded=True):
            st.markdown(
                "**문제.** 평균의 표준오차는 σ/√n 이라는 공식이 있습니다. 그러면 중앙값이나 90번째 백분위수의 표준오차는? 공식이 없거나 매우 복잡합니다.\n\n"
                "**부트스트랩(1979년, 지금은 표준 도구).** 표본이 하나뿐이니 **그 표본을 모집단인 척** 삼습니다.\n"
                "1. 가진 표본에서 같은 크기만큼 **복원추출**(뽑은 것을 다시 넣고 뽑기)한다\n"
                "2. 그 '다시 뽑은 표본'에서 원하는 통계량(중앙값 등)을 계산한다\n"
                "3. 1~2 를 2,000번 반복한다\n"
                "4. 2,000개 값의 2.5번째와 97.5번째 백분위수가 95% 신뢰구간이다\n\n"
                "**신뢰구간의 정확한 뜻.** '참값이 이 구간에 있을 확률이 95%' 가 아닙니다. "
                "'같은 방법으로 표본을 뽑아 구간을 만드는 일을 반복하면, 그 구간들의 95% 가 참값을 포함한다'입니다. 아래 포함률 실험이 바로 그 반복입니다.")
        bci = t("stat10_bootstrap_ci.csv")
        st.dataframe(bci, hide_index=True, alt="부트스트랩 신뢰구간",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["표본 추정값", "부트스트랩 표준오차", "95% CI 하한", "95% CI 상한", "전체 자료의 값(참값)"]}
                     | {"구간 폭 / 추정값(%)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("2만 건 무작위 표본에서 계산했습니다. 전체 자료의 값(참값)과 비교할 수 있는 것은 이 실습이 171만 건 전체를 갖고 있기 때문입니다. "
                   "평균의 부트스트랩 구간과 고전적 t 구간이 거의 같다는 점도 확인해 보세요. 표본이 크면 두 방법이 일치합니다. "
                   "참값을 놓친 행이 있다면, 95% 구간도 20번에 한 번은 놓치는 것이 정상입니다.")
        cov = t("stat10_ci_coverage.csv")
        p1, p2 = st.columns([3, 2])
        with p1, st.container(border=True):
            st.markdown("**포함률 실험: 크기 50 인 표본을 400번 뽑았을 때 95% 구간이 참값을 맞힌 비율**")
            cl = cov.melt(id_vars="대상", value_vars=["t 구간 포함률(%)", "부트스트랩 구간 포함률(%)"], var_name="방법", value_name="포함률").dropna()
            cl["방법"] = cl["방법"].str.replace(" 포함률(%)", "", regex=False)
            bar = alt.Chart(cl).mark_bar().encode(
                y=alt.Y("대상:N", title=None, axis=alt.Axis(labelLimit=240)), yOffset="방법:N", x=alt.X("포함률:Q", title="포함률 (%)", scale=alt.Scale(domain=[60, 100])),
                color=alt.Color("방법:N", scale=alt.Scale(range=[ORANGE, BLUE]), title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["대상:N", "방법:N", alt.Tooltip("포함률:Q", format=".1f")])
            goal = alt.Chart(pd.DataFrame({"x": [95]})).mark_rule(strokeDash=[5, 3], color="#0b0b0b").encode(x="x:Q")
            st.altair_chart((bar + goal).properties(height=230), alt="신뢰구간 포함률 비교")
        with p2, st.container(border=True):
            craw = cov[cov["대상"] == "처리 시간의 산술평균"].iloc[0]
            st.markdown(
                f"처리 시간의 평균에 대한 '95%' 구간은 실제로 **{craw['t 구간 포함률(%)']:.0f}%** (t 구간), **{craw['부트스트랩 구간 포함률(%)']:.0f}%** (부트스트랩)만 맞힙니다. "
                "표본 50건에는 몇 달짜리 극단값이 들어오지 않는 경우가 많아, 어느 방법이든 평균을 낮게 잡기 때문입니다.\n\n"
                "반면 **중앙값**이나 **로그의 평균**은 95% 근처를 지킵니다. 꼬리가 긴 자료에서는 평균보다 이런 대상을 추정하는 편이 정직합니다. "
                "**부트스트랩은 공식이 없는 통계량에 쓸 수 있다는 것이 강점이지, 만능은 아닙니다.**")
        pci = t("stat10_proportion_ci.csv")
        if pci is not None:
            with st.container(border=True):
                st.markdown("**비율의 신뢰구간 (Wilson): 자치구별 24시간 내 처리율**")
                which = st.segmented_control("표본 크기", ["실제 건수", "100건뿐이었다면"], default="실제 건수", required=True, key="pci_n")
                d = pci[pci["표본"] == which].rename(columns={"24시간 내 처리율(%)": "p", "Wilson 하한": "lo", "Wilson 상한": "hi"})
                base = alt.Chart(d).encode(y=alt.Y("자치구:N", sort=KO, title=None))
                st.altair_chart((base.mark_rule(strokeWidth=3, color=BLUE).encode(x=alt.X("lo:Q", title="24시간 내 처리율 (%)", scale=alt.Scale(domain=[40, 80])), x2="hi:Q")
                                 + base.mark_circle(size=90, color=ORANGE, opacity=1).encode(
                                     x="p:Q", tooltip=["자치구:N", alt.Tooltip("p:Q", format=".2f"), alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f")])
                                 ).properties(height=210), alt="자치구별 처리율과 Wilson 신뢰구간")
                st.caption("실제 건수(수만~수십만)에서는 구간이 점 크기보다 좁습니다. 100건뿐이었다면 ±10%p 가까이 넓어져 자치구끼리 구별되지 않습니다. "
                           "표본이 크면 우연 오차는 거의 사라지지만, 신고 습관이나 기록 방식 같은 **자료 자체의 치우침**은 신뢰구간이 알려 주지 못합니다.")
    concept_footer("s5")

# ============================================================ 7단계: 현대적 방법
with tab_s7:
    stage_header("s7")
    if need("stat_11_modern.py", "stat11_quantreg_coef.csv", "stat11_model_compare.csv", "stat11_permutation_importance.csv"):
        with st.expander("처음 배우는 사람을 위한 설명: 6단계 회귀의 두 가지 아쉬움", expanded=True):
            st.markdown(
                "1. **OLS 는 평균만 설명합니다.** 그런데 처리 시간의 평균은 좋은 대푯값이 아니었고(2단계), 정책적으로는 '가장 오래 걸리는 10%'가 더 궁금할 때가 많습니다. "
                "→ **분위수 회귀**\n"
                "2. **선형 모형은 식의 모양을 사람이 정해야 합니다.** 기관마다 자치구 효과가 다른 것(상호작용)이나 휘어진 관계는 일일이 넣어 줘야 합니다. "
                "→ **그래디언트 부스팅**\n\n"
                "| | 회귀 (통계 모형) | 부스팅 (기계학습) |\n|---|---|---|\n"
                "| 주된 목적 | 설명: 무엇이 얼마나 영향을 주나 | 예측: 새 자료를 얼마나 잘 맞히나 |\n"
                "| 결과물 | 계수, 신뢰구간, p 값 | 예측값 (계수 없음) |\n"
                "| 유연성 | 사람이 정한 식 안에서만 | 상호작용·비선형을 스스로 찾음 |\n"
                "| 평가 | 가정 점검, 유의성 | 학습에 안 쓴 자료에서의 성능 |\n"
                "| 약점 | 식을 잘못 세우면 틀린다 | 과적합 위험, 속을 들여다보기 어렵다 |\n\n"
                "둘은 경쟁이 아니라 용도가 다릅니다. '왜'를 묻는다면 회귀, '얼마나 잘 맞히나'를 묻는다면 부스팅이 먼저입니다.")

        st.subheader("① 분위수 회귀: 평균 말고 분포 전체를")
        with st.expander("처음 배우는 사람을 위한 설명: 분위수 회귀"):
            st.markdown(
                "- **OLS** 는 잔차의 **제곱**의 합을 최소화합니다 → 조건부 **평균**을 추정합니다.\n"
                "- **중앙값 회귀** 는 잔차의 **절댓값**의 합을 최소화합니다 → 조건부 **중앙값**을 추정합니다.\n"
                "- **분위수 회귀** 는 위로 틀린 것과 아래로 틀린 것에 다른 벌점을 줍니다(핀볼 손실). τ = 0.9 라면 '실제가 더 큰' 실수에 9배 무거운 벌점을 주므로, "
                "예측선이 자료의 90% 를 아래에 두는 높이까지 올라갑니다.\n\n"
                "**읽는 법.** 같은 변수의 계수가 τ 에 따라 달라지면, 그 변수는 분포를 통째로 옮기는 것이 아니라 **모양**을 바꾸는 것입니다. "
                "예를 들어 빨리 끝나는 민원(τ = 0.1)에서는 차이가 없는데 오래 끄는 민원(τ = 0.9)에서만 차이가 크다면, "
                "'평균이 길다'가 아니라 '유난히 오래 끄는 민원이 많다'가 정확한 표현입니다. OLS 계수 하나로는 이 둘을 구별할 수 없습니다.\n\n"
                "**장점**: 극단값에 덜 흔들림, 정규성 가정 불필요, 분포 전체를 봄 · **단점**: 계산이 느림, 양 끝 분위수에서 불안정")
        qc = t("stat11_quantreg_coef.csv")
        qinfo = t("stat11_quantreg_info.csv")
        qv = [v for v in qc["변수"].unique() if v != "const"]
        q1, q2 = st.columns([3, 2])
        with q1, st.container(border=True):
            default_v = [v for v in ["agency_group=HPD", "agency_group=DPR", "agency_group=DSNY"] if v in qv]
            chosen = st.multiselect("변수 (기준 범주 대비 계수)", qv, default=default_v, key="qr_vars")
            if chosen:
                qq = qc[qc["변수"].isin(chosen) & qc["분위수 τ"].notna()]
                ols_q = qc[qc["변수"].isin(chosen) & qc["분위수 τ"].isna()]
                cscale = alt.Scale(domain=chosen, range=CATEGORICAL[:len(chosen)])
                band = alt.Chart(qq).mark_area(opacity=0.15).encode(x="분위수 τ:Q", y="CI 2.5%:Q", y2="CI 97.5%:Q", color=alt.Color("변수:N", scale=cscale, legend=None))
                line = alt.Chart(qq).mark_line(point=True).encode(
                    x=alt.X("분위수 τ:Q", title="분위수 τ (0.1 = 빨리 끝나는 쪽, 0.9 = 오래 걸리는 쪽)", axis=alt.Axis(values=[0.1, 0.25, 0.5, 0.75, 0.9])),
                    y=alt.Y("계수:Q", title="계수 (로그 처리시간)"), color=alt.Color("변수:N", scale=cscale, title=None, legend=alt.Legend(orient="bottom", columns=2)),
                    tooltip=["변수:N", "분위수 τ:Q", alt.Tooltip("계수:Q", format=".3f"), alt.Tooltip("배율 exp(계수):Q", format=".2f")])
                ols_r = alt.Chart(ols_q).mark_rule(strokeDash=[5, 3]).encode(y="계수:Q", color=alt.Color("변수:N", scale=cscale, legend=None),
                                                                              tooltip=["변수:N", alt.Tooltip("계수:Q", title="OLS 계수", format=".3f")])
                st.altair_chart((band + line + ols_r).properties(height=330), alt="분위수별 회귀계수")
                st.caption("선은 분위수 회귀 계수(띠는 95% 신뢰구간), 같은 색 점선은 OLS(평균) 계수입니다. 선이 오른쪽으로 올라가면 '오래 걸리는 쪽에서 차이가 더 크다'는 뜻입니다.")
        with q2, st.container(border=True):
            if qinfo is not None:
                st.caption(f"표본 {qinfo['적합 표본'].iloc[0]:,.0f}건 (8월까지 접수분) · 목표변수: {qinfo['목표변수'].iloc[0]} · 기준 범주: {qinfo['기준 범주'].iloc[0]}")
            qs = t("stat11_quantreg_summary.csv")
            st.dataframe(qs, hide_index=True, height=380, alt="분위수별 계수 요약",
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                        ["τ=0.10 계수", "τ=0.50 계수 (중앙값)", "τ=0.90 계수", "OLS 계수 (평균)", "τ=0.90 - τ=0.10"]}
                         | {"읽는 법": st.column_config.TextColumn(width="large")})
        qsi = t("stat11_quantreg_summary.csv").set_index("변수")
        if "agency_group=DPR" in qsi.index and "agency_group=HPD" in qsi.index:
            dpr, hpd = qsi.loc["agency_group=DPR"], qsi.loc["agency_group=HPD"]
            st.info(
                f"공원국(DPR)의 계수는 빨리 끝나는 쪽(τ=0.1)에서 {dpr['τ=0.10 계수']:.2f}, 오래 걸리는 쪽(τ=0.9)에서 {dpr['τ=0.90 계수']:.2f} 입니다. "
                f"로그 척도라서 배율로 바꾸면 각각 약 {np.exp(dpr['τ=0.10 계수']):.0f}배와 {np.exp(dpr['τ=0.90 계수']):,.0f}배입니다(기준: NYPD). "
                "공원국 민원은 '조금 느린' 것이 아니라 '일부가 극단적으로 오래 끄는' 것입니다.  \n"
                f"주거 시설(HPD)은 반대로 τ=0.1 에서도 {hpd['τ=0.10 계수']:.2f} 로 이미 큽니다. 빨리 끝나는 민원조차 느리다는 뜻입니다. "
                "자치구 계수는 τ 에 따라 크게 변하지 않아, 자치구는 분포를 통째로 조금 옮기는 정도입니다. "
                "OLS 계수 하나로는 이 차이가 보이지 않습니다.",
                title="현재 값이 뜻하는 것", icon=":material/lightbulb:")

        st.subheader("② 그래디언트 부스팅: 더 잘 맞히기, 그리고 무엇 덕분인가")
        with st.expander("처음 배우는 사람을 위한 설명: 부스팅, 과적합, 순열 중요도"):
            st.markdown(
                "**결정나무**: '기관이 NYPD 인가? → 예/아니오 → 민원 유형이 난방인가? → …' 처럼 질문을 이어 가며 나누는 모형. 한 그루로는 거칠다.\n\n"
                "**부스팅**: 나무를 한 그루씩 더해 가며, 새 나무가 **지금까지의 나무들이 틀린 부분**을 맞히도록 배운다. 수백 그루를 쌓는다. "
                "표 형태의 자료에서 지금도 가장 잘 맞는 방법 가운데 하나다 (XGBoost, LightGBM, CatBoost 가 모두 이 계열).\n\n"
                "**과적합**: 학습 자료의 잡음까지 외워 새 자료에서 못 맞히는 것. 학습 AUC 와 평가 AUC 의 차이로 확인한다. "
                "이 분석은 8월까지로 배우고 9월로 평가했다.\n\n"
                "**AUC**: '24시간 내 처리' 한 건과 '아닌' 한 건을 골랐을 때 전자에 더 높은 확률을 줄 비율. 0.5 무작위, 1 완벽. "
                "**Brier 점수**: 예측 확률과 실제(0/1)의 차이 제곱 평균. 작을수록 확률이 정직하다.\n\n"
                "**순열 중요도**: 부스팅에는 계수가 없다. 대신 변수 하나를 **무작위로 뒤섞어** 결과와의 관계를 끊고 성능이 얼마나 떨어지는지 본다. "
                "주의: 서로 겹치는 변수(기관과 민원 유형)는 함께 과소평가되고, 중요도는 '예측에 쓸모'이지 '원인'이 아니다.\n\n"
                "**비교 설계**: A → B 는 변수(민원 유형)를 더한 효과, B → C 는 모형을 유연하게 바꾼 효과, C → D 는 시각·요일·좌표를 더한 효과다.")
        mc = t("stat11_model_compare.csv")
        b1, b2 = st.columns([3, 2])
        with b1, st.container(border=True):
            st.markdown("**모형별 평가 AUC (9월 접수분)**")
            mcp = mc[mc["종류"] != "기준"].rename(columns={"AUC (평가, 9월)": "auc"})
            st.altair_chart(alt.Chart(mcp).mark_bar().encode(
                y=alt.Y("모형:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)), x=alt.X("auc:Q", title="AUC", scale=alt.Scale(domain=[0.9, 1.0])),
                color=alt.Color("종류:N", scale=alt.Scale(domain=["통계 모형", "기계학습"], range=[ORANGE, BLUE]), title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["모형:N", alt.Tooltip("auc:Q", format=".4f")]).properties(height=200), alt="모형별 평가 AUC")
            st.caption("가로축이 0.9 에서 시작합니다. 차이를 보려고 확대한 것이니 막대 길이를 그대로 비율로 읽으면 안 됩니다.")
            st.dataframe(mc, hide_index=True, alt="모형 비교표",
                         column_order=["모형", "변수", "AUC (학습)", "AUC (평가, 9월)", "정확도 (평가)", "Brier 점수 (평가)", "과적합 정도 (학습 AUC - 평가 AUC)", "앞 단계 대비 AUC 변화"],
                         column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                        ["AUC (학습)", "AUC (평가, 9월)", "정확도 (평가)", "Brier 점수 (평가)", "과적합 정도 (학습 AUC - 평가 AUC)", "앞 단계 대비 AUC 변화"]})
        with b2:
            with st.container(border=True):
                st.markdown("**순열 중요도** (모형 D, 뒤섞었을 때 AUC 하락)")
                imp = t("stat11_permutation_importance.csv")
                st.altair_chart(hbar(imp, "변수", "AUC 하락 (평균)", label_fmt=".3f", height=230), alt="변수별 순열 중요도")
            cal = t("stat11_calibration.csv")
            if cal is not None:
                with st.container(border=True):
                    st.markdown("**보정: '70%' 라고 하면 정말 70% 인가**")
                    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 3], color="#898781").encode(x="x:Q", y="y:Q")
                    pts = alt.Chart(cal).mark_circle(color=BLUE, opacity=1).encode(
                        x=alt.X("예측 확률 평균:Q", title="예측 확률"), y=alt.Y("실제 비율:Q", title="실제 24시간 내 처리 비율"),
                        size=alt.Size("건수:Q", legend=None, scale=alt.Scale(range=[40, 400])),
                        tooltip=["구간:N", alt.Tooltip("건수:Q", format=","), alt.Tooltip("예측 확률 평균:Q", format=".3f"), alt.Tooltip("실제 비율:Q", format=".3f")])
                    st.altair_chart((diag + pts).properties(height=230), alt="예측 확률 보정 그림")
                    st.caption("점이 대각선 위에 있을수록 모형이 말하는 확률이 정직합니다. 원 크기는 건수입니다.")
        get = lambda p: mc[mc["모형"].str.startswith(p)]["AUC (평가, 9월)"].iloc[0]
        a, b, c, d = get("A."), get("B."), get("C."), get("D.")
        st.info(
            f"평가 AUC 는 로지스틱 {a:.3f} → 민원 유형 추가 {b:.3f} → 부스팅 {c:.3f} → 시각·좌표 추가 {d:.3f} 입니다. "
            f"향상 폭은 **변수를 더한 단계(+{b - a:.3f})** 가 **모형을 바꾼 단계(+{c - b:.3f})** 보다 "
            + (f"약 {(b - a) / (c - b):.0f}배 큽니다. " if c - b > 0 else "훨씬 큽니다. ") +
            "좋은 변수가 좋은 알고리즘보다 먼저라는 것을 보여 줍니다. 학습 AUC 와 평가 AUC 가 거의 같아 과적합도 보이지 않습니다.  \n"
            f"순열 중요도 1위는 **{imp['변수'].iloc[0]}** 이고 {imp['변수'].iloc[1]} 이 뒤를 잇습니다. 둘은 서로 겹치는 정보(유형마다 담당 기관이 정해짐)라 "
            "기관의 중요도는 과소평가되어 있습니다. 8단계에서 '처리 시간은 기관과 유형이 결정한다'를 다른 방법으로 다시 확인합니다.",
            title="현재 값이 뜻하는 것", icon=":material/lightbulb:")
    concept_footer("s7")

# ============================================================ 단계 꼬리말 (기존 탭)
with tab4:
    concept_footer("s1")
with tab1:
    concept_footer("s3")
with tab2:
    concept_footer("s4")
with tab3:
    concept_footer("s6")
with tab7:
    concept_footer("s8")
with tab6:
    concept_footer("s9")
with tab5:
    concept_footer("s10")
with tab8:
    concept_footer("s11")
with tab9:
    concept_footer("s12")
