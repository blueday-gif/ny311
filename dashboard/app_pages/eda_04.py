import pandas as pd
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("1부 데이터 알기 · 4단계", "언제, 어디서, 어떤 민원이 많은가? 변수끼리는 어떤 관계인가? 자치구끼리 차이가 있나?",
       "빈도와 비율, 주기성, 피어슨·스피어만 상관, 상관 ≠ 인과, 다중 비교", "borough, problem, hour, weekday, month, hours",
       "eda_04_patterns.py", "1 둘러보기, 6 검정·회귀, 8 기관 비교, 9 시계열")

view = st.segmented_control("볼 내용", ["범주", "인구당 비교", "시간 패턴", "변수 관계", "자치구 비교"], default="범주",
                            label_visibility="collapsed")

if view == "범주":
    term("빈도와 비율", "빈도 = 몇 건인가, 비율 = 전체 중 몇 %인가. 집단 크기가 다르면 **비율**로 비교해야 공정합니다.")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**자치구별 민원**")
        b = table("eda04_borough_counts")
        st.bar_chart(b, x="borough", y="건수", horizontal=True, sort="-건수", alt="자치구별 민원 건수")
        show_table(b, alt="자치구별 건수와 비율")
    with c2, st.container(border=True):
        st.markdown("**민원 유형 상위 15개**")
        p = table("eda04_problem_counts").head(15)
        st.bar_chart(p, x="problem", y="비율(%)", horizontal=True, sort="-비율(%)", alt="민원 유형 상위 15개 비율")
        show_table(p, alt="민원 유형 상위 15개 건수, 비율, 누적 비율")
    meaning(f"유형은 197개지만 상위 15개가 전체의 {p['누적 비율(%)'].iloc[-1]:.1f}%입니다. 소수 유형에 민원이 몰려 있습니다. "
            "자치구 건수는 브루클린이 가장 많지만, 인구가 5배까지 차이 나서 건수만으로는 비교할 수 없습니다. "
            "외부 인구 자료를 붙인 **주민 1만 명당 민원**은 '인구당 비교' 탭에서 봅니다.")

elif view == "인구당 비교":
    st.caption("데이터: 전처리본(중복 의심 제외) + 자치구 인구(2020 인구총조사, `docs/borough_context.csv`) · "
               "스크립트 `analysis/eda_borough_per_capita.py`")
    term("정규화(1만 명당)", "집단 크기가 다를 때 건수를 크기로 나눠 같은 잣대로 맞추는 것.  \n"
         "예) 신고 16만 건 / 인구 50만 명 × 1만 = 1만 명당 3,262건. 인구가 5배 차이 나는 브루클린과 스태튼아일랜드를 공정하게 비교할 수 있습니다.")
    pc = table("edac09_borough_per_capita")
    top, low = pc.iloc[0], pc.iloc[-1]
    with st.container(horizontal=True):
        st.metric(f"1만 명당 최다 · {top['borough_ko']}", f"{top['1만명당 신고']:,.0f}건",
                  f"시 평균의 {top['시 평균 대비(배)']:.2f}배", delta_color="off", border=True)
        st.metric(f"1만 명당 최소 · {low['borough_ko']}", f"{low['1만명당 신고']:,.0f}건",
                  f"시 평균의 {low['시 평균 대비(배)']:.2f}배", delta_color="off", border=True)
        bx = pc.set_index("borough").loc["BRONX"]
        st.metric("브롱크스 순위", f"건수 {bx['건수 순위']}위 → 인구당 {bx['1만명당 순위']}위", border=True)
    figure("edac09_borough_per_capita", alt="자치구별 신고 수와 주민 1만 명당 신고 수 비교")
    show_table(pc[["borough_ko", "신고수", "population_census2020", "1만명당 신고", "시 평균 대비(배)", "1만명당 사건",
                   "사건당 신고", "건수 순위", "1만명당 순위", "1만명당 순위(대략 인구)", "사건 기준 순위"]],
               alt="자치구별 신고 수, 인구, 1만 명당 신고·사건 수와 순위",
               column_config={"borough_ko": "자치구",
                              "population_census2020": st.column_config.NumberColumn("인구(2020)", format="localized"),
                              "신고수": st.column_config.NumberColumn(format="localized"),
                              "1만명당 신고": st.column_config.NumberColumn(format="%.0f"),
                              "1만명당 사건": st.column_config.NumberColumn(format="%.0f"),
                              "시 평균 대비(배)": st.column_config.NumberColumn(format="%.2f"),
                              "사건당 신고": st.column_config.NumberColumn(format="%.2f")})
    meaning("건수 3위인 브롱크스가 인구당으로는 1위(시 평균의 약 1.3배)이고, 스태튼아일랜드는 인구당으로도 꼴찌(약 0.7배)입니다. "
            "같은 날·같은 곳·같은 유형을 하나로 묶은 **사건** 기준으로 봐도 브롱크스가 1위라서, 한 건물의 반복 신고만으로 생긴 차이는 아닙니다.")
    caution("맨해튼(2위)과 브루클린(3위)은 차이가 2%뿐이라, 인구를 최근 대략값으로 바꾸면 순위가 뒤집힙니다(표의 '대략 인구' 순위). "
            "차이가 작은 순위는 인구 기준 하나에 기대어 말하지 마세요. 또 맨해튼은 통근자·관광객이 많아 주민 수로 나누면 부풀려집니다.")

    st.markdown("**어떤 민원에서 차이가 나나: 대표 유형별 시 평균 대비 배율**")
    figure("edac09_group_ratio", alt="자치구별 대표 민원 유형의 1만 명당 신고 수, 시 평균 대비 배율 히트맵")
    with st.expander("기관별 1만 명당 신고와 배율 표"):
        show_table(table("edac09_agency_per_capita"), alt="자치구 × 기관 1만 명당 신고 수")
        show_table(table("edac09_agency_ratio"), alt="자치구 × 기관 시 평균 대비 배율")
    meaning("임대 주거가 많은 브롱크스는 난방·온수, 실내 설비, 주거 소음이 시 평균의 약 2배입니다(HPD 전체도 약 2배). "
            "자가·단독주택이 많은 스태튼아일랜드는 실내 설비 민원이 시 평균의 0.2~0.3배인 대신 가로수(2.3배), 도로 파손(1.5배), "
            "청소(DSNY 1.8배)가 많습니다. 맨해튼은 거리·상업 소음이 가장 높습니다(1.5배). 주거 형태와 민원 성격이 맞물려 있다는 가설과 맞는 모습입니다.")
    caution("자치구는 5개뿐이라 '임대 비율이 높을수록 난방 민원이 많다'는 상관을 말하기엔 표본이 너무 작습니다. "
            "또 311 신고는 문제가 많은 곳이 아니라 **신고하는 곳**을 보여 줍니다(신고 습관·언어·접근성의 영향). "
            "배경 자료와 가설은 `docs/borough_context.md`에 있습니다.")

elif view == "시간 패턴":
    term("주기성", "일정한 간격으로 같은 모양이 반복되는 성질. 예) 매일 새벽에 줄고 낮에 늘어나는 패턴 = 하루 주기.")
    h = table("eda04_by_hour")
    w = table("eda04_by_weekday").rename(columns={"Unnamed: 0": "요일"})
    m = table("eda04_by_month")
    m["created"] = m["created"].astype(str)
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**시간대별**")
        st.line_chart(h, x="hour", y="건수", alt="시간대별 접수 건수")
    with c2, st.container(border=True):
        st.markdown("**요일별**")
        w["순서"] = range(len(w))
        st.bar_chart(w, x="요일", y="건수", sort="순서", alt="요일별 접수 건수")
    with c3, st.container(border=True):
        st.markdown("**월별**")
        st.bar_chart(m, x="created", y="건수", alt="월별 접수 건수")
    meaning(f"새벽 {h.loc[h['건수'].idxmin(), 'hour']}시에 가장 적고 {h.loc[h['건수'].idxmax(), 'hour']}시 무렵 가장 많습니다 "
            "(사람의 활동 시간을 따라가는 하루 주기). 평일이 많고 주말이 적습니다(1주 주기). 요일별로는 화요일이 가장 많고 월요일이 그다음입니다."
            "월별 차이는 크지 않습니다. 첫 달(2025-11)은 1일 밤부터 시작해 하루 가까이 덜 들어 있습니다.")

elif view == "변수 관계":
    term("피어슨 vs 스피어만", "**피어슨 r**: 두 숫자가 '직선'으로 함께 움직이는 정도. 극단값에 크게 흔들립니다.  \n"
         "**스피어만 ρ**: 값을 순위로 바꾼 뒤 구한 상관. 방향만 보며 극단값에 강합니다.  \n"
         "예) 1,2,3,4 와 1,4,9,1000 은 직선은 아니지만 순서가 같아서 ρ = 1.")
    c = table("eda04_correlation")
    c1, c2 = st.columns([2, 3])
    with c1:
        with st.container(horizontal=True):
            st.metric("피어슨 r", f"{c.loc[0, '상관계수']:.2f}", border=True)
            st.metric("스피어만 ρ", f"{c.loc[1, '상관계수']:.2f}", border=True)
        meaning("하루 단위로 '접수 건수'와 '그날 민원의 처리 시간 중앙값'을 비교했습니다. 둘 다 양수라 "
                "'민원이 많은 날 처리도 길다'는 방향은 같습니다. 피어슨이 더 큰 것은 (가능성으로) 민원이 폭증한 소수의 극단적인 날이 "
                "직선 관계를 끌어올렸기 때문입니다.")
        caution("상관관계는 인과관계가 아닙니다. 두 값을 함께 움직이게 하는 제3의 원인(폭설, 공휴일, 민원 유형 구성 변화)이 있을 수 있습니다.")
    with c2:
        figure("eda04_scatter_daily", alt="하루 접수 건수와 처리 시간 중앙값의 산점도")

else:
    g = table("eda04_borough_hours")
    term("다중 비교 문제", "비교를 여러 번 하면 우연히 '유의한' 결과가 나올 확률이 커집니다.  \n"
         "예) 5개 자치구를 둘씩 비교하면 10번. 각각 5% 확률로 우연히 유의하면, 하나 이상 걸릴 확률은 약 40%.  \n"
         "해결: **Bonferroni 보정** = 기준 0.05 를 비교 횟수로 나눠(0.005) 더 엄격하게 봅니다.")
    c1, c2 = st.columns([2, 3])
    with c1:
        show_table(g, alt="자치구별 처리 시간 건수, 중앙값, 평균, 24시간 처리율",
                   column_config={"건수": st.column_config.NumberColumn(format="localized")})
    with c2:
        figure("eda04_borough_boxplot", alt="자치구별 처리 시간 상자그림")
    meaning(f"중앙값은 {g.iloc[0]['borough']} {g.iloc[0]['중앙값']:.1f}시간 ~ {g.iloc[-1]['borough']} {g.iloc[-1]['중앙값']:.1f}시간입니다. "
            "그런데 평균 순서는 중앙값 순서와 다릅니다. 건수도 최대 7.6배 차이가 나서, 작은 집단의 값은 덜 안정적입니다.")
    caution("Kruskal-Wallis 검정의 p값은 0에 가깝지만, 표본이 수백만 건이면 아주 작은 차이도 '유의'하게 나옵니다. "
            "차이의 크기는 중앙값 차이(몇 시간인가)로 판단하세요. 또 자치구마다 민원 유형 구성이 달라서, 이 차이가 "
            "처리 속도 때문인지 유형 구성 때문인지는 이 표만으로 알 수 없습니다(11단계 모델 해석 참고).")

next_step("5. 무엇을 예측하나: 머신러닝 문제 정의")
