import pandas as pd
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("심화 탐구 · 자치구 성격", "민원 유형 구성만 보고 각 자치구가 어떤 곳인지 짐작할 수 있을까? 그 특징은 자치구 전체의 모습인가, 한두 동네의 모습인가?",
       "배율(특화 지수·입지계수), 로그 눈금, 사건 기준 재확인, 동네(CB) 일관성, 코사인 유사도, 계절성",
       "borough, problem, community_board, incident_group_id, created_date", "eda_borough_character.py")

BORO = ["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"]
KO = {"MANHATTAN": "맨해튼", "BRONX": "브롱크스", "BROOKLYN": "브루클린", "QUEENS": "퀸스", "STATEN ISLAND": "스태튼아일랜드"}

# 민원 유형만으로 추정한 성격(분석 결과를 읽고 쓴 해석). 숫자는 아래 표에서 매번 다시 읽는다
PERSONA = {
    "MANHATTAN": dict(
        icon=":material/location_city:", key="도심·관광·노숙",
        one="사는 사람보다 오가는 사람이 많은 도심. 차와 집 앞 공간이 없다.",
        body="분실물·택시·호출 차량(교통수단·방문객 주제 4.8배), 노숙 텐트·노숙인 지원·구걸·노점(노숙·거리 질서 4.5배)이 "
             "다른 자치구보다 훨씬 많습니다. 반대로 차고 앞 주차(0.11배)·방치 차량·가로수·눈 민원은 거의 없습니다. "
             "통근자·관광객·상점이 몰리고, 고층 건물이 빽빽하며, 차를 덜 타는 도시 중심부의 모습입니다. "
             "노숙 텐트·노숙인 지원은 12개 커뮤니티 보드(CB) 가운데 11~12곳에서 배율이 1을 넘어, 한 동네가 아니라 맨해튼 전체의 특징입니다. "
             "반면 헬기 소음은 7번 CB(어퍼웨스트사이드) 한 곳이 70%를 차지하는 국지적 특징입니다.",
        check="배경 자료: 1인 가구·임차 비중이 압도적이고 가구 소득이 5개 구 중 가장 높습니다. 주거 설비 민원이 1.2배로 약간 많은 것은 "
              "북부(할렘·워싱턴하이츠)의 임대 아파트 영향입니다."),
    "BRONX": dict(
        icon=":material/apartment:", key="노후 임대 아파트",
        one="엘리베이터 있는 오래된 임대 아파트에 사람이 빽빽이 사는 곳.",
        body="배율 상위 10개가 **모두 주택국(HPD)·건물국(DOB) 소관의 집 안 하자**입니다(가전·엘리베이터·벽·바닥·배관·난방·창문·누수·위생, 1.8~2.2배). "
             "주거 소음도 2.0배로, 민원 6건 중 1건이 이웃 소음입니다. 노숙·가로수·호출 차량 민원은 적습니다. "
             "이 특징은 12개 CB 가운데 10~11곳에서 똑같이 나타나 브롱크스 전체가 같은 성격입니다. "
             "주민 수로 나눠도 주거 설비·주거 소음이 시 평균의 약 2배라서, 신고 구성만이 아니라 양 자체가 많습니다.",
        check="배경 자료: 다세대 임대 비율이 높고 빈곤율이 가장 높으며 가구 소득이 가장 낮습니다. 관리가 덜 된 오래된 임대 건물이라는 추정과 맞습니다. "
              "단, 세입자가 HPD 신고로 권리를 지키는 습관도 숫자를 키웁니다."),
    "BROOKLYN": dict(
        icon=":material/holiday_village:", key="평균형 대형 주거지",
        one="가장 '보통의 뉴욕'. 주차난과 공사가 조금 더 많은 오래된 주거 밀집지.",
        body="배율이 큰 유형이 거의 없습니다(대부분 1.3~1.8배). 시 전체 신고의 31%를 차지하기도 하지만, 자기를 뺀 나머지 자치구와 비교해도 "
             "구성 차이(JS 거리 0.14)가 5개 구 중 가장 작습니다. 그래도 거리 청소 민원(2.9배, 18개 CB 전부), 그라피티·주택 폐기물·공사·쥐, "
             "그리고 불법 주차(1.45배, 신고 5건 중 1건)가 많아 **주차 공간이 부족하고 공사가 잦은 오래된 주거지**로 읽힙니다. "
             "노숙·노점·거리 소음은 적습니다. 퀸스와 구성이 가장 닮았습니다(코사인 유사도 0.97).",
        check="배경 자료: 신규 유입층과 저소득·이민자 주거지가 공존합니다. 자치구 안 차이가 커서 CB 단위로 보면 북부(젠트리피케이션 지역)와 남부의 성격이 다릅니다(8단계 지역 쏠림)."),
    "QUEENS": dict(
        icon=":material/directions_car:", key="차량·교외 주택·공항",
        one="차고 딸린 주택에서 차를 타고 다니는 넓은 주거지, 그리고 공항.",
        body="차고 앞 주차(2.2배, 14개 CB 전부)·방치 차량·차량 소음·도로 파손(2.2배, 14개 CB 전부)·하수 민원이 많아 "
             "**개인 주택과 자동차, 넓은 도로망**의 모습입니다. 반대로 아파트 실내 하자(바닥·벽·누수·문창문 0.36~0.46배)는 적습니다. "
             "건물 용도 위반(3.4배)은 불법 개조·쪼개기 임대를 뜻하는데 14개 CB 중 13곳에서 많아 퀸스 전체의 특징입니다. "
             "반면 택시 민원(2.5배)은 배율 1을 넘는 CB가 1곳뿐이고 사건 기준으로 다시 세면 0.9배로 떨어집니다. "
             "공항(JFK·라과디아) 주변의 반복 신고가 만든 착시에 가깝습니다. 마약·차량 소음도 8번 CB 한 곳이 50~60%를 차지합니다.",
        check="배경 자료: 면적이 가장 넓고 이민자 비율이 매우 높은 중산층 주거지입니다. 공항 2곳이 있습니다."),
    "STATEN ISLAND": dict(
        icon=":material/forest:", key="교외 단독주택·자연",
        one="뉴욕 안의 교외. 집 앞 나무·눈·가로등을 주민이 직접 신고한다.",
        body="눈·얼음(7.3배), 불법 게시물, 보도·뿌리, 쓰레기 미수거, 가로등, 나무(3.3~3.7배), 동물 사체 같은 "
             "**집 밖 공공시설·자연** 민원이 많고, 모두 3개 CB 전부에서 나타납니다. 나무·공원·눈 주제는 3.9배입니다. "
             "반대로 노숙 텐트(0.09배)·노점·엘리베이터·난방(0.24배) 민원은 거의 없습니다. 아파트보다 단독주택, 걷기보다 자가용의 생활입니다. "
             "다른 자치구와 구성이 가장 다르고(JS 거리 0.36), 맨해튼과는 정반대입니다(코사인 유사도 0.64).",
        check="배경 자료: 자가 보유율이 가장 높고(약 65% 이상) 단독주택·자가용 의존이 큽니다. 주민 1만 명당 신고는 가장 적지만, 나무·공원·눈은 인구당으로도 시 평균의 2.5배입니다."),
}

sig = table("edac10_signature")
lq = table("edac10_problem_ratio")
theme = table("edac10_theme")
dist = table("edac10_distance").set_index("borough")
sim = table("edac10_similarity").set_index("자치구")
monthly = table("edac10_theme_monthly")

term("배율 (특화 지수)", "**배율 = 이 자치구 안에서 그 유형의 비중 ÷ 나머지 4개 자치구에서 그 유형의 비중**  \n"
     "예) 맨해튼 민원의 4.41%가 노숙 텐트이고, 나머지 자치구는 0.42%라면 배율 = 4.41 ÷ 0.42 ≈ 10.5배.  \n"
     "건수가 아니라 **비중끼리** 나누므로 자치구 크기(신고량) 차이는 걷어 냅니다. 1보다 크면 '유난히 많다', 작으면 '유난히 적다'.  \n"
     "지역경제학의 입지계수(LQ)와 같은 생각입니다. 비교 대상을 '시 전체'가 아니라 '나머지 자치구'로 둔 것은, "
     "시 전체의 31%인 브루클린이 시 평균을 자기 쪽으로 끌어당기기 때문입니다.")

# ── 한 줄 성격 카드 ──
with st.container(horizontal=True):
    for b in BORO:
        p = PERSONA[b]
        top = sig[(sig["borough"] == b) & (sig["방향"] == "많음")].head(3)
        with st.container(border=True):
            st.markdown(f"**{p['icon']} {KO[b]}**  \n:primary[{p['key']}]")
            st.caption(p["one"])
            st.markdown("  \n".join(f"`{r.problem}` {r.배율:.1f}배" for r in top.itertuples()))

view = st.segmented_control("볼 내용", ["한눈에 보기", "자치구 깊이 보기", "인구당 비교", "계절 변화", "닮은 정도"],
                            default="한눈에 보기", label_visibility="collapsed")

if view == "한눈에 보기":
    st.markdown("**비슷한 유형을 11개 주제로 묶어 본 배율**")
    figure("edac10_theme_ratio", alt="자치구 × 주제 배율 히트맵. 빨강은 많음, 파랑은 적음")
    meaning("다섯 자치구가 서로 다른 칸에서 빨갛습니다. **맨해튼**은 노숙·거리 질서(4.5배)와 교통수단·방문객(4.8배), "
            "**브롱크스**는 주거 설비(1.8배)와 주거 소음(2.0배), **퀸스**는 주차·차량과 도로·상하수도(약 1.6배), "
            "**스태튼아일랜드**는 나무·공원·눈(3.9배)과 도로·청소(약 2배)가 많습니다. **브루클린**은 모든 칸이 0.4~1.4배로 가장 '평범'합니다. "
            "집 안 문제(임대 아파트) → 집 앞 문제(자동차·도로) → 집 밖 자연(나무·눈)으로, 도심에서 멀어질수록 민원의 무대가 바깥으로 옮겨 갑니다.")

    st.markdown("**주제 구성 비교 (각 자치구 민원 = 100%)**")
    tl = theme.copy()
    tl["자치구"] = tl["borough_ko"]
    st.bar_chart(tl, x="자치구", y="비중(%)", color="theme", stack="normalize", horizontal=True, sort=False,
                 alt="자치구별 주제 구성 100% 누적 막대")
    with st.expander("주제별 비중·배율 표, 주제에 들어간 유형"):
        pv = theme.pivot(index="borough_ko", columns="theme", values="비중(%)").loc[[KO[b] for b in BORO]]
        st.dataframe(pv.style.format("{:.1f}").background_gradient(cmap="Blues", axis=0), alt="자치구 × 주제 비중(%)")
        rv = theme.pivot(index="borough_ko", columns="theme", values="배율").loc[[KO[b] for b in BORO]]
        st.dataframe(rv.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=0, vmax=2, axis=None),
                     alt="자치구 × 주제 배율")
        mem = table("edac10_theme_members").groupby("theme")["problem"].apply(", ".join).reset_index()
        show_table(mem, alt="주제별 포함 유형", column_config={"theme": "주제", "problem": "포함 유형"})

    st.markdown("**자치구별 특징 유형 (로그 눈금: 2배와 ½배가 같은 길이)**")
    figure("edac10_signature", alt="자치구별로 많은 유형과 적은 유형의 배율 막대")
    term("로그 눈금", "배율 4배와 ¼배는 '같은 정도로 치우친' 것인데, 보통 눈금에서는 4배 막대가 훨씬 길어 보입니다. "
         "로그 눈금은 1을 가운데 두고 2배·4배·8배와 ½·¼·⅛을 같은 간격으로 놓아, 많음과 적음을 공정하게 비교하게 해 줍니다.")

elif view == "자치구 깊이 보기":
    b = st.segmented_control("자치구", BORO, format_func=KO.get, default="MANHATTAN", key="boro_pick")
    b = b or "MANHATTAN"
    p, d = PERSONA[b], dist.loc[b]
    with st.container(horizontal=True):
        st.metric("신고 수", f"{d['신고수']:,.0f}건", f"시 전체의 {d['시 전체 신고 중 비중(%)']:.1f}%", delta_color="off", border=True)
        st.metric("나머지 자치구와 구성 차이 (JS 거리)", f"{d['나머지 자치구와의 JS 거리']:.2f}",
                  f"5개 구 중 {int(dist['나머지 자치구와의 JS 거리'].rank(ascending=False)[b])}위", delta_color="off", border=True)
        st.metric("50%를 채우는 유형 수", f"{d['50% 도달 유형 수']:.0f}개", f"유효 유형 수 {d['유효 유형 수']:.1f}개",
                  delta_color="off", border=True)
    st.markdown(f"#### {p['icon']} {KO[b]}: {p['key']}")
    st.markdown(p["body"])
    st.caption(p["check"])

    s = sig[sig["borough"] == b]
    hi, lo = s[s["방향"] == "많음"], s[s["방향"] == "적음"]
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**유난히 많은 유형 (배율 상위 10)**")
        st.bar_chart(hi, x="problem", y="배율", horizontal=True, sort="-배율", color="#D6604D", alt=f"{KO[b]}에 많은 유형 배율")
    with c2, st.container(border=True):
        st.markdown("**유난히 적은 유형 (배율 하위 8)**")
        st.bar_chart(lo, x="problem", y="배율", horizontal=True, sort="배율", color="#4393C3", alt=f"{KO[b]}에 적은 유형 배율")

    st.markdown("**특징이 진짜인지 두 번 확인하기**")
    term("사건 기준 배율", "한 건물·한 장소에서 같은 날 같은 유형을 여러 번 신고하면 신고 수가 부풀려집니다. "
         "같은 사건(같은 날·같은 곳·같은 유형)을 1건으로 다시 세서 배율을 구해, 반복 신고가 만든 착시인지 확인합니다.")
    term("배율>1 CB 비율", "자치구 안의 커뮤니티 보드(CB, 우리의 '동' 정도) 가운데 그 유형의 배율이 1을 넘는 곳의 비율. "
         "100%면 자치구 어디서나 나타나는 특징, 10~30%면 한두 동네의 특징입니다. '최다 CB 점유'는 그 유형 신고 중 가장 많은 한 CB가 차지하는 몫입니다.")
    show = s[["방향", "problem", "theme", "건수", "자치구 비중(%)", "나머지 비중(%)", "배율", "사건 기준 배율",
              "배율>1 CB 비율(%)", "최다 CB", "최다 CB 점유(%)"]]
    show_table(show, alt=f"{KO[b]} 특징 유형 상세 표",
               column_config={"problem": "유형", "theme": "주제",
                              "건수": st.column_config.NumberColumn(format="localized"),
                              "자치구 비중(%)": st.column_config.NumberColumn(format="%.2f"),
                              "나머지 비중(%)": st.column_config.NumberColumn(format="%.2f"),
                              "배율": st.column_config.NumberColumn(format="%.2f"),
                              "사건 기준 배율": st.column_config.NumberColumn(format="%.2f"),
                              "배율>1 CB 비율(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                              "최다 CB 점유(%)": st.column_config.NumberColumn(format="%.0f%%")})

    flags = []
    for _, r in hi.iterrows():
        g, cbr, top_cb, top_sh = r["사건 기준 배율"], r["배율>1 CB 비율(%)"], r["최다 CB"], r["최다 CB 점유(%)"]
        if pd.notna(g) and g < r["배율"] * 0.6:
            flags.append(f"`{r['problem']}`: 사건 기준으로 다시 세면 {r['배율']:.1f}배 → {g:.1f}배. 반복 신고가 부풀린 부분이 큽니다.")
        if cbr < 40:
            where = (f" 가장 많은 CB 한 곳({top_cb})이 {top_sh:.0f}%를 차지합니다." if pd.notna(top_cb)
                     else " 대부분 CB 번호가 없는 장소(고속도로·공항 등)에서 접수됐습니다.")
            flags.append(f"`{r['problem']}`: 배율 1을 넘는 CB가 {cbr:.0f}%뿐인 국지적 특징입니다.{where}")
    if flags:
        caution("  \n".join(dict.fromkeys(flags)))
    else:
        meaning("상위 특징 유형이 사건 기준으로 다시 세도 비슷하고, 대부분의 CB에서 나타납니다. 자치구 전체의 성격으로 봐도 됩니다.")

elif view == "인구당 비교":
    term("비중 배율 vs 인구당 배율", "**비중 배율**은 '민원 구성'의 차이(전체 민원 중 몇 %인가)이고, **인구당 배율**은 '양'의 차이(주민 1만 명당 몇 건인가)입니다.  \n"
         "예) 신고가 전체적으로 적은 스태튼아일랜드는 비중이 높아도 인구당으로는 평범할 수 있습니다. 두 값이 함께 높아야 '정말 많다'고 말할 수 있습니다.")
    pc = theme.pivot(index="borough_ko", columns="theme", values="1만명당 시 평균 대비").loc[[KO[b] for b in BORO]]
    st.markdown("**주제별 주민 1만 명당 신고: 시 평균 대비 배율** (2020 인구총조사)")
    st.dataframe(pc.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=0, vmax=2, axis=None),
                 alt="자치구 × 주제 1만 명당 신고의 시 평균 대비 배율")
    raw = theme.pivot(index="borough_ko", columns="theme", values="1만명당").loc[[KO[b] for b in BORO]]
    with st.expander("주제별 주민 1만 명당 신고 수(건)"):
        st.dataframe(raw.style.format("{:,.0f}"), alt="자치구 × 주제 1만 명당 신고 수")
    meaning("구성 배율과 인구당 배율이 같은 방향입니다. 브롱크스 주거 설비(1.96배)·주거 소음(2.07배), 스태튼아일랜드 나무·공원·눈(2.53배), "
            "맨해튼 교통수단·방문객(2.80배)·노숙·거리 질서(2.71배)는 **구성도 양도 많습니다**. "
            "반면 퀸스의 주차·차량은 구성 배율 1.63배에 비해 인구당은 1.26배로 줄어듭니다. 퀸스 주민이 신고를 덜 하는 편이라, 구성에서 주차가 도드라져 보이는 몫이 있습니다.")
    caution("맨해튼은 낮에 통근자·관광객이 몰려 주민 수보다 '그곳에 있는 사람'이 훨씬 많습니다. 주민 수로 나눈 맨해튼 값은 부풀려집니다.")

elif view == "계절 변화":
    c1, c2 = st.columns([1, 2])
    with c1:
        b = st.radio("자치구", BORO, format_func=KO.get, key="season_boro")
        ths = st.multiselect("주제", sorted(monthly["theme"].unique()),
                             default=["주거 설비(실내)", "거리·상업 소음", "주거 소음", "나무·공원·눈", "주차·차량"])
    with c2:
        mm = monthly[(monthly["borough"] == b) & monthly["theme"].isin(ths)]
        st.line_chart(mm, x="ym", y="비중(%)", color="theme", alt=f"{KO[b]} 월별 주제 비중",
                      x_label="접수 월", y_label="그 달 민원 중 비중(%)")
    meaning("성격은 계절에 따라 '세기'가 달라집니다. 겨울(12~2월)에는 난방 때문에 주거 설비 비중이 크게 오르고(브롱크스 1월 58%), "
            "스태튼아일랜드는 눈·얼음 때문에 나무·공원·눈이 치솟습니다. 여름(6~8월)에는 거리·상업 소음이 늘어, "
            "브롱크스도 7월에는 거리 소음 비중(20%)이 맨해튼(18%)을 넘습니다. 하지만 1년 내내 **순서**(어느 자치구가 어떤 주제에서 높은가)는 크게 바뀌지 않습니다.")
    caution("데이터가 2025-11 ~ 2026-10의 1년뿐입니다. 한 번의 겨울(폭설·한파 정도)에 따라 겨울 값이 해마다 달라질 수 있습니다.")

else:
    term("코사인 유사도", "각 자치구의 민원을 '유형별 비중' 194개 숫자로 된 화살표로 보고, 두 화살표가 같은 방향인지 재는 값. "
         "1 = 구성이 똑같음, 0 = 전혀 겹치지 않음. 신고량(화살표 길이)은 무시하고 구성(방향)만 봅니다.")
    term("JS 거리 (젠슨-섀넌)", "두 비중 분포가 얼마나 다른지 재는 거리. 0 = 똑같음, 1 = 전혀 다름. "
         "여기서는 각 자치구를 '자기를 뺀 나머지 자치구'와 비교합니다. 값이 클수록 다른 곳과 구성이 다른 '개성 강한' 자치구입니다.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**자치구끼리 민원 구성 유사도**")
        st.dataframe(sim.style.format("{:.2f}").background_gradient(cmap="Greens", vmin=0.6, vmax=1, axis=None),
                     alt="자치구 쌍별 코사인 유사도 행렬")
    with c2:
        st.markdown("**개성 강한 순서 (나머지 자치구와의 JS 거리)**")
        dd = dist.reset_index().sort_values("나머지 자치구와의 JS 거리", ascending=False)
        st.bar_chart(dd, x="borough_ko", y="나머지 자치구와의 JS 거리", horizontal=True, sort="-나머지 자치구와의 JS 거리",
                     alt="자치구별 나머지 자치구와의 JS 거리", x_label="", y_label="")
    show_table(dist.reset_index()[["borough_ko", "나머지 자치구와의 JS 거리", "시 전체와의 JS 거리", "50% 도달 유형 수", "유효 유형 수"]],
               alt="자치구별 구성 거리와 다양성",
               column_config={"borough_ko": "자치구",
                              "나머지 자치구와의 JS 거리": st.column_config.NumberColumn(format="%.3f"),
                              "시 전체와의 JS 거리": st.column_config.NumberColumn(format="%.3f"),
                              "유효 유형 수": st.column_config.NumberColumn(format="%.1f")})
    meaning("가장 닮은 쌍은 **브루클린–퀸스(0.97)**: 둘 다 주차·주택가 중심입니다. 의외로 **맨해튼–브롱크스(0.91)**도 꽤 닮았는데, "
            "둘 다 난방·주거 소음·거리 소음 같은 '아파트 도시' 유형이 상위에 있기 때문입니다(맨해튼 북부는 브롱크스와 생활권이 이어집니다). "
            "가장 다른 쌍은 **맨해튼–스태튼아일랜드(0.64)**. 개성은 스태튼아일랜드(0.36) > 맨해튼(0.31) > 브롱크스(0.27) > 퀸스(0.24) > 브루클린(0.14) 순입니다. "
            "브롱크스는 상위 5개 유형만으로 민원의 절반을 채워(다른 곳은 7~9개) 가장 '한쪽으로 쏠린' 자치구이기도 합니다.")

st.divider()
caution("이 페이지의 '성격'은 **민원 유형만으로 한 추정**입니다. 311 신고는 문제가 많은 곳이 아니라 **신고하는 곳**을 보여 줍니다. "
        "신고 습관·언어·앱 사용·기관 신뢰도가 자치구마다 다릅니다. 실제 주거 형태·소득과의 관계를 말하려면 인구조사(ACS) 자료를 CB 단위로 붙여 "
        "상관을 봐야 합니다. 자치구는 5개뿐이라 자치구 단위 상관은 표본이 너무 작습니다.")
next_step("5. 무엇을 예측하나: 머신러닝 문제 정의")
