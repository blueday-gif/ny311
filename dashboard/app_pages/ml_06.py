import streamlit as st

from dash_common import header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 6단계", "머신러닝 모델이 '잘했다'고 말하려면, 무엇과 비교해야 하나?",
       "기준선, 정확도, 다수결 예측, 규칙 기반 예측", "within_24h, problem, created", "ml_06_split_baseline.py",
       "6 검정·회귀")

base = table("ml06_baselines")
monthly = table("ml06_monthly_rate")
monthly["created"] = monthly["created"].astype(str)

term("기준선", "아주 단순한 방법으로 낸 점수. 모델은 적어도 이것보다 좋아야 쓸 이유가 있습니다.  \n"
     "예) 일기예보 모델이 '내일도 오늘과 같다'보다 못 맞히면 쓸모가 없습니다.")
term("정확도", "전체 예측 중 맞힌 비율. 이해하기 쉽지만, 정답이 한쪽으로 쏠린 문제에서는 속기 쉽습니다(10단계).")

with st.container(horizontal=True):
    st.metric("① 다수결 (항상 '24시간 안')", f"{base.loc[0, '평가 정확도']:.3f}", border=True)
    st.metric("② 민원 유형별 다수결 규칙", f"{base.loc[1, '평가 정확도']:.3f}", border=True)

meaning(f"아무 정보도 쓰지 않아도 {base.loc[0, '평가 정확도'] * 100:.1f}%를 맞힙니다. 민원 유형 하나만 보고 찍으면 "
        f"{base.loc[1, '평가 정확도'] * 100:.1f}%로 크게 오릅니다. 이 문제의 핵심 정보는 **민원 유형**입니다. "
        f"앞으로 만들 모델은 0.58이 아니라 **{base.loc[1, '평가 정확도']:.2f}보다** 나아야 의미가 있습니다.")

c1, c2 = st.columns([3, 2])
with c1, st.container(border=True):
    st.markdown("**월별 24시간 처리율** (학습 기간 vs 평가 기간)")
    st.bar_chart(monthly, x="created", y="처리율", color="구분", alt="월별 24시간 처리율, 학습과 평가 기간 구분")
with c2, st.container(border=True):
    st.markdown("**민원 유형별 규칙** (학습 기간 처리율)")
    rule = table("ml06_problem_rule").rename(columns={"Unnamed: 0": "민원 유형"})
    show_table(rule, alt="민원 유형별 24시간 처리율과 규칙 예측", height=320,
               column_config={"24시간 처리율(학습)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

meaning(f"1월 {monthly['처리율'].min():.1f}% ~ 5월 {monthly['처리율'].max():.1f}%로 월마다 처리율이 크게 오르내립니다. "
        "겨울에는 오래 걸리는 난방·온수 민원이 늘어 처리율이 낮아집니다. 모델은 학습 기간의 패턴으로 평가 기간을 맞혀야 합니다.")

next_step("7. 첫 모델: 로지스틱 회귀")
