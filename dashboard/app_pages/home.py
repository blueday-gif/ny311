import pandas as pd
import streamlit as st

from dash_common import meaning, show_table, table

overview = table("eda01_overview").set_index("항목")["값"]
split = table("ml05_split", index_col=0)

st.markdown(
    "뉴욕시 311 민원 약 **396만 건**으로 데이터 분석과 머신러닝을 처음부터 차례대로 익히는 대시보드입니다. "
    "왼쪽 메뉴는 공부 순서입니다. 처음이라면 **1. 데이터 점검**부터 차례대로 보세요."
)

with st.container(horizontal=True):
    st.metric("민원 건수", f"{int(overview['행 수']):,}건", border=True)
    st.metric("열", f"{int(overview['열 수'])}개", border=True)
    st.metric("관측 기간", f"{int(overview['기간(일)']) + 1}일", border=True)
    st.metric("학습 / 평가", f"{int(split.loc['train', '건수']) / 1e4:,.0f}만 / {int(split.loc['test', '건수']) / 1e4:,.0f}만", border=True)

st.caption(f"접수 기간 {overview['접수 시작'][:10]} ~ {overview['접수 끝'][:10]} · 출처 NYC Open Data (311 Service Requests) · 전체 민원 유형")

st.subheader("학습 지도: 3부 15단계", anchor=False)
roadmap = pd.DataFrame([
    ("1부 데이터 알기", "1~4", "표 점검 → 숫자 요약 → 분포·이상치 → 패턴·비교", "평균 vs 중앙값, 왜도, IQR, 상관 ≠ 인과"),
    ("2부 머신러닝 기본기", "5~12", "문제 정의 → 기준선 → 모델 3종 → 평가 → 해석 → 튜닝", "누수, 기준선, 과적합, AUC, 교차검증"),
    ("3부 응용", "13~15", "처리 시간 회귀 → 내일 건수 예측 → 동네 군집", "회귀 지표, 지연 특징, K-means"),
], columns=["부", "단계", "흐름", "핵심 개념"])
show_table(roadmap, alt="학습 지도")

st.subheader("모든 머신러닝 단계는 같은 7단계 틀로 진행합니다", anchor=False)
with st.container(horizontal=True):
    for i, s in enumerate(["질문", "데이터 준비", "기준선", "모델", "평가", "해석", "한계"], start=1):
        st.metric(f"{i}단계", s, border=True)

meaning(
    "같은 틀을 반복하면 머신러닝 절차가 몸에 익습니다. 특히 **3단계 기준선**을 꼭 먼저 봅니다. "
    "'모델 없이 찍으면 몇 점인가'를 알아야 모델이 정말 잘했는지 판단할 수 있습니다."
)

with st.expander(":material/info: 기존 대시보드와 무엇이 다른가요?"):
    st.markdown(
        "- 기존 앱(2024년 4~9월, 13단계)은 통계 개념을 넓게 다룹니다. 이 앱은 **한 단계에 새 개념 3개 이하**로 줄이고 "
        "**머신러닝 절차**(기준선, 과적합, 교차검증, 누수)를 따로 배웁니다.\n"
        "- 데이터가 **12개월 전체**라서 겨울 난방 민원처럼 계절에 따른 차이도 볼 수 있습니다.\n"
        "- 각 페이지 머리말에 '기존 앱 참고 탭'을 적어 두었습니다. 개념을 더 깊이 보고 싶으면 그 탭을 함께 보세요."
    )

with st.expander(":material/terminal: 직접 실행하려면"):
    st.code("py -3.12 analysis/run_all.py\npy -3.12 -m streamlit run dashboard/app.py", language="bash")
    st.caption("분석 스크립트를 다시 실행하면 이 대시보드의 숫자와 그림이 자동으로 바뀝니다 (seed 고정이라 같은 결과).")
