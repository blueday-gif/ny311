import altair as alt
import streamlit as st

from dash_common import header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 12단계", "모델의 설정값은 어떻게 골라야 하나? 평가 데이터로 고르면 왜 안 되나?",
       "하이퍼파라미터, 교차검증, 시계열 교차검증, 그리드 탐색", "ml_05 의 특징 9개, created", "ml_12_tuning.py")

folds = table("ml12_cv_folds")
grid = table("ml12_grid_results")
final = table("ml12_final_test")

term("하이퍼파라미터", "모델이 데이터에서 스스로 배우지 않고, 사람이 미리 정해 주는 설정값. 예) 나무 깊이, 학습률, 나무 개수.")
term("교차검증", "학습 데이터를 여러 조각으로 나눠 돌아가며 한 조각은 검증용, 나머지는 학습용으로 씁니다. 점수를 여러 번 재서 평균을 냅니다.  \n"
     "**평가 데이터는 끝까지 손대지 않고 마지막 시험에만** 씁니다. 평가 데이터로 설정을 고르면 시험 문제를 보고 공부한 셈입니다.")
term("시계열 교차검증", "시간 순서를 지키는 교차검증. 항상 '앞 기간으로 학습 → 바로 뒤 기간으로 검증'합니다.")

st.subheader("시계열 교차검증의 조각", anchor=False)
show_table(folds, alt="시계열 교차검증 조각별 학습·검증 기간")
meaning("조각이 뒤로 갈수록 학습 기간이 길어지고, 검증 기간은 항상 학습 기간 바로 다음입니다. 실제 운영과 같은 순서입니다.")

st.subheader("설정 9가지 × 조각 4개 = 학습 36번", anchor=False)
c1, c2 = st.columns([3, 2])
with c1:
    heat = alt.Chart(grid).mark_rect().encode(
        x=alt.X("최대 잎 개수:O", title="최대 잎 개수 (나무 크기)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("학습률:O"),
        color=alt.Color("검증 AUC 평균:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["학습률", "최대 잎 개수", "검증 AUC 평균", "검증 AUC 표준편차", "순위"]).properties(height=280)
    labels = heat.mark_text(fontSize=13).encode(text=alt.Text("검증 AUC 평균:Q", format=".4f"), color=alt.value("black"))
    st.altair_chart(heat + labels, alt="학습률과 잎 개수 조합별 교차검증 AUC 히트맵")
with c2:
    show_table(grid, alt="그리드 탐색 결과표")

spread = grid["검증 AUC 평균"].max() - grid["검증 AUC 평균"].min()
meaning(f"1등과 꼴찌의 차이는 {spread:.4f}뿐인데, 조각마다 흔들리는 폭(표준편차)은 약 {grid['검증 AUC 표준편차'].mean():.3f}로 "
        f"{grid['검증 AUC 표준편차'].mean() / spread:.0f}배 큽니다. 즉 이 순위는 우연일 가능성이 큽니다. "
        "설정을 열심히 고르는 것보다 좋은 특징을 찾는 것이 더 중요합니다.")

st.subheader("마지막 시험: 기본 설정 vs 고른 설정", anchor=False)
with st.container(horizontal=True):
    st.metric("기본 설정 평가 AUC", f"{final.loc[0, '평가 AUC']:.4f}", border=True)
    st.metric("탐색 결과 평가 AUC", f"{final.loc[1, '평가 AUC']:.4f}",
              round(final.loc[1, "평가 AUC"] - final.loc[0, "평가 AUC"], 4), delta_description="기본 설정 대비", border=True)
st.caption(f"탐색 결과: {final.loc[1, '모델']}")
meaning("두 점수는 사실상 같습니다. '기본 설정도 충분히 좋다'는 결론입니다. 교차검증 점수와 평가 점수가 다른 것은 정상입니다. "
        "검증 기간(학습 기간 안쪽)과 평가 기간(8~10월)이 다르기 때문입니다.")

next_step("13. 처리 시간 예측: 몇 시간 걸릴까")
