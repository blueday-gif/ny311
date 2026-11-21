import altair as alt
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, table, term

header("2부 머신러닝 기본기 · 11단계", "모델이 가장 많이 기대는 정보는 무엇인가? 접수 시각이 바뀌면 예측 확률은 어떻게 변하나?",
       "순열 중요도, 부분의존도, 중요도 ≠ 인과", "ml_05 의 특징 9개 (그래디언트 부스팅)", "ml_11_interpret.py",
       "7 현대 방법(순열 중요도)")

imp = table("ml11_permutation_importance")

term("순열 중요도", "한 열의 값만 무작위로 뒤섞은 뒤 점수가 얼마나 떨어지는지 잽니다. 많이 떨어질수록 모델이 그 열에 많이 기댑니다.  \n"
     "예) 민원 유형을 뒤섞었더니 AUC 가 크게 떨어졌다면, 민원 유형은 매우 중요한 열입니다.")

st.subheader("순열 중요도: 열을 뒤섞으면 AUC 가 얼마나 떨어지나", anchor=False)
bars = alt.Chart(imp).mark_bar(color="#4C72B0").encode(
    x=alt.X("AUC 감소(평균):Q", title="AUC 감소량"), y=alt.Y("열:N", sort="-x", title=None),
    tooltip=["열", "AUC 감소(평균)", "AUC 감소(표준편차)"])
text = bars.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text("AUC 감소(평균):Q", format=".4f"))
st.altair_chart(bars + text, alt="열별 순열 중요도 막대그래프")

top = imp.iloc[0]
second = imp.iloc[1]
meaning(f"{top['열']}(민원 유형)을 뒤섞으면 AUC 가 {top['AUC 감소(평균)']:.3f} 떨어집니다. {second['열']}(장소 유형)은 {second['AUC 감소(평균)']:.3f}. "
        f"나머지 열은 모두 {imp.iloc[2]['AUC 감소(평균)']:.3f} 이하입니다. 모델은 거의 '민원 유형 + 장소 유형' 두 열로 판단합니다.")
caution("중요도는 '모델이 예측에 많이 쓴 정보'이지 '처리 속도를 바꾸는 원인'이 아닙니다. 장소 유형이 중요하게 나와도 "
        "장소를 바꾸면 처리가 빨라진다는 뜻이 아니라, 장소 유형이 담당 기관을 대신 알려 줄 뿐입니다. "
        "또 비슷한 정보를 담은 열끼리는 중요도가 나뉘어 각각 작게 보일 수 있습니다.")

st.subheader("부분의존도: 이 값만 바꾸면 확률이 어떻게 변하나", anchor=False)
term("부분의존도 (PDP)", "다른 정보는 그대로 두고 한 열의 값만 바꿔 가며 '평균 예측 확률'이 어떻게 변하는지 그린 선입니다.")
figure("ml11_partial_dependence", alt="시간대, 요일, 자치구의 부분의존도")
meaning("세 그림의 y축은 같은 범위(약 0.545~0.595)로 맞췄습니다. 폭이 5%p 정도뿐입니다.  \n"
        "- **요일**: 토요일(5) 접수가 가장 낮습니다. 주말 민원은 처리가 월요일로 밀리는 것으로 보입니다(폭 약 4.5%p).  \n"
        "- **시간대**: 밤·새벽 접수가 약 0.59, 오전 10시 무렵이 약 0.57(폭 약 3%p).  \n"
        "- **자치구**: 폭 약 1.6%p. 민원 유형을 알고 나면 자치구 차이는 작습니다. 4단계의 자치구별 중앙값 차이와 비교해 보세요.")

next_step("12. 교차검증과 튜닝")
