import altair as alt
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("3부 응용 · 13단계", "'24시간 안인가'가 아니라 '몇 시간 걸릴까'를 맞힐 수 있을까?",
       "회귀, MAE, 중앙절대오차, R², 로그 타깃, 중도절단", "hours, ml_05 의 특징 9개", "ml_13_regression.py",
       "6 검정·회귀, 7 현대 방법(분위수 회귀), 10 생존분석")

reg = table("ml13_regression_results")

term("회귀", "숫자 값(처리 시간)을 맞히는 문제. 분류(0/1 맞히기)와 짝을 이루는 머신러닝의 두 기본 문제 중 하나입니다.")
term("MAE, 중앙절대오차, R²", "**MAE**: |실제 − 예측|의 평균. '평균적으로 몇 시간 틀리나'.  \n"
     "**중앙절대오차**: |실제 − 예측|의 중앙값. '보통 민원에서 몇 시간 틀리나'. 극단값에 덜 흔들립니다.  \n"
     "**R²**: 예측이 '평균값 하나로 찍기'보다 얼마나 나은지. 1 = 완벽, 0 = 평균 찍기, 음수 = 그보다 못함.")
term("로그 타깃", "처리 시간은 왜도가 6 이상입니다(3단계). 그대로 맞히면 수천 시간짜리 소수 민원에 모델이 끌려갑니다. "
     "log1p(처리 시간)을 맞힌 뒤 되돌려서 시간 단위로 평가합니다.")

metric = st.segmented_control("지표", ["중앙절대오차(시간)", "MAE(시간)", "R²(로그 기준)"], default="중앙절대오차(시간)")
chart = alt.Chart(reg).mark_bar(color="#4C72B0").encode(
    x=alt.X(f"{metric}:Q"), y=alt.Y("방법:N", sort=list(reg["방법"]), title=None), tooltip=list(reg.columns))
text = chart.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text(f"{metric}:Q", format=".2f"))
st.altair_chart(chart + text, alt=f"방법별 {metric} 막대그래프")
show_table(reg, alt="회귀 방법별 MAE, 중앙절대오차, R²")

meaning(f"MAE 는 {reg['MAE(시간)'].min():.0f}~{reg['MAE(시간)'].max():.0f}시간으로 큽니다. 몇 달 걸린 소수 민원을 크게 틀리기 때문입니다. "
        f"중앙절대오차는 전체 중앙값 {reg.iloc[0]['중앙절대오차(시간)']:.1f}시간 → 유형별 중앙값 {reg.iloc[1]['중앙절대오차(시간)']:.1f} → "
        f"부스팅 {reg.iloc[3]['중앙절대오차(시간)']:.1f}시간으로 줄어듭니다. "
        "**지표마다 1등이 다를 수 있습니다**(MAE 는 선형 회귀, 중앙절대오차와 R² 는 부스팅). "
        "이번에도 '민원 유형'이 대부분을 설명하고, 모델은 그 위에 조금 더합니다.")

c1, c2 = st.columns([2, 3])
with c1:
    figure("ml13_actual_vs_pred", alt="그래디언트 부스팅 실제 대 예측 산점도")
with c2:
    meaning("점들이 가로 줄무늬처럼 몰려 있다면, 모델이 '민원 유형별 대표 시간' 몇 개만 말하고 있다는 뜻입니다. "
            "같은 유형 안에서 처리 시간이 크게 갈리는 이유(담당자, 현장 상황)는 이 데이터에 없어서 맞히기 어렵습니다.")
    term("중도절단", "'아직 끝나지 않아서 최종 값을 모르는' 관측. 예) 30일째 처리 중인 민원은 '최소 30일'이라는 것만 압니다.")
    caution("미종결 민원(약 4%)을 뺐습니다. 오래 걸리는 민원이 덜 들어가서 처리 시간을 짧게 예측하는 쪽으로 치우칩니다. "
            "제대로 다루려면 기존 앱 10단계의 생존분석이 필요합니다.")

next_step("14. 내일 건수 예측")
