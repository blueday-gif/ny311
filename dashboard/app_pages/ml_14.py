import altair as alt
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("3부 응용 · 14단계", "지난 며칠의 접수 건수로 다음 날 접수 건수를 맞힐 수 있을까? 단순한 방법보다 나을까?",
       "지연 특징(lag), 이동평균, 단순 예측, 계절 단순 예측, MAPE", "created (일별 건수로 집계)", "ml_14_forecast.py",
       "9 시계열 (거기서는 ARIMA)")

fc = table("ml14_forecast_results")
coef = table("ml14_linear_coef")

term("지연 특징", "과거 값을 오늘의 입력으로 쓰는 것. lag1 = 어제 건수, lag7 = 일주일 전 같은 요일 건수.")
term("단순 예측과 계절 단순 예측", "**단순 예측**: '내일은 오늘과 같다'. 시계열 예측의 가장 기본 기준선입니다.  \n"
     "**계절 단순 예측**: '다음 월요일은 이번 월요일과 같다'처럼 주기만큼 전 값을 씁니다.")
term("MAPE", "|실제 − 예측| ÷ 실제의 평균을 %로. '평균적으로 몇 % 틀리나'. 예) 실제 10,000, 예측 9,500 → 5%.")
st.caption("이 단계는 '하루 앞' 예측입니다. 매일 실제 값이 들어오면 그 값을 써서 다음 날을 맞힙니다. "
           "학습: 2025-11-09 ~ 2026-07-31, 평가: 2026-08-01 ~ 2026-10-30.")

c1, c2 = st.columns([2, 3])
with c1:
    show_table(fc, alt="예측 방법별 MAE와 MAPE")
with c2:
    chart = alt.Chart(fc).mark_bar(color="#4C72B0").encode(
        x=alt.X("MAPE(%):Q"), y=alt.Y("방법:N", sort=list(fc["방법"]), title=None), tooltip=list(fc.columns))
    st.altair_chart(chart + chart.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text("MAPE(%):Q", format=".2f")),
                    alt="예측 방법별 MAPE 막대그래프")

best = fc.loc[fc["MAPE(%)"].idxmin()]
meaning(f"{best['방법']}가 MAPE {best['MAPE(%)']:.1f}%로 가장 정확합니다(하루 평균 약 {best['MAE(건)']:,.0f}건 오차). "
        "'어제 값'은 요일 패턴(월요일에 늘고 토요일에 줄어듦)을 무시해서 가장 많이 틀립니다. "
        "부스팅이 선형 회귀보다 못한 이유: 학습할 날이 265일뿐이고, 나무 모델은 학습 때 본 범위 밖의 값을 예측하지 못합니다. "
        "**자료가 적을 때는 단순한 모델이 더 강할 수 있습니다.**")

figure("ml14_forecast", alt="평가 기간의 실제 접수 건수와 예측 비교 선그래프")

with st.expander(":material/functions: 선형 회귀 가중치 보기"):
    show_table(coef, alt="선형 회귀 특징별 가중치")
    st.caption(f"lag1 {coef.loc[0, '가중치']:.2f} = 어제 건수가 100건 많으면 오늘 예측이 {coef.loc[0, '가중치'] * 100:.0f}건 많아집니다. "
               "wd_0(월요일)은 크게 양수, wd_5(토요일)는 크게 음수입니다.")
caution("갑자기 튀는 날(폭우, 폭염, 행사)은 과거 건수만으로는 맞힐 수 없습니다. 날씨·행사 정보를 더하면 개선될 수 있습니다.")

next_step("15. 동네 유형 묶기")
