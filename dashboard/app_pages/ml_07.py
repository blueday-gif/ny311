import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 7단계", "여러 정보를 함께 쓰면 민원 유형 하나만 쓸 때보다 더 잘 맞힐까? 무엇이 '24시간 안 처리' 쪽으로 미나?",
       "로지스틱 회귀, 확률 예측, 혼동행렬, 오즈비", "problem, borough, channel, location_type, season, hour, weekday",
       "ml_07_logistic.py", "6 검정·회귀(로지스틱 회귀)")

acc = table("ml07_accuracy").iloc[0]
base = table("ml06_baselines")
cm = table("ml07_confusion_matrix", index_col=0)
odds = table("ml07_odds_ratio")

term("로지스틱 회귀", "입력마다 점수(가중치)를 매겨 더한 뒤, 그 합을 0~1 확률로 바꾸는 모델. 확률이 0.5 이상이면 1로 예측합니다.  \n"
     "예) 점수 = (불법주차면 +2) + (주말이면 −0.3) + … → 확률 0.85 → '24시간 안 처리될 것'")
st.caption("위도·경도는 뺐습니다. 로지스틱 회귀는 '북쪽일수록 빠르다' 같은 직선 관계만 배워서 위치를 잘 다루지 못합니다. "
           "시간·요일도 직선이 아니라서(23시 다음은 0시) 범주로 바꿔 넣었습니다.")

with st.container(horizontal=True):
    st.metric("학습 정확도", f"{acc['학습 정확도']:.3f}", border=True)
    st.metric("평가 정확도", f"{acc['평가 정확도']:.3f}", round((acc["평가 정확도"] - base.loc[1, "평가 정확도"]) * 100, 1),
              delta_description="%p, 민원 유형 규칙 대비", border=True)

meaning("학습과 평가 점수가 비슷합니다 = 외우지 않고 일반적인 규칙을 배웠다는 뜻입니다. "
        "민원 유형 규칙보다 약 1.4%p 더 맞혔습니다. 여러 정보를 함께 쓴 효과지만, 큰 폭은 아닙니다.")

term("혼동행렬", "예측과 실제를 2×2 표로 나눠 센 것. 어떤 종류의 실수를 하는지 보여 줍니다.  \n"
     "실제 1인데 0으로 예측 = **놓친 것**, 실제 0인데 1로 예측 = **헛짚은 것**.")
c1, c2 = st.columns([2, 3])
with c1:
    figure("ml07_confusion_matrix", alt="로지스틱 회귀 혼동행렬")
with c2:
    fp = int(cm.iloc[0, 1])
    fn = int(cm.iloc[1, 0])
    meaning(f"헛짚기(실제로는 오래 걸렸는데 '빨리 끝난다'): {fp:,}건  \n"
            f"놓치기(빨리 끝났는데 '오래 걸린다'): {fn:,}건  \n\n"
            f"이 모델은 '놓치는' 실수를 약 {fn / fp:.0f}배 더 많이 합니다. 어느 실수가 더 곤란한지는 쓰임새에 따라 다릅니다(10단계).")

st.subheader("무엇이 '24시간 안 처리' 쪽으로 미나: 오즈비", anchor=False)
term("오즈와 오즈비", "**오즈** = 일어날 확률 ÷ 안 일어날 확률. 확률 0.8 → 오즈 4.  \n"
     "**오즈비** = exp(가중치). 그 조건이 켜질 때 오즈가 몇 배가 되는가. 오즈비 3 = 3배, 0.5 = 절반.")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown(":green[**빨리 처리되는 쪽 상위 10**]")
    show_table(odds.head(10), alt="오즈비 상위 10개 조건")
with c2, st.container(border=True):
    st.markdown(":red[**오래 걸리는 쪽 상위 10**]")
    show_table(odds.tail(10).iloc[::-1], alt="오즈비 하위 10개 조건")
meaning("양쪽 모두 민원 유형(problem_…)과 장소 유형(location_type_…)이 차지합니다. "
        "'Residential Building/House'(빠름), 'RESIDENTIAL BUILDING'(느림)처럼 **뜻이 같은 장소가 표기만 다르게** 따로 있습니다. "
        "접수 기관마다 쓰는 표기가 달라서, 장소 유형이 사실상 '담당 기관'을 알려 줍니다.")
caution("오즈비는 '연관'이지 '원인'이 아닙니다. 그 유형이라서 빨라진 것인지, 그 유형을 맡은 기관이 빨라서인지 모델은 구분하지 못합니다.")

next_step("8. 결정나무와 과적합")
