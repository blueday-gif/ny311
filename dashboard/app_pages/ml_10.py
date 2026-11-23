import altair as alt
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 10단계", "'좋은 모델'을 정확도 하나로 말해도 되나? 기준값 0.5 를 바꾸면 무엇이 달라지나?",
       "정밀도, 재현율, F1, ROC 곡선, 기준값, 보정, 클래스 불균형", "ml_05 의 특징 9개 (그래디언트 부스팅)",
       "ml_10_evaluation.py", "7 현대 방법(AUC, 보정)")

thr = table("ml10_threshold_table")
summ = table("ml10_summary").set_index("항목")["값"]

st.info("이 단계는 **'오래 걸릴 민원(24시간 초과)'을 찾는 것**을 목표로 봅니다. 담당자가 미리 챙겨야 할 민원을 골라내는 상황입니다.",
        icon=":material/target:")

term("정밀도와 재현율", "**정밀도**: '오래 걸린다'고 찍은 것 중 실제로 오래 걸린 비율. 헛짚기를 얼마나 안 하나.  \n"
     "**재현율**: 실제로 오래 걸린 민원 중 모델이 찾아낸 비율. 놓치기를 얼마나 안 하나.  \n"
     "**F1**: 둘의 조화평균. 하나만 높고 다른 하나가 낮으면 F1 이 낮아집니다.")
term("기준값", "확률이 몇 이상이면 '양성'으로 찍을지 정하는 선. 기본은 0.5 지만 쓰임새에 따라 바꿉니다.")

st.subheader("기준값을 바꿔 보세요", anchor=False)
t = st.select_slider("기준값 (이 확률 이상이면 '오래 걸린다'로 찍음)", options=list(thr["기준값"]), value=0.5)
row = thr[thr["기준값"] == t].iloc[0]
with st.container(horizontal=True):
    st.metric("찍은 건수", f"{int(row['찍은 건수']):,}", border=True)
    st.metric("정밀도", f"{row['정밀도']:.3f}", border=True)
    st.metric("재현율", f"{row['재현율']:.3f}", border=True)
    st.metric("F1", f"{row['F1']:.3f}", border=True)
    st.metric("정확도", f"{row['정확도']:.3f}", border=True)

long = thr.melt(id_vars="기준값", value_vars=["정밀도", "재현율", "F1"], var_name="지표", value_name="값")
lines = alt.Chart(long).mark_line(point=True).encode(
    x=alt.X("기준값:Q"), y=alt.Y("값:Q", scale=alt.Scale(zero=False)), color=alt.Color("지표:N", legend=alt.Legend(orient="bottom")),
    tooltip=["기준값", "지표", "값"])
rule = alt.Chart().mark_rule(strokeDash=[4, 4], color="gray").encode(x=alt.datum(t))
st.altair_chart(lines + rule, alt="기준값에 따른 정밀도, 재현율, F1 변화")
meaning("기준값을 낮추면 더 많이 찍어서 재현율이 오르고(놓치는 민원↓) 정밀도가 내려갑니다(헛짚기↑). 올리면 반대입니다. "
        f"정확도와 F1 은 기준값 {thr.loc[thr['F1'].idxmax(), '기준값']} 근처에서 가장 높습니다. 기본값 0.5 가 항상 최선은 아닙니다. "
        "**어느 기준이 정답인지는 데이터가 아니라 쓰임새(업무 비용)가 정합니다.**")

st.subheader("ROC · 정밀도-재현율 · 보정 곡선", anchor=False)
term("ROC 곡선과 보정", "**ROC 곡선**: 기준값을 1→0으로 내리며 (헛짚은 비율, 찾아낸 비율)을 이은 선. 왼쪽 위로 붙을수록 좋고, 아래 넓이가 AUC.  \n"
     "**보정**: 모델이 '확률 0.7'이라고 한 민원들 중 실제로 70%가 그렇게 되는가. 대각선에 붙을수록 확률을 믿을 수 있습니다.")
figure("ml10_roc_pr_calibration", alt="ROC 곡선, 정밀도-재현율 곡선, 보정 곡선")
meaning(f"AUC {summ['AUC (24시간 초과 찾기)']:.3f}: 오래 걸린 민원 하나와 빨리 끝난 민원 하나를 뽑으면, "
        f"{summ['AUC (24시간 초과 찾기)'] * 100:.0f}%의 확률로 오래 걸린 쪽에 더 높은 위험도를 줍니다.")

st.subheader("클래스 불균형: 정답이 쏠리면 정확도는 속인다", anchor=False)
term("클래스 불균형", "정답 중 한쪽이 훨씬 많은 상태. 예) 1000건 중 10건만 양성. 이때 '항상 음성'이라고 하면 정확도 99%지만 양성을 하나도 못 찾습니다.")
acc_all0 = summ["'항상 30일 이내' 정확도"]
with st.container(horizontal=True):
    st.metric("30일 넘게 걸린 민원 비율", f"{summ['30일 초과 비율(평가)'] * 100:.1f}%", border=True)
    st.metric("'항상 30일 이내' 정확도", f"{acc_all0:.3f}", border=True)
    st.metric("그때 재현율", "0.000", border=True)
caution(f"아무것도 배우지 않은 '항상 30일 이내'가 정확도 {acc_all0:.3f}를 받습니다. 정확도가 높아 보여도 찾아내는 것이 하나도 없어 쓸모는 0입니다. 이 값은 다른 목표(24시간)의 모델 정확도와 직접 비교하지 않습니다. "
        "드문 사건을 찾는 문제에서는 정확도 대신 재현율·정밀도·AUC 를 봐야 합니다.")

next_step("11. 모델 해석: 모델은 무엇을 보고 판단했나")
