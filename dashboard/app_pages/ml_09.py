import altair as alt
import streamlit as st

from dash_common import header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 9단계", "나무 하나 대신 여러 그루를 합치면 더 잘 맞힐까? 지금까지 만든 모델 중 무엇이 가장 나은가?",
       "앙상블, 랜덤포레스트, 그래디언트 부스팅, 모델 비교표", "ml_05 의 특징 9개", "ml_09_ensemble.py",
       "7 현대 방법(그래디언트 부스팅)")

comp = table("ml09_model_comparison")

term("앙상블", "여러 모델의 답을 모아 최종 답을 정하는 방법. 한 사람보다 여러 사람의 의견을 모으면 덜 틀리는 것과 같습니다.")
term("랜덤포레스트", "데이터를 조금씩 다르게 뽑아 깊은 나무를 수백 그루 키우고 다수결로 답합니다. 나무 하나의 과적합이 서로 상쇄됩니다.")
term("그래디언트 부스팅", "얕은 나무를 순서대로 키우되, 다음 나무는 '앞 나무들이 틀린 부분'을 고치도록 배웁니다. 오답 노트를 계속 보완하는 방식.")
term("AUC", "'1인 민원'과 '0인 민원'을 하나씩 뽑았을 때, 모델이 1인 쪽에 더 높은 확률을 줄 가능성. 0.5 = 찍기, 1.0 = 완벽.  \n"
     "정확도와 달리 0.5 같은 기준값 없이 '순서를 얼마나 잘 매기나'를 봅니다.")

metric = st.segmented_control("비교 지표", ["평가 정확도", "평가 AUC"], default="평가 정확도")
data = comp.dropna(subset=[metric]).copy()
data["구분"] = data["모델"].str.contains("기준선").map({True: "기준선", False: "모델"})
# clip=True: x축을 0.5부터 시작하게 했으므로, 0부터 그려지는 막대가 축 밖(모델 이름 위)으로 넘치지 않게 자른다
chart = alt.Chart(data).mark_bar(clip=True).encode(
    x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0.5, 1.0]), title=metric),
    y=alt.Y("모델:N", sort=list(data["모델"]), title=None, axis=alt.Axis(labelLimit=260)),
    color=alt.Color("구분:N", scale=alt.Scale(domain=["기준선", "모델"], range=["#BBBBBB", "#4C72B0"]),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=["모델", metric],
) + alt.Chart(data).mark_text(align="left", dx=4).encode(
    x=f"{metric}:Q", y=alt.Y("모델:N", sort=list(data["모델"])), text=alt.Text(f"{metric}:Q", format=".3f"))
st.altair_chart(chart, alt=f"모델별 {metric} 막대그래프")

show_table(comp, alt="모델 비교표: 학습·평가 정확도, AUC, 학습 시간")

best = comp.dropna(subset=["평가 AUC"]).iloc[1:]
top_auc = best.loc[best["평가 AUC"].idxmax()]
rf = comp[comp["모델"].str.contains("랜덤")].iloc[0]
meaning(f"모든 모델이 다수결(0.58)보다 크게 좋고 민원 유형 규칙(0.88)보다 조금 좋습니다. AUC는 {top_auc['모델']}가 {top_auc['평가 AUC']:.3f}로 가장 높습니다. "
        f"랜덤포레스트는 학습 {rf['학습 정확도']:.3f} / 평가 {rf['평가 정확도']:.3f}로 차이가 가장 커서 약간 외웠습니다. "
        "하지만 모델 사이 차이는 0.4%p 안쪽입니다. '복잡한 모델이 항상 훨씬 좋다'가 아니라 "
        "'**핵심 정보(민원 유형)가 대부분을 설명한다**'가 이 데이터의 교훈입니다. 이럴 때는 해석하기 쉬운 로지스틱 회귀도 좋은 선택입니다.")

next_step("10. 평가 지표: 정확도만 보면 왜 위험한가")
