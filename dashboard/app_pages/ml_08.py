import altair as alt
import streamlit as st

from dash_common import figure, header, meaning, next_step, table, term

header("2부 머신러닝 기본기 · 8단계", "사람이 읽을 수 있는 '만약 ~라면' 규칙으로 맞힐 수 있을까? 나무를 깊게 키우면 무조건 좋아질까?",
       "결정나무, 나무 깊이, 과적합, 과소적합", "problem, borough, channel, location_type, season, hour, weekday, lat, lon",
       "ml_08_tree.py")

curve = table("ml08_depth_curve")
curve["설정 깊이"] = curve["설정 깊이"].astype(str)
best = curve.loc[curve["평가 정확도"].idxmax()]
last = curve.iloc[-1]

term("결정나무", "'민원 유형이 불법주차인가? → 예 → 접수 시각이 22시 이후인가? → …'처럼 예/아니오 질문을 이어 붙여 답을 내는 모델.  \n"
     "그림으로 그리면 사람이 그대로 읽을 수 있다는 것이 큰 장점입니다.")

st.subheader("깊이 3 나무: 질문 3번으로 만든 규칙", anchor=False)
figure("ml08_tree_depth3", alt="깊이 3 결정나무 그림")
with st.expander(":material/description: 규칙을 글자로 보기"):
    st.code("\n".join(table("ml08_tree_depth3_rules")["규칙"]), language=None)
meaning(f"맨 처음 질문이 가장 중요한 갈림길입니다. 이 나무는 '장소 유형'을 먼저 묻습니다(7단계에서 본 것처럼 담당 기관을 알려 주기 때문). "
        f"'location_type_Street/Sidewalk <= 0.50'은 '장소 유형이 Street/Sidewalk가 아니면'이라는 뜻입니다. "
        f"질문 3번만으로 정확도 {curve.loc[curve['설정 깊이'] == '3', '평가 정확도'].iloc[0]:.3f}지만, 민원 유형 규칙(0.882)보다는 낮습니다.")

st.subheader("깊어질수록 무슨 일이 생기나: 과적합", anchor=False)
term("과적합과 과소적합", "**과적합**: 학습 데이터를 너무 꼼꼼히 '외워서' 처음 보는 데이터에서는 오히려 못 맞히는 상태. "
     "기출문제 답만 외운 학생과 같습니다.  \n**과소적합**: 너무 단순해서 학습 데이터조차 못 맞히는 상태.")

c1, c2 = st.columns([3, 2])
with c1, st.container(border=True):
    long = curve.melt(id_vars="설정 깊이", value_vars=["학습 정확도", "평가 정확도"], var_name="데이터", value_name="정확도")
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("설정 깊이:N", sort=list(curve["설정 깊이"]), title="나무 최대 깊이", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("정확도:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("데이터:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["설정 깊이", "데이터", "정확도"],
    )
    st.altair_chart(chart, alt="나무 깊이에 따른 학습·평가 정확도")
with c2:
    st.metric(f"가장 좋은 깊이 ({best['설정 깊이']})의 평가 정확도", f"{best['평가 정확도']:.3f}", border=True)
    st.metric("깊이 제한 없음의 평가 정확도", f"{last['평가 정확도']:.3f}",
              round(last["평가 정확도"] - best["평가 정확도"], 3), delta_description="가장 좋은 깊이 대비", border=True)
    st.metric("깊이 제한 없음의 학습 정확도", f"{last['학습 정확도']:.3f}", border=True)
    st.metric("깊이 제한 없음 잎 개수 (= 규칙 수)", f"{int(last['잎 개수']):,}개", border=True)

meaning(f"깊이 1~2는 학습·평가 모두 낮습니다(과소적합). 깊이 {best['설정 깊이']}까지는 둘 다 오르다가 평가 {best['평가 정확도']:.3f}에서 정점을 찍습니다. "
        f"그 뒤로는 학습 정확도만 계속 올라 {last['학습 정확도']:.3f}이 되고, 평가 정확도는 {last['평가 정확도']:.3f}로 떨어집니다. "
        f"제한 없는 나무는 잎이 {int(last['잎 개수']):,}개 = 학습 데이터를 거의 통째로 외운 것입니다.")

next_step("9. 모델 비교: 나무를 여러 그루 쓰면")
