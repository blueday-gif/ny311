import altair as alt
import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("3부 응용 · 15단계", "정답 없이, 민원 구성이 비슷한 동네끼리 묶으면 몇 가지 유형이 나올까?",
       "비지도학습, K-means, 표준화, 엘보 방법, 실루엣 점수", "lat, lon (약 1km 격자), problem (동네 프로필)",
       "ml_15_clustering.py", "12 군집·PCA")

ks = table("ml15_k_selection")
# 저장된 표는 행 = 무리, 열 = 민원 유형. 화면에서는 행 = 민원 유형, 열 = 무리로 뒤집어 본다
prof = table("ml15_cluster_profile", index_col=0).T
prof.columns = prof.columns.astype(str)

term("비지도학습", "정답(타깃) 없이 데이터 안의 구조를 찾는 방법. 5~14단계는 정답이 있는 '지도학습'이었습니다.")
term("K-means", "점들을 K개 무리로 나눕니다. ① 중심 K개를 놓고 ② 각 점을 가장 가까운 중심에 붙이고 ③ 중심을 무리의 평균으로 옮기기를 반복합니다.")
term("표준화", "각 열을 '평균 0, 표준편차 1'로 맞춥니다. 비율이 큰 유형(불법주차)이 거리 계산을 독차지하지 않게 합니다.")
term("엘보 방법과 실루엣 점수", "**엘보 방법**: K를 늘려 가며 '무리 안 거리 합'을 그리고, 줄어드는 속도가 꺾이는 지점을 봅니다.  \n"
     "**실루엣 점수**: 점이 자기 무리와는 가깝고 다른 무리와는 먼가를 −1~1로 잰 것. 1에 가까울수록 잘 나뉘었습니다.")

st.caption("동네 = 약 1km 격자 칸. 민원 300건 이상인 칸만 쓰고, 각 칸을 '상위 10개 민원 유형 + 기타'의 비율로 나타냈습니다.")

st.subheader("무리 개수 K 고르기", anchor=False)
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**엘보 방법**")
    st.line_chart(ks, x="K", y="무리 안 거리 합(inertia)", alt="K에 따른 무리 안 거리 합")
with c2, st.container(border=True):
    st.markdown("**실루엣 점수**")
    st.line_chart(ks, x="K", y="실루엣 점수", alt="K에 따른 실루엣 점수")
best_k = int(ks.loc[ks["실루엣 점수"].idxmax(), "K"])
meaning(f"실루엣 점수는 K={best_k}에서 {ks['실루엣 점수'].max():.3f}로 가장 높고, K가 늘수록 낮아집니다. "
        "0.3 정도면 '무리가 아주 뚜렷하지는 않다'는 뜻입니다. K=2 는 '크게 두 종류'라는 거친 구분입니다. "
        "더 세밀한 유형이 필요하면 점수를 조금 양보하고 K=3~4 를 고를 수도 있습니다(사람의 판단).")

st.subheader(f"K = {best_k} 군집 결과", anchor=False)
c1, c2 = st.columns([3, 2])
with c1:
    long = (prof.drop(index="격자 칸 수").reset_index(names="민원 유형")
            .melt(id_vars="민원 유형", var_name="무리", value_name="비율(%)"))
    long["무리"] = "무리 " + long["무리"].astype(str)
    chart = alt.Chart(long[long["민원 유형"] != "기타"]).mark_bar().encode(
        x=alt.X("비율(%):Q"), y=alt.Y("민원 유형:N", title=None), color=alt.Color("무리:N", legend=alt.Legend(orient="bottom")),
        yOffset="무리:N", tooltip=["무리", "민원 유형", "비율(%)"])
    st.altair_chart(chart, alt="무리별 민원 유형 구성 비율 묶음 막대그래프")
    show_table(prof.reset_index(names="민원 유형"), alt="무리별 평균 민원 구성 비율과 격자 칸 수")
with c2:
    figure("ml15_cluster_map", alt="격자 칸을 무리 색으로 칠한 뉴욕 지도")

meaning(f"무리 0({int(prof.loc['격자 칸 수', '0'])}칸)은 난방·온수, 주거 소음, 비위생, 배관 비율이 높습니다 → **'아파트 주거 민원' 동네**.  \n"
        f"무리 1({int(prof.loc['격자 칸 수', '1'])}칸)은 불법주차, 방치 차량, 도로 상태 비율이 높습니다 → **'도로·차량 민원' 동네**.  \n"
        "위치 정보를 쓰지 않았는데도 같은 무리가 지도에서 뭉쳐 보인다면, 가까운 동네끼리 민원 구성이 비슷하다는 뜻입니다.")
caution("이름 붙이기는 사람의 해석입니다. 군집은 '비슷한 것끼리 묶었다'까지만 말해 줍니다.")

next_step("정리와 한계")
