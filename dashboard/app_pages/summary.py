import streamlit as st

from dash_common import show_table, table

findings = table("summary_findings")
limits = table("summary_limitations")
mapping = table("summary_dashboard_mapping")

st.subheader("발견 5가지", anchor=False)
for _, r in findings.iterrows():
    with st.container(border=True):
        st.markdown(f"**{r['번호']}.** {r['발견']}")

st.subheader("한계", anchor=False)
for _, r in limits.iterrows():
    st.warning(r["내용"], title=r["한계"], icon=":material/warning:")

st.subheader("스크립트 ↔ 대시보드 연결표", anchor=False)
show_table(mapping, alt="분석 스크립트, 기존 앱 탭, 새 대시보드 탭 연결표")

st.subheader("전체를 다시 실행하려면", anchor=False)
st.code("py -3.12 analysis/run_all.py\npy -3.12 -m streamlit run dashboard/app.py", language="bash")
st.caption("run_all.py 는 16개 스크립트를 번호 순서대로 실행합니다(약 6분). seed 가 고정되어 있어 다시 실행해도 같은 결과가 나옵니다.")
