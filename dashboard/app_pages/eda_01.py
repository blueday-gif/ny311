import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("1부 데이터 알기 · 1단계", "이 표에는 무엇이 들어 있고, 믿고 써도 되는 상태인가?",
       "행과 열, 자료형, 결측률, 관측 기간, 중복 행", "44개 열 전체, created, unique_key", "eda_01_check.py",
       "0 시작하기")

ov = table("eda01_overview").set_index("항목")["값"]
missing = table("eda01_missing")

with st.container(horizontal=True):
    st.metric("행 (민원 1건 = 1행)", f"{int(ov['행 수']):,}", border=True)
    st.metric("열", f"{int(ov['열 수'])}", border=True)
    st.metric("민원 번호 중복", f"{int(ov['민원 번호 중복']):,}건", border=True)
    st.metric("결측 90% 초과 열", f"{int(ov['결측률 90% 초과 열 수'])}개", border=True)

term("결측치", "값이 비어 있는 칸. **왜 비었는지**가 중요합니다.  \n"
     "예) 종결 시각이 비어 있으면 '아직 처리 중'이라는 뜻일 수 있습니다. 실수로 빠진 값과는 다릅니다.")

st.subheader("열별 결측률", anchor=False)
c1, c2 = st.columns([3, 2])
with c1:
    show_table(missing, alt="열별 결측 건수와 결측률", height=420,
               column_config={"결측률(%)": st.column_config.ProgressColumn("결측률(%)", min_value=0, max_value=100,
                                                                         format="%.1f%%")})
with c2:
    figure("eda01_missing_bar", alt="열별 결측률 상위 25개 막대그래프")

meaning(f"44개 열 중 {int((missing['결측률(%)'] == 0).sum())}개는 비어 있는 칸이 없고, "
        f"{int((missing['결측률(%)'] > 90).sum())}개는 90% 넘게 비어 있습니다. 택시·다리·도로 관련 칸처럼 "
        "특정 민원에만 쓰는 열이라 일반 분석에서는 뺍니다. 종결 시각(Closed Date)의 결측 4%는 '아직 처리 중'이라는 정보입니다.")

st.subheader("관측 기간", anchor=False)
st.markdown(f"**{ov['접수 시작']}** ~ **{ov['접수 끝']}**")
caution("파일 이름은 '2020년부터'지만 실제로는 최근 약 1년만 들어 있습니다. 연도별 추세는 볼 수 없습니다. "
        "또 첫날은 21시부터 시작해서 '부분 일자'입니다. 일별 건수를 셀 때 첫날·마지막 날을 뺍니다.")

st.subheader("중복", anchor=False)
log = table("eda01_exclusion_log")
show_table(log, alt="중복 신고 후보 기록")
meaning(f"민원 번호는 하나도 겹치지 않습니다. 같은 시각·같은 유형·같은 위치의 민원은 {int(ov['같은 시각·유형·위치 중복']):,}건"
        f"({log['시작 행 대비 비율(%)'].iloc[0]:.1f}%)인데, 여러 사람이 같은 일을 신고한 것일 수 있어 **지우지 않고 기록만** 했습니다.")

next_step("2. 숫자로 요약: 평균만 보면 왜 위험한가")
