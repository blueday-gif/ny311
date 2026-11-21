import streamlit as st

from dash_common import figure, header, meaning, next_step, show_table, table, term

header("1부 데이터 알기 · 2단계", "처리 시간과 하루 접수 건수를 숫자 몇 개로 요약하면? 평균만 보면 왜 위험한가?",
       "평균, 중앙값, 표준편차, 사분위수, 절사평균", "hours (= closed − created), date", "eda_02_summary.py",
       "2 대푯값, 3 퍼짐·기술통계")

s = table("eda02_hours_summary").set_index("통계량")["값"]
daily = table("eda02_daily_summary", index_col=0).iloc[:, 0]

with st.container(horizontal=True):
    st.metric("처리 시간 중앙값", f"{s['중앙값']:.1f}시간", border=True)
    st.metric("처리 시간 평균", f"{s['평균']:.1f}시간", border=True)
    st.metric("평균 ÷ 중앙값", f"{s['평균 / 중앙값 (배)']:.0f}배", border=True)
    st.metric("하루 접수 (평균)", f"{daily['mean']:,.0f}건", border=True)

term("평균과 중앙값", "**평균**: 모두 더해 개수로 나눈 값. 아주 큰 값 하나에도 크게 움직입니다.  \n"
     "**중앙값**: 줄 세웠을 때 정가운데 값. 극단값에 거의 영향을 받지 않습니다.  \n"
     "예) 1, 2, 3, 4, 1000 → 평균 202, 중앙값 3.")
term("표준편차와 사분위수", "**표준편차**: 값들이 평균에서 얼마나 떨어져 있는지. 클수록 넓게 흩어져 있습니다.  \n"
     "**사분위수**: 줄 세워 4등분하는 경계. Q1(25%), Q2(50% = 중앙값), Q3(75%).  \n"
     f"예) Q1 = {s['Q1 (25%)']:.1f}시간 → '민원 4건 중 1건은 {s['Q1 (25%)']:.1f}시간 안에 끝났다'.")
term("절사평균", "위아래 극단값을 일정 비율(여기서는 각 5%) 잘라내고 구한 평균. 평균과 중앙값의 중간 성격입니다.")

c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("처리 시간 기술통계", anchor=False)
    show_table(s.reset_index(), alt="처리 시간 기술통계표")
with c2:
    st.subheader("평균과 중앙값이 멀리 떨어져 있다", anchor=False)
    figure("eda02_mean_vs_median", alt="처리 시간 히스토그램에 평균선과 중앙값선 표시")

meaning(f"평균({s['평균']:.0f}시간 ≈ {s['평균'] / 24:.1f}일)이 중앙값({s['중앙값']:.1f}시간)의 약 {s['평균 / 중앙값 (배)']:.0f}배입니다. "
        "보통 민원은 반나절 안에 끝나지만, 몇 주~몇 달 걸리는 소수의 민원이 평균을 크게 끌어올립니다. "
        "민원의 82.5%가 '평균'보다 빨리 끝나므로, 평균은 '보통 민원'을 대표하지 못합니다. 상위 1%만 빼도 평균은 "
        "159.6 → 124.1시간으로 줄지만 중앙값은 7.9 → 7.5시간으로 거의 그대로입니다.")

st.subheader("하루 접수 건수", anchor=False)
show_table(daily.round(1).rename("하루 접수 건수").reset_index().rename(columns={"index": "통계량"}),
           alt="하루 접수 건수 기술통계")
meaning(f"하루 접수 건수는 평균({daily['mean']:,.0f})과 중앙값({daily['50%']:,.0f})이 거의 같아 치우침이 작습니다. "
        f"다만 최댓값({daily['max']:,.0f}건)은 평소의 2배가 넘습니다. 특별한 날(폭설 등)일 수 있습니다.")

with st.expander(":material/filter_alt: 제외한 데이터와 기준"):
    show_table(table("eda02_exclusion_log"), alt="처리 시간 계산에서 제외한 기준과 건수")
    show_table(table("eda02_daily_exclusion_log"), alt="일별 집계에서 제외한 기준과 건수")

next_step("3. 분포와 이상치: 로그 변환이 필요한가")
