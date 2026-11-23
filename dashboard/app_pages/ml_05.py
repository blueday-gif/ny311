import streamlit as st

from dash_common import caution, figure, header, meaning, next_step, show_table, table, term

header("2부 머신러닝 기본기 · 5단계", "민원이 접수되는 순간, 그 민원이 24시간 안에 처리될지 맞힐 수 있을까?",
       "타깃(정답), 특징(입력), 데이터 누수, 학습/평가 기간 나누기",
       "created, closed, problem, borough, channel, location_type, lat, lon", "ml_05_problem.py", "6 검정·회귀(④)")

split = table("ml05_split", index_col=0)
leak = table("ml05_leakage_demo")

term("머신러닝", "정답이 있는 과거 사례를 많이 보여 주고, 컴퓨터가 '입력 → 정답' 규칙을 스스로 찾게 하는 방법.  \n"
     "예) 과거 민원 수십만 건의 (유형, 자치구, 시간) → (24시간 안 처리 여부)를 보고 새 민원의 결과를 맞힙니다.")
term("타깃과 특징", "**타깃(정답)** = 맞히고 싶은 값. 여기서는 `within_24h` (1 = 24시간 안 종결, 0 = 아님).  \n"
     "**특징(입력)** = 맞힐 때 참고하는 값. 민원 유형, 자치구, 접수 시각 등.")

st.subheader("학습 기간과 평가 기간", anchor=False)
with st.container(horizontal=True):
    st.metric("학습 (2025-11 ~ 2026-07)", f"{int(split.loc['train', '건수']):,}건", border=True)
    st.metric("학습 24시간 처리율", f"{split.loc['train', '처리율_24h']:.1f}%", border=True)
    st.metric("평가 (2026-08 ~ 2026-10)", f"{int(split.loc['test', '건수']):,}건", border=True)
    st.metric("평가 24시간 처리율", f"{split.loc['test', '처리율_24h']:.1f}%", border=True)
term("학습 데이터와 평가 데이터", "학습 데이터로 규칙을 배우고, **한 번도 보지 않은** 평가 데이터로 실력을 잽니다. 연습 문제와 시험 문제를 나누는 것.  \n"
     "시간 순서가 있는 자료는 '과거로 배우고 미래로 시험'해야 합니다. 무작위로 섞으면 미래 정보로 과거를 맞히는 셈이 됩니다.")
meaning("정답 1(24시간 안 처리)이 약 57%로, 1과 0이 반반에 가깝습니다. 두 기간의 처리율도 거의 같아서 "
        "평가 기간에 세상이 크게 바뀌지는 않았습니다.")

st.subheader("어떤 열을 특징으로 쓸 수 있나", anchor=False)
show_table(table("ml05_features"), alt="특징 후보 열과 접수 시점에 알 수 있는지 여부")
st.caption("민원 유형은 197개 중 학습 기간 상위 40개 + 기타, 장소 유형은 상위 20개 + 기타 + 없음으로 묶었습니다. "
           "상위 범주를 학습 기간에서만 고른 것도 누수를 막기 위해서입니다. 월 대신 계절을 쓴 이유: 평가 기간의 8~10월이 학습 기간에 없기 때문입니다.")

st.subheader("누수 실습: 답지를 몰래 넣으면", anchor=False)
term("데이터 누수", "실제로 예측하는 순간에는 알 수 없는 정보가 학습에 섞여 들어가는 것. 점수가 비정상적으로 좋아집니다.  \n"
     "예) '처리 결과가 갱신된 시각'은 처리 후에야 생깁니다. 이것을 넣으면 시험 문제와 답지를 같이 주는 셈입니다.")
c1, c2 = st.columns([2, 3])
with c1:
    with st.container(horizontal=True):
        st.metric("정상 특징만", f"{leak.loc[0, '평가 AUC']:.3f}", border=True)
        st.metric("+ 누수 특징", f"{leak.loc[1, '평가 AUC']:.3f}", border=True)
    st.caption("AUC: 0.5 = 찍기, 1.0 = 완벽 (10단계에서 자세히)")
with c2:
    figure("ml05_leakage_demo", alt="정상 특징과 누수 특징을 넣었을 때의 AUC 비교")
meaning(f"정상 특징만으로도 AUC {leak.loc[0, '평가 AUC']:.2f}로 이미 높습니다. 민원 유형이 '어느 기관이 어떻게 처리하는지'를 (가능성으로) "
        f"거의 정해 주기 때문입니다. 누수 특징을 넣으면 {leak.loc[1, '평가 AUC']:.2f}로 더 오르고, 틀리는 정도(1 − AUC)가 약 4분의 1로 줄어듭니다. "
        "실력이 좋아진 것이 아니라 답을 본 것입니다.")
caution("점수가 높다고 곧 누수는 아니고, 누수가 없다고 점수가 낮은 것도 아닙니다. 그래서 '접수 시점에 아는가'를 열마다 직접 점검합니다.")

with st.expander(":material/filter_alt: 제외한 데이터와 기준"):
    show_table(table("ml05_exclusion_log"), alt="머신러닝 데이터에서 제외한 기준과 건수")
    st.caption("관측 끝 24시간 안에 접수됐는데 아직 미종결인 민원은 24시간 안에 끝날지 '아직 모르므로' 뺐습니다.")

next_step("6. 기준선: 모델 없이 찍으면 몇 점인가")
