import streamlit as st

from dash_common import figure, header, meaning, next_step, show_table, table, term

header("1부 데이터 알기 · 3단계", "처리 시간은 어떤 모양으로 퍼져 있고, 로그 변환이 필요한가? 이상치는 얼마나 되나?",
       "히스토그램, 상자그림, 왜도, 로그 변환(log1p), IQR 이상치 규칙", "hours", "eda_03_distribution.py",
       "3 퍼짐·기술통계, 4 분포·변환·표준화")

skew = table("eda03_skewness").set_index("값")["왜도"]
out = table("eda03_outliers_iqr")

with st.container(horizontal=True):
    st.metric("왜도 (원래 값)", f"{skew.iloc[0]:.2f}", border=True)
    st.metric("왜도 (log1p 후)", f"{skew.iloc[1]:.2f}", border=True)
    st.metric("IQR 이상치 (원래 값)", f"{out['이상치 비율(%)'].iloc[0]:.1f}%", border=True)
    st.metric("IQR 이상치 (log1p 후)", f"{out['이상치 비율(%)'].iloc[1]:.1f}%", border=True)

term("왜도", "분포가 한쪽으로 치우친 정도. 0이면 좌우 대칭, 양수면 오른쪽 꼬리가 깁니다(큰 값이 드물게 있다).  \n"
     "보통 |왜도| > 1 이면 많이 치우쳤다고 봅니다. 예) 소득 분포.")
term("로그 변환 (log1p)", "큰 값을 압축하는 방법. 1, 10, 100, 1000 이 0, 1, 2, 3 으로 고르게 펴집니다.  \n"
     "0에는 로그를 쓸 수 없어서 log(1 + x) 인 **log1p** 를 씁니다.")

st.subheader("히스토그램: 원래 값 vs 로그 변환", anchor=False)
figure("eda03_hist_raw_vs_log", alt="처리 시간 원래 값과 log1p 변환 값의 히스토그램 비교")
meaning(f"원래 값은 왜도 {skew.iloc[0]:.1f}로 오른쪽 꼬리가 매우 깁니다. log1p 를 씌우면 {skew.iloc[1]:.2f}로 줄어 대칭에 가까워집니다. "
        "그래서 처리 시간으로 모델을 만들 때(13단계)는 로그 변환 값을 씁니다. 오른쪽 그림에 봉우리가 여럿 보이는 것은 (가설로서) "
        "'빨리 끝나는 민원'과 '오래 걸리는 민원'이 섞여 있기 때문입니다.")

st.subheader("상자그림", anchor=False)
figure("eda03_boxplot", alt="처리 시간 원래 값과 로그 값의 상자그림")

term("IQR 이상치 규칙", "IQR = Q3 − Q1. **Q1 − 1.5×IQR** 보다 작거나 **Q3 + 1.5×IQR** 보다 크면 이상치 후보로 표시합니다.  \n"
     "이상치는 '틀린 값'이 아니라 '드문 값'입니다. 여기서는 **지우지 않고** 개수만 기록합니다.")
st.subheader("IQR 규칙으로 센 이상치", anchor=False)
show_table(out, alt="IQR 이상치 경계와 개수")
meaning(f"원래 값 기준으로는 {out['위 경계'].iloc[0]:.0f}시간 넘게 걸린 민원 {out['위 이상치 수'].iloc[0]:,}건"
        f"({out['이상치 비율(%)'].iloc[0]:.1f}%)이 '이상치'로 잡힙니다. 그런데 로그 변환 후에는 0건입니다. "
        "즉 이 값들은 틀린 값이 아니라 **긴 꼬리의 일부**입니다. 치우친 분포에 IQR 규칙을 그대로 쓰면 꼬리 전체를 이상치로 오해합니다.")

next_step("4. 패턴과 비교: 언제, 어디서, 무엇이 많은가")
