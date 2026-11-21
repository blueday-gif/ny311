"""NYC 311 민원 EDA·머신러닝 입문 대시보드.

실행: py -3.12 -m streamlit run dashboard/app.py
(먼저 py -3.12 analysis/run_all.py 로 outputs/ 결과를 만들어 둔다)
"""

import streamlit as st

st.set_page_config(page_title="NYC 311 머신러닝 입문", page_icon=":material/school:", layout="wide")

page = st.navigation(
    {
        "": [
            st.Page("app_pages/home.py", title="시작하기", icon=":material/home:", default=True),
        ],
        "1부 데이터 알기": [
            st.Page("app_pages/eda_01.py", title="1. 데이터 점검", icon=":material/fact_check:"),
            st.Page("app_pages/eda_02.py", title="2. 숫자로 요약", icon=":material/functions:"),
            st.Page("app_pages/eda_03.py", title="3. 분포와 이상치", icon=":material/bar_chart:"),
            st.Page("app_pages/eda_04.py", title="4. 패턴과 비교", icon=":material/schedule:"),
        ],
        "2부 머신러닝 기본기": [
            st.Page("app_pages/ml_05.py", title="5. 무엇을 예측하나", icon=":material/flag:"),
            st.Page("app_pages/ml_06.py", title="6. 기준선", icon=":material/horizontal_rule:"),
            st.Page("app_pages/ml_07.py", title="7. 첫 모델: 로지스틱", icon=":material/show_chart:"),
            st.Page("app_pages/ml_08.py", title="8. 결정나무와 과적합", icon=":material/account_tree:"),
            st.Page("app_pages/ml_09.py", title="9. 모델 비교", icon=":material/forest:"),
            st.Page("app_pages/ml_10.py", title="10. 평가 지표", icon=":material/rule:"),
            st.Page("app_pages/ml_11.py", title="11. 모델 해석", icon=":material/search_insights:"),
            st.Page("app_pages/ml_12.py", title="12. 교차검증과 튜닝", icon=":material/tune:"),
        ],
        "3부 응용": [
            st.Page("app_pages/ml_13.py", title="13. 처리 시간 예측", icon=":material/timer:"),
            st.Page("app_pages/ml_14.py", title="14. 내일 건수 예측", icon=":material/trending_up:"),
            st.Page("app_pages/ml_15.py", title="15. 동네 유형 묶기", icon=":material/hub:"),
        ],
        "마무리": [
            st.Page("app_pages/summary.py", title="정리와 한계", icon=":material/summarize:"),
        ],
    },
    position="sidebar",
)

st.title(page.title, anchor=False)
page.run()
