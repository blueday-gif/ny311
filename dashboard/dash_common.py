"""대시보드 공통 도구: 결과 표·그림 읽기, 단계 머리말, 용어 상자, 해석 상자.

대시보드는 원본 데이터(수백 MB)를 읽지 않는다. analysis/ 스크립트가 만든
outputs/tables/*.csv 와 outputs/figures/*.png 만 읽어서 가볍게 동작한다.
분석을 다시 실행하면(py -3.12 analysis/run_all.py) 화면의 숫자도 자동으로 바뀐다.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# 프로젝트 루트 = dashboard/ 의 부모
ROOT = Path(__file__).resolve().parent.parent
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"


@st.cache_data(ttl="1h", max_entries=200)
def table(name: str, index_col=None) -> pd.DataFrame:
    """outputs/tables/{name}.csv 를 읽는다. 결과 표는 작아서 캐시해 두면 탭 이동이 빠르다."""
    return pd.read_csv(TAB_DIR / f"{name}.csv", encoding="utf-8-sig", index_col=index_col)


def figure(name: str, alt: str) -> None:
    """outputs/figures/{name}.png 를 보여 준다. 파일이 없으면 무엇을 실행해야 하는지 알려 준다."""
    path = FIG_DIR / f"{name}.png"
    if path.exists():
        st.image(str(path), alt=alt)
    else:
        st.warning(f"그림 파일이 없습니다: outputs/figures/{name}.png", title="먼저 분석을 실행하세요")


def header(part: str, question: str, concepts: str, columns: str, script: str, legacy: str | None = None) -> None:
    """기존 앱과 같은 틀의 머리말: 핵심 질문, 배우는 개념, 쓰는 열, 스크립트."""
    st.caption(part)
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**핵심 질문**  \n{question}")
            st.markdown(f"**이 단계에서 배우는 개념**  \n{concepts}")
        with c2:
            st.markdown(f"**쓰는 열**  \n{columns}")
            st.markdown(f"**분석 스크립트**  \n`analysis/{script}`")
            if legacy:
                st.markdown(f"**기존 앱 참고 탭**  \n{legacy}")


def term(name: str, body: str) -> None:
    """처음 나오는 용어를 펼치기 상자로 설명한다."""
    with st.expander(f":material/menu_book: 용어: {name}"):
        st.markdown(body)


def meaning(body: str) -> None:
    """결과 바로 아래 '이 숫자가 뜻하는 것' 상자."""
    st.info(body, title="이 숫자가 뜻하는 것", icon=":material/lightbulb:")


def caution(body: str) -> None:
    """해석할 때 주의할 점."""
    st.warning(body, title="주의", icon=":material/warning:")


def show_table(df: pd.DataFrame, alt: str, **kwargs) -> None:
    """표를 보여 준다. 인덱스는 숨기고 화면 폭에 맞춘다."""
    st.dataframe(df, hide_index=kwargs.pop("hide_index", True), alt=alt, **kwargs)


def next_step(text: str) -> None:
    """다음 단계 안내."""
    st.divider()
    st.caption(f":material/arrow_forward: 다음 단계 · {text}")
