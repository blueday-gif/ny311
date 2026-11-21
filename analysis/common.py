"""
공통 도구 모음 (모든 eda_ / ml_ 스크립트가 import 해서 쓴다)

- 이 파일은 직접 실행하지 않는다.
- 하는 일: 경로 정하기, 데이터 불러오기, 파생 열 만들기, 그래프 한글 설정, 표/그림/제외 로그 저장.
- 같은 코드를 15개 파일에 복사하지 않으려고 한 곳에 모았다. 고칠 일이 생기면 여기만 고치면 된다.
"""

# Path: 운영체제(Windows/Mac)와 상관없이 경로를 안전하게 다루기 위해 쓴다
from pathlib import Path

# matplotlib: 그래프를 그리는 기본 라이브러리
import matplotlib

# "Agg"는 화면 없이 그림 파일만 저장하는 방식. 스크립트를 끝까지 멈춤 없이 돌리기 위해 쓴다
matplotlib.use("Agg")

# pyplot: matplotlib에서 그림을 그릴 때 쓰는 함수 모음
import matplotlib.pyplot as plt

# numpy: 숫자 배열 계산용
import numpy as np

# pandas: 표(DataFrame)를 다루는 핵심 라이브러리
import pandas as pd

# ─────────────────────────────────────────────
# 1. 경로: 모두 "프로젝트 루트" 기준으로 만든다
# ─────────────────────────────────────────────

# 이 파일(analysis/common.py)의 부모의 부모 = 프로젝트 루트. 어느 폴더에서 실행해도 같은 경로가 된다
ROOT = Path(__file__).resolve().parent.parent

# Parquet으로 변환해 둔 원본 데이터 (scripts/convert_csv_to_parquet.py 가 만든 파일)
DATA_PATH = ROOT / "data" / "processed" / "311_requests_2020_present.parquet"

# ml_05 가 만드는 머신러닝용 데이터. ml_06 ~ ml_13 이 이 파일을 읽는다
ML_PATH = ROOT / "data" / "processed" / "ml_within24h.parquet"

# 그래프 저장 폴더
FIG_DIR = ROOT / "outputs" / "figures"

# 표(CSV) 저장 폴더
TAB_DIR = ROOT / "outputs" / "tables"

# 폴더가 없으면 만든다. exist_ok=True 는 "이미 있어도 오류 내지 말라"는 뜻
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 표 폴더도 같은 방식으로 만든다
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2. 재현성: 난수 seed 고정
# ─────────────────────────────────────────────

# seed(씨앗값)를 고정하면 무작위 표본 추출, 모델 학습 결과가 매번 똑같이 나온다
SEED = 42

# 머신러닝 학습에 쓰는 표본 크기. 396만 건 전체로 학습하면 느려서 공부용으로는 표본을 쓴다
TRAIN_SAMPLE = 200_000

# 평가용 표본 크기
TEST_SAMPLE = 100_000

# 학습/평가를 나누는 날짜. 이 날짜 "전"에 접수된 민원으로 배우고, "이후" 민원으로 시험 본다
SPLIT_DATE = pd.Timestamp("2026-08-01")

# ─────────────────────────────────────────────
# 3. 열 이름: 원본의 긴 영어 이름을 짧은 이름으로 바꾼다
# ─────────────────────────────────────────────

# 원본 열 이름 → 분석에서 쓸 짧은 이름. 공백과 괄호가 있으면 코드가 읽기 어려워서 바꾼다
RENAME = {
    "Unique Key": "unique_key",
    "Created Date": "created",
    "Closed Date": "closed",
    "Agency": "agency",
    "Problem (formerly Complaint Type)": "problem",
    "Borough": "borough",
    "Status": "status",
    "Open Data Channel Type": "channel",
    "Location Type": "location_type",
    "Latitude": "lat",
    "Longitude": "lon",
}


def setup_plot() -> None:
    """그래프에 한글이 깨지지 않도록 글꼴을 설정한다."""
    # Windows 기본 한글 글꼴. 지정하지 않으면 한글이 네모(□)로 나온다
    plt.rcParams["font.family"] = "Malgun Gothic"
    # 한글 글꼴에서 마이너스 기호(-)가 깨지는 문제를 막는다
    plt.rcParams["axes.unicode_minus"] = False
    # 그림 해상도. 100이면 화면에서 보기 적당하다
    plt.rcParams["figure.dpi"] = 100


def load_data() -> pd.DataFrame:
    """Parquet에서 분석에 쓸 열만 읽고, 파생 열을 붙여 돌려준다."""
    # 44개 열 중 필요한 11개만 읽는다. 전부 읽으면 메모리를 수 GB 더 쓴다
    df = pd.read_parquet(DATA_PATH, columns=list(RENAME.keys()))
    # 긴 열 이름을 짧은 이름으로 바꾼다
    df = df.rename(columns=RENAME)
    # 글자 열을 category(범주형)로 바꾼다. 같은 글자가 반복되는 열은 메모리가 크게 줄어든다
    for col in ["agency", "problem", "borough", "status", "channel", "location_type"]:
        df[col] = df[col].astype("category")
    # 처리 시간(시간 단위) = 종결 시각 - 접수 시각. 아직 종결 안 된 민원은 NaN(빈 값)이 된다
    df["hours"] = (df["closed"] - df["created"]).dt.total_seconds() / 3600
    # 접수 시각의 '시'(0~23). 하루 중 언제 민원이 많은지 볼 때 쓴다
    df["hour"] = df["created"].dt.hour
    # 요일 번호(0=월요일 ~ 6=일요일). 숫자지만 이름표라서 평균을 내면 안 된다
    df["weekday"] = df["created"].dt.day_of_week
    # 월(1~12)
    df["month"] = df["created"].dt.month
    # 접수 날짜(시각 제외). 일별 건수를 셀 때 쓴다
    df["date"] = df["created"].dt.normalize()
    # 완성된 표를 돌려준다
    return df


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """표를 outputs/tables/ 에 CSV로 저장한다. utf-8-sig 는 엑셀에서 한글이 안 깨지게 한다."""
    # 저장할 전체 경로를 만든다
    path = TAB_DIR / f"{name}.csv"
    # CSV로 쓴다. 매번 덮어쓰므로 다시 실행해도 파일이 쌓이지 않는다
    df.to_csv(path, index=index, encoding="utf-8-sig")
    # 어디에 저장했는지 화면에 알려 준다
    print(f"  [표 저장] {path.relative_to(ROOT)}")
    # 경로를 돌려준다
    return path


def save_fig(fig, name: str) -> Path:
    """그림을 outputs/figures/ 에 PNG로 저장하고 메모리에서 닫는다."""
    # 저장할 전체 경로
    path = FIG_DIR / f"{name}.png"
    # 여백을 자동 정리한다. 제목·축 글자가 잘리지 않게 한다
    fig.tight_layout()
    # 파일로 저장한다
    fig.savefig(path)
    # 그림을 닫는다. 닫지 않으면 그림이 쌓여 메모리를 계속 쓴다
    plt.close(fig)
    # 저장 위치를 알려 준다
    print(f"  [그림 저장] {path.relative_to(ROOT)}")
    # 경로를 돌려준다
    return path


class ExclusionLog:
    """제외하거나 대체한 데이터의 기준과 건수를 기록하는 도구."""

    def __init__(self, step: str, n_start: int):
        # 어느 단계의 로그인지 (파일 이름에 쓴다)
        self.step = step
        # 시작 행 수
        self.n_start = n_start
        # 기록을 모아 둘 목록
        self.rows = []

    def add(self, rule: str, n: int, action: str = "제외") -> None:
        """규칙 하나와 해당 건수를 기록한다."""
        # 전체 대비 비율(%)도 같이 남긴다. 건수만 보면 큰지 작은지 판단하기 어렵다
        self.rows.append({
            "단계": self.step,
            "기준": rule,
            "처리": action,
            "건수": int(n),
            "시작 행 대비 비율(%)": round(n / self.n_start * 100, 3),
        })

    def save(self) -> pd.DataFrame:
        """outputs/tables/{step}_exclusion_log.csv 로 저장한다."""
        # 기록이 하나도 없으면 "제외 없음"이라고 한 줄 남긴다. 빈 파일보다 의도가 분명하다
        if not self.rows:
            self.add("제외한 행 없음", 0, "없음")
        # 목록을 표로 바꾼다
        log = pd.DataFrame(self.rows)
        # 저장한다
        save_table(log, f"{self.step}_exclusion_log")
        # 화면에도 보여 준다
        print(log.to_string(index=False))
        # 표를 돌려준다
        return log


def load_ml(sample: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ml_05 가 만든 머신러닝용 데이터를 읽어 학습용/평가용으로 나눠 돌려준다."""
    # 파일이 없으면 무엇을 먼저 실행해야 하는지 알려 주고 멈춘다
    if not ML_PATH.exists():
        raise FileNotFoundError("먼저 py -3.12 analysis/ml_05_problem.py 를 실행하세요.")
    # 머신러닝용 데이터를 읽는다
    ml = pd.read_parquet(ML_PATH)
    # split 열이 "train"인 행 = 학습용
    train = ml[ml["split"] == "train"]
    # split 열이 "test"인 행 = 평가용
    test = ml[ml["split"] == "test"]
    # 표본을 쓰기로 했으면 seed를 고정해 같은 행을 뽑는다
    if sample:
        # 학습용에서 20만 건을 뽑는다 (전체가 더 적으면 전체 사용)
        train = train.sample(n=min(TRAIN_SAMPLE, len(train)), random_state=SEED)
        # 평가용에서 10만 건을 뽑는다
        test = test.sample(n=min(TEST_SAMPLE, len(test)), random_state=SEED)
    # 학습용과 평가용을 함께 돌려준다
    return train, test


# 머신러닝 특징(feature) 목록: "접수하는 순간 알 수 있는 정보"만 넣는다
CAT_FEATURES = ["problem", "borough", "channel", "location_type", "season"]

# 숫자 특징. hour는 0~23, weekday는 0~6, lat/lon은 위치
NUM_FEATURES = ["hour", "weekday", "lat", "lon"]

# 모든 특징을 합친 목록
FEATURES = CAT_FEATURES + NUM_FEATURES

# 예측할 대상(타깃): 24시간 안에 종결됐으면 1, 아니면 0
TARGET = "within_24h"


def season_of(month: pd.Series) -> pd.Series:
    """월(1~12)을 계절 이름으로 바꾼다.

    월을 그대로 특징으로 쓰면, 학습 기간(11월~7월)에 없던 8·9·10월을 평가할 때 모델이 처음 보는 값이 된다.
    계절로 묶으면 여름(6~8월)과 가을(9~11월)이 학습 기간에도 들어 있어 이 문제가 줄어든다.
    """
    # 월 번호 → 계절 이름 사전
    table = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
             6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
    # map 으로 바꾸고 범주형으로 만든다
    return month.map(table).astype("category")


def make_onehot_preprocess(numeric: list[str] | None = None, extra_cat: list[str] | None = None):
    """범주형 열은 원-핫 인코딩하고, 숫자 열은 그대로 통과시키는 전처리 도구를 만든다.

    [용어] 원-핫 인코딩(one-hot encoding): 범주 하나를 0/1 열 여러 개로 펼치는 방법.
      예) 자치구 = 퀸스 → [브롱크스 0, 브루클린 0, 맨해튼 0, 퀸스 1, 스태튼 0].
      모델은 글자를 계산할 수 없어서 숫자로 바꿔 줘야 한다.
    """
    # ColumnTransformer: 열마다 다른 전처리를 적용하는 도구
    from sklearn.compose import ColumnTransformer
    # OneHotEncoder: 원-핫 인코딩 도구
    from sklearn.preprocessing import OneHotEncoder

    # 숫자 열 목록이 없으면 기본값(NUM_FEATURES)을 쓴다
    numeric = NUM_FEATURES if numeric is None else numeric
    # 숫자처럼 보이지만 범주로 다룰 열(예: hour, weekday)을 추가로 받는다
    extra_cat = [] if extra_cat is None else extra_cat
    # 범주형은 원-핫, 숫자는 그대로(passthrough). handle_unknown="ignore" 는 평가 때 처음 보는 범주가 와도 오류 없이 0으로 처리
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES + extra_cat),
        ("num", "passthrough", numeric),
    ], verbose_feature_names_out=False)


def make_hgb(**params):
    """그래디언트 부스팅 분류 모델(HistGradientBoostingClassifier)을 만든다.

    범주형 열을 pandas category 자료형 그대로 받을 수 있어서(categorical_features="from_dtype") 원-핫이 필요 없다.
    결측값(위도·경도 빈 칸)도 알아서 처리한다.
    """
    # 부스팅 모델
    from sklearn.ensemble import HistGradientBoostingClassifier

    # 기본 설정. random_state 로 결과를 고정한다
    base = dict(categorical_features="from_dtype", random_state=SEED, max_iter=200)
    # 사용자가 넘긴 설정으로 덮어쓴다 (ml_12 튜닝에서 쓴다)
    base.update(params)
    # 모델을 돌려준다
    return HistGradientBoostingClassifier(**base)


def print_header(title: str) -> None:
    """단계 제목을 눈에 띄게 출력한다."""
    # 줄을 그어 단계 구분을 쉽게 한다
    print("\n" + "=" * 70)
    # 제목 출력
    print(title)
    # 아래 줄
    print("=" * 70)


# 다른 파일에서 np 를 따로 import 하지 않아도 되도록 노출 (선택 사항)
__all__ = [
    "ROOT", "DATA_PATH", "ML_PATH", "FIG_DIR", "TAB_DIR", "SEED", "SPLIT_DATE",
    "TRAIN_SAMPLE", "TEST_SAMPLE", "setup_plot", "load_data", "save_table",
    "save_fig", "ExclusionLog", "load_ml", "CAT_FEATURES", "NUM_FEATURES",
    "FEATURES", "TARGET", "print_header", "np", "pd", "plt",
    "season_of", "make_onehot_preprocess", "make_hgb",
]
