"""기초통계 포트폴리오(stat_00 ~ stat_04) 공용 설정: 경로, 상수, 그래프 스타일, 작은 도우미.

기존 01~04 스크립트(맨해튼 소음 군집분석)와 구분하기 위해 파일은 stat_ 접두어,
산출물은 stat1_ ~ stat4_ 접두어를 쓴다.
"""

import os
from pathlib import Path

import pandas as pd

# 기본은 프로젝트 루트. NYC311_ROOT 환경변수는 다른 폴더의 data/, outputs/로 시험 실행할 때만 쓴다.
ROOT = Path(os.environ.get("NYC311_ROOT") or Path(__file__).resolve().parents[1])
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

# ---- 분석 기간 / 파일 ----
START = "2024-04-01"          # 포함
END = "2024-10-01"            # 미포함
RAW_311 = RAW / "nyc311_all_2024_apr_sep.csv"
RAW_WEATHER = RAW / "nyc_daily_weather_2024_apr_sep.csv"
CLEAN = PROC / "nyc311_all_clean.parquet"
DAILY = PROC / "nyc311_all_daily.csv"
AGG = PROC / "nyc311_all_agg_daily_borough_type.csv"
MAP_SAMPLE = PROC / "nyc311_all_map_sample.csv"

SEED = 42

# 자치구 인구: 2020 Decennial Census (2020-04-01 기준, U.S. Census Bureau)
BOROUGH_POP = {
    "BRONX": 1_472_654,
    "BROOKLYN": 2_736_074,
    "MANHATTAN": 1_694_251,
    "QUEENS": 2_405_464,
    "STATEN ISLAND": 495_747,
}
BOROUGHS = list(BOROUGH_POP)
BOROUGH_KO = {
    "BRONX": "브롱크스",
    "BROOKLYN": "브루클린",
    "MANHATTAN": "맨해튼",
    "QUEENS": "퀸스",
    "STATEN ISLAND": "스태튼아일랜드",
}

TIME_BANDS = ["새벽(00-05)", "오전(06-11)", "오후(12-17)", "저녁(18-23)"]

# ---- 색상 / 스타일 (기존 02_descriptive_stats.py 와 동일한 팔레트) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_GRAY = "#b5b3ab"
SINGLE = CATEGORICAL[0]
ACCENT = CATEGORICAL[1]
BOROUGH_COLOR = dict(zip(BOROUGHS, CATEGORICAL))


def setup_style():
    """matplotlib 전역 스타일. 그래프를 그리는 스크립트에서만 호출한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
    return plt


def save_fig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  그림 저장: {path.relative_to(ROOT)}")


def save_table(df, name, index=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"  표 저장:   {path.relative_to(ROOT)}  ({len(df)}행)")


def rnd(df, n):
    """숫자 열만 반올림한다 (날짜 열이 섞인 표에 DataFrame.round 를 쓰면 경고가 난다)."""
    num = df.select_dtypes("number").columns
    return df.round(dict.fromkeys(num, n))


def top_n_other(s, n, other="기타"):
    """빈도 상위 n개 범주만 남기고 나머지는 other 로 묶는다."""
    top = s.value_counts().index[:n]
    return s.where(s.isin(top), other)


def time_band(hour):
    return pd.cut(hour, bins=[-1, 5, 11, 17, 23], labels=TIME_BANDS).astype(str)


def load_clean():
    if not CLEAN.exists():
        raise SystemExit(f"{CLEAN.relative_to(ROOT)} 가 없습니다. 먼저 analysis/stat_01_clean_descriptive.py 를 실행하세요.")
    return pd.read_parquet(CLEAN)


HOUR_TICKS = [0, 1, 6, 24, 72, 168, 720, 2160]
HOUR_TICK_LABELS = ["0", "1h", "6h", "1일", "3일", "1주", "30일", "90일"]


def hour_axis(ax, max_hours=None):
    """x축을 log1p(시간) 좌표로 쓰는 그래프에 사람이 읽기 쉬운 눈금(1h, 1일, 1주 ...)을 단다.

    처리 시간은 몇 분에서 몇 달까지 퍼져 있어 그냥 그리면 짧은 쪽이 전부 뭉개진다.
    log1p(x) = log(1 + x) 로 바꾼 좌표에 그리면 짧은 시간대가 넓게 펼쳐지고, 0시간도 그릴 수 있다.
    """
    import numpy as np

    pairs = [(t, s) for t, s in zip(HOUR_TICKS, HOUR_TICK_LABELS) if max_hours is None or t <= max_hours]
    ax.set_xticks(np.log1p([t for t, _ in pairs]), [s for _, s in pairs])
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


# ============================================================================ 공간 분석 공용 (stat_08, stat_09)
# [좌표 변환] 위도·경도는 '각도'라서 그대로 거리 계산에 쓰면 안 된다. 위도 1도는 어디서나 약 111km 지만,
# 경도 1도는 적도에서 111km, 뉴욕(북위 40.7도)에서는 약 84km 로 줄어든다(위도의 코사인만큼).
# 뉴욕처럼 좁은 지역에서는 기준점에서의 동서·남북 거리를 km 로 바꾸는 단순한 변환(등장방형 근사)으로 충분하다.
LAT0, LON0 = 40.49, -74.27                     # 기준점: 뉴욕시 남서쪽 모서리 바깥
KM_PER_DEG_LAT = 111.05                        # 북위 40.7도 부근에서 위도 1도의 길이
KM_PER_DEG_LON = 111.32 * 0.7581               # 경도 1도의 길이 = 111.32 x cos(40.7도)

# 민원을 9개 분류로 묶는다. 184개 유형을 그대로 쓰면 칸마다 대부분 0 이 되어 비율이 불안정하다.
CATEGORIES = ["주거 소음", "거리·상업 소음", "주차·차량", "주거 시설(HPD)", "위생·쓰레기(DSNY)",
              "도로·교통(DOT)", "환경·수도(DEP)", "공원·나무(DPR)", "기타"]
_PARKING = {"Illegal Parking", "Blocked Driveway", "Abandoned Vehicle", "Derelict Vehicles"}
_AGENCY_CAT = {"HPD": "주거 시설(HPD)", "DSNY": "위생·쓰레기(DSNY)", "DOT": "도로·교통(DOT)",
               "DEP": "환경·수도(DEP)", "DPR": "공원·나무(DPR)"}


def to_km(lat, lon):
    """위도·경도 -> 기준점에서 동쪽으로 x km, 북쪽으로 y km."""
    return (lon - LON0) * KM_PER_DEG_LON, (lat - LAT0) * KM_PER_DEG_LAT


def to_latlon(x, y):
    return LAT0 + y / KM_PER_DEG_LAT, LON0 + x / KM_PER_DEG_LON


def add_category(df):
    """complaint_type 과 agency 로 9개 분류 열(category)을 만든다. 위에서부터 먼저 맞는 규칙이 적용된다."""
    import numpy as np

    t, a = df["complaint_type"], df["agency"]
    cat = a.map(_AGENCY_CAT).fillna("기타")
    cat = cat.mask(t.isin(_PARKING), "주차·차량")
    cat = cat.mask(t.str.startswith("Noise - "), "거리·상업 소음")       # NYPD 소관 소음 (거리, 상업시설, 차량 등)
    cat = cat.mask(t == "Noise - Residential", "주거 소음")
    out = df.copy()
    out["category"] = np.asarray(cat)
    return out


def load_points():
    """좌표가 있는 종결 민원 + km 좌표(x, y) + 9개 분류."""
    df = load_clean()
    df = add_category(df[df["has_coord"]])
    df["x"], df["y"] = to_km(df["latitude"], df["longitude"])
    return df


def assign_cells(df, size_km):
    """점을 한 변 size_km 의 정사각형 격자 칸에 배정한다 (ix: 동서 번호, iy: 남북 번호)."""
    import numpy as np

    out = df.copy()
    out["ix"] = np.floor(out["x"] / size_km).astype(int)
    out["iy"] = np.floor(out["y"] / size_km).astype(int)
    return out


def cell_center(ix, iy, size_km):
    """격자 칸 중심의 (위도, 경도)."""
    return to_latlon((ix + 0.5) * size_km, (iy + 0.5) * size_km)
