"""8단계(추가 분석 4): 공간 분석 - KDE 핫스팟, DBSCAN, Moran's I.

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력. 좌표가 있는 종결 민원을 쓴다)
출력:  outputs/tables/stat8_*.csv, outputs/figures/stat8_*.png
실행:  프로젝트 루트에서  python analysis/stat_08_spatial.py

======================================================================================
세 방법은 서로 다른 질문에 답한다 (먼저 이 표를 읽고 시작하자)
======================================================================================
  방법        질문                                   입력        결과                      검정(p 값)
  ----------  -------------------------------------  ----------  ------------------------  ----------
  KDE         민원이 '어디에' 몰려 있는가             점(좌표)    매끄러운 밀도 지도        없음
  DBSCAN      밀집된 '덩어리'는 몇 개이고 어디인가    점(좌표)    군집 번호 + 잡음점        없음
  Moran's I   가까운 지역끼리 값이 '비슷한가'         면(격자 칸)  상관계수 하나 + 국지 지도  있음

  - KDE 는 "그림"을 준다. 어디가 진한지 한눈에 보이지만, 어디까지가 핫스팟인지 경계는 정해 주지 않는다.
  - DBSCAN 은 "경계"를 준다. 점마다 어느 덩어리에 속하는지(또는 어디에도 속하지 않는지) 딱 잘라 말한다.
  - Moran's I 는 "증거"를 준다. 눈에 보이는 뭉침이 우연히도 생길 수 있는 수준인지를 p 값으로 판단한다.
  그래서 보통 KDE 로 훑어보고 -> DBSCAN 으로 덩어리를 확정하고 -> Moran's I 로 우연이 아님을 확인한다.

======================================================================================
[가장 중요한 주의] 이 자료의 점은 '문제가 있는 곳'이 아니라 '신고가 들어온 곳'이다
======================================================================================
  - 사람이 많이 살수록, 신고를 적극적으로 하는 동네일수록 점이 많다. 밀도가 높다고 문제가 심하다는 뜻은 아니다.
    그래서 건수 지도와 함께 '입지계수(LQ)' 지도를 그린다. LQ 는 "그곳 민원 중 이 유형의 비중이 시 전체
    평균의 몇 배인가"로, 신고량 자체의 차이를 걷어내고 민원 구성의 차이만 본다.
  - 좌표는 건물 주소 단위다. 같은 건물에서 100번 신고하면 점 100개가 한 좌표에 겹친다 (이 자료는 76%가 겹친 좌표).
    이 때문에 DBSCAN 은 '민원 건'이 아니라 '서로 다른 신고 위치'를 대상으로 돌린다 (아래 (2) 참고).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from stat_common import (
    ACCENT, BOROUGH_KO, CATEGORICAL, GRID, INK, MUTED, OTHER_GRAY, SEED, SINGLE, SURFACE,
    assign_cells, cell_center, load_points, p_text, rnd, save_fig, save_table, setup_style, to_km, to_latlon,
)

# ---- KDE 설정 ----
KDE_CELL = 0.25                    # 밀도를 계산하는 격자 간격 (km)
KDE_BW = 0.5                       # 기본 대역폭 (km)
KDE_BW_COMPARE = [0.2, 0.5, 2.0]   # 대역폭 민감도 비교용
KDE_LAYERS = ["전체", "주거 소음", "거리·상업 소음", "주차·차량", "주거 시설(HPD)"]
LAND_MIN_DENSITY = 20              # 전체 민원 밀도가 이보다 낮은 곳(물, 공항, 큰 공원)은 지도에서 뺀다 (건/km²)
# ---- DBSCAN 설정 ----
DBSCAN_TYPE = "Noise - Commercial" # 상업시설 소음: 술집·클럽·식당 주변에 뭉치는 유형이라 덩어리 찾기에 알맞다
MIN_SAMPLES = 10
DBSCAN_EPS_M = 150                 # 이웃 반경 (m). 정한 이유는 dbscan_analysis 안의 주석 참고
EARTH_RADIUS_M = 6_371_000
REPEAT_TOP = 15                    # 반복 신고 위치 상위 몇 곳을 표로 남길지
# ---- Moran's I 설정 ----
MORAN_CELL = 1.0                   # 격자 칸 한 변 (km)
MORAN_MIN_N = 100                  # 칸에 민원이 이보다 적으면 비율·평균이 불안정해 뺀다
N_PERM = 999                       # 순열 검정 반복 수
MORAN_VARS = {
    "log_n": "민원 건수 (log10)",
    "share_주거 소음": "주거 소음 비중",
    "share_주거 시설(HPD)": "주거 시설(HPD) 비중",
    "share_주차·차량": "주차·차량 비중",
    "within_24h": "24시간 내 처리율",
    "mean_log_res": "평균 로그 처리시간",
    "adj_log_res": "유형 보정 로그 처리시간",
}


# ============================================================================ (1) KDE 핫스팟
def kde_surface(x, y, shape, bandwidth):
    """커널 밀도 추정 (Kernel Density Estimation, KDE).

    [아이디어] 점 하나하나 위에 작은 '모래 언덕'(커널)을 쌓고, 모든 언덕을 더한 높이를 그 지점의 밀도로 본다.
        점이 몰린 곳은 언덕이 겹쳐 높아지고, 드문 곳은 낮다. 히스토그램을 매끄럽게 만든 것이라고 생각하면 된다.
        히스토그램은 칸의 경계를 어디에 긋느냐에 따라 모양이 달라지지만 KDE 는 그 문제가 훨씬 작다.

    [대역폭(bandwidth)] 언덕의 너비다. KDE 에서 가장 중요한 선택이며, 커널의 모양보다 결과에 훨씬 큰 영향을 준다.
        - 너무 좁으면: 건물 하나하나가 뾰족하게 솟아 큰 흐름이 안 보인다 (과소평활, 잡음을 무늬로 착각)
        - 너무 넓으면: 도시 전체가 한 덩어리로 뭉개져 동네 수준의 차이가 사라진다 (과대평활)
        정답은 없다. "어느 정도 크기의 현상을 보고 싶은가"로 정한다. 동네 수준이면 수백 m, 자치구 수준이면 수 km.
        자동 선택 규칙(Scott, Silverman)도 있지만 자료가 정규분포에 가깝다는 가정에서 나온 값이라,
        도시처럼 여러 봉우리가 있는 자료에서는 대개 너무 넓게 나온다.

    [계산 방법] 점이 170만 개라 점마다 언덕을 쌓으면 너무 느리다. 그래서 먼저 250m 격자로 건수를 세고(2차원
        히스토그램), 그 격자에 가우시안 필터를 씌운다. 격자 간격이 대역폭보다 충분히 작으면 결과는 거의 같다.
        이 방법을 구간화 KDE(binned KDE)라고 한다.

    [단위] 결과를 칸 넓이로 나눠 '1km² 당 민원 건수'로 만든다. 필터는 총합을 보존하므로 지도 전체를 적분하면 총 건수다.
    """
    counts, _, _ = np.histogram2d(y, x, bins=shape, range=[[0, shape[0] * KDE_CELL], [0, shape[1] * KDE_CELL]])
    return gaussian_filter(counts, sigma=bandwidth / KDE_CELL, mode="constant") / KDE_CELL ** 2


def concentration(cell_counts):
    """민원이 얼마나 좁은 면적에 몰려 있는가.

    [상위 면적 점유율] 칸을 건수가 많은 순으로 줄 세웠을 때 상위 p% 면적이 전체 민원의 몇 %를 차지하는지.
    [지니계수] 0 이면 모든 칸에 똑같이 퍼져 있고, 1 에 가까울수록 소수의 칸에 몰려 있다.
        소득 불평등을 재는 그 지니계수와 같은 계산이다 (로렌츠 곡선과 대각선 사이 넓이의 2배).
    """
    v = np.sort(np.asarray(cell_counts, dtype=float))[::-1]
    total, n = v.sum(), len(v)
    out = {f"상위 {p}% 면적의 민원 점유율(%)": v[: max(1, int(round(n * p / 100)))].sum() / total * 100 for p in (1, 5, 10, 20)}
    asc = v[::-1]
    out["지니계수"] = 1 - 2 * (np.cumsum(asc) / total).sum() / n + 1 / n
    return out


def repeat_locations(pts):
    """같은 좌표에서 반복해 들어온 신고. KDE 봉우리를 해석하기 전에 반드시 확인해야 한다.

    311 자료에서는 한 주소(또는 한 사람)가 수백~수천 건을 신고하는 일이 흔하다. 그런 주소 하나가 KDE 지도에
    커다란 봉우리를 만들면 "이 동네가 시끄럽다"로 잘못 읽게 된다. 봉우리가 '넓은 지역의 현상'인지
    '점 하나의 반복'인지는 지도만 봐서는 구별되지 않으므로 표로 따로 확인한다.
    """
    g = pts.assign(lat=pts["latitude"].round(5), lon=pts["longitude"].round(5)).groupby(["lat", "lon"])
    top = g.size().nlargest(REPEAT_TOP).rename("민원 건수").reset_index()
    rows = []
    for r in top.itertuples():
        sub = g.get_group((r.lat, r.lon))
        vc = sub["complaint_type"].value_counts()
        rows.append({"lat": r.lat, "lon": r.lon, "자치구": BOROUGH_KO[sub["borough"].iloc[0]], "민원 건수": len(sub),
                     "가장 많은 유형": vc.index[0], "그 유형 비중(%)": vc.iloc[0] / len(sub) * 100,
                     "신고한 날 수": sub["date"].nunique(), "전체 민원 중 비중(%)": len(sub) / len(pts) * 100})
    return pd.DataFrame(rows)


def kde_analysis(pts):
    shape = (int(np.ceil(pts["y"].max() / KDE_CELL)) + 4, int(np.ceil(pts["x"].max() / KDE_CELL)) + 4)
    dens = {}
    for layer in KDE_LAYERS:
        sub = pts if layer == "전체" else pts[pts["category"] == layer]
        dens[layer] = kde_surface(sub["x"].to_numpy(), sub["y"].to_numpy(), shape, KDE_BW)
    land = dens["전체"] > LAND_MIN_DENSITY
    bw_compare = {bw: kde_surface(pts["x"].to_numpy(), pts["y"].to_numpy(), shape, bw) for bw in KDE_BW_COMPARE}

    # 참고: Scott 의 규칙이 권하는 대역폭 (2차원: n^(-1/6) x 표준편차)
    scott = len(pts) ** (-1 / 6) * np.sqrt((pts["x"].var() + pts["y"].var()) / 2)
    print(f"  격자 {shape[1]} x {shape[0]} 칸 (칸 {KDE_CELL * 1000:.0f}m), 기본 대역폭 {KDE_BW}km, 참고: Scott 규칙 {scott:.2f}km")

    # 입지계수(LQ, location quotient) = (그 지점 민원 중 이 유형의 비중) / (시 전체에서 이 유형의 비중)
    #   LQ = 1: 시 평균과 같은 구성,  LQ = 2: 이 유형이 시 평균의 2배만큼 두드러짐,  LQ = 0.5: 절반
    #   건수 지도는 '사람 많은 곳'을 보여 주는 경향이 있고, LQ 지도는 '그 동네 민원의 성격'을 보여 준다.
    rows, conc_rows, peak_rows = [], [], []
    iy, ix = np.nonzero(land)
    lat, lon = to_latlon((ix + 0.5) * KDE_CELL, (iy + 0.5) * KDE_CELL)
    keep = (ix % 2 == 0) & (iy % 2 == 0)             # 대시보드용으로는 500m 간격으로 솎아 저장한다
    # 봉우리가 어느 자치구인지: 250m 칸마다 가장 많이 나온 자치구
    cell = assign_cells(pts[["x", "y", "borough"]], KDE_CELL)
    boro = cell.groupby(["ix", "iy"])["borough"].agg(lambda s: s.mode().iloc[0])
    coarse = assign_cells(pts[["x", "y", "category"]], 0.5)
    for layer in KDE_LAYERS:
        d = dens[layer]
        share_city = 1.0 if layer == "전체" else (pts["category"] == layer).mean()
        lq = np.where(land, d / np.where(land, dens["전체"], 1) / share_city, np.nan)
        rows.append(pd.DataFrame({"분류": layer, "lat": lat[keep], "lon": lon[keep], "밀도(건/km²)": d[land][keep],
                                  "입지계수(LQ)": lq[land][keep]}))
        c = coarse if layer == "전체" else coarse[coarse["category"] == layer]
        conc_rows.append({"분류": layer, "민원 건수": len(c), **concentration(c.groupby(["ix", "iy"]).size())})
        # 봉우리: 반경 약 1km 안에서 가장 높은 지점들
        peaks = (d == maximum_filter(d, size=int(2 / KDE_CELL) + 1)) & land
        py, px = np.nonzero(peaks)
        order = np.argsort(d[py, px])[::-1][:5]
        for rank, k in enumerate(order, 1):
            plat, plon = to_latlon((px[k] + 0.5) * KDE_CELL, (py[k] + 0.5) * KDE_CELL)
            b = boro.get((px[k], py[k]), None)
            peak_rows.append({"분류": layer, "순위": rank, "자치구": BOROUGH_KO.get(b, ""), "lat": plat, "lon": plon,
                              "밀도(건/km²)": d[py[k], px[k]], "시 평균 대비 배율": d[py[k], px[k]] / d[land].mean()})
    return dens, land, bw_compare, pd.concat(rows), pd.DataFrame(conc_rows), pd.DataFrame(peak_rows)


# ============================================================================ (2) DBSCAN
def k_distance(coords_rad, k):
    """k-거리 그래프: eps 를 고르는 표준적인 방법.

    각 점에서 k번째로 가까운 이웃까지의 거리를 구해 작은 순으로 늘어놓는다.
      - 덩어리 안의 점들은 이 거리가 짧고 비슷하다 (그래프의 평평한 부분)
      - 외딴 점들은 이 거리가 갑자기 길어진다 (그래프가 급히 솟는 부분)
    둘이 갈리는 '무릎(knee)' 지점의 거리를 eps 로 쓴다. 무릎보다 가까이에 이웃 k개가 있으면 덩어리 안,
    없으면 외딴 점이라는 뜻이 된다. 무릎은 그래프 양 끝을 이은 직선에서 가장 멀리 떨어진 점으로 찾는다.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree").fit(coords_rad)
    dist, _ = nn.kneighbors(coords_rad)                       # 0번째 이웃은 자기 자신
    kd = np.sort(dist[:, k]) * EARTH_RADIUS_M
    xs = np.linspace(0, 1, len(kd))
    ys = (kd - kd[0]) / (kd[-1] - kd[0])
    knee = int(np.argmax(xs - ys))
    return kd, knee


def describe_clusters(labels, xy, weight):
    """군집화 결과를 한 줄로 요약한다. labels 가 -1 인 점은 잡음."""
    lab = np.asarray(labels)
    mask = lab >= 0
    ids, sizes = np.unique(lab[mask], return_counts=True)
    radii = [np.sqrt(((xy[lab == i] - xy[lab == i].mean(axis=0)) ** 2).sum(axis=1)).mean() * 1000 for i in ids]
    sil = np.nan
    if len(ids) >= 2:
        # 실루엣 계수: 점마다 (가장 가까운 다른 군집까지의 평균 거리 - 자기 군집 안 평균 거리) / 둘 중 큰 값.
        # 1 에 가까우면 자기 군집에 꼭 맞고, 0 이면 경계에 있고, 음수면 다른 군집이 더 가깝다. 잡음점은 빼고 잰다.
        rng = np.random.default_rng(SEED)
        idx = np.flatnonzero(mask)
        idx = rng.choice(idx, size=min(6000, len(idx)), replace=False)
        if len(np.unique(lab[idx])) >= 2:
            sil = silhouette_score(xy[idx], lab[idx])
    return {
        "군집 수": len(ids), "잡음 위치 비율(%)": (~mask).mean() * 100,
        "잡음에 속한 민원 비율(%)": weight[~mask].sum() / weight.sum() * 100,
        "최대 군집의 위치 점유율(%)": sizes.max() / len(lab) * 100 if len(ids) else 0.0,
        "군집 반경 중앙값(m)": float(np.median(radii)) if len(ids) else np.nan, "실루엣 (잡음 제외)": sil,
    }


def dbscan_analysis(pts):
    """DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    [아이디어] "주변에 이웃이 충분히 많은 점들이 이어진 것"을 하나의 군집으로 본다. 정해야 할 것은 두 가지뿐이다.
        eps:         이웃으로 칠 거리 (반경)
        min_samples: 반경 안에 이웃이 최소 몇 개 있어야 '밀집'이라고 볼 것인가 (자기 자신 포함)

    [점의 세 종류]
        핵심점(core):   반경 eps 안에 점이 min_samples 개 이상 있는 점. 군집의 뼈대다.
        경계점(border): 자기는 핵심점이 아니지만 어떤 핵심점의 반경 안에 있는 점. 군집의 가장자리다.
        잡음점(noise):  둘 다 아닌 점. 어느 군집에도 속하지 않으며 번호 -1 을 받는다.
      핵심점끼리 반경 안에 있으면 같은 군집으로 이어 붙인다. 사슬처럼 이어지므로 군집 모양은 자유롭다.

    [K-means 와의 차이: 같은 '군집분석'이지만 생각이 정반대다]
                      K-means                          DBSCAN
      군집의 정의     중심에 가까운 점들의 모임         밀집된 점들이 이어진 덩어리
      군집 수         미리 정해야 한다 (k)             결과로 나온다
      군집 모양       둥근 덩어리만                     길쭉하든 휘었든 자유
      모든 점 배정?   그렇다. 외딴 점도 억지로 배정     아니다. 외딴 점은 잡음으로 남긴다
      약점            이상값에 중심이 끌려간다          밀도가 다른 군집이 섞이면 eps 하나로 못 맞춘다
      정할 것         k                                eps, min_samples

    [HDBSCAN] DBSCAN 의 약점(eps 하나로 모든 곳을 재단)을 고친 방법. eps 를 여러 값으로 바꿔 가며 군집이 생기고
        합쳐지는 과정을 나무 구조로 기록한 뒤, 오래 유지되는(안정적인) 군집만 고른다. 번화가처럼 촘촘한 곳과
        외곽처럼 성긴 곳의 군집을 함께 찾을 수 있다. 정할 것은 min_cluster_size(군집으로 인정할 최소 크기)다.

    [이 자료에서의 설계]
      - 대상: 상업시설 소음(Noise - Commercial). 술집·클럽·식당이 모인 곳에 뭉치므로 덩어리의 뜻이 분명하다.
      - 단위: 민원 건이 아니라 '서로 다른 신고 위치'. 좌표가 건물 단위라 같은 업소에 대한 신고 수십 건이 한 점에
        겹치는데, 그대로 돌리면 "신고가 반복된 건물 하나"가 군집이 되어 버린다. 위치를 한 번씩만 세면
        군집의 뜻이 "소음 신고 대상 업소가 반경 eps 안에 min_samples 곳 이상 모인 구역", 즉 유흥가가 된다.
        각 위치의 민원 건수는 군집을 만든 뒤 요약할 때 가중치로 쓴다.
      - 거리: 위경도는 각도이므로 지구 표면 거리(haversine)를 쓴다. sklearn 은 라디안 좌표를 요구하고
        eps 도 라디안이어야 한다 (eps_m / 지구 반지름).
    """
    sub = pts[pts["complaint_type"] == DBSCAN_TYPE]
    loc = (sub.assign(lat=sub["latitude"].round(5), lon=sub["longitude"].round(5))
              .groupby(["lat", "lon"]).agg(민원건수=("borough", "size"), borough=("borough", "first")).reset_index())
    x, y = to_km(loc["lat"].to_numpy(), loc["lon"].to_numpy())
    xy = np.column_stack([x, y])
    rad = np.radians(loc[["lat", "lon"]].to_numpy())
    w = loc["민원건수"].to_numpy()
    print(f"  {DBSCAN_TYPE}: 민원 {len(sub):,}건 -> 서로 다른 위치 {len(loc):,}곳 (위치당 평균 {len(sub) / len(loc):.1f}건)")

    kd, knee = k_distance(rad, MIN_SAMPLES)
    eps_knee = kd[knee]
    # [무릎 값을 그대로 쓰지 않은 이유] k-거리 그래프의 무릎은 '도시 전체에서 평균적으로' 덩어리와 외딴 점이
    # 갈리는 거리다. 그런데 이 자료는 맨해튼처럼 아주 촘촘한 곳과 외곽처럼 성긴 곳이 섞여 있다. 무릎 값(수백 m)을
    # 쓰면 촘촘한 곳에서는 반경이 너무 커서 여러 유흥가가 사슬처럼 이어져 거대한 한 덩어리가 된다.
    # 그래서 "걸어서 한두 블록" 거리인 150m 로 정했다 (맨해튼의 남북 한 블록이 약 80m, 동서 한 블록이 약 250m).
    # 아래 민감도 표에서도 150~200m 부근에서 군집 수가 가장 많다. 그보다 작으면 군집이 잘게 부서져 사라지고,
    # 크면 서로 합쳐지기 때문이다. 이렇게 eps 는 자동 규칙 + 분야 지식 + 민감도 확인으로 함께 정한다.
    eps_m = DBSCAN_EPS_M
    print(f"  {MIN_SAMPLES}-거리 그래프의 무릎: {eps_knee:.0f}m (참고값). eps 는 분야 지식으로 {eps_m:.0f}m 로 정함")
    idx = np.unique(np.linspace(0, len(kd) - 1, 300).astype(int))
    kd_table = pd.DataFrame({"순위 비율": idx / (len(kd) - 1), "k-거리(m)": kd[idx]})

    # 민감도: eps 와 min_samples 를 바꾸면 결과가 얼마나 달라지는가 (DBSCAN 의 최대 약점을 직접 확인한다)
    sweep = []
    for ms in (5, 10, 20):
        for e in sorted({75, 100, 150, 200, 300, 400, 500, eps_m}):
            lab = DBSCAN(eps=e / EARTH_RADIUS_M, min_samples=ms, metric="haversine", algorithm="ball_tree").fit_predict(rad)
            d = describe_clusters(lab, xy, w)
            sweep.append({"min_samples": ms, "eps(m)": e, **{k: d[k] for k in
                          ["군집 수", "잡음 위치 비율(%)", "최대 군집의 위치 점유율(%)"]}})
    sweep = pd.DataFrame(sweep)

    # 같은 점에 세 방법을 적용해 비교한다
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=MIN_SAMPLES, metric="haversine", algorithm="ball_tree").fit_predict(rad)
    # HDBSCAN 과 K-means 는 km 로 바꾼 평면 좌표에 유클리드 거리를 쓴다 (도시 크기에서는 haversine 과 차이가 거의 없다)
    hdb = HDBSCAN(min_cluster_size=MIN_SAMPLES * 2, min_samples=MIN_SAMPLES).fit_predict(xy)
    k = int(np.clip(len(np.unique(db[db >= 0])), 2, 200))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(xy)      # 군집 수를 DBSCAN 과 같게 맞춰 비교
    compare = pd.DataFrame([
        {"방법": f"K-means (k={k}, DBSCAN 군집 수에 맞춤)", **describe_clusters(km, xy, w)},
        {"방법": f"DBSCAN (eps={eps_m:.0f}m, min_samples={MIN_SAMPLES})", **describe_clusters(db, xy, w)},
        {"방법": f"HDBSCAN (min_cluster_size={MIN_SAMPLES * 2})", **describe_clusters(hdb, xy, w)},
    ])

    loc["dbscan"], loc["hdbscan"], loc["kmeans"] = db, hdb, km
    cl = loc[loc["dbscan"] >= 0]
    top = (cl.groupby("dbscan").apply(lambda g: pd.Series({
        "신고 위치 수": len(g), "민원 건수": g["민원건수"].sum(),
        "자치구": BOROUGH_KO[g["borough"].mode().iloc[0]],
        "중심 lat": np.average(g["lat"], weights=g["민원건수"]), "중심 lon": np.average(g["lon"], weights=g["민원건수"]),
        "동서 폭(m)": (to_km(g["lat"], g["lon"])[0].max() - to_km(g["lat"], g["lon"])[0].min()) * 1000,
        "남북 폭(m)": (to_km(g["lat"], g["lon"])[1].max() - to_km(g["lat"], g["lon"])[1].min()) * 1000,
    }), include_groups=False).sort_values("민원 건수", ascending=False).reset_index().rename(columns={"dbscan": "군집"}))
    top["민원 점유율(%)"] = top["민원 건수"] / w.sum() * 100
    info = {"eps": eps_m, "knee": eps_knee, "n_loc": len(loc), "n": len(sub), "k": k}
    return loc, xy, kd_table, sweep, compare, top, info


# ============================================================================ (3) Moran's I
def cell_table(pts, size, min_n):
    """격자 칸별 집계. 비율과 평균은 칸의 민원이 min_n 건 이상일 때만 믿을 만하다."""
    d = assign_cells(pts, size)
    # 유형 보정 처리시간: 각 민원의 로그 처리시간에서 '그 민원 유형의 시 전체 평균'을 뺀 값.
    # 0 보다 크면 "같은 유형의 다른 민원보다 오래 걸렸다"는 뜻이다. 7단계에서 본 것처럼 처리 시간은 유형이
    # 거의 결정하므로, 이 값을 써야 '유형 구성' 때문이 아닌 '그 지역'의 빠르고 느림을 볼 수 있다.
    d["adj"] = d["log_resolution"] - d.groupby("complaint_type")["log_resolution"].transform("mean")
    g = d.groupby(["ix", "iy"])
    out = pd.DataFrame({"n": g.size(), "within_24h": g["within_24h"].mean(), "mean_log_res": g["log_resolution"].mean(),
                        "adj_log_res": g["adj"].mean(), "borough": g["borough"].agg(lambda s: s.mode().iloc[0])})
    shares = pd.crosstab([d["ix"], d["iy"]], d["category"], normalize="index").add_prefix("share_")
    out = out.join(shares).reset_index()
    out = out[out["n"] >= min_n].reset_index(drop=True)
    out["log_n"] = np.log10(out["n"])
    out["lat"], out["lon"] = cell_center(out["ix"].to_numpy(), out["iy"].to_numpy(), size)
    return out


def queen_weights(cells):
    """공간 가중치 행렬 W: "누가 누구의 이웃인가"를 숫자로 적은 표. 공간 통계의 출발점이다.

    [퀸 인접] 체스의 퀸처럼 상하좌우와 대각선으로 맞닿은 칸(최대 8개)을 이웃으로 본다.
              (룩 인접은 상하좌우 4칸만. 거리 기준, 가까운 k개 기준 등 다른 정의도 있다.)
    [행 표준화] 각 행의 합이 1 이 되게 나눈다. 그러면 W 에 값을 곱한 결과(공간 시차, spatial lag)가
              "내 이웃들 값의 평균"이 된다. 이웃이 3개인 가장자리 칸과 8개인 안쪽 칸을 공평하게 다루는 효과도 있다.
    W 를 어떻게 정의하느냐에 따라 결과가 달라질 수 있으므로, 보고할 때 반드시 정의를 함께 적는다.
    """
    index = {(ix, iy): i for i, (ix, iy) in enumerate(zip(cells["ix"], cells["iy"]))}
    rows, cols = [], []
    for (ix, iy), i in index.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = index.get((ix + dx, iy + dy))
                if j is not None and j != i:
                    rows.append(i)
                    cols.append(j)
    n = len(cells)
    W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    has_nb = deg > 0
    W = sparse.diags(np.where(has_nb, 1 / np.where(has_nb, deg, 1), 0)) @ W
    return W, has_nb, deg


def morans_i(values, W, rng):
    """전역 Moran's I: 지도 전체에 대해 "가까운 곳끼리 값이 비슷한가"를 숫자 하나로 요약한다.

    [정의] z = 값 - 평균 이라 할 때 (행 표준화된 W 기준)
              I = 합_i z_i * (이웃들의 z 평균)_i  /  합_i z_i²
        보통의 상관계수가 "x 와 y 가 같이 움직이는가"라면, Moran's I 는 "내 값과 내 이웃들의 값이 같이 움직이는가"다.
        그래서 공간 자기상관(spatial autocorrelation) 계수라고 부른다. 6단계의 시간 자기상관(ACF)의 공간판이다.

    [읽는 법]
        I > 0 : 비슷한 값끼리 모여 있다 (높은 곳 옆에 높은 곳). 군집형. 대부분의 사회·지리 자료가 여기에 속한다.
        I ≈ 0 : 공간적으로 무작위. 정확한 기댓값은 0 이 아니라 -1/(n-1) 이다.
        I < 0 : 높은 값과 낮은 값이 번갈아 놓인다 (체스판 무늬). 드물다.
        크기는 대략 0.3 이상이면 뚜렷한 군집, 0.1 안팎이면 약한 군집으로 본다.

    [순열 검정] "이 정도 I 가 우연히도 나올 수 있는가"를 확인하는 방법.
        귀무가설 H0: 값들이 위치와 무관하게 놓여 있다 (공간적 무작위).
        값들을 칸에 무작위로 다시 뿌리고 I 를 계산하는 일을 999번 반복해 '우연일 때의 I 분포'를 만든다.
        관측한 I 가 그 분포에서 얼마나 극단적인지가 p 값이다 (유사 p 값). 정규분포 가정이 필요 없어서 널리 쓴다.
        반복 수가 999 면 p 값의 최솟값은 1/1000 = 0.001 이다.
    """
    z = values - values.mean()
    denom = (z ** 2).sum()
    lag = W @ z
    obs = (z * lag).sum() / denom
    perm = np.array([rng.permutation(z) for _ in range(N_PERM)]).T          # (n, 반복 수)
    sims = (perm * (W @ perm)).sum(axis=0) / denom
    expected = -1 / (len(z) - 1)
    p = ((np.abs(sims - expected) >= abs(obs - expected)).sum() + 1) / (N_PERM + 1)
    z_score = (obs - sims.mean()) / sims.std()
    return obs, expected, z_score, p, z, lag


def local_moran(z, W, rng):
    """국지 Moran (LISA, Local Indicators of Spatial Association): "어디가" 뭉쳐 있는지 칸마다 판정한다.

    전역 I 는 지도 전체에 대한 숫자 하나라서, 뭉침이 어디에 있는지는 알려 주지 않는다. LISA 는 전역 I 를
    칸별 기여분으로 쪼갠 것이다:  I_i = z_i * (이웃들의 z 평균)_i / (z² 의 평균).  모두 더해 n 으로 나누면 전역 I 가 된다.

    [네 가지 유형] 내 값(z_i)과 이웃 평균의 부호로 나눈다.
        HH (높음-높음): 나도 높고 이웃도 높다 -> 핫스팟
        LL (낮음-낮음): 나도 낮고 이웃도 낮다 -> 콜드스팟
        HL (높음-낮음): 나만 높고 이웃은 낮다 -> 공간 이상점 (낮은 지역 속의 섬)
        LH (낮음-높음): 나만 낮고 이웃은 높다 -> 공간 이상점 (높은 지역 속의 구멍)
    [유의성] 이웃 자리에 다른 칸들의 값을 무작위로 넣어 보는 순열 검정으로 칸마다 p 값을 구한다.
        p < 0.05 인 칸만 유형을 표시하고 나머지는 '유의하지 않음'으로 둔다.
    [다중검정 주의] 칸이 수백 개이므로 p < 0.05 기준이면 아무 무늬가 없어도 5% 정도는 우연히 유의하게 나온다.
        낱개의 칸보다 같은 유형이 여러 칸 이어진 '덩어리'를 믿어야 한다.
    [KDE 핫스팟과의 차이] KDE 는 "여기가 진하다"는 그림이고, LISA 의 HH 는 "여기와 주변이 함께 높으며
        그것이 우연으로 보기 어렵다"는 검정 결과다.
    """
    lag = W @ z
    m2 = (z ** 2).mean()
    local = z * lag / m2
    perm = np.array([rng.permutation(z) for _ in range(N_PERM)]).T
    lag_perm = W @ perm
    higher = (lag_perm >= lag[:, None]).sum(axis=1)
    p = (np.minimum(higher, N_PERM - higher) + 1) / (N_PERM + 1)
    kind = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0), (z < 0) & (lag > 0)],
                     ["HH (핫스팟)", "LL (콜드스팟)", "HL (낮은 곳 속 높은 섬)", "LH (높은 곳 속 낮은 구멍)"], default="유의하지 않음")
    kind = np.where(p < 0.05, kind, "유의하지 않음")
    return local, lag, p, kind


def moran_analysis(pts):
    rng = np.random.default_rng(SEED)
    cells = cell_table(pts, MORAN_CELL, MORAN_MIN_N)
    W, has_nb, deg = queen_weights(cells)
    cells, W = cells[has_nb].reset_index(drop=True), W[has_nb][:, has_nb]
    print(f"  {MORAN_CELL:.0f}km 격자: 민원 {MORAN_MIN_N}건 이상인 칸 {len(cells)}개 (이웃이 없는 칸 {int((~has_nb).sum())}개 제외), "
          f"칸당 이웃 평균 {deg[has_nb].mean():.1f}개")

    glob, lisa = [], []
    for col, label in MORAN_VARS.items():
        v = cells[col].to_numpy(dtype=float)
        obs, exp, zs, p, z, lag = morans_i(v, W, rng)
        strength = "강한 군집" if obs >= 0.5 else "뚜렷한 군집" if obs >= 0.3 else "약한 군집" if obs >= 0.1 else "거의 무작위"
        glob.append({"변수": label, "칸 수": len(v), "Moran's I": obs, "기댓값 (무작위일 때)": exp, "z 점수": zs,
                     "유사 p 값": p, "판정 (α=0.05)": "공간 군집 있음" if p < 0.05 and obs > exp else
                     ("분산 패턴" if p < 0.05 else "무작위와 구별 안 됨"), "세기": strength})
        local, lag2, lp, kind = local_moran(z, W, rng)
        sd = v.std()
        lisa.append(pd.DataFrame({"변수": label, "ix": cells["ix"], "iy": cells["iy"], "lat": cells["lat"], "lon": cells["lon"],
                                  "자치구": cells["borough"].map(BOROUGH_KO), "값": v, "표준화 값": z / sd,
                                  "이웃 평균(표준화)": lag2 / sd, "국지 I": local, "p": lp, "유형": kind}))
    glob = pd.DataFrame(glob).sort_values("Moran's I", ascending=False)
    lisa = pd.concat(lisa)

    # 격자 크기를 바꾸면 결과가 달라지는가 (MAUP: 공간 단위 임의성 문제)
    # 같은 점 자료라도 어떤 크기·모양의 구역으로 묶느냐에 따라 통계가 달라진다. 우편번호든 격자든 피할 수 없는
    # 문제이므로, 한 가지 크기의 결과만 보고 단정하지 말고 크기를 바꿔도 결론이 유지되는지 확인한다.
    maup = []
    for size, min_n in [(0.5, 30), (1.0, 100), (2.0, 400)]:
        c = cell_table(pts, size, min_n)
        Wm, nb, _ = queen_weights(c)
        c, Wm = c[nb].reset_index(drop=True), Wm[nb][:, nb]
        row = {"격자 크기(km)": size, "최소 건수": min_n, "칸 수": len(c)}
        for col in ("log_n", "share_주거 소음", "adj_log_res"):
            row[f"I: {MORAN_VARS[col]}"] = morans_i(c[col].to_numpy(dtype=float), Wm, rng)[0]
        maup.append(row)
    return cells, glob, lisa, pd.DataFrame(maup)


# ============================================================================ 그래프
def map_axes(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figures(plt, dens, land, bw_compare, loc, xy, kd_table, compare, info, lisa):
    from matplotlib.colors import LogNorm

    ext = [0, land.shape[1] * KDE_CELL, 0, land.shape[0] * KDE_CELL]
    vmax = np.percentile(dens["전체"][land], 99.5)

    # (1) 대역폭 비교
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (bw, d) in zip(axes, bw_compare.items()):
        ax.imshow(np.where(d > LAND_MIN_DENSITY, d, np.nan), origin="lower", extent=ext, cmap="magma_r",
                  norm=LogNorm(vmin=LAND_MIN_DENSITY, vmax=vmax * 2))
        note = "너무 좁음: 점들이 낱낱이 보임" if bw < KDE_BW else "기본값" if bw == KDE_BW else "너무 넓음: 동네 차이가 사라짐"
        map_axes(ax, f"대역폭 {bw}km  ({note})")
    fig.suptitle("같은 자료, 다른 대역폭: KDE 결과는 대역폭이 결정한다", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "stat8_01_kde_bandwidth.png")

    # (2) 유형별 밀도와 입지계수
    layers = [l for l in dens if l != "전체"]
    fig, axes = plt.subplots(2, len(layers), figsize=(3.4 * len(layers), 7.4))
    for j, layer in enumerate(layers):
        d = dens[layer]
        axes[0, j].imshow(np.where(land, d, np.nan), origin="lower", extent=ext, cmap="magma_r",
                          norm=LogNorm(vmin=max(1, np.percentile(d[land], 20)), vmax=np.percentile(d[land], 99.8)))
        map_axes(axes[0, j], f"{layer}\n밀도 (건/km²)")
        share = d[land].sum() / dens["전체"][land].sum()
        lq = np.where(land, d / np.where(land, dens["전체"], 1) / share, np.nan)
        axes[1, j].imshow(np.log2(np.clip(lq, 0.125, 8)), origin="lower", extent=ext, cmap="RdBu_r", vmin=-2, vmax=2)
        map_axes(axes[1, j], "입지계수 LQ\n(빨강: 시 평균보다 두드러짐)")
    fig.suptitle("건수 지도와 입지계수 지도는 다른 이야기를 한다", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "stat8_02_kde_density_lq.png")

    # (3) k-거리 그래프
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(kd_table["순위 비율"] * 100, kd_table["k-거리(m)"], color=SINGLE, lw=1.8)
    ax.axhline(info["eps"], color=ACCENT, lw=1.2, ls="--")
    ax.text(2, info["eps"], f" 선택한 eps = {info['eps']:.0f}m", va="bottom", fontsize=9, color=ACCENT)
    ax.axhline(info["knee"], color=MUTED, lw=1, ls=":")
    ax.text(2, info["knee"], f" 자동 규칙의 무릎 = {info['knee']:.0f}m (너무 커서 쓰지 않음)", va="bottom", fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.set_xlabel("위치 (k-거리가 짧은 순, %)")
    ax.set_ylabel(f"{MIN_SAMPLES}번째 이웃까지의 거리 (m, 로그 축)")
    ax.set_title("k-거리 그래프: 급히 솟기 시작하는 무릎이 eps 후보")
    ax.yaxis.grid(True, color=GRID)
    save_fig(fig, "stat8_03_dbscan_k_distance.png")

    # (4) 세 방법 비교 (맨해튼 남부~중부 확대: 가장 큰 DBSCAN 군집 주변 12km)
    big = loc.loc[loc["dbscan"] >= 0, "dbscan"].value_counts().index[0]
    cx, cy = xy[(loc["dbscan"] == big).to_numpy()].mean(axis=0)
    win = (np.abs(xy[:, 0] - cx) < 6) & (np.abs(xy[:, 1] - cy) < 6)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    rng = np.random.default_rng(SEED)
    for ax, col, row in zip(axes, ["kmeans", "dbscan", "hdbscan"], compare.itertuples()):
        lab = loc[col].to_numpy()[win]
        p = xy[win]
        noise = lab < 0
        ax.scatter(p[noise, 0], p[noise, 1], s=3, color=OTHER_GRAY, alpha=0.7, linewidths=0)
        ids = np.unique(lab[~noise])
        palette = {i: CATEGORICAL[k % len(CATEGORICAL)] for k, i in enumerate(rng.permutation(ids))}
        ax.scatter(p[~noise, 0], p[~noise, 1], s=5, color=[palette[i] for i in lab[~noise]], linewidths=0)
        map_axes(ax, f"{row.방법}\n군집 {row._2}개 · 잡음 위치 {row._3:.0f}%")
    fig.suptitle(f"같은 점({DBSCAN_TYPE} 신고 위치), 세 가지 군집화 (회색 = 잡음)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "stat8_04_dbscan_vs_kmeans_hdbscan.png")

    # (5) Moran 산점도와 LISA 지도
    kinds = {"HH (핫스팟)": "#e34948", "LL (콜드스팟)": "#2a78d6", "HL (낮은 곳 속 높은 섬)": "#eda100",
             "LH (높은 곳 속 낮은 구멍)": "#1baf7a", "유의하지 않음": "#dcdbd3"}
    show = ["민원 건수 (log10)", "주거 시설(HPD) 비중", "유형 보정 로그 처리시간"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    for j, var in enumerate(show):
        d = lisa[lisa["변수"] == var]
        ax = axes[0, j]
        ax.scatter(d["표준화 값"], d["이웃 평균(표준화)"], s=9, color=[kinds[k] for k in d["유형"]], linewidths=0)
        slope = np.polyfit(d["표준화 값"], d["이웃 평균(표준화)"], 1)[0]      # 이 직선의 기울기가 곧 Moran's I 다
        xs = np.array([d["표준화 값"].min(), d["표준화 값"].max()])
        ax.plot(xs, slope * xs, color=INK, lw=1.2)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{var}\nMoran 산점도 (기울기 = I = {slope:.2f})", fontsize=11)
        ax.set_xlabel("내 값 (표준화)")
        ax.set_ylabel("이웃 평균 (표준화)")
        ax = axes[1, j]
        x, y = to_km(d["lat"].to_numpy(), d["lon"].to_numpy())
        ax.scatter(x, y, s=22, marker="s", color=[kinds[k] for k in d["유형"]], linewidths=0)
        map_axes(ax, "LISA 지도")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in kinds.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Moran's I: 가까운 칸끼리 값이 비슷한가", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_fig(fig, "stat8_05_moran_lisa.png")


# ============================================================================ 실행
def main():
    pts = load_points()
    print(f"좌표가 있는 종결 민원 {len(pts):,}건, 겹친 좌표 비율 {pts.duplicated(['latitude', 'longitude']).mean():.0%}")

    print("[1] KDE 핫스팟")
    dens, land, bw_compare, kde_map, conc, peaks = kde_analysis(pts)
    save_table(rnd(kde_map, 5).round({"밀도(건/km²)": 1, "입지계수(LQ)": 3}), "stat8_kde_map.csv")
    save_table(conc.round(3), "stat8_kde_concentration.csv")
    save_table(rnd(peaks, 5).round({"밀도(건/km²)": 0, "시 평균 대비 배율": 1}), "stat8_kde_peaks.csv")
    print(conc.round(2).to_string(index=False))
    print(peaks[peaks["순위"] <= 2].round(4).to_string(index=False))
    rep = repeat_locations(pts)
    save_table(rnd(rep, 2), "stat8_repeat_locations.csv")
    print(rep.head(6).round(2).to_string(index=False))

    print("[2] DBSCAN")
    loc, xy, kd_table, sweep, compare, top, info = dbscan_analysis(pts)
    save_table(kd_table.round(2), "stat8_dbscan_k_distance.csv")
    save_table(sweep.round(2), "stat8_dbscan_sweep.csv")
    save_table(compare.round(3), "stat8_cluster_method_compare.csv")
    save_table(rnd(top, 5).round({"동서 폭(m)": 0, "남북 폭(m)": 0, "민원 점유율(%)": 2}), "stat8_dbscan_clusters.csv")
    save_table(loc[["lat", "lon", "민원건수", "dbscan", "hdbscan", "kmeans"]], "stat8_dbscan_points.csv")
    pd.DataFrame([{"대상 유형": DBSCAN_TYPE, "민원 건수": info["n"], "신고 위치 수": info["n_loc"], "min_samples": MIN_SAMPLES,
                   "k-거리 무릎(m)": round(info["knee"], 1), "선택한 eps(m)": info["eps"], "K-means k": info["k"]}]
                 ).pipe(save_table, "stat8_dbscan_info.csv")
    print(sweep.pivot(index="eps(m)", columns="min_samples", values="군집 수").to_string())
    print(compare.round(2).to_string(index=False))
    print(top.head(6)[["군집", "신고 위치 수", "민원 건수", "자치구", "중심 lat", "중심 lon", "민원 점유율(%)"]].round(4).to_string(index=False))

    print("[3] Moran's I")
    cells, glob, lisa, maup = moran_analysis(pts)
    save_table(glob.round(4), "stat8_moran_global.csv")
    save_table(rnd(lisa, 4).round({"lat": 5, "lon": 5}), "stat8_moran_lisa.csv")
    save_table(maup.round(3), "stat8_moran_maup.csv")
    print(glob.round(3).to_string(index=False))
    print(maup.round(3).to_string(index=False))
    counts = lisa.groupby(["변수", "유형"]).size().unstack(fill_value=0)
    save_table(counts.reset_index(), "stat8_moran_lisa_counts.csv")
    print(counts.to_string())

    print("[4] 그래프")
    figures(setup_style(), dens, land, bw_compare, loc, xy, kd_table, compare, info, lisa)


if __name__ == "__main__":
    main()
