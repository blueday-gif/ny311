"""9단계(추가 분석 5): PCA 와 K-means - 동네를 민원 구성으로 요약하고 유형으로 묶기.

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력)
출력:  outputs/tables/stat9_*.csv, outputs/figures/stat9_*.png
실행:  프로젝트 루트에서  python analysis/stat_09_cluster_pca.py

======================================================================================
8단계와 무엇이 다른가: '어디에'가 아니라 '어떤 동네인가'
======================================================================================
8단계(KDE, DBSCAN, Moran's I)는 지도 위의 '위치'를 분석했다. 이번에는 위치를 잠시 잊고, 1km 격자 칸 하나하나를
"민원 구성이 이러저러한 동네"라는 프로필로 본다. 칸마다 변수 11개(분류별 비중 9개 + 야간 비중 + 주말 비중)가 있다.

  PCA     질문: 변수 11개를 몇 개의 축으로 줄여 말할 수 있는가      결과: 새 축(주성분)과 각 칸의 좌표
  K-means 질문: 프로필이 비슷한 칸끼리 묶으면 몇 가지 유형이 나오는가  결과: 칸마다 군집 번호

  둘 다 '정답(레이블)이 없는' 자료에서 구조를 찾는 비지도 학습(unsupervised learning)이다.
  회귀처럼 "맞혔다/틀렸다"를 잴 정답이 없으므로, 결과가 얼마나 안정적이고 해석 가능한지로 평가한다.

  [둘의 관계] PCA 는 '변수'를 줄이고(열 방향 요약), K-means 는 '관측값'을 묶는다(행 방향 요약).
      경쟁하는 방법이 아니라 함께 쓰는 방법이다: PCA 로 축을 줄여 그림을 그리고, 그 위에 K-means 군집을 색칠한다.

  [8단계 DBSCAN 과의 차이] 8단계에서는 지도 좌표(2차원)에 군집화를 적용했다. 여기서는 변수 공간(11차원)에 적용한다.
      군집화는 "무엇을 좌표로 넣느냐"에 따라 전혀 다른 질문이 된다.
      좌표가 위치면 '붙어 있는 곳'을 찾고, 좌표가 프로필이면 '닮은 곳'을 찾는다.
      닮은 동네는 서로 멀리 떨어져 있을 수 있다 (예: 브롱크스의 주거지와 브루클린의 주거지).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from stat_08_spatial import queen_weights
from stat_common import (
    BOROUGH_KO, BOROUGHS, CATEGORICAL, CATEGORIES, GRID, INK, MUTED, OTHER_GRAY, SEED, SINGLE, ACCENT,
    assign_cells, cell_center, load_points, rnd, save_fig, save_table, setup_style, to_km,
)

CELL = 1.0                 # 격자 칸 한 변 (km)
MIN_N = 300                # 칸의 민원이 이보다 적으면 비중이 불안정해 제외
MAX_ONE_LOCATION = 0.5     # 한 주소가 칸 민원의 절반을 넘으면 '동네'가 아니라 '한 주소'의 프로필이므로 제외
K_RANGE = range(2, 9)
K_CANDIDATES = range(3, 8) # 최종 k 후보. k=2 는 표에는 싣지만 너무 거칠어 후보에서 뺀다 (분석자의 판단)
N_BOOT = 30                # 부트스트랩 안정성 반복 수
MIN_STABILITY = 0.80
SHARE_COLS = [f"share_{c}" for c in CATEGORIES]
FEATURES = SHARE_COLS + ["night_share", "weekend_share"]
LABELS = dict(zip(SHARE_COLS, CATEGORIES)) | {"night_share": "야간(22-05시)", "weekend_share": "주말"}


# ============================================================================ 분석 단위 만들기
def build_units(pts):
    d = assign_cells(pts, CELL)
    d["is_night"] = ((d["hour"] >= 22) | (d["hour"] < 6)).astype(int)
    d["adj"] = d["log_resolution"] - d.groupby("complaint_type")["log_resolution"].transform("mean")
    d["loc"] = d["latitude"].round(5).astype(str) + "," + d["longitude"].round(5).astype(str)
    g = d.groupby(["ix", "iy"])
    u = pd.DataFrame({
        "n": g.size(), "night_share": g["is_night"].mean(), "weekend_share": g["is_weekend"].mean(),
        "within_24h": g["within_24h"].mean(), "adj_log_res": g["adj"].mean(),
        "borough": g["borough"].agg(lambda s: s.mode().iloc[0]),
        "max_location_share": d.groupby(["ix", "iy", "loc"]).size().groupby(["ix", "iy"]).max() / g.size(),
    })
    shares = pd.crosstab([d["ix"], d["iy"]], d["category"], normalize="index").reindex(columns=CATEGORIES, fill_value=0)
    u = u.join(shares.add_prefix("share_")).reset_index()

    small = u["n"] < MIN_N
    dominated = ~small & (u["max_location_share"] > MAX_ONE_LOCATION)
    log = pd.DataFrame([
        ("민원이 1건 이상 있는 칸", len(u), int(u["n"].sum()), ""),
        (f"제외: 민원 {MIN_N}건 미만", int(small.sum()), int(u.loc[small, "n"].sum()), "비중 변수가 불안정"),
        (f"제외: 한 주소가 칸 민원의 {MAX_ONE_LOCATION:.0%} 초과", int(dominated.sum()), int(u.loc[dominated, "n"].sum()),
         "동네가 아니라 반복 신고 주소 하나의 프로필"),
        ("군집분석 대상", int((~small & ~dominated).sum()), int(u.loc[~small & ~dominated, "n"].sum()), ""),
    ], columns=["항목", "칸 수", "민원 건수", "비고"])
    u = u[~small & ~dominated].reset_index(drop=True)
    u["lat"], u["lon"] = cell_center(u["ix"].to_numpy(), u["iy"].to_numpy(), CELL)
    return u, log


# ============================================================================ (1) PCA
def run_pca(Z):
    """주성분 분석 (Principal Component Analysis, PCA).

    [왜 필요한가] 변수 11개는 서로 얽혀 있다. 예를 들어 주거 시설(HPD) 민원이 많은 동네는 주차 민원 비중이 낮다.
        변수들이 같이 움직인다면 11개를 따로 볼 필요 없이 "같이 움직이는 방향" 몇 개만 보면 된다.
        11차원은 그림으로 그릴 수 없지만 2~3차원으로 줄이면 산점도로 볼 수 있다는 실용적인 이유도 크다.

    [하는 일] 자료가 가장 넓게 퍼져 있는 방향을 찾아 첫 번째 축(PC1)으로 삼는다. 그다음, PC1 과 직각이면서
        남은 퍼짐이 가장 큰 방향을 PC2 로 삼는다. 이를 반복한다. 길쭉한 구름 모양의 점들을 가장 긴 방향,
        그다음 긴 방향 순으로 돌려 보는 것이라고 생각하면 된다. 새 축들은 서로 상관이 0 이다.

    [용어]
        주성분(PC):          새로 만든 축. 원래 변수들의 가중합이다.  PC1 = w1·x1 + w2·x2 + ... + w11·x11
        적재값(loading):     그 가중치 w. 어떤 변수가 그 축을 만드는 데 크게 기여했는지 보여 준다. 축의 '뜻'을 읽는 열쇠다.
                             부호가 반대인 변수들은 그 축에서 서로 맞서는 관계다.
        점수(score):         각 관측값(칸)을 새 축에서 본 좌표.
        설명 분산 비율:       전체 퍼짐(분산) 가운데 그 축이 담아낸 비율. PC1 이 40% 면 축 하나로 정보의 40% 를 요약한 것이다.
        고윳값(eigenvalue):  그 축 방향의 분산. 표준화한 자료에서는 변수 하나의 분산이 1 이므로,
                             고윳값 > 1 이면 "원래 변수 하나보다 많은 정보를 담은 축"이다 (카이저 기준).

    [몇 개의 축을 쓸 것인가] 정답은 없고 흔히 세 가지 기준을 함께 본다.
        누적 설명 분산 80% 이상 / 고윳값 1 초과(카이저 기준) / 스크리 그림에서 꺾이는 지점(팔꿈치)

    [반드시 표준화] PCA 는 분산이 큰 방향을 찾으므로, 단위가 큰 변수가 있으면 그 변수가 PC1 을 독차지한다.
        모든 변수를 평균 0, 표준편차 1 로 맞춘 뒤에 해야 한다 (상관행렬 기반 PCA).

    [한계]
        - 직선적인 관계만 잡는다. 휘어진 관계는 놓친다.
        - 축의 뜻은 사람이 적재값을 보고 붙이는 해석이다. 깔끔하게 해석되지 않는 축도 흔하다.
        - 분산이 크다고 중요한 것은 아니다. 분산이 작은 방향에 군집을 가르는 정보가 있을 수도 있다.
        - 비중 자료의 함정: 분류별 비중 9개는 합이 항상 1 이다. 하나가 늘면 나머지가 반드시 줄어 '기계적인'
          음의 상관이 생기고, 9개 중 하나는 나머지로 완전히 결정되므로 마지막 고윳값 하나는 0 에 가깝게 나온다.
          이런 자료를 구성 자료(compositional data)라고 하며, 엄밀하게는 로그비 변환(CLR)을 한 뒤 분석한다.
          아래 (3)에서 CLR 변환을 해도 군집 결과가 비슷한지 확인한다.
    """
    pca = PCA().fit(Z)
    var = pd.DataFrame({
        "주성분": [f"PC{i + 1}" for i in range(len(FEATURES))],
        "고윳값": pca.explained_variance_, "설명 분산 비율(%)": pca.explained_variance_ratio_ * 100,
        "누적(%)": np.cumsum(pca.explained_variance_ratio_) * 100,
    })
    var["카이저 기준 (고윳값 > 1)"] = np.where(var["고윳값"] > 1, "채택", "")
    n80 = int((var["누적(%)"] < 80).sum()) + 1
    n_kaiser = int((var["고윳값"] > 1).sum())
    # 적재값: 고유벡터에 sqrt(고윳값)을 곱하면 '원래 변수와 주성분 점수의 상관계수'가 되어 -1~1 로 읽기 쉽다
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=[LABELS[f] for f in FEATURES],
                            columns=var["주성분"]).iloc[:, :max(4, n_kaiser)]
    scores = pca.transform(Z)
    return pca, var, loadings, scores, n80, n_kaiser


def name_axis(loadings, pc):
    """적재값이 큰 변수들로 축의 양 끝에 이름을 붙인다 (자동 요약일 뿐, 최종 해석은 사람이 한다)."""
    s = loadings[pc].sort_values()
    neg = [f"{i}({v:+.2f})" for i, v in s.head(2).items() if v < -0.3]
    pos = [f"{i}({v:+.2f})" for i, v in s.tail(2)[::-1].items() if v > 0.3]
    return f"(+) {', '.join(pos) or '뚜렷한 변수 없음'}  ↔  (-) {', '.join(neg) or '뚜렷한 변수 없음'}"


# ============================================================================ (2) K-means
def bootstrap_stability(Z, k, base_labels, rng):
    """부트스트랩 안정성: "자료가 조금 달라져도 같은 군집이 나오는가".

    칸들을 복원추출로 다시 뽑아(부트스트랩 표본) K-means 를 새로 돌리고, 그 결과로 전체 칸에 군집을 매긴 뒤
    원래 군집과 얼마나 일치하는지 ARI 로 잰다. 이를 여러 번 반복해 평균한다.

    [ARI, 조정 랜드 지수] 두 군집 결과가 얼마나 같은지 재는 값. 군집 번호가 서로 달라도 상관없이 '같은 짝끼리
        묶였는가'만 본다. 1 이면 완전히 같고, 0 이면 무작위로 묶은 것과 다를 바 없다 (우연 일치를 보정한 값이다).

    안정성이 낮은 k 는 표본이 바뀔 때마다 군집이 달라진다는 뜻이므로, 실루엣이 조금 높더라도 믿기 어렵다.
    """
    scores = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(Z), len(Z))
        km = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1_000_000))).fit(Z[idx])
        scores.append(adjusted_rand_score(base_labels, km.predict(Z)))
    return float(np.mean(scores))


def select_k(Z):
    """K-means 와 군집 수 k 고르기.

    [K-means 알고리즘]
        1. 중심점 k개를 놓는다 (k-means++ 방식으로 서로 멀리 떨어지게 고른다)
        2. 각 관측값을 가장 가까운 중심점의 군집에 배정한다
        3. 각 군집의 평균 위치로 중심점을 옮긴다
        4. 배정이 더 바뀌지 않을 때까지 2~3 을 반복한다
      줄이려는 값은 군집 내 제곱합(inertia, WCSS) = "각 점에서 자기 군집 중심까지 거리의 제곱"의 합이다.
      시작 위치에 따라 결과가 달라질 수 있어 여러 번(n_init) 돌려 가장 좋은 것을 쓴다.

    [k 는 자료가 정해 주지 않는다] k 를 늘리면 군집 내 제곱합은 무조건 줄어든다 (k = 관측값 수면 0 이 된다).
      그래서 "작을수록 좋다"로는 k 를 고를 수 없고, 여러 지표를 함께 본다.
        엘보우(팔꿈치): 제곱합이 줄어드는 폭이 갑자기 완만해지는 k. 그림으로 판단하므로 주관적이다.
        실루엣 계수:    -1~1. 자기 군집에 가깝고 남의 군집과 멀수록 크다. 0.5 이상 뚜렷, 0.25~0.5 약함, 그 아래는 구조가 희미.
        Calinski-Harabasz: 군집 간 분산 / 군집 내 분산. 클수록 좋다.
        Davies-Bouldin:    군집끼리 얼마나 겹치는가. 작을수록 좋다.
        부트스트랩 안정성: 자료가 달라져도 같은 군집이 나오는가. 1 에 가까울수록 좋다.
      지표들이 서로 다른 k 를 가리키는 일이 흔하다. 그때는 "해석할 수 있는가", "군집이 너무 작지 않은가"를 함께 본다.
      군집분석의 결과는 '발견'이라기보다 자료를 요약하는 하나의 '유용한 방법'이라는 점을 기억하자.

    [이 분석의 선택 규칙] k = 3~7 가운데 안정성이 0.80 이상이고 가장 작은 군집이 전체의 3% 이상인 것 중
      실루엣이 가장 큰 k. 조건을 만족하는 k 가 없으면 안정성이 가장 높은 k.
    """
    rng = np.random.default_rng(SEED)
    rows, fitted = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        fitted[k] = km
        sizes = np.bincount(km.labels_)
        rows.append({
            "k": k, "군집 내 제곱합": km.inertia_, "실루엣": silhouette_score(Z, km.labels_),
            "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_), "Davies-Bouldin": davies_bouldin_score(Z, km.labels_),
            "부트스트랩 안정성(ARI)": bootstrap_stability(Z, k, km.labels_, rng),
            "최소 군집 크기": int(sizes.min()), "최소 군집 비율(%)": sizes.min() / len(Z) * 100,
        })
    t = pd.DataFrame(rows)
    t["제곱합 감소율(%)"] = -t["군집 내 제곱합"].pct_change() * 100
    ok = t[t["k"].isin(K_CANDIDATES) & (t["부트스트랩 안정성(ARI)"] >= MIN_STABILITY) & (t["최소 군집 비율(%)"] >= 3)]
    if len(ok):
        k_best, why = int(ok.loc[ok["실루엣"].idxmax(), "k"]), "안정성 0.80 이상·최소 군집 3% 이상인 k 중 실루엣 최대"
    else:
        cand = t[t["k"].isin(K_CANDIDATES)]
        k_best, why = int(cand.loc[cand["부트스트랩 안정성(ARI)"].idxmax(), "k"]), "조건을 만족하는 k 가 없어 안정성 최대인 k"
    t["선택"] = np.where(t["k"] == k_best, "선택", "")
    return t, fitted[k_best], k_best, why


# ============================================================================ (3) 방법 비교
def clr(shares):
    """중심 로그비 변환 (centered log-ratio, CLR): 합이 1 인 비중 자료를 위한 변환.
    각 비중에 로그를 취한 뒤 그 칸의 로그 평균을 뺀다. 0 은 로그를 취할 수 없어 아주 작은 값을 더한다."""
    logp = np.log(shares + 0.005)
    return logp - logp.mean(axis=1, keepdims=True)


def compare_methods(u, Z, scores, n80, base, k):
    """같은 자료에 여러 군집화 방법을 적용해 결과가 얼마나 같은지 본다.

    방법이 달라도 비슷한 군집이 나오면 그 구조는 자료에 실제로 있는 것이고, 방법마다 제각각이면
    군집이 방법이 만들어 낸 인공물일 가능성이 크다. 그래서 한 가지 방법의 결과만 보고하지 않는다.

      K-means                중심에서 가까운 것끼리. 둥글고 크기가 비슷한 군집을 가정한다.
      PCA 후 K-means         분산이 작은 축(잡음일 가능성이 큰 축)을 버리고 군집화. 결과가 더 안정적일 수 있다.
      CLR 변환 후 K-means    비중 자료의 '합이 1' 문제를 풀고 군집화. 결과가 비슷하면 그 문제의 영향이 작다는 뜻이다.
      Ward 계층적 군집       가장 닮은 둘을 차례로 합쳐 나무(덴드로그램)를 만든다. k 를 나중에 정할 수 있고 결과가 항상 같다.
      가우시안 혼합(GMM)     군집마다 타원 모양과 크기가 달라도 된다. 각 관측값이 군집에 속할 '확률'을 준다 (소프트 군집).
      DBSCAN                밀집된 덩어리를 찾고 나머지는 잡음. 변수 공간에서는 점들이 고르게 퍼져 있는 경우가 많아
                            대개 "큰 덩어리 하나 + 잡음"이 되어 버린다. 지도 좌표(8단계)에서는 잘 통했던 것과 대조적이다.
                            방법의 좋고 나쁨은 자료의 모양에 달려 있다는 것을 보여 주는 예다.
    """
    def row(name, labels, note=""):
        lab = np.asarray(labels)
        mask = lab >= 0
        n_cl = len(np.unique(lab[mask]))
        return {"방법": name, "군집 수": n_cl, "잡음 칸 비율(%)": (~mask).mean() * 100,
                "실루엣 (표준화 변수 기준)": silhouette_score(Z[mask], lab[mask]) if n_cl >= 2 else np.nan,
                "기준 K-means 와의 일치도(ARI)": adjusted_rand_score(base, lab), "비고": note}

    Zc = StandardScaler().fit_transform(np.column_stack([clr(u[SHARE_COLS].to_numpy()), u[["night_share", "weekend_share"]]]))
    nn = NearestNeighbors(n_neighbors=6).fit(Z)
    kd = np.sort(nn.kneighbors(Z)[0][:, 5])
    xs = np.linspace(0, 1, len(kd))
    eps = float(kd[int(np.argmax(xs - (kd - kd[0]) / (kd[-1] - kd[0])))])
    rows = [
        row(f"K-means (k={k}, 기준)", base, "표준화한 변수 11개"),
        row(f"PCA {n80}개 축 → K-means", KMeans(k, n_init=20, random_state=SEED).fit_predict(scores[:, :n80]), "누적 분산 80% 까지의 축"),
        row("CLR 변환 → K-means", KMeans(k, n_init=20, random_state=SEED).fit_predict(Zc), "비중 9개를 로그비로 변환"),
        row("Ward 계층적 군집", AgglomerativeClustering(k, linkage="ward").fit_predict(Z), "같은 k 로 자름"),
        row("가우시안 혼합 (GMM)", GaussianMixture(k, covariance_type="diag", n_init=5, random_state=SEED).fit_predict(Z), "대각 공분산"),
        row(f"DBSCAN (eps={eps:.2f}, min_samples=5)", DBSCAN(eps=eps, min_samples=5).fit_predict(Z), "eps 는 5-거리 그래프의 무릎"),
    ]
    return pd.DataFrame(rows)


# ============================================================================ (4) 군집 해석
def profile(u, Z):
    """군집 프로필: 군집마다 변수 평균을 구하고, 전체 평균에서 표준편차 몇 배 떨어졌는지(z)로 특징을 읽는다.
    이름은 z 가 큰 변수 두 개로 자동으로 붙인다. 군집분석에서 가장 중요한 단계는 이 해석이다."""
    zdf = pd.DataFrame(Z, columns=[LABELS[f] for f in FEATURES])
    zmean = zdf.groupby(u["cluster"].to_numpy()).mean()
    names = {}
    for c, r in zmean.iterrows():
        top = r.sort_values(ascending=False)
        feats = [i for i, v in top.head(2).items() if v > 0.4] or [top.index[0]]
        names[c] = " · ".join(feats) + " 두드러짐"
    raw = u.groupby("cluster")[FEATURES].mean() * 100
    raw.columns = [f"{LABELS[f]} (%)" for f in FEATURES]
    g = u.groupby("cluster")
    out = pd.DataFrame({
        "군집": list(names), "이름": list(names.values()), "칸 수": g.size().to_numpy(), "민원 건수": g["n"].sum().to_numpy(),
        "칸당 민원 중앙값": g["n"].median().to_numpy(), "24시간 내 처리율(%)": (g["within_24h"].mean() * 100).to_numpy(),
        "유형 보정 로그 처리시간": g["adj_log_res"].mean().to_numpy(),
    })
    boro = pd.crosstab(u["cluster"], u["borough"], normalize="index").reindex(columns=BOROUGHS, fill_value=0) * 100
    boro.columns = [f"{BOROUGH_KO[b]} 칸 비율(%)" for b in boro.columns]
    out = pd.concat([out, raw.reset_index(drop=True), boro.reset_index(drop=True)], axis=1)
    zt = zmean.reset_index().rename(columns={"index": "군집"})
    zt.insert(1, "이름", zt["군집"].map(names))
    return out, zt, names


def spatial_coherence(u):
    """K-means 에는 위치를 넣지 않았다. 그런데도 같은 군집의 칸들이 지도에서 붙어 있는가?

    인접한 칸 쌍 가운데 두 칸이 같은 군집인 비율을 구하고, 군집 번호를 무작위로 섞었을 때의 비율과 비교한다
    (조인 카운트 통계의 단순한 형태, 순열 999번). 관측 비율이 훨씬 높으면 "닮은 동네는 붙어 있다"는 뜻이다.
    8단계 Moran's I 의 범주형 판이라고 보면 된다.
    """
    W, has_nb, _ = queen_weights(u)
    A = (W > 0).tocoo()
    lab = u["cluster"].to_numpy()
    obs = (lab[A.row] == lab[A.col]).mean()
    rng = np.random.default_rng(SEED)
    sims = np.array([(p[A.row] == p[A.col]).mean() for p in (rng.permutation(lab) for _ in range(999))])
    return pd.DataFrame([{
        "인접한 칸 쌍 수": A.nnz // 2, "같은 군집인 쌍 비율(%)": obs * 100, "무작위일 때 기대 비율(%)": sims.mean() * 100,
        "배율": obs / sims.mean(), "유사 p 값": ((sims >= obs).sum() + 1) / 1000,
    }])


# ============================================================================ 그래프
def figures(plt, u, var, loadings, scores, ksel, k_best, zt, names):
    colors = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(sorted(names))}

    # 스크리 그림
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var["설명 분산 비율(%)"], color=SINGLE, width=0.62)
    ax2 = ax.twinx()
    ax2.plot(x, var["누적(%)"], color=ACCENT, marker="o", ms=4, lw=1.5)
    ax2.axhline(80, color=OTHER_GRAY, lw=1, ls="--")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("누적 (%)", color=ACCENT)
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(x, var["주성분"], fontsize=8)
    ax.set_ylabel("설명 분산 비율 (%)")
    ax.set_title("스크리 그림: 주성분별 설명 분산 (점선 = 누적 80%)")
    save_fig(fig, "stat9_01_pca_scree.png")

    # 적재값 히트맵
    fig, ax = plt.subplots(figsize=(1.5 * loadings.shape[1] + 3, 5))
    ax.imshow(loadings.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            v = loadings.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(loadings.shape[1]), [f"{c}\n{var.loc[j, '설명 분산 비율(%)']:.0f}%" for j, c in enumerate(loadings.columns)])
    ax.set_yticks(range(loadings.shape[0]), loadings.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("적재값: 각 주성분과 원래 변수의 상관 (빨강 +, 파랑 -)")
    save_fig(fig, "stat9_02_pca_loadings.png")

    # k 선택
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, col, title in zip(axes, ["군집 내 제곱합", "실루엣", "부트스트랩 안정성(ARI)"],
                              ["엘보우: 군집 내 제곱합", "실루엣 (클수록 좋음)", "부트스트랩 안정성 (클수록 좋음)"]):
        ax.plot(ksel["k"], ksel[col], color=SINGLE, marker="o", ms=5, lw=1.6)
        ax.scatter([k_best], ksel.loc[ksel["k"] == k_best, col], color=ACCENT, s=70, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("군집 수 k")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[2].axhline(MIN_STABILITY, color=OTHER_GRAY, lw=1, ls="--")
    fig.suptitle(f"군집 수 고르기 (주황 점 = 선택한 k = {k_best})", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "stat9_03_k_selection.png")

    # PCA 평면 위의 군집 + 지도
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for c in sorted(names):
        m = (u["cluster"] == c).to_numpy()
        axes[0].scatter(scores[m, 0], scores[m, 1], s=14, color=colors[c], alpha=0.8, linewidths=0, label=f"{c}: {names[c]}")
        x, y = to_km(u.loc[m, "lat"].to_numpy(), u.loc[m, "lon"].to_numpy())
        axes[1].scatter(x, y, s=24, marker="s", color=colors[c], linewidths=0)
    top = loadings.iloc[:, :2]
    scale = np.abs(scores[:, :2]).max() * 0.8
    for name, (a, b) in top.iterrows():
        if a * a + b * b > 0.3:                                    # 두 축에 뚜렷하게 실리는 변수만 화살표로 그린다
            axes[0].annotate("", xy=(a * scale, b * scale), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
            axes[0].text(a * scale * 1.06, b * scale * 1.06, name, fontsize=8.5, color=INK, ha="center")
    axes[0].axhline(0, color=GRID, lw=0.8)
    axes[0].axvline(0, color=GRID, lw=0.8)
    axes[0].set_xlabel(f"PC1 ({var.loc[0, '설명 분산 비율(%)']:.0f}%)")
    axes[0].set_ylabel(f"PC2 ({var.loc[1, '설명 분산 비율(%)']:.0f}%)")
    axes[0].set_title("PCA 평면에서 본 군집 (화살표 = 변수 방향)", fontsize=11)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for s in axes[1].spines.values():
        s.set_visible(False)
    axes[1].set_title("같은 군집을 지도에 표시 (위치는 군집화에 쓰지 않았다)", fontsize=11)
    save_fig(fig, "stat9_04_clusters_pca_map.png")

    # 프로필 히트맵
    zm = zt.set_index("군집").drop(columns="이름")
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(zm) + 2))
    ax.imshow(zm.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(zm.shape[0]):
        for j in range(zm.shape[1]):
            v = zm.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 1.2 else INK)
    ax.set_xticks(range(zm.shape[1]), zm.columns, rotation=30, ha="right")
    ax.set_yticks(range(zm.shape[0]), [f"{c}: {names[c]}" for c in zm.index])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("군집 프로필: 전체 평균에서 표준편차 몇 배 떨어져 있는가")
    save_fig(fig, "stat9_05_cluster_profiles.png")


# ============================================================================ 실행
def main():
    pts = load_points()
    print("[1] 분석 단위: 1km 격자 칸")
    u, log = build_units(pts)
    save_table(log, "stat9_input_log.csv")
    print(log.to_string(index=False))

    # 표준화: 변수마다 평균 0, 표준편차 1. PCA 와 K-means 는 모두 '거리'에 바탕을 두므로 단위를 맞추지 않으면
    # 값의 범위가 큰 변수가 결과를 좌우한다 (3단계에서 배운 표준화가 여기서 쓰인다).
    Z = StandardScaler().fit_transform(u[FEATURES])
    corr = pd.DataFrame(np.corrcoef(Z.T), index=[LABELS[f] for f in FEATURES], columns=[LABELS[f] for f in FEATURES])
    save_table(corr.round(3).rename_axis("변수").reset_index(), "stat9_feature_corr.csv")
    strong = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=np.abs, ascending=False).head(4)
    print("  상관이 강한 변수 쌍: " + ", ".join(f"{a}~{b} {v:+.2f}" for (a, b), v in strong.items()))

    print("[2] PCA")
    pca, var, loadings, scores, n80, n_kaiser = run_pca(Z)
    save_table(var.round(3), "stat9_pca_variance.csv")
    save_table(loadings.round(3).rename_axis("변수").reset_index(), "stat9_pca_loadings.csv")
    axis_names = pd.DataFrame([{"주성분": pc, "설명 분산 비율(%)": var.loc[i, "설명 분산 비율(%)"], "축의 양 끝 (적재값)": name_axis(loadings, pc)}
                               for i, pc in enumerate(loadings.columns)])
    save_table(axis_names.round(2), "stat9_pca_axis_names.csv")
    print(var.head(6).round(2).to_string(index=False))
    print(f"  누적 80% 까지 {n80}개 축, 카이저 기준 {n_kaiser}개 축, 마지막 고윳값 {var['고윳값'].iloc[-1]:.4f} (비중의 합이 1 이라 0 에 가깝다)")
    for _, r in axis_names.iterrows():
        print(f"  {r['주성분']}: {r['축의 양 끝 (적재값)']}")

    print("[3] K-means: k 고르기")
    ksel, km, k_best, why = select_k(Z)
    save_table(ksel.round(3), "stat9_k_selection.csv")
    print(ksel.round(3).to_string(index=False))
    print(f"  선택: k = {k_best} ({why})")
    # 군집 번호를 큰 군집부터 1, 2, 3 ... 으로 다시 매긴다 (K-means 가 주는 번호 자체에는 뜻이 없다)
    order = pd.Series(km.labels_).value_counts().index
    u["cluster"] = pd.Series(km.labels_).map({old: new + 1 for new, old in enumerate(order)}).to_numpy()

    print("[4] 방법 비교")
    cmp = compare_methods(u, Z, scores, n80, u["cluster"].to_numpy(), k_best)
    save_table(cmp.round(3), "stat9_method_compare.csv")
    print(cmp.drop(columns="비고").round(3).to_string(index=False))

    print("[5] 군집 해석")
    prof, zt, names = profile(u, Z)
    save_table(prof.round(2), "stat9_cluster_profiles.csv")
    save_table(zt.round(3), "stat9_cluster_profiles_z.csv")
    coh = spatial_coherence(u)
    save_table(coh.round(4), "stat9_spatial_coherence.csv")
    print(prof.iloc[:, :7].round(2).to_string(index=False))
    print(coh.round(3).to_string(index=False))

    u["cluster_name"] = u["cluster"].map(names)
    for i in range(3):
        u[f"pc{i + 1}"] = scores[:, i]
    out = u[["ix", "iy", "lat", "lon", "borough", "n", "cluster", "cluster_name", "pc1", "pc2", "pc3",
             "within_24h", "adj_log_res", *FEATURES]]
    save_table(rnd(out, 4).round({"lat": 5, "lon": 5}), "stat9_units.csv")

    print("[6] 그래프")
    figures(setup_style(), u, var, loadings, scores, ksel, k_best, zt, names)


if __name__ == "__main__":
    main()
