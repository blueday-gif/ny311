"""
[ML 15단계] 군집: 비슷한 동네끼리 묶으면

핵심 질문 : 정답 없이, 민원 구성이 비슷한 동네끼리 묶으면 몇 가지 유형이 나올까?
배우는 개념: 비지도학습, K-means, 표준화, 엘보 방법, 실루엣 점수
사용하는 열: lat, lon(격자 만들기), problem(동네 프로필)
실행 방법 : py -3.12 analysis/ml_15_clustering.py
기존 앱 참고: 12 군집·PCA 탭 (거기서는 PCA·부트스트랩까지, 여기서는 K-means 기본만)

[용어] 비지도학습(unsupervised learning)
  정답(타깃) 없이 데이터 안의 구조를 찾는 방법. 지금까지(ml_05~14)는 정답이 있는 '지도학습'이었다.
  예) 손님의 구매 기록만 보고 '비슷한 손님 무리'를 찾는 것.

[용어] K-means
  점들을 K개 무리로 나눈다. ① 중심 K개를 아무 데나 놓고 ② 각 점을 가장 가까운 중심에 붙이고 ③ 중심을 무리의 평균으로 옮기기를 반복한다.

[용어] 표준화(standardization)
  각 열에서 평균을 빼고 표준편차로 나눠 '평균 0, 표준편차 1'로 맞추는 것. 단위가 큰 열이 거리 계산을 독차지하지 않게 한다.

[용어] 실루엣 점수(silhouette score)
  점이 '자기 무리와는 가깝고 다른 무리와는 먼가'를 -1~1 로 잰 것. 1에 가까울수록 무리가 잘 나뉘었다.
"""

# K-means
from sklearn.cluster import KMeans

# 실루엣 점수
from sklearn.metrics import silhouette_score

# 표준화 도구
from sklearn.preprocessing import StandardScaler

# 공통 도구
from common import SEED, ExclusionLog, load_data, np, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# ─────────────────────────────────────────────
# 1. 동네 만들기: 약 1km 격자
# ─────────────────────────────────────────────
print_header("1. 약 1km 격자로 동네 나누기")

# 데이터 불러오기
df = load_data()

# 제외 로그
log = ExclusionLog("ml15", len(df))

# 좌표가 없는 민원은 격자에 넣을 수 없다
no_xy = df["lat"].isna() | df["lon"].isna()

# 기록
log.add("위도·경도 없음", no_xy.sum())

# 좌표 있는 민원만
geo = df.loc[~no_xy, ["lat", "lon", "problem"]].copy()

# 위도 0.009도 ≈ 1km, 뉴욕 위도에서 경도 0.012도 ≈ 1km. 나눈 몫(정수)이 같은 민원은 같은 칸
geo["gx"] = np.floor(geo["lon"] / 0.012).astype(int)

# 세로 칸 번호
geo["gy"] = np.floor(geo["lat"] / 0.009).astype(int)

# 칸별 민원 수
cell_n = geo.groupby(["gx", "gy"]).size()

# 민원이 300건 미만인 칸은 비율이 불안정해서 뺀다 (예: 3건 중 1건 = 33%는 우연일 수 있다)
small = cell_n[cell_n < 300]

# 기록
log.add("민원 300건 미만 격자 칸의 민원 (비율이 불안정)", int(small.sum()))

# 로그 저장
log.save()

# 남길 칸
keep = cell_n[cell_n >= 300].index

# 출력
print(f"격자 칸 {len(cell_n):,}개 중 {len(keep):,}개 사용")

# ─────────────────────────────────────────────
# 2. 동네 프로필: 민원 유형 구성 비율
# ─────────────────────────────────────────────
print_header("2. 동네 프로필 (상위 10개 유형의 비율)")

# 전체에서 가장 많은 민원 유형 10개
top10 = geo["problem"].value_counts().head(10).index.astype(str)

# 유형 이름: 상위 10개가 아니면 '기타'
geo["ptype"] = geo["problem"].astype(str).where(geo["problem"].astype(str).isin(top10), "기타")

# 칸 × 유형 건수표
counts = geo.groupby(["gx", "gy", "ptype"]).size().unstack(fill_value=0)

# 사용할 칸만
counts = counts.loc[keep]

# 건수 → 비율 (행 합으로 나눈다). 민원이 많은 칸과 적은 칸을 공정하게 비교하려고 비율을 쓴다
profile = counts.div(counts.sum(axis=1), axis=0)

# 출력
print(profile.describe().loc[["mean", "std"]].round(3).T.to_string())
# 해석: 유형마다 동네 간 평균 비율과 흩어짐(표준편차)이 다르다. 표준편차가 큰 유형이 동네를 구분하는 데 쓸모가 크다.

# 표준화: 비율이 큰 유형(불법주차)이 거리 계산을 독차지하지 않게 한다
X = StandardScaler().fit_transform(profile.values)

# ─────────────────────────────────────────────
# 3. K 고르기: 엘보 방법과 실루엣 점수
# ─────────────────────────────────────────────
print_header("3. 무리 개수 K 고르기")

# [용어] 엘보 방법(elbow method): K를 늘려 가며 '무리 안 거리 합(inertia)'을 그린다.
#        줄어드는 속도가 갑자기 느려지는 '팔꿈치' 지점이 적당한 K 후보다.

# 결과 목록
rows = []

# K = 2 ~ 8
for k in range(2, 9):
    # n_init=10: 시작 위치를 10번 바꿔 가장 좋은 결과를 쓴다 (시작 위치 운에 덜 좌우되게)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    # 학습 + 무리 번호
    labels = km.fit_predict(X)
    # 기록
    rows.append({"K": k, "무리 안 거리 합(inertia)": round(km.inertia_, 1),
                 "실루엣 점수": round(silhouette_score(X, labels, random_state=SEED), 4)})

# 표로
ks = pd.DataFrame(rows)

# 출력
print(ks.to_string(index=False))
# 해석: inertia 는 K가 커지면 항상 줄어든다(무리가 많을수록 가까워지니까). 그래서 '얼마나 줄었나'의 꺾임을 본다.
#       실루엣 점수는 K=2 에서 0.325로 가장 높고, K가 늘수록 0.19까지 내려간다. 0.3 정도면 '무리가 아주 뚜렷하지는 않다'는 뜻이다.
#       K=2 는 '크게 두 종류'라는 거친 구분이다. 더 세밀한 유형이 필요하면 점수를 조금 양보하고 K=3~4 를 고를 수도 있다(사람의 판단).

# 저장
save_table(ks, "ml15_k_selection")

# 실루엣 점수가 가장 높은 K
best_k = int(ks.loc[ks["실루엣 점수"].idxmax(), "K"])

# 출력
print(f"선택한 K = {best_k}")

# 그림 두 칸
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 엘보 그래프
axes[0].plot(ks["K"], ks["무리 안 거리 합(inertia)"], marker="o")

# 제목
axes[0].set_title("엘보 방법")

# x축
axes[0].set_xlabel("K")

# 실루엣 그래프
axes[1].plot(ks["K"], ks["실루엣 점수"], marker="o", color="#55A868")

# 선택한 K 표시
axes[1].axvline(best_k, color="gray", ls="--")

# 제목
axes[1].set_title("실루엣 점수 (높을수록 잘 나뉨)")

# x축
axes[1].set_xlabel("K")

# 저장
save_fig(fig, "ml15_k_selection")

# ─────────────────────────────────────────────
# 4. 최종 군집과 해석
# ─────────────────────────────────────────────
print_header(f"4. K = {best_k} 군집 결과")

# 최종 모델
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED)

# 무리 번호
profile["cluster"] = km.fit_predict(X)

# 무리별 평균 프로필 (비율, %) + 칸 수
summary = (profile.groupby("cluster").mean() * 100).round(1)

# 칸 수 추가
summary["격자 칸 수"] = profile["cluster"].value_counts().sort_index()

# 출력
print(summary.T.to_string())
# 해석: 무리 0(205칸)은 난방·온수 12.9%, 주거 소음 12.4%, 비위생 5.3%, 배관 3.2%가 높다 → '아파트 주거 민원' 동네.
#       무리 1(514칸)은 불법주차 20.1%, 방치 차량 3.9%, 도로 상태 3.9%가 높다 → '도로·차량 민원' 동네.
#       이름 붙이기는 사람의 해석이다. 군집은 '비슷한 것끼리 묶었다'까지만 말해 준다.

# 저장
save_table(summary, "ml15_cluster_profile", index=True)

# 지도: 격자 칸 중심점을 무리 색으로
fig, ax = plt.subplots(figsize=(7, 7))

# 칸 중심 좌표 (칸 번호 + 0.5) × 칸 크기
cx = (profile.index.get_level_values("gx") + 0.5) * 0.012

# 위도
cy = (profile.index.get_level_values("gy") + 0.5) * 0.009

# 산점도
sc = ax.scatter(cx, cy, c=profile["cluster"], cmap="tab10", s=18, marker="s")

# 범례 (무리 번호)
ax.legend(*sc.legend_elements(), title="무리", loc="upper left")

# 축 이름
ax.set_xlabel("경도")

# 축 이름
ax.set_ylabel("위도")

# 가로세로 비율을 실제 거리에 가깝게 (뉴욕 위도에서 경도 1도는 위도 1도의 약 0.76배)
ax.set_aspect(1 / 0.76)

# 제목
ax.set_title(f"민원 구성으로 나눈 동네 유형 (K = {best_k}, 약 1km 격자)")

# 저장
save_fig(fig, "ml15_cluster_map")
# 해석: 위치 정보를 하나도 쓰지 않았는데도 같은 무리가 지도에서 뭉쳐 보인다면, 가까운 동네끼리 민원 구성이 비슷하다는 뜻이다.
