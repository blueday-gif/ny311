"""
[EDA 4단계] 범주·시간 패턴, 변수 관계, 집단 비교

핵심 질문 : 언제, 어디서, 어떤 민원이 많은가? 변수끼리는 어떤 관계인가? 자치구끼리 차이가 있나?
배우는 개념: 빈도와 비율, 주기성, 피어슨·스피어만 상관, 상관≠인과, 집단 크기 차이, 다중 비교
사용하는 열: borough, problem, hour, weekday, month, date, hours
실행 방법 : py -3.12 analysis/eda_04_patterns.py
기존 앱 참고: 1 둘러보기, 6 검정·회귀, 8 기관 비교, 9 시계열 탭

[용어] 빈도와 비율
  빈도 = 몇 건인가, 비율 = 전체 중 몇 %인가. 집단 크기가 다르면 빈도보다 비율로 비교해야 공정하다.
  예) A구 100건 중 50건, B구 1000건 중 100건 → 빈도는 B가 많지만 비율은 A(50%)가 B(10%)보다 높다.
"""

# scipy.stats: 상관계수와 검정 함수
from scipy import stats

# 공통 도구
from common import ExclusionLog, load_data, np, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 데이터 불러오기
df = load_data()

# 전체 건수 (비율 계산의 분모)
N = len(df)

# ─────────────────────────────────────────────
# 1. 자치구별 빈도와 비율
# ─────────────────────────────────────────────
print_header("1. 자치구(borough)별 빈도와 비율")

# value_counts(): 값마다 몇 건인지 센다
borough = df["borough"].value_counts().rename("건수").to_frame()

# 비율(%) 열 추가
borough["비율(%)"] = (borough["건수"] / N * 100).round(2)

# 출력
print(borough.to_string())
# 해석: 브루클린이 가장 많고 스태튼아일랜드가 가장 적다. 하지만 이것은 '인구 차이'를 그대로 반영한 것일 수 있다.
#       이 데이터에는 인구가 없어서 '주민 1인당 민원'은 계산할 수 없다 (한계로 기록).

# 저장
save_table(borough, "eda04_borough_counts", index=True)

# ─────────────────────────────────────────────
# 2. 민원 유형별 빈도와 비율 (상위 15개)
# ─────────────────────────────────────────────
print_header("2. 민원 유형(problem) 상위 15개")

# 유형별 건수
problem = df["problem"].value_counts().rename("건수").to_frame()

# 비율
problem["비율(%)"] = (problem["건수"] / N * 100).round(2)

# 누적 비율: 위에서부터 더해 나간 비율. "상위 몇 개 유형이 전체의 몇 %인가"를 알려 준다
problem["누적 비율(%)"] = problem["비율(%)"].cumsum().round(2)

# 상위 15개 출력
print(problem.head(15).to_string())
# 해석: 유형은 197개나 되지만 상위 15개가 전체의 약 3분의 2를 차지한다. 소수 유형에 민원이 몰려 있다.

# 전체 유형 표 저장
save_table(problem, "eda04_problem_counts", index=True)

# 가로 막대그래프
fig, ax = plt.subplots(figsize=(8, 6))

# 상위 15개를 아래→위로 그리기 위해 순서를 뒤집는다
top15 = problem.head(15).iloc[::-1]

# 막대
ax.barh(top15.index.astype(str), top15["비율(%)"], color="#4C72B0")

# x축
ax.set_xlabel("전체 민원 대비 비율 (%)")

# 제목
ax.set_title("민원 유형 상위 15개")

# 저장
save_fig(fig, "eda04_problem_top15")

# ─────────────────────────────────────────────
# 3. 시간 패턴: 시간대, 요일, 월
# ─────────────────────────────────────────────
print_header("3. 시간 패턴")

# [용어] 주기성(periodicity): 일정한 간격으로 같은 모양이 반복되는 성질.
#        예) 매일 새벽에 줄고 낮에 늘어나는 패턴 = 하루 주기, 주말마다 줄어드는 패턴 = 1주 주기.

# 시간대별 건수 (0~23시)
by_hour = df.groupby("hour").size().rename("건수")

# 요일별 건수. 첫날·마지막 날 같은 부분 일자의 영향은 52주에 나눠져 작다
by_wd = df.groupby("weekday").size().rename("건수")

# 요일 번호에 이름을 붙인다. 0=월 ... 6=일
by_wd.index = ["월", "화", "수", "목", "금", "토", "일"]

# 월별 건수. 연-월 순서로 정렬하려고 date 에서 '연-월' 기간을 만든다
by_month = df.groupby(df["created"].dt.to_period("M")).size().rename("건수")

# 출력
print("[요일별]\n" + by_wd.to_string())
# 해석: 평일(특히 월요일)이 많고 주말이 적다. 관공서·업무 관련 민원이 평일에 몰리기 때문으로 보인다(추측, 검증 필요).

# 월별 출력
print("[월별]\n" + by_month.to_string())
# 해석: 월별 차이는 크지 않다(약 30만~35만 건). 첫 달(2025-11)은 1일 밤부터 시작해 하루 가까이 덜 들어 있다.

# 세 그래프를 한 그림에
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 시간대 선그래프
axes[0].plot(by_hour.index, by_hour.values, marker="o")

# 제목
axes[0].set_title("시간대별 접수 건수")

# x축
axes[0].set_xlabel("접수 시각 (시)")

# 요일 막대
axes[1].bar(by_wd.index, by_wd.values, color="#55A868")

# 제목
axes[1].set_title("요일별 접수 건수")

# 월 막대. 기간 값을 글자로 바꿔 x축에 쓴다
axes[2].bar(by_month.index.astype(str), by_month.values, color="#C44E52")

# 제목
axes[2].set_title("월별 접수 건수")

# 월 글자가 겹치지 않게 45도 기울인다
axes[2].tick_params(axis="x", rotation=45)

# y축 이름은 모두 같다
for ax in axes:
    # y축
    ax.set_ylabel("건수")

# 저장
save_fig(fig, "eda04_time_patterns")
# 해석: 시간대 그래프는 새벽 3~4시에 가장 낮고 오전 10시 무렵 가장 높다 = 사람의 활동 시간을 따라가는 하루 주기.

# 시간 패턴 표 저장
save_table(by_hour.to_frame(), "eda04_by_hour", index=True)

# 요일 저장
save_table(by_wd.to_frame(), "eda04_by_weekday", index=True)

# 월 저장
save_table(by_month.to_frame(), "eda04_by_month", index=True)

# ─────────────────────────────────────────────
# 4. 변수 관계: 상관계수 (피어슨 vs 스피어만)
# ─────────────────────────────────────────────
print_header("4. 변수 관계: 상관")

# [용어] 피어슨 상관계수(Pearson r)
#   두 숫자가 '직선' 관계로 함께 움직이는 정도. -1 ~ 1. 극단값 하나에 크게 흔들린다.
#   예) 키와 몸무게: 키가 크면 몸무게도 대체로 커서 r 이 양수.
# [용어] 스피어만 상관계수(Spearman ρ)
#   값을 순위로 바꾼 뒤 구한 상관. '한쪽이 커지면 다른 쪽도 커지는가(방향)'만 본다. 극단값에 강하다.
#   예) 1,2,3,4 와 1,4,9,1000 은 직선은 아니지만 순서는 완전히 같아서 ρ = 1.

# 날짜별로 (1) 접수 건수와 (2) 처리 시간 중앙값을 만든다. 두 숫자의 관계를 보기 위해서다
daily = df.groupby("date").agg(접수건수=("unique_key", "size"), 처리시간_중앙값=("hours", "median"))

# 제외 로그
log = ExclusionLog("eda04_daily", len(daily))

# 부분 일자 2일 제외
log.add("첫날·마지막 날 (부분 일자)", 2)

# 첫날·마지막 날 제거
daily = daily.iloc[1:-1]

# 로그 저장
log.save()

# 피어슨 상관
r_p, p_p = stats.pearsonr(daily["접수건수"], daily["처리시간_중앙값"])

# 스피어만 상관
r_s, p_s = stats.spearmanr(daily["접수건수"], daily["처리시간_중앙값"])

# 출력
print(f"일별 접수 건수 vs 처리 시간 중앙값: 피어슨 r = {r_p:.3f}, 스피어만 ρ = {r_s:.3f}")
# 해석: 피어슨 0.50, 스피어만 0.34. 둘 다 양수라 '민원이 많은 날 처리 시간 중앙값도 길다'는 방향은 같다.
#       피어슨이 더 큰 것은 민원이 폭증한 소수의 극단적인 날이 직선 관계를 강하게 끌어올렸기 때문이다. 순위로 보면 관계는 중간 정도다.

# 상관계수 표 저장
save_table(pd.DataFrame([
    {"방법": "피어슨", "상관계수": round(r_p, 4), "p값": p_p},
    {"방법": "스피어만", "상관계수": round(r_s, 4), "p값": p_s},
]), "eda04_correlation")

# 주의 문구를 출력한다
print("주의: 상관관계는 인과관계가 아니다. '민원이 많은 날 처리가 느리다'가 보여도,")
print("      두 값을 함께 움직이게 하는 제3의 원인(예: 폭설, 공휴일, 민원 유형 구성 변화)이 있을 수 있다.")

# 산점도
fig, ax = plt.subplots(figsize=(6, 5))

# 점 하나 = 하루
ax.scatter(daily["접수건수"], daily["처리시간_중앙값"], s=12, alpha=0.6)

# x축
ax.set_xlabel("하루 접수 건수")

# y축
ax.set_ylabel("그날 접수 민원의 처리 시간 중앙값 (시간)")

# 제목에 두 상관계수를 함께 적는다
ax.set_title(f"하루 단위 관계 (피어슨 {r_p:.2f}, 스피어만 {r_s:.2f})")

# 저장
save_fig(fig, "eda04_scatter_daily")

# ─────────────────────────────────────────────
# 5. 집단 비교: 자치구별 처리 시간
# ─────────────────────────────────────────────
print_header("5. 집단 비교: 자치구별 처리 시간")

# 비교 대상: 처리 시간이 있고 음수가 아닌 민원, 자치구가 'Unspecified'가 아닌 민원
grp = df[(df["hours"] >= 0) & (df["borough"] != "Unspecified")]

# 제외 로그
log_g = ExclusionLog("eda04_group", len(df))

# 처리 시간 계산 불가/오류 제외 건수
log_g.add("처리 시간 없음 또는 < 0", (~(df["hours"] >= 0)).sum())

# 자치구 미상 제외 건수
log_g.add("borough = Unspecified (자치구 미상)", ((df["hours"] >= 0) & (df["borough"] == "Unspecified")).sum())

# 로그 저장
log_g.save()

# 자치구별 건수, 중앙값, 평균, 24시간 안 처리 비율
g = grp.groupby("borough", observed=True)["hours"].agg(
    건수="size", 중앙값="median", 평균="mean",
    처리율_24h=lambda s: (s <= 24).mean() * 100,
).round(2).sort_values("중앙값")

# 출력
print(g.to_string())
# 해석: 중앙값은 퀸스 4.9시간 ~ 스태튼아일랜드 13.1시간으로 차이가 난다. 그런데 평균 순서는 중앙값 순서와 다르다(평균은 맨해튼이 가장 길다).
#       건수도 최대 7.6배 다르다(브루클린 vs 스태튼아일랜드). 건수가 적은 집단의 값은 덜 안정적이다.
#       또 자치구마다 '민원 유형 구성'이 달라서, 차이가 처리 속도 때문인지 유형 구성 때문인지 이 표만으로는 알 수 없다.

# 저장
save_table(g, "eda04_borough_hours", index=True)

# [용어] Kruskal-Wallis 검정: 세 집단 이상의 '분포 위치(대략 중앙값)'가 모두 같은지 검정한다.
#        정규분포를 가정하지 않아서 치우친 처리 시간에 알맞다.
# [용어] p값: '정말 차이가 없다면' 지금만큼 큰 차이가 우연히 나올 확률. 작을수록 우연으로 보기 어렵다.

# 자치구별 처리 시간 목록을 만든다 (검정 함수에 넣기 위해)
groups = [s.values for _, s in grp.groupby("borough", observed=True)["hours"]]

# 검정 실행
H, p = stats.kruskal(*groups)

# 출력
print(f"Kruskal-Wallis H = {H:,.1f}, p = {p:.3g}")
# 해석: 표본이 수백만 건이면 아주 작은 차이도 p값이 0에 가깝게 나온다. 'p가 작다 = 차이가 크다'가 아니다.
#       실제로 의미 있는 차이인지는 위 표의 중앙값 차이(몇 시간인가)로 판단해야 한다.

# [용어] 다중 비교 문제: 비교를 여러 번 하면 우연히 '유의한' 결과가 나올 확률이 커진다.
#   예) 5개 자치구를 둘씩 비교하면 10번. 각각 5% 확률로 우연히 유의하면, 하나 이상 걸릴 확률은 약 40%다.
#   해결: Bonferroni 보정 = 기준 0.05 를 비교 횟수로 나눠 더 엄격하게 본다 (0.05 / 10 = 0.005).

# 자치구 쌍 비교 횟수
k = len(groups)

# 쌍의 개수 = k × (k-1) / 2
n_pairs = k * (k - 1) // 2

# 보정 기준
alpha_bonf = 0.05 / n_pairs

# 출력
print(f"자치구 {k}개 → 둘씩 비교 {n_pairs}번 → Bonferroni 보정 기준 p < {alpha_bonf:.4f}")
# 해석: 둘씩 비교할 때는 p < 0.05 가 아니라 이 보정 기준으로 판단해야 '우연히 하나 걸리는' 오류를 줄인다.

# 자치구별 상자그림 (로그 눈금, 이상치 점 생략)
fig, ax = plt.subplots(figsize=(8, 4))

# 자치구 이름 순서 = 표의 중앙값 순서
order = list(g.index)

# 각 자치구의 log1p(처리 시간)
data = [np.log1p(grp.loc[grp["borough"] == b, "hours"]) for b in order]

# 상자그림
ax.boxplot(data, tick_labels=[f"{b}\n(n={g.loc[b, '건수']:,})" for b in order], showfliers=False)

# y축
ax.set_ylabel("log(1 + 처리 시간)")

# 제목
ax.set_title("자치구별 처리 시간 (n = 집단 크기)")

# 저장
save_fig(fig, "eda04_borough_boxplot")
