"""
[EDA 2단계] 기술통계: 숫자 몇 개로 자료를 요약하기

핵심 질문 : 처리 시간과 하루 접수 건수를 숫자 몇 개로 요약하면? 평균만 보면 왜 위험한가?
배우는 개념: 평균, 중앙값, 표준편차, 사분위수, 절사평균
사용하는 열: hours(처리 시간, closed - created에서 계산), date(접수 날짜)
실행 방법 : py -3.12 analysis/eda_02_summary.py
기존 앱 참고: 2 대푯값, 3 퍼짐·기술통계 탭

[용어] 평균(mean)
  모든 값을 더해 개수로 나눈 값. 아주 큰 값 하나에도 크게 움직인다.
  예) 1, 2, 3, 4, 1000 의 평균은 202. 다섯 개 중 네 개가 4 이하인데도 평균은 202다.

[용어] 중앙값(median)
  값을 크기 순서로 줄 세웠을 때 정가운데 값. 극단값에 거의 영향을 받지 않는다.
  예) 1, 2, 3, 4, 1000 의 중앙값은 3.

[용어] 표준편차(standard deviation)
  값들이 평균에서 평균적으로 얼마나 떨어져 있는지. 클수록 값이 넓게 흩어져 있다.
  예) 5, 5, 5 의 표준편차는 0. 1, 5, 9 는 약 4.

[용어] 사분위수(quartile)
  값을 줄 세워 4등분하는 경계. Q1(25%), Q2(50% = 중앙값), Q3(75%).
  예) Q1 = 1.2시간이면 "민원 4건 중 1건은 1.2시간 안에 끝났다"는 뜻.
"""

# scipy.stats: 통계 함수 모음. 여기서는 절사평균(trim_mean)을 쓴다
from scipy import stats

# 공통 도구
from common import ExclusionLog, load_data, pd, plt, print_header, save_fig, save_table, setup_plot, np

# 그래프 한글 설정
setup_plot()

# 데이터 불러오기 (hours, date 등 파생 열 포함)
df = load_data()

# ─────────────────────────────────────────────
# 1. 처리 시간 계산 대상 정하기 (제외 기준 기록)
# ─────────────────────────────────────────────
print_header("1. 처리 시간 계산 대상")

# 제외 로그를 시작한다. 시작 행 수 = 전체 민원 수
log = ExclusionLog("eda02", len(df))

# 종결 시각이 없는 민원(아직 처리 중)은 처리 시간을 계산할 수 없다
no_close = df["hours"].isna()

# 기록한다
log.add("closed(종결 시각) 없음 → 처리 시간 계산 불가", no_close.sum())

# 처리 시간이 음수 = 종결이 접수보다 먼저. 현실에서 불가능하므로 입력 오류로 본다
negative = df["hours"] < 0

# 기록한다
log.add("처리 시간 < 0 (종결이 접수보다 빠름, 입력 오류)", negative.sum())

# 두 조건에 해당하지 않는 행의 처리 시간만 남긴다. ~ 는 '아니다(not)'라는 뜻
hours = df.loc[~no_close & ~negative, "hours"]

# 로그 저장
log.save()

# 남은 건수 출력
print(f"처리 시간 분석 대상: {len(hours):,}건")
# 해석: 약 96%의 민원으로 처리 시간을 요약한다. 빠진 4%는 '오래 걸리는 중인' 민원일 가능성이 높아서
#       아래 결과는 실제보다 처리 시간을 약간 짧게 보여 줄 수 있다.

# ─────────────────────────────────────────────
# 2. 처리 시간 기술통계
# ─────────────────────────────────────────────
print_header("2. 처리 시간 기술통계 (단위: 시간)")

# 평균
mean = hours.mean()

# 중앙값
median = hours.median()

# 표준편차
std = hours.std()

# 사분위수. quantile(0.25) = 25% 지점 값
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)

# [용어] 절사평균(trimmed mean): 위아래 극단값을 일정 비율(여기서는 각 5%) 잘라내고 구한 평균.
#        평균과 중앙값의 중간 성격이다. 예) 1,2,3,4,1000 에서 양끝 하나씩 빼면 (2+3+4)/3 = 3
trimmed = stats.trim_mean(hours, 0.05)

# 결과를 표로 만든다
summary = pd.DataFrame([
    {"통계량": "건수", "값": len(hours)},
    {"통계량": "평균", "값": round(mean, 2)},
    {"통계량": "절사평균(양끝 5%)", "값": round(trimmed, 2)},
    {"통계량": "중앙값", "값": round(median, 2)},
    {"통계량": "표준편차", "값": round(std, 2)},
    {"통계량": "Q1 (25%)", "값": round(q1, 2)},
    {"통계량": "Q3 (75%)", "값": round(q3, 2)},
    {"통계량": "최솟값", "값": round(hours.min(), 2)},
    {"통계량": "최댓값", "값": round(hours.max(), 2)},
    {"통계량": "평균 / 중앙값 (배)", "값": round(mean / median, 1)},
])

# 출력
print(summary.to_string(index=False))
# 해석: 평균(약 160시간 ≈ 6.7일)이 중앙값(약 8시간)의 약 20배다.
#       "보통 민원은 반나절 안에 끝나지만, 몇 주~몇 달 걸리는 소수의 민원이 평균을 크게 끌어올린다"는 뜻이다.

# 저장
save_table(summary, "eda02_hours_summary")

# ─────────────────────────────────────────────
# 3. 왜 평균만 보면 위험한가: 직접 보여 주기
# ─────────────────────────────────────────────
print_header("3. 평균만 보면 위험한 이유")

# 평균보다 처리 시간이 짧은 민원의 비율
below_mean = (hours < mean).mean() * 100

# 출력
print(f"평균({mean:.1f}시간)보다 빨리 끝난 민원: {below_mean:.1f}%")
# 해석: 민원의 82.5%가 '평균'보다 빨리 끝난다. 평균은 '보통 민원'을 대표하지 못한다.

# 상위 1% 민원을 빼면 평균이 얼마나 바뀌는지 본다
p99 = hours.quantile(0.99)

# 상위 1% 미만만 남긴 평균
mean_wo_top1 = hours[hours < p99].mean()

# 출력
print(f"상위 1%(>{p99:.0f}시간) 제외 시 평균: {mean:.1f} → {mean_wo_top1:.1f}시간")
# 해석: 민원 100건 중 1건만 빼도 평균이 159.6 → 124.1시간으로 22% 줄어든다. 평균은 극단값에 민감하다.

# 같은 내용을 중앙값으로 확인한다
median_wo_top1 = hours[hours < p99].median()

# 출력
print(f"상위 1% 제외 시 중앙값: {median:.1f} → {median_wo_top1:.1f}시간")
# 해석: 중앙값은 7.9 → 7.5시간으로 거의 그대로다. 그래서 치우친 자료는 중앙값으로 요약한다.

# 히스토그램에 평균선과 중앙값선을 함께 그린다
fig, ax = plt.subplots(figsize=(8, 4))

# x축을 로그 눈금으로 쓰기 위해 0보다 큰 값만 그린다 (로그는 0에서 정의되지 않는다)
pos = hours[hours > 0]

# 로그 간격으로 나눈 구간 50개. 0.01시간(36초) ~ 최댓값
edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 50)

# 히스토그램
ax.hist(pos, bins=edges, color="#BBBBBB")

# x축 로그 눈금
ax.set_xscale("log")

# 눈금 글자를 10⁻² 같은 수식 대신 0.01, 1, 100 처럼 보통 숫자로 쓴다 (한글 글꼴에서 수식 마이너스가 깨지는 것도 막는다)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

# 중앙값 세로선
ax.axvline(median, color="#2E7D32", lw=2, label=f"중앙값 {median:.1f}시간")

# 평균 세로선
ax.axvline(mean, color="#C62828", lw=2, label=f"평균 {mean:.1f}시간")

# 축 이름
ax.set_xlabel("처리 시간 (시간, 로그 눈금)")

# y축 이름
ax.set_ylabel("민원 건수")

# 제목
ax.set_title("처리 시간 분포: 평균과 중앙값이 멀리 떨어져 있다")

# 범례
ax.legend()

# 저장
save_fig(fig, "eda02_mean_vs_median")

# ─────────────────────────────────────────────
# 4. 하루 접수 건수 기술통계
# ─────────────────────────────────────────────
print_header("4. 하루 접수 건수")

# 날짜별 민원 수를 센다
daily = df.groupby("date").size()

# 제외 로그 (일별 집계용)
log_d = ExclusionLog("eda02_daily", len(daily))

# 첫날·마지막 날은 하루가 다 차지 않은 '부분 일자'라서 뺀다 (eda_01 에서 확인)
log_d.add("첫날·마지막 날 (부분 일자)", 2)

# 첫날과 마지막 날을 뺀다. iloc[1:-1] 은 '처음과 끝 하나씩 빼고'라는 뜻
daily = daily.iloc[1:-1]

# 로그 저장
log_d.save()

# describe(): 건수, 평균, 표준편차, 최소, 사분위수, 최대를 한 번에 계산한다
d = daily.describe()

# 출력
print(d.round(1).to_string())
# 해석: 하루 접수 건수는 평균(약 10,890건)과 중앙값(약 10,804건)이 거의 같다. 처리 시간과 달리 치우침이 작은 자료다.
#       다만 최댓값(22,805건)은 평소의 2배가 넘는다. 폭설 같은 특별한 날일 수 있어 ml_14 예측에서 큰 오차를 만들 수 있다.

# 저장 (index=True 로 통계량 이름도 같이 저장)
save_table(d.round(2).rename("하루 접수 건수").to_frame(), "eda02_daily_summary", index=True)
