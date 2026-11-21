"""
[EDA 3단계] 분포 확인과 이상치

핵심 질문 : 처리 시간은 어떤 모양으로 퍼져 있고, 로그 변환이 필요한가? 이상치는 얼마나 되나?
배우는 개념: 히스토그램, 상자그림, 왜도, 로그 변환(log1p), IQR 이상치 규칙
사용하는 열: hours(처리 시간)
실행 방법 : py -3.12 analysis/eda_03_distribution.py
기존 앱 참고: 3 퍼짐·기술통계, 4 분포·변환·표준화 탭

[용어] 히스토그램(histogram)
  값을 구간으로 나눠 각 구간에 몇 개가 들어가는지 막대로 그린 그림. 분포의 '모양'을 본다.
  예) 0~1시간 막대가 가장 높으면 "1시간 안에 끝나는 민원이 가장 많다"는 뜻.

[용어] 왜도(skewness)
  분포가 한쪽으로 치우친 정도. 0이면 좌우 대칭, 양수면 오른쪽 꼬리가 길다(큰 값이 드물게 있다).
  예) 소득 분포는 왜도가 양수다. 대부분은 평범하고 소수가 매우 높다. 보통 |왜도| > 1 이면 많이 치우쳤다고 본다.

[용어] 로그 변환(log transform)
  값에 로그를 씌워 큰 값을 압축하는 방법. 1, 10, 100, 1000 이 0, 1, 2, 3 으로 고르게 펴진다.
  0에는 로그를 쓸 수 없어서 log(1 + x) 인 log1p 를 쓴다. 예) log1p(0) = 0.

[용어] IQR 이상치 규칙
  IQR(사분위범위) = Q3 - Q1. 'Q1 - 1.5×IQR' 보다 작거나 'Q3 + 1.5×IQR' 보다 크면 이상치로 표시한다.
  예) Q1=1, Q3=79 이면 IQR=78, 위 경계 = 79 + 117 = 196시간. 196시간 넘게 걸린 민원이 이상치 후보.
  이상치는 '틀린 값'이 아니라 '드문 값'이다. 여기서는 지우지 않고 개수와 비율만 기록한다.
"""

# 공통 도구
from common import ExclusionLog, load_data, np, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 데이터 불러오기
df = load_data()

# ─────────────────────────────────────────────
# 1. 분석 대상: eda_02 와 같은 기준
# ─────────────────────────────────────────────
print_header("1. 분석 대상")

# 제외 로그 시작
log = ExclusionLog("eda03", len(df))

# 종결 시각 없음 → 처리 시간 계산 불가
log.add("closed(종결 시각) 없음", df["hours"].isna().sum())

# 음수 처리 시간 → 입력 오류
log.add("처리 시간 < 0", (df["hours"] < 0).sum())

# 처리 시간이 0 이상인 값만 남긴다 (NaN 은 비교 결과가 False 라 자동으로 빠진다)
hours = df.loc[df["hours"] >= 0, "hours"]

# 로그 저장
log.save()

# ─────────────────────────────────────────────
# 2. 히스토그램: 원래 값 vs 로그 변환 값
# ─────────────────────────────────────────────
print_header("2. 히스토그램과 왜도")

# 로그 변환. log1p(x) = log(1 + x). 처리 시간 0 도 계산할 수 있다
log_hours = np.log1p(hours)

# 원래 값의 왜도. pandas 의 skew() 가 계산해 준다
skew_raw = hours.skew()

# 로그 변환 후 왜도
skew_log = log_hours.skew()

# 출력
print(f"왜도: 원래 값 {skew_raw:.2f} → log1p 변환 후 {skew_log:.2f}")
# 해석: 원래 값은 왜도 6.17로 오른쪽 꼬리가 매우 길다. log1p 를 씌우면 0.57로 줄어 훨씬 대칭에 가깝다.
#       그래서 처리 시간으로 그래프를 그리거나 회귀 모델을 만들 때는 log1p 변환 값을 쓴다 (ml_13).

# 표로 저장
save_table(pd.DataFrame([
    {"값": "처리 시간(원래)", "왜도": round(skew_raw, 3)},
    {"값": "log1p(처리 시간)", "왜도": round(skew_log, 3)},
]), "eda03_skewness")

# 그림 두 칸을 나란히 만든다
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 왼쪽: 원래 값 히스토그램. 99% 지점까지만 그린다 (끝까지 그리면 막대가 왼쪽 끝에 몰려 안 보인다)
axes[0].hist(hours[hours <= hours.quantile(0.99)], bins=60, color="#4C72B0")

# 왼쪽 제목
axes[0].set_title(f"원래 값 (왜도 {skew_raw:.1f})")

# 왼쪽 x축 이름
axes[0].set_xlabel("처리 시간 (시간, 상위 1% 제외)")

# 오른쪽: 로그 변환 값 히스토그램
axes[1].hist(log_hours, bins=60, color="#55A868")

# 오른쪽 제목
axes[1].set_title(f"log1p 변환 (왜도 {skew_log:.1f})")

# 오른쪽 x축 이름
axes[1].set_xlabel("log(1 + 처리 시간)")

# 두 그림 모두 y축은 건수
for ax in axes:
    # y축 이름
    ax.set_ylabel("민원 건수")

# 저장
save_fig(fig, "eda03_hist_raw_vs_log")
# 해석: 오른쪽 그림에 봉우리가 두 개 보이면 "빨리 끝나는 민원 무리"와 "오래 걸리는 민원 무리"가 섞여 있다는 뜻이다.
#       민원 유형마다 처리 방식이 다르기 때문이다 (eda_04 에서 유형별로 본다).

# ─────────────────────────────────────────────
# 3. 상자그림
# ─────────────────────────────────────────────

# [용어] 상자그림(box plot): 상자 = Q1~Q3, 가운데 선 = 중앙값, 수염 = IQR 규칙 경계, 점 = 이상치 후보.
#        분포를 한눈에 요약하고 여러 집단을 나란히 비교할 때 좋다.

# 그림 두 칸
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# 원래 값 상자그림. 점이 수십만 개라 그리기 느려서 이상치 점은 생략(showfliers=False)
axes[0].boxplot(hours, orientation="horizontal", showfliers=False)

# 제목
axes[0].set_title("원래 값 (이상치 점 생략)")

# x축
axes[0].set_xlabel("처리 시간 (시간)")

# 로그 값 상자그림
axes[1].boxplot(log_hours, orientation="horizontal", showfliers=False)

# 제목
axes[1].set_title("log1p 변환 값")

# x축
axes[1].set_xlabel("log(1 + 처리 시간)")

# 저장
save_fig(fig, "eda03_boxplot")
# 해석: 원래 값 상자그림은 상자가 왼쪽에 납작하게 붙고 수염이 오른쪽으로 길다 = 오른쪽 꼬리가 긴 분포.

# ─────────────────────────────────────────────
# 4. IQR 규칙으로 이상치 세기 (제거하지 않음)
# ─────────────────────────────────────────────
print_header("4. IQR 이상치")


def iqr_outliers(s: pd.Series, name: str) -> dict:
    """IQR 규칙으로 이상치 경계와 개수를 계산한다."""
    # 1사분위수
    q1 = s.quantile(0.25)
    # 3사분위수
    q3 = s.quantile(0.75)
    # 사분위범위
    iqr = q3 - q1
    # 아래 경계
    low = q1 - 1.5 * iqr
    # 위 경계
    high = q3 + 1.5 * iqr
    # 아래 경계보다 작은 값 개수
    n_low = int((s < low).sum())
    # 위 경계보다 큰 값 개수
    n_high = int((s > high).sum())
    # 결과를 사전(dict)으로 돌려준다
    return {
        "값": name, "Q1": round(q1, 3), "Q3": round(q3, 3), "IQR": round(iqr, 3),
        "아래 경계": round(low, 3), "위 경계": round(high, 3),
        "아래 이상치 수": n_low, "위 이상치 수": n_high,
        "이상치 비율(%)": round((n_low + n_high) / len(s) * 100, 2),
    }


# 원래 값과 로그 값 각각에 규칙을 적용한다
out = pd.DataFrame([
    iqr_outliers(hours, "처리 시간(원래, 시간)"),
    iqr_outliers(log_hours, "log1p(처리 시간)"),
])

# 출력
print(out.to_string(index=False))
# 해석: 원래 값 기준으로는 민원의 15.4%(약 58만 건, 196시간 초과)가 '이상치'로 잡힌다. 치우친 분포에서는 IQR 규칙이 꼬리 전체를 이상치로 본다.
#       log1p 변환 후에는 이상치가 0건이다. 즉 이 값들은 '틀린 값'이 아니라 '긴 꼬리'의 일부다. 그래서 지우지 않는다.

# 저장
save_table(out, "eda03_outliers_iqr")

# 이상치는 지우지 않았다는 것을 로그로 남긴다
log_o = ExclusionLog("eda03_outliers", len(hours))

# 원래 값 기준 이상치 건수 기록
log_o.add("IQR 규칙 이상치 (원래 값 기준)", out.loc[0, "아래 이상치 수"] + out.loc[0, "위 이상치 수"], "유지(기록만)")

# 로그 값 기준 이상치 건수 기록
log_o.add("IQR 규칙 이상치 (log1p 기준)", out.loc[1, "아래 이상치 수"] + out.loc[1, "위 이상치 수"], "유지(기록만)")

# 저장
log_o.save()
