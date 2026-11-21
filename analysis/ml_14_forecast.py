"""
[ML 14단계] 예측: 내일 민원은 몇 건 들어올까

핵심 질문 : 지난 며칠의 접수 건수로 다음 날 접수 건수를 맞힐 수 있을까? 단순한 방법보다 나을까?
배우는 개념: 시계열 예측, 지연 특징(lag), 이동평균, 단순 예측(naive), 계절 단순 예측, MAPE
사용하는 열: created(일별 건수로 집계)
실행 방법 : py -3.12 analysis/ml_14_forecast.py
기존 앱 참고: 9 시계열 탭 (거기서는 ARIMA, 여기서는 머신러닝 방식)

[용어] 지연 특징(lag feature)
  과거 값을 오늘의 입력으로 쓰는 것. lag1 = 어제 건수, lag7 = 일주일 전 같은 요일 건수.
  예) 오늘 건수를 맞힐 때 어제 10,500건, 지난주 같은 요일 11,200건이었다는 정보를 넣는다.

[용어] 단순 예측(naive forecast)
  "내일은 오늘과 같다"고 찍는 방법. 시계열 예측의 가장 기본 기준선이다.
  계절 단순 예측(seasonal naive)은 "다음 월요일은 이번 월요일과 같다"처럼 주기만큼 전 값을 쓴다.

[용어] MAPE (평균 절대 백분율 오차)
  |실제 - 예측| / 실제 의 평균을 %로. "평균적으로 몇 % 틀리나". 규모가 다른 시계열끼리 비교하기 좋다.
  예) 실제 10,000, 예측 9,500 → 5%.

※ 이 단계는 '하루 앞(1-step ahead)' 예측이다. 매일 실제 값이 들어오면 그 값을 써서 다음 날을 맞힌다.
"""

# 선형 회귀
from sklearn.linear_model import LinearRegression

# 그래디언트 부스팅 회귀
from sklearn.ensemble import HistGradientBoostingRegressor

# 평가 지표
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 공통 도구
from common import (SEED, SPLIT_DATE, ExclusionLog, load_data, np, pd, plt, print_header, save_fig, save_table,
                    setup_plot)

# 그래프 한글 설정
setup_plot()

# ─────────────────────────────────────────────
# 1. 일별 건수 만들기
# ─────────────────────────────────────────────
print_header("1. 일별 접수 건수")

# 데이터 불러오기 (날짜 열만 있으면 된다)
df = load_data()

# 날짜별 건수. 한 행 = 하루
daily = df.groupby("date").size().rename("y").to_frame()

# 제외 로그
log = ExclusionLog("ml14", len(daily))

# 첫날·마지막 날은 부분 일자라 뺀다 (eda_01 참고)
log.add("첫날·마지막 날 (부분 일자)", 2)

# 제거
daily = daily.iloc[1:-1]

# ─────────────────────────────────────────────
# 2. 특징 만들기
# ─────────────────────────────────────────────
print_header("2. 지연 특징 만들기")

# 어제 건수. shift(1) 은 한 칸 아래로 밀어서 '어제 값'을 오늘 행에 놓는다
daily["lag1"] = daily["y"].shift(1)

# 일주일 전 같은 요일 건수
daily["lag7"] = daily["y"].shift(7)

# 어제까지 7일 평균. shift(1) 을 먼저 해야 '오늘 값'이 평균에 섞이지 않는다 (섞이면 누수)
daily["roll7"] = daily["y"].shift(1).rolling(7).mean()

# 요일 번호
daily["weekday"] = daily.index.day_of_week

# 요일을 0/1 열 7개로 펼친다 (선형 회귀가 요일을 숫자 크기로 오해하지 않도록)
wd = pd.get_dummies(daily["weekday"], prefix="wd", dtype=int)

# 원래 표에 붙인다
daily = pd.concat([daily, wd], axis=1)

# 앞쪽 7일은 lag7 이 비어 있으므로 뺀다
n_before = len(daily)

# 빈 값이 있는 행 제거
daily = daily.dropna()

# 기록
log.add("앞쪽 7일 (lag7·roll7 계산 불가)", n_before - len(daily))

# 로그 저장
log.save()

# 특징 목록
feats = ["lag1", "lag7", "roll7"] + list(wd.columns)

# 학습: 2026-08-01 전, 평가: 이후
train = daily[daily.index < SPLIT_DATE]

# 평가 기간
test = daily[daily.index >= SPLIT_DATE]

# 출력
print(f"학습 {len(train)}일 ({train.index.min():%Y-%m-%d} ~ {train.index.max():%Y-%m-%d}), "
      f"평가 {len(test)}일 ({test.index.min():%Y-%m-%d} ~ {test.index.max():%Y-%m-%d})")

# ─────────────────────────────────────────────
# 3. 기준선과 모델
# ─────────────────────────────────────────────
print_header("3. 예측과 평가")

# 예측 모음: 이름 → 평가 기간 예측값
preds = {}

# 기준선 ①: 내일 = 오늘
preds["① 단순 예측 (어제 값)"] = test["lag1"].values

# 기준선 ②: 내일 = 지난주 같은 요일
preds["② 계절 단순 예측 (지난주 같은 요일)"] = test["lag7"].values

# 모델 ③: 선형 회귀
lin = LinearRegression()

# 학습
lin.fit(train[feats], train["y"])

# 예측
preds["③ 선형 회귀 (lag + 요일)"] = lin.predict(test[feats])

# 모델 ④: 그래디언트 부스팅. 학습 데이터가 260일 정도로 적어서 나무를 작게(잎 8개) 키운다
hgb = HistGradientBoostingRegressor(max_leaf_nodes=8, max_iter=200, learning_rate=0.05, random_state=SEED)

# 학습
hgb.fit(train[feats], train["y"])

# 예측
preds["④ 그래디언트 부스팅 (lag + 요일)"] = hgb.predict(test[feats])

# 평가표
rows = []

# 방법마다 지표
for name, p in preds.items():
    # 기록
    rows.append({"방법": name, "MAE(건)": round(mean_absolute_error(test["y"], p), 0),
                 "MAPE(%)": round(mean_absolute_percentage_error(test["y"], p) * 100, 2)})

# 표로
fc = pd.DataFrame(rows)

# 출력
print(fc.to_string(index=False))
# 해석: 어제 값 7.2% → 지난주 같은 요일 6.6% → 부스팅 6.1% → 선형 회귀 5.1%. 하루 평균 약 536건 틀린다.
#       '어제 값'은 요일 패턴(월요일에 늘고 토요일에 줄어듦)을 무시해서 가장 많이 틀린다.
#       부스팅이 선형 회귀보다 못한 이유: 학습할 날이 265일뿐이고, 나무 모델은 학습 때 본 범위 밖의 값을 예측하지 못한다.
#       자료가 적을 때는 단순한 모델이 더 강할 수 있다.

# 저장
save_table(fc, "ml14_forecast_results")

# 선형 회귀 가중치 (어떤 정보에 얼마나 기대는지)
coef = pd.DataFrame({"특징": feats, "가중치": lin.coef_.round(3)})

# 출력
print(coef.to_string(index=False))
# 해석: lag1 0.64 = 어제 건수가 100건 많으면 오늘 예측이 64건 많아진다. 어제 값에 가장 크게 기댄다.
#       요일 가중치: 월요일(wd_0) +1,471건, 토요일(wd_5) -1,291건. 주말을 지나 월요일에 민원이 몰린다.

# 저장
save_table(coef, "ml14_linear_coef")

# ─────────────────────────────────────────────
# 4. 그림
# ─────────────────────────────────────────────

# 그림
fig, ax = plt.subplots(figsize=(13, 4.5))

# 실제 값 (학습 기간 끝 30일 + 평가 기간)
ax.plot(daily.index[-(len(test) + 30):], daily["y"].iloc[-(len(test) + 30):], color="black", lw=1.5, label="실제")

# 계절 단순 예측
ax.plot(test.index, preds["② 계절 단순 예측 (지난주 같은 요일)"], lw=1, alpha=0.8, label="지난주 같은 요일")

# 부스팅 예측
ax.plot(test.index, preds["④ 그래디언트 부스팅 (lag + 요일)"], lw=1, alpha=0.8, label="그래디언트 부스팅")

# 선형 회귀 예측
ax.plot(test.index, preds["③ 선형 회귀 (lag + 요일)"], lw=1, alpha=0.8, label="선형 회귀")

# 학습/평가 경계선
ax.axvline(SPLIT_DATE, color="gray", ls="--")

# 축 이름
ax.set_ylabel("하루 접수 건수")

# 제목
ax.set_title("하루 앞 접수 건수 예측 (점선 오른쪽 = 평가 기간)")

# 범례
ax.legend(loc="upper left")

# 저장
save_fig(fig, "ml14_forecast")
# 해석: 갑자기 튀는 날(폭우, 폭염, 행사)은 과거 건수만으로는 맞힐 수 없다. 날씨·행사 정보를 더하면 개선될 수 있다(한계).
