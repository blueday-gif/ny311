"""
[ML 6단계] 기준선: 모델 없이 찍으면 몇 점인가

핵심 질문 : 머신러닝 모델이 '잘했다'고 말하려면, 무엇과 비교해야 하나?
배우는 개념: 기준선(baseline), 정확도(accuracy), 다수결 예측, 규칙 기반 예측
사용하는 열: within_24h(타깃), problem, created(월별 비율 확인)
실행 방법 : py -3.12 analysis/ml_06_split_baseline.py   (먼저 ml_05 실행)
기존 앱 참고: 6 검정·회귀 탭

[용어] 기준선(baseline)
  아주 단순한 방법으로 낸 점수. 모델은 적어도 이것보다 좋아야 쓸 이유가 있다.
  예) 일기예보 모델이 '내일도 오늘과 같다'보다 못 맞히면 쓸모가 없다.

[용어] 정확도(accuracy)
  전체 예측 중 맞힌 비율. 이해하기 쉽지만, 정답이 한쪽으로 쏠린 문제에서는 속기 쉽다 (ml_10).
  예) 100건 중 90건이 0인 문제에서 '무조건 0'이라고 하면 정확도 90%지만 아무것도 배우지 않은 것이다.
"""

# DummyClassifier: '항상 가장 많은 정답을 말하는' 가짜 모델. 기준선용으로 sklearn 이 제공한다
from sklearn.dummy import DummyClassifier

# 정확도 계산 함수
from sklearn.metrics import accuracy_score

# 공통 도구
from common import FEATURES, ML_PATH, TARGET, load_ml, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 학습용 20만, 평가용 10만 표본을 불러온다 (seed 고정이라 매번 같은 행)
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]

# 결과를 모을 목록
results = []

# ─────────────────────────────────────────────
# 1. 기준선 ①: 다수결 (항상 더 많은 쪽으로 찍기)
# ─────────────────────────────────────────────
print_header("1. 기준선 ①: 다수결")

# strategy="most_frequent": 학습 데이터에서 가장 많은 정답(여기서는 1)을 항상 말한다
dummy = DummyClassifier(strategy="most_frequent")

# '학습' = 1과 0 중 어느 쪽이 많은지 세는 것뿐이다
dummy.fit(X_tr, y_tr)

# 평가 데이터 예측 (모두 같은 값)
pred = dummy.predict(X_te)

# 정확도
acc = accuracy_score(y_te, pred)

# 기록
results.append({"방법": "① 다수결 (항상 1)", "평가 정확도": round(acc, 4)})

# 출력
print(f"항상 '24시간 안 처리됨(1)'이라고 찍으면 정확도 {acc:.3f}")
# 해석: 아무 정보도 쓰지 않아도 57.9%를 맞힌다. 정확도 58%짜리 모델은 '배운 것이 없다'는 뜻이다.

# ─────────────────────────────────────────────
# 2. 기준선 ②: 민원 유형 하나만 보고 규칙 만들기
# ─────────────────────────────────────────────
print_header("2. 기준선 ②: 민원 유형별 다수결 규칙")

# 학습 데이터에서 민원 유형별 24시간 처리율을 계산한다
rate = y_tr.groupby(X_tr["problem"], observed=True).mean()

# 처리율이 50% 이상인 유형은 1, 아니면 0 으로 찍는 규칙
rule = (rate >= 0.5).astype(int)

# 평가 데이터의 각 민원 유형에 규칙을 적용한다. 학습에 없던 유형은 다수결 값(1)으로 채운다
pred_rule = X_te["problem"].astype(str).map(rule.rename(index=str)).fillna(1).astype(int)

# 정확도
acc_rule = accuracy_score(y_te, pred_rule)

# 기록
results.append({"방법": "② 민원 유형별 다수결 규칙", "평가 정확도": round(acc_rule, 4)})

# 출력
print(f"민원 유형만 보고 찍으면 정확도 {acc_rule:.3f}")
# 해석: 열 하나(민원 유형)만으로 정확도가 0.58 → 0.88로 크게 오른다. 이 문제의 핵심 정보는 민원 유형이다.
#       앞으로 만들 모델은 '0.58'이 아니라 '0.88'보다 나아야 의미가 있다. 남은 12%를 얼마나 줄이는지가 진짜 실력이다.

# 규칙표 저장: 유형별 처리율과 규칙
rule_tab = pd.DataFrame({"24시간 처리율(학습)": rate.round(3), "규칙 예측": rule}).sort_values("24시간 처리율(학습)")

# 저장
save_table(rule_tab, "ml06_problem_rule", index=True)

# ─────────────────────────────────────────────
# 3. 기준선 결과 저장
# ─────────────────────────────────────────────

# 표로 만든다
res = pd.DataFrame(results)

# 출력
print(res.to_string(index=False))

# 저장 (ml_09 모델 비교표에서 함께 보여 준다)
save_table(res, "ml06_baselines")

# ─────────────────────────────────────────────
# 4. 왜 '시간 기준'으로 나눴나: 월별 정답 비율 확인
# ─────────────────────────────────────────────
print_header("4. 월별 24시간 처리율")

# 전체(표본 아님) 데이터에서 월별 처리율을 본다
full = pd.read_parquet(ML_PATH, columns=["created", "split", TARGET])

# 월별 평균(= 처리율)과 기간 구분
monthly = full.groupby(full["created"].dt.to_period("M")).agg(
    처리율=(TARGET, "mean"), 구분=("split", "first"))

# 퍼센트로
monthly["처리율"] = (monthly["처리율"] * 100).round(2)

# 출력
print(monthly.to_string())
# 해석: 1월 47.6% ~ 5월 62.2%로 월마다 처리율이 크게 오르내린다. 겨울에는 오래 걸리는 난방·온수 민원이 늘어 처리율이 낮아진다.
#       평가 기간(8~10월)은 57~58%로 학습 기간 평균과 비슷하다.
#       모델은 학습 기간의 패턴으로 평가 기간을 맞혀야 한다. 실제 운영도 '과거로 미래를 예측'하므로 이 방식이 정직하다.

# 저장
save_table(monthly, "ml06_monthly_rate", index=True)

# 그림
fig, ax = plt.subplots(figsize=(9, 4))

# 학습 기간은 파랑, 평가 기간은 주황
colors = ["#4C72B0" if s == "train" else "#DD8452" for s in monthly["구분"]]

# 막대
ax.bar(monthly.index.astype(str), monthly["처리율"], color=colors)

# y축
ax.set_ylabel("24시간 안 처리율 (%)")

# 제목
ax.set_title("월별 24시간 처리율 (파랑 = 학습 기간, 주황 = 평가 기간)")

# x축 글자 기울이기
ax.tick_params(axis="x", rotation=45)

# y축 범위를 좁혀 차이를 보기 쉽게 한다
ax.set_ylim(monthly["처리율"].min() - 5, monthly["처리율"].max() + 5)

# 저장
save_fig(fig, "ml06_monthly_rate")
