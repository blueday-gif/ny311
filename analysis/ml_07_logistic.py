"""
[ML 7단계] 첫 모델: 로지스틱 회귀

핵심 질문 : 여러 정보를 함께 쓰면, 민원 유형 하나만 쓸 때보다 더 잘 맞힐까? 어떤 정보가 '24시간 안 처리' 쪽으로 미나?
배우는 개념: 로지스틱 회귀, 확률 예측, 혼동행렬, 오즈비(odds ratio)
사용하는 열: within_24h(타깃), problem, borough, channel, location_type, season, hour, weekday
실행 방법 : py -3.12 analysis/ml_07_logistic.py   (먼저 ml_05 실행)
기존 앱 참고: 6 검정·회귀(로지스틱 회귀) 탭

[용어] 로지스틱 회귀(logistic regression)
  입력마다 '점수(가중치)'를 매겨 더한 뒤, 그 합을 0~1 사이 확률로 바꾸는 모델. 확률이 0.5 이상이면 1로 예측한다.
  예) 점수 = (불법주차면 +2) + (주말이면 -0.3) + ... → 확률 0.85 → "24시간 안 처리될 것"

[용어] 혼동행렬(confusion matrix)
  예측과 실제를 2×2 표로 나눠 센 것. 어떤 종류의 실수를 하는지 보여 준다.
  예) 실제 1인데 0으로 예측 = '놓친 것', 실제 0인데 1로 예측 = '헛짚은 것'.

※ 위도·경도는 이 모델에서 뺀다. 로지스틱 회귀는 '숫자가 커질수록 확률이 한 방향으로 변한다'는 직선 관계만 배워서,
  "북쪽으로 갈수록 빨리 처리된다" 같은 뜻 없는 규칙이 된다. 위치는 나무 모델(ml_08 이후)이 더 잘 다룬다.
  시간(hour)과 요일(weekday)도 같은 이유로 숫자가 아니라 범주로 바꿔 넣는다 (23시 다음은 0시라 직선이 아니다).
"""

# 시간 측정
import time

# 로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression

# 평가 지표: 정확도, 혼동행렬
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

# Pipeline: 전처리 → 모델을 한 줄로 묶는 도구. 학습/예측 때 같은 전처리를 자동으로 적용해 실수를 줄인다
from sklearn.pipeline import Pipeline

# 공통 도구
from common import (FEATURES, TARGET, load_ml, make_onehot_preprocess, np, pd, plt, print_header, save_fig,
                    save_table, setup_plot)

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]

# ─────────────────────────────────────────────
# 1. 모델 만들기와 학습
# ─────────────────────────────────────────────
print_header("1. 로지스틱 회귀 학습")

# 전처리: 범주형 + hour, weekday 를 원-핫으로, 숫자 열(위도·경도)은 쓰지 않음(numeric=[])
pre = make_onehot_preprocess(numeric=[], extra_cat=["hour", "weekday"])

# max_iter: 최적의 가중치를 찾을 때 반복할 최대 횟수. 기본값(100)으로는 끝까지 못 찾을 수 있어 늘린다
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])

# 학습 시작 시각
t0 = time.time()

# 학습: 20만 건을 보고 가중치를 찾는다
model.fit(X_tr, y_tr)

# 걸린 시간 출력
print(f"학습 시간: {time.time() - t0:.1f}초, 원-핫 후 입력 열 수: {len(model[:-1].get_feature_names_out())}")
# 해석: 범주 9개 열이 0/1 열 100개 남짓으로 펼쳐졌다. 모델은 이 열마다 가중치 하나씩을 배웠다.

# ─────────────────────────────────────────────
# 2. 정확도: 학습 vs 평가
# ─────────────────────────────────────────────
print_header("2. 정확도")

# 학습 데이터 정확도 (연습 문제 점수)
acc_tr = accuracy_score(y_tr, model.predict(X_tr))

# 평가 데이터 정확도 (시험 점수)
acc_te = accuracy_score(y_te, model.predict(X_te))

# 출력
print(f"학습 정확도 {acc_tr:.3f} / 평가 정확도 {acc_te:.3f}")
# 해석: 학습 0.882, 평가 0.896으로 비슷하다 = '외우지 않고 일반적인 규칙을 배웠다' (ml_08 과적합과 비교).
#       평가가 오히려 조금 높은 것은 평가 기간(8~10월)에 맞히기 어려운 겨울 민원이 적기 때문으로 보인다.
#       기준선과 비교: 다수결 0.580, 민원 유형 규칙 0.882 → 로지스틱 0.896. 여러 정보를 함께 써서 약 1.4%p 더 맞혔다.

# 결과 저장
save_table(pd.DataFrame([{"모델": "로지스틱 회귀", "학습 정확도": round(acc_tr, 4), "평가 정확도": round(acc_te, 4)}]),
           "ml07_accuracy")

# ─────────────────────────────────────────────
# 3. 혼동행렬: 어떤 실수를 하나
# ─────────────────────────────────────────────
print_header("3. 혼동행렬")

# 평가 데이터 예측
pred = model.predict(X_te)

# 2×2 표: 행 = 실제(0, 1), 열 = 예측(0, 1)
cm = confusion_matrix(y_te, pred)

# 각 칸에 이름을 붙여 표로 만든다
cm_tab = pd.DataFrame(cm, index=["실제 0 (24h 초과)", "실제 1 (24h 이내)"], columns=["예측 0", "예측 1"])

# 출력
print(cm_tab.to_string())
# 해석: 오른쪽 위(약 2,200건) = 실제로는 오래 걸렸는데 '빨리 끝난다'고 헛짚은 건수,
#       왼쪽 아래(약 8,200건) = 빨리 끝났는데 '오래 걸린다'고 놓친 건수. 이 모델은 '놓치는' 실수를 약 4배 더 많이 한다.
#       두 실수 중 어느 쪽이 더 곤란한지는 쓰임새에 따라 다르다 (ml_10 에서 다룬다).

# 저장
save_table(cm_tab, "ml07_confusion_matrix", index=True)

# 그림
fig, ax = plt.subplots(figsize=(5, 4))

# sklearn 이 혼동행렬 그림을 그려 준다
ConfusionMatrixDisplay(cm, display_labels=["24h 초과(0)", "24h 이내(1)"]).plot(ax=ax, values_format=",", colorbar=False)

# 제목
ax.set_title("로지스틱 회귀 혼동행렬 (평가 데이터)")

# x축
ax.set_xlabel("예측")

# y축
ax.set_ylabel("실제")

# 저장
save_fig(fig, "ml07_confusion_matrix")

# ─────────────────────────────────────────────
# 4. 가중치 읽기: 오즈비
# ─────────────────────────────────────────────
print_header("4. 무엇이 '24시간 안 처리' 쪽으로 미나 (오즈비)")

# [용어] 오즈(odds) = 일어날 확률 / 안 일어날 확률. 확률 0.8 이면 오즈 = 0.8/0.2 = 4.
# [용어] 오즈비(odds ratio) = exp(가중치). 그 조건이 '켜질 때' 오즈가 몇 배가 되는가.
#        예) 오즈비 3 = 그 조건이면 24시간 안 처리될 오즈가 3배. 오즈비 0.5 = 절반.
#        원-핫 범주의 오즈비는 '같은 열의 다른 범주들 평균과 비교한' 상대값이라 크기만 대략 비교하는 데 쓴다.

# 열 이름 (예: problem_Illegal Parking)
names = model[:-1].get_feature_names_out()

# 가중치. coef_[0] 은 열마다 하나씩 있는 숫자
coef = model[-1].coef_[0]

# 표로 만든다
odds = pd.DataFrame({"입력 열": names, "가중치": coef.round(3), "오즈비": np.exp(coef).round(3)})

# 오즈비 큰 순서로 정렬
odds = odds.sort_values("오즈비", ascending=False)

# 위쪽(24시간 안 처리 쪽으로 강하게 미는 조건) 10개
print("[24시간 안 처리 쪽으로 미는 조건 상위 10]\n" + odds.head(10).to_string(index=False))

# 아래쪽(오래 걸리는 쪽으로 미는 조건) 10개
print("[오래 걸리는 쪽으로 미는 조건 상위 10]\n" + odds.tail(10).iloc[::-1].to_string(index=False))
# 해석: 위·아래 모두 민원 유형(problem_...)과 장소 유형(location_type_...)이 차지한다. 처리 속도는 '어떤 민원인가'에 가장 크게 좌우된다.
#       눈여겨볼 점: 'Residential Building/House'(빠름), 'RESIDENTIAL BUILDING'(느림), 'Residential Building'(느림)처럼
#       뜻이 같은 장소가 표기만 다르게 따로 있다. 접수 기관마다 쓰는 표기가 달라서, 장소 유형이 사실상 '담당 기관'을 알려 준다.
#       이런 표기 불일치는 데이터 정제 단계에서 확인해야 할 품질 문제다.
#       단, 이것은 '연관'이지 '원인'이 아니다. 그 유형이라서 빨라진 것인지, 그 유형을 맡은 기관이 빨라서인지 이 모델은 구분하지 못한다.

# 전체 오즈비 표 저장
save_table(odds, "ml07_odds_ratio")
