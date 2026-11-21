"""
[ML 11단계] 모델 해석: 모델은 무엇을 보고 판단했나

핵심 질문 : 모델이 가장 많이 기대는 정보는 무엇인가? 접수 시각이 바뀌면 예측 확률은 어떻게 변하나?
배우는 개념: 순열 중요도(permutation importance), 부분의존도(PDP), 중요도 ≠ 인과
사용하는 열: within_24h(타깃), FEATURES 전체 (그래디언트 부스팅 모델 사용)
실행 방법 : py -3.12 analysis/ml_11_interpret.py   (먼저 ml_05 실행)
기존 앱 참고: 7 현대 방법(순열 중요도) 탭

[용어] 순열 중요도(permutation importance)
  한 열의 값만 무작위로 뒤섞은 뒤 점수가 얼마나 떨어지는지 잰다. 많이 떨어질수록 모델이 그 열에 많이 기댄다.
  예) '민원 유형'을 뒤섞었더니 AUC 가 0.96 → 0.70 으로 떨어졌다면, 민원 유형은 매우 중요한 열이다.

[용어] 부분의존도(partial dependence, PDP)
  다른 정보는 그대로 두고 한 열의 값만 바꿔 가며 '평균 예측 확률'이 어떻게 변하는지 그린 선.
  예) 접수 시각을 0시 → 23시로 바꿔 가며 평균 확률을 그리면, 시간대가 예측에 주는 영향의 모양이 보인다.
"""

# 순열 중요도 계산 함수
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# 공통 도구
from common import FEATURES, SEED, TARGET, load_ml, make_hgb, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가에서 2만 건만 쓴다. 순열 중요도는 열마다 여러 번 다시 예측해서 느리기 때문이다
sub = test.sample(20_000, random_state=SEED)

# 평가 입력/정답
X_te, y_te = sub[FEATURES], sub[TARGET]

# ─────────────────────────────────────────────
# 1. 모델 학습
# ─────────────────────────────────────────────
print_header("1. 그래디언트 부스팅 학습")

# 모델
model = make_hgb()

# 학습
model.fit(X_tr, y_tr)

# 완료 표시
print("학습 완료")

# ─────────────────────────────────────────────
# 2. 순열 중요도
# ─────────────────────────────────────────────
print_header("2. 순열 중요도 (AUC 가 얼마나 떨어지나)")

# n_repeats=5: 열마다 5번 뒤섞어 평균을 낸다 (한 번만 섞으면 우연의 영향이 크다)
# scoring="roc_auc": 떨어지는 정도를 AUC 로 잰다
r = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=-1)

# 표로: 평균 감소량과 표준편차
imp = pd.DataFrame({"열": FEATURES, "AUC 감소(평균)": r.importances_mean.round(4),
                    "AUC 감소(표준편차)": r.importances_std.round(4)}).sort_values("AUC 감소(평균)", ascending=False)

# 출력
print(imp.to_string(index=False))
# 해석: 민원 유형(problem)을 뒤섞으면 AUC 가 0.251 떨어진다. 장소 유형(location_type)은 0.071. 나머지 열은 모두 0.004 미만이다.
#       즉 모델은 거의 '민원 유형 + 장소 유형' 두 열로 판단한다. 계절(season)은 0.000 = 빼도 차이가 없다.
#       표준편차가 평균보다 훨씬 작아서, 5번 뒤섞어도 순위는 흔들리지 않는다.

# 저장
save_table(imp, "ml11_permutation_importance")

# 그림
fig, ax = plt.subplots(figsize=(7, 4.5))

# 아래→위 순서
p = imp.iloc[::-1]

# 가로 막대 + 오차 막대(표준편차)
ax.barh(p["열"], p["AUC 감소(평균)"], xerr=p["AUC 감소(표준편차)"], color="#4C72B0")

# x축
ax.set_xlabel("열을 뒤섞었을 때 AUC 감소량")

# 제목
ax.set_title("순열 중요도: 모델이 많이 기대는 열")

# 저장
save_fig(fig, "ml11_permutation_importance")

# 주의 문구
print("주의: 중요도는 '모델이 예측에 많이 쓴 정보'이지 '처리 속도를 바꾸는 원인'이 아니다.")
print("      예) 장소 유형이 중요하게 나와도, 장소를 바꾸면 처리가 빨라진다는 뜻이 아니다. 장소 유형이 담당 기관을 대신 알려 줄 뿐이다.")
print("      또 서로 비슷한 정보를 담은 열(민원 유형 ↔ 장소 유형)이 있으면 중요도가 둘 사이에 나뉘어 각각 작게 보일 수 있다.")

# ─────────────────────────────────────────────
# 3. 부분의존도: 접수 시각, 요일
# ─────────────────────────────────────────────
print_header("3. 부분의존도 (시간대, 요일, 자치구)")

# 부분의존도 함수는 정수(int) 열을 받지 않는다(값을 바꿔 넣을 때 반올림 문제가 생길 수 있어서). 그래서 실수(float)로 바꾼 복사본을 쓴다
X_pd = X_te.copy()

# 시간·요일을 실수로 바꾼다. 값(0, 1, 2 ...)은 그대로이고 자료형만 바뀐다
X_pd[["hour", "weekday"]] = X_pd[["hour", "weekday"]].astype(float)

# 그림 세 칸
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 시간대·요일(선)과 자치구(막대)를 한 번에 그린다
# method="brute": 값을 실제로 바꿔 넣고 예측을 다시 하는 방식. 부스팅의 빠른 계산법(recursion)은 글자 범주를 지원하지 않고,
#                 결과도 확률이 아니라 로그 오즈 단위로 나와서 읽기 어렵다
# response_method="predict_proba": 세 그림 모두 '확률(0~1)' 단위로 맞춘다
disp = PartialDependenceDisplay.from_estimator(model, X_pd, ["hour", "weekday", "borough"],
                                               categorical_features=["borough"], method="brute",
                                               response_method="predict_proba", ax=axes)

# 세 그림의 y축 범위를 똑같이 맞춘다. 칸마다 범위가 다르면 작은 차이가 크게 보이는 착시가 생긴다
lows, highs = zip(*[a.get_ylim() for a in disp.axes_.ravel()])

# 공통 범위
for a in disp.axes_.ravel():
    # 가장 낮은 값 ~ 가장 높은 값
    a.set_ylim(min(lows), max(highs))
    # y축 이름을 한국어로
    a.set_ylabel("평균 예측 확률 (24시간 안 처리)")

# 제목
fig.suptitle("부분의존도: 이 값만 바꾸면 '24시간 안 처리' 평균 확률이 어떻게 변하나")

# 저장
save_fig(fig, "ml11_partial_dependence")
# 해석: 선이 출렁이는 폭이 그 열의 영향 크기다. 세 그림의 y축은 0.545~0.595로 폭이 5%p 정도뿐이다(일부러 같은 범위로 맞췄다).
#       - 요일: 토요일(5) 접수가 0.545로 가장 낮다. 주말에 들어온 민원은 처리가 월요일로 밀리는 것으로 보인다. 폭 약 4.5%p.
#       - 시간대: 밤·새벽 접수가 약 0.59, 오전 10시 무렵이 약 0.57. 폭 약 3%p.
#       - 자치구: 브루클린 0.567 ~ 스태튼아일랜드 0.583. 폭 약 1.6%p. 민원 유형을 알고 나면 자치구 차이는 작다(eda_04 의 중앙값 차이와 비교).
#       민원 유형을 바꿀 때 확률이 수십 %p 움직이는 것(ml_07 오즈비)과 비교하면, 시간·요일·자치구의 영향은 작다.
#       단, 부분의존도는 다른 열과 독립이라고 가정하고 그린다. 새벽에는 민원 유형 구성 자체가 달라서 해석에 주의가 필요하다.
