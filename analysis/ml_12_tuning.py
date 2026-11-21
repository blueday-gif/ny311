"""
[ML 12단계] 교차검증과 하이퍼파라미터 튜닝

핵심 질문 : 모델의 설정값(학습률, 나무 크기)은 어떻게 골라야 하나? 평가 데이터로 고르면 왜 안 되나?
배우는 개념: 하이퍼파라미터, 검증 데이터, 교차검증, 시계열 교차검증(TimeSeriesSplit), 그리드 탐색
사용하는 열: within_24h(타깃), FEATURES 전체, created(시간 순 정렬)
실행 방법 : py -3.12 analysis/ml_12_tuning.py   (먼저 ml_05 실행, 몇 분 걸린다)

[용어] 하이퍼파라미터(hyperparameter)
  모델이 데이터에서 스스로 배우지 않고, 사람이 미리 정해 주는 설정값.
  예) 나무 깊이, 학습률(한 번에 얼마나 크게 고칠지), 나무 개수.

[용어] 교차검증(cross-validation)
  학습 데이터를 여러 조각으로 나눠, 돌아가며 한 조각은 검증용으로 쓰고 나머지로 학습한다. 점수를 여러 번 재서 평균을 낸다.
  평가(test) 데이터는 끝까지 손대지 않고 '마지막 시험'에만 쓴다. 평가 데이터로 설정을 고르면 시험 문제를 보고 공부한 셈이 된다.

[용어] 시계열 교차검증(TimeSeriesSplit)
  시간 순서를 지키는 교차검증. 항상 '앞 기간으로 학습 → 바로 뒤 기간으로 검증'한다.
  예) [11~12월 학습 → 1월 검증], [11~1월 학습 → 2~3월 검증], ... 미래로 과거를 맞히는 일이 없다.
"""

# 시간 측정
import time

# 그리드 탐색과 시계열 교차검증
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 평가 지표
from sklearn.metrics import roc_auc_score

# 공통 도구
from common import FEATURES, SEED, TARGET, load_ml, make_hgb, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 튜닝은 여러 번 학습하므로 학습 표본을 10만 건으로 줄인다. 시간 순서로 정렬해야 TimeSeriesSplit 이 '앞→뒤'를 지킨다
train = train.sample(100_000, random_state=SEED).sort_values("created")

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]

# ─────────────────────────────────────────────
# 1. 시계열 교차검증 구조 보기
# ─────────────────────────────────────────────
print_header("1. 시계열 교차검증: 각 조각의 기간")

# 4개 조각으로 나누는 시계열 교차검증
tscv = TimeSeriesSplit(n_splits=4)

# 조각별 기간 기록
folds = []

# split() 은 (학습 행 번호, 검증 행 번호) 쌍을 차례로 돌려준다
for i, (tr_idx, va_idx) in enumerate(tscv.split(X_tr), start=1):
    # 학습 기간과 검증 기간을 날짜로 기록
    folds.append({
        "조각": i,
        "학습 기간": f"{train['created'].iloc[tr_idx].min():%Y-%m-%d} ~ {train['created'].iloc[tr_idx].max():%Y-%m-%d}",
        "검증 기간": f"{train['created'].iloc[va_idx].min():%Y-%m-%d} ~ {train['created'].iloc[va_idx].max():%Y-%m-%d}",
        "학습 건수": len(tr_idx), "검증 건수": len(va_idx),
    })

# 표로
folds = pd.DataFrame(folds)

# 출력
print(folds.to_string(index=False))
# 해석: 조각이 뒤로 갈수록 학습 기간이 길어지고, 검증 기간은 항상 학습 기간 '바로 다음'이다. 실제 운영과 같은 순서다.

# 저장
save_table(folds, "ml12_cv_folds")

# ─────────────────────────────────────────────
# 2. 그리드 탐색
# ─────────────────────────────────────────────
print_header("2. 그리드 탐색 (설정 9가지 × 조각 4개 = 학습 36번)")

# 시험할 설정 조합. learning_rate = 학습률, max_leaf_nodes = 나무 하나의 최대 잎 개수(나무 크기)
grid = {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63]}

# GridSearchCV: 모든 조합을 교차검증으로 평가하고 가장 좋은 조합을 고른다
search = GridSearchCV(make_hgb(), grid, cv=tscv, scoring="roc_auc", n_jobs=-1)

# 시작 시각
t0 = time.time()

# 탐색 실행 (평가 데이터는 쓰지 않는다)
search.fit(X_tr, y_tr)

# 걸린 시간
print(f"탐색 시간: {time.time() - t0:.0f}초")

# 결과 표: 설정별 검증 AUC 평균과 표준편차
cv = pd.DataFrame(search.cv_results_)[["param_learning_rate", "param_max_leaf_nodes", "mean_test_score",
                                       "std_test_score", "rank_test_score"]]

# 열 이름을 한국어로
cv.columns = ["학습률", "최대 잎 개수", "검증 AUC 평균", "검증 AUC 표준편차", "순위"]

# 순위 순으로 정렬하고 반올림
cv = cv.sort_values("순위").round(4)

# 출력
print(cv.to_string(index=False))
# 해석: 1등(0.9486)과 꼴찌(0.9469)의 차이가 0.0017뿐이다. 그런데 조각마다 흔들리는 폭(표준편차)은 약 0.012로 7배 크다.
#       즉 이 순위는 우연일 가능성이 크다. 설정을 열심히 고르는 것보다 좋은 특징을 찾는 것이 더 중요하다.

# 저장
save_table(cv, "ml12_grid_results")

# ─────────────────────────────────────────────
# 3. 마지막 시험: 기본 설정 vs 고른 설정
# ─────────────────────────────────────────────
print_header("3. 평가 데이터로 마지막 시험")

# 기본 설정 모델
default = make_hgb()

# 같은 10만 건으로 학습 (공정한 비교)
default.fit(X_tr, y_tr)

# 기본 설정 평가 AUC
auc_default = roc_auc_score(y_te, default.predict_proba(X_te)[:, 1])

# 탐색이 고른 설정 (GridSearchCV 는 끝나면 그 설정으로 학습 데이터 전체를 다시 학습해 둔다)
auc_best = roc_auc_score(y_te, search.best_estimator_.predict_proba(X_te)[:, 1])

# 결과표
final = pd.DataFrame([
    {"모델": "기본 설정 (학습률 0.1, 잎 31)", "검증 AUC(교차검증)": None, "평가 AUC": round(auc_default, 4)},
    {"모델": f"탐색 결과 {search.best_params_}", "검증 AUC(교차검증)": round(search.best_score_, 4), "평가 AUC": round(auc_best, 4)},
])

# 출력
print(final.to_string(index=False))
# 해석: 평가 AUC 0.9588(기본) vs 0.9589(탐색 결과). 0.0001 차이로 사실상 같다. '기본 설정도 충분히 좋다'는 결론이다.
#       교차검증 점수와 평가 점수가 다른 것은 정상이다. 검증 기간(학습 기간 안쪽)과 평가 기간(8~10월)이 다르기 때문이다.

# 저장
save_table(final, "ml12_final_test")

# 그림: 설정별 검증 AUC 히트맵
fig, ax = plt.subplots(figsize=(5.5, 4))

# 표를 (학습률 × 잎 개수) 모양으로 바꾼다
heat = cv.pivot(index="학습률", columns="최대 잎 개수", values="검증 AUC 평균")

# 색 칠하기
im = ax.imshow(heat.values, cmap="Blues")

# x축 눈금
ax.set_xticks(range(heat.shape[1]), heat.columns)

# y축 눈금
ax.set_yticks(range(heat.shape[0]), heat.index)

# 칸마다 숫자 쓰기
for i in range(heat.shape[0]):
    # 열 반복
    for j in range(heat.shape[1]):
        # 숫자
        ax.text(j, i, f"{heat.values[i, j]:.4f}", ha="center", va="center", fontsize=9)

# 축 이름
ax.set_xlabel("최대 잎 개수 (나무 크기)")

# 축 이름
ax.set_ylabel("학습률")

# 제목
ax.set_title("설정별 교차검증 AUC")

# 색 막대
fig.colorbar(im, ax=ax)

# 저장
save_fig(fig, "ml12_grid_heatmap")
