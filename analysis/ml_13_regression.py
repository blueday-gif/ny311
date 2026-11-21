"""
[ML 13단계] 회귀: 처리 시간 자체를 맞힐 수 있나

핵심 질문 : '24시간 안인가 아닌가'가 아니라 '몇 시간 걸릴까'를 맞힐 수 있을까?
배우는 개념: 회귀(regression), MAE, 중앙절대오차, R², 로그 타깃, 중도절단(censoring)
사용하는 열: hours(처리 시간), FEATURES 전체
실행 방법 : py -3.12 analysis/ml_13_regression.py   (먼저 ml_05 실행)
기존 앱 참고: 6 검정·회귀(선형 회귀), 7 현대 방법(분위수 회귀), 10 생존분석 탭

[용어] 회귀(regression)
  숫자 값(처리 시간)을 맞히는 문제. 분류(0/1 맞히기)와 짝을 이루는 머신러닝의 두 기본 문제 중 하나다.

[용어] MAE (평균 절대 오차)
  |실제 - 예측| 의 평균. "평균적으로 몇 시간 틀리나"로 바로 읽을 수 있다.
  예) 실제 10, 20 / 예측 12, 15 → (2 + 5) / 2 = 3.5시간.

[용어] R² (결정계수)
  예측이 '평균값 하나로 찍기'보다 얼마나 나은지. 1 = 완벽, 0 = 평균 찍기와 같음, 음수 = 평균 찍기보다 못함.

[용어] 로그 타깃
  처리 시간은 왜도가 6 이상으로 매우 치우쳐 있다(eda_03). 그대로 맞히면 수천 시간짜리 소수 민원에 모델이 끌려간다.
  log1p(처리 시간)을 맞힌 뒤, 예측값을 expm1 으로 되돌려 '시간' 단위로 평가한다.
"""

# 선형 회귀(Ridge: 가중치가 너무 커지지 않게 살짝 제한한 선형 회귀)
from sklearn.linear_model import Ridge

# 그래디언트 부스팅 회귀
from sklearn.ensemble import HistGradientBoostingRegressor

# 평가 지표
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

# 전처리+모델 묶음
from sklearn.pipeline import Pipeline

# 공통 도구
from common import (FEATURES, SEED, ExclusionLog, load_ml, make_onehot_preprocess, np, pd, plt, print_header,
                    save_fig, save_table, setup_plot)

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# ─────────────────────────────────────────────
# 1. 대상: 종결된 민원만
# ─────────────────────────────────────────────
print_header("1. 회귀 대상 정하기")

# 제외 로그 (표본 기준)
log = ExclusionLog("ml13", len(train) + len(test))

# 미종결 민원 수
n_open = int(train["hours"].isna().sum() + test["hours"].isna().sum())

# 기록
log.add("미종결 민원 (처리 시간을 아직 모름)", n_open)

# 로그 저장
log.save()

# [용어] 중도절단(censoring): '아직 끝나지 않아서 최종 값을 모르는' 관측. 예) 30일째 처리 중인 민원은 최소 30일이 걸린다는 것만 안다.
#        이런 민원을 빼면 오래 걸리는 민원이 덜 들어가서 모델이 처리 시간을 짧게 예측하는 쪽으로 치우친다.
#        제대로 다루려면 생존분석(기존 앱 10단계)이 필요하다. 여기서는 빼고, 이 한계를 기록한다.

# 종결된 민원만 남긴다
tr = train.dropna(subset=["hours"])

# 평가도 같은 기준
te = test.dropna(subset=["hours"])

# 학습 입력
X_tr = tr[FEATURES]

# 학습 정답: log1p(처리 시간)
y_tr = np.log1p(tr["hours"])

# 평가 입력
X_te = te[FEATURES]

# 평가 정답(시간 단위 그대로, 마지막 평가에 쓴다)
h_te = te["hours"].values

# ─────────────────────────────────────────────
# 2. 기준선과 모델
# ─────────────────────────────────────────────
print_header("2. 기준선과 모델 비교")

# 예측값을 모을 사전: 이름 → 시간 단위 예측
preds = {}

# 기준선 ①: 학습 데이터 처리 시간의 중앙값 하나로 모두 찍기
preds["① 전체 중앙값 하나로 찍기"] = np.full(len(te), tr["hours"].median())

# 기준선 ②: 민원 유형별 중앙값으로 찍기 (ml_06 의 '유형 규칙'과 같은 생각)
med_by_problem = tr.groupby("problem", observed=True)["hours"].median()

# 평가 민원의 유형에 맞는 중앙값을 가져온다. 학습에 없던 유형은 전체 중앙값으로
preds["② 민원 유형별 중앙값"] = te["problem"].astype(str).map(med_by_problem.rename(index=str)).fillna(
    tr["hours"].median()).values

# 모델 ③: Ridge 선형 회귀 (로그 타깃). alpha 는 가중치를 얼마나 제한할지
ridge = Pipeline([("pre", make_onehot_preprocess(numeric=[], extra_cat=["hour", "weekday"])), ("reg", Ridge(alpha=1.0))])

# 학습
ridge.fit(X_tr, y_tr)

# 예측(로그) → expm1 으로 시간 단위로 되돌린다
preds["③ 선형 회귀 (Ridge, 로그 타깃)"] = np.expm1(ridge.predict(X_te))

# 모델 ④: 그래디언트 부스팅 회귀 (로그 타깃)
hgb = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300, random_state=SEED)

# 학습
hgb.fit(X_tr, y_tr)

# 예측 → 시간 단위
preds["④ 그래디언트 부스팅 (로그 타깃)"] = np.expm1(hgb.predict(X_te))

# 평가표
rows = []

# 예측마다 지표 계산
for name, p in preds.items():
    # 음수 예측은 0으로 자른다 (처리 시간은 음수가 될 수 없다)
    p = np.clip(p, 0, None)
    # 지표 기록
    rows.append({
        "방법": name,
        "MAE(시간)": round(mean_absolute_error(h_te, p), 1),
        "중앙절대오차(시간)": round(median_absolute_error(h_te, p), 2),
        "R²(로그 기준)": round(r2_score(np.log1p(h_te), np.log1p(p)), 3),
    })

# 표로
reg = pd.DataFrame(rows)

# 출력
print(reg.to_string(index=False))
# 해석: MAE(평균 오차)는 76~99시간으로 크다. 몇 달 걸린 소수 민원을 크게 틀리기 때문이다.
#       중앙절대오차는 '보통 민원'에서 몇 시간 틀리는지다. 전체 중앙값 8.4시간 → 유형별 중앙값 6.7 → 부스팅 5.2시간으로 줄어든다.
#       MAE 는 선형 회귀(76.0)가, 중앙절대오차와 R² 는 부스팅(5.21, 0.648)이 가장 좋다. 지표마다 1등이 다를 수 있다.
#       R²(로그 기준): 전체 중앙값 -0.03(평균 찍기보다도 못함) → 유형별 중앙값 0.615 → 모델 0.64~0.65.
#       이번에도 '민원 유형'이 대부분을 설명하고, 모델은 그 위에 조금 더한다.

# 저장
save_table(reg, "ml13_regression_results")

# ─────────────────────────────────────────────
# 3. 그림: 실제 vs 예측 (로그 눈금)
# ─────────────────────────────────────────────

# 점이 너무 많아서 5천 개만 그린다
idx = np.random.default_rng(SEED).choice(len(te), 5000, replace=False)

# 그림
fig, ax = plt.subplots(figsize=(6, 6))

# 부스팅 예측 (로그)
p_log = np.log1p(np.clip(preds["④ 그래디언트 부스팅 (로그 타깃)"], 0, None))

# 산점도: x = 실제(로그), y = 예측(로그)
ax.scatter(np.log1p(h_te[idx]), p_log[idx], s=4, alpha=0.3)

# 완벽한 예측선 (y = x)
lim = [0, np.log1p(h_te).max()]

# 대각선
ax.plot(lim, lim, "r--", lw=1, label="완벽한 예측 (y = x)")

# 축 이름
ax.set_xlabel("실제 log(1 + 처리 시간)")

# 축 이름
ax.set_ylabel("예측 log(1 + 처리 시간)")

# 제목
ax.set_title("그래디언트 부스팅: 실제 vs 예측 (5천 건)")

# 범례
ax.legend()

# 저장
save_fig(fig, "ml13_actual_vs_pred")
# 해석: 점들이 가로 줄무늬처럼 몰려 있다면, 모델이 '민원 유형별 대표 시간' 몇 개만 말하고 있다는 뜻이다.
#       같은 유형 안에서 처리 시간이 크게 갈리는 이유(담당자, 현장 상황)는 이 데이터에 없어서 맞히기 어렵다.
