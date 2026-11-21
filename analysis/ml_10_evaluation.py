"""
[ML 10단계] 평가 지표: 정확도만 보면 왜 위험한가

핵심 질문 : '좋은 모델'을 숫자 하나(정확도)로 말해도 되나? 기준값 0.5 를 바꾸면 무엇이 달라지나?
배우는 개념: 정밀도(precision), 재현율(recall), F1, ROC 곡선, 기준값(threshold), 보정(calibration), 클래스 불균형
사용하는 열: within_24h(타깃), FEATURES 전체 (그래디언트 부스팅 모델 사용)
실행 방법 : py -3.12 analysis/ml_10_evaluation.py   (먼저 ml_05 실행)
기존 앱 참고: 7 현대 방법(AUC, 보정) 탭

이 단계에서는 '오래 걸릴 민원(24시간 초과)'을 찾는 것을 목표로 본다.
  → 담당자가 미리 챙겨야 할 민원을 골라내는 상황. 그래서 '양성(찾고 싶은 쪽)' = 24시간 초과(타깃 0)로 뒤집어 계산한다.

[용어] 정밀도(precision)
  모델이 "오래 걸린다"고 찍은 것 중 실제로 오래 걸린 비율. 헛짚기를 얼마나 안 하나.
  예) 100건을 골랐는데 80건이 진짜 오래 걸렸다면 정밀도 0.8.

[용어] 재현율(recall)
  실제로 오래 걸린 민원 중 모델이 찾아낸 비율. 놓치기를 얼마나 안 하나.
  예) 실제 오래 걸린 민원이 200건인데 그중 150건을 찾았다면 재현율 0.75.

[용어] F1 점수
  정밀도와 재현율의 조화평균. 둘 중 하나만 높고 다른 하나가 낮으면 F1 이 낮아진다.

[용어] 기준값(threshold)
  확률이 몇 이상이면 '양성'으로 찍을지 정하는 선. 기본은 0.5 지만 쓰임새에 따라 바꾼다.
  예) 기준을 0.3 으로 낮추면 더 많이 찍어서 재현율↑ 정밀도↓. 0.7 로 올리면 반대.
"""

# 평가 지표 함수들
from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# 보정 곡선 그림
from sklearn.calibration import CalibrationDisplay

# 공통 도구
from common import FEATURES, TARGET, load_ml, make_hgb, np, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]

# ─────────────────────────────────────────────
# 1. 모델 학습 (ml_09 에서 가장 나았던 그래디언트 부스팅)
# ─────────────────────────────────────────────
print_header("1. 그래디언트 부스팅 학습")

# 모델 만들기
model = make_hgb()

# 학습
model.fit(X_tr, y_tr)

# '24시간 초과(타깃 0)일 확률' = 1 - '24시간 이내일 확률'. 찾고 싶은 쪽을 양성으로 놓기 위해 뒤집는다
p_slow = model.predict_proba(X_te)[:, 0]

# 정답도 뒤집는다: 24시간 초과 = 1
y_slow = 1 - y_te.values

# 양성 비율 출력
print(f"평가 데이터에서 '24시간 초과' 비율: {y_slow.mean() * 100:.1f}%")
# 해석: 42.1%. 0과 1이 심하게 쏠리지 않은 문제라서 정확도가 크게 속이지는 않는다. 아래 4절에서 쏠린 경우를 따로 본다.

# ─────────────────────────────────────────────
# 2. 기준값에 따른 지표 변화
# ─────────────────────────────────────────────
print_header("2. 기준값(threshold)에 따른 정밀도·재현율")

# 시험할 기준값들
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 결과 목록
rows = []

# 기준값마다 계산
for t in thresholds:
    # 확률이 기준 이상이면 '오래 걸린다(1)'로 찍는다
    pred = (p_slow >= t).astype(int)
    # 지표 계산 후 기록. zero_division=0 은 하나도 안 찍었을 때 오류 대신 0을 쓰라는 뜻
    rows.append({
        "기준값": t,
        "찍은 건수": int(pred.sum()),
        "정확도": round(accuracy_score(y_slow, pred), 4),
        "정밀도": round(precision_score(y_slow, pred, zero_division=0), 4),
        "재현율": round(recall_score(y_slow, pred), 4),
        "F1": round(f1_score(y_slow, pred), 4),
    })

# 표로
thr = pd.DataFrame(rows)

# 출력
print(thr.to_string(index=False))
# 해석: 기준값 0.2 → 재현율 0.98 / 정밀도 0.81, 기준값 0.8 → 재현율 0.71 / 정밀도 0.91. 낮추면 놓치는 민원이 줄고 헛짚기가 는다.
#       정확도와 F1 은 기준값 0.3 근처에서 가장 높다. 기본값 0.5 가 항상 최선은 아니다.
#       '오래 걸릴 민원을 하나도 놓치면 안 된다'면 낮은 기준값, '담당자 시간이 부족해 확실한 것만 보겠다'면 높은 기준값.
#       어느 기준이 '정답'인지는 데이터가 아니라 쓰임새(업무 비용)가 정한다.

# 저장
save_table(thr, "ml10_threshold_table")

# ─────────────────────────────────────────────
# 3. ROC 곡선, 정밀도-재현율 곡선, 보정 곡선
# ─────────────────────────────────────────────
print_header("3. ROC·PR·보정 곡선")

# [용어] ROC 곡선: 기준값을 1→0 으로 내리며 (헛짚은 비율, 찾아낸 비율)을 이은 선. 왼쪽 위로 붙을수록 좋다. 아래 넓이 = AUC.
# [용어] 보정(calibration): 모델이 "확률 0.7"이라고 한 민원들 중 실제로 70%가 그렇게 되는가. 대각선에 붙을수록 확률을 믿을 수 있다.

# AUC 계산
auc = roc_auc_score(y_slow, p_slow)

# 출력
print(f"AUC = {auc:.3f}")
# 해석: AUC 0.960. 오래 걸린 민원 하나와 빨리 끝난 민원 하나를 뽑으면, 96%의 확률로 오래 걸린 쪽에 더 높은 위험도를 준다.

# 그림 세 칸
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ROC 곡선
RocCurveDisplay.from_predictions(y_slow, p_slow, ax=axes[0], name="그래디언트 부스팅")

# 찍기(대각선) 참고선
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="찍기 (AUC 0.5)")

# 제목
axes[0].set_title(f"ROC 곡선 (AUC {auc:.3f})")

# 범례
axes[0].legend(loc="lower right")

# 정밀도-재현율 곡선
PrecisionRecallDisplay.from_predictions(y_slow, p_slow, ax=axes[1], name="그래디언트 부스팅")

# 양성 비율(찍기 수준) 참고선
axes[1].axhline(y_slow.mean(), color="k", ls="--", lw=1, label=f"찍기 수준 ({y_slow.mean():.2f})")

# 제목
axes[1].set_title("정밀도-재현율 곡선")

# 범례
axes[1].legend(loc="lower left")

# 보정 곡선: 확률을 10구간으로 나눠 구간별 실제 비율을 그린다
CalibrationDisplay.from_predictions(y_slow, p_slow, n_bins=10, ax=axes[2], name="그래디언트 부스팅")

# 제목
axes[2].set_title("보정 곡선 (대각선에 가까울수록 확률을 믿을 수 있다)")

# 저장
save_fig(fig, "ml10_roc_pr_calibration")
# 해석: 보정 곡선이 대각선 근처에 있으면 "확률 0.8"이라는 말을 그대로 믿고 업무 우선순위에 쓸 수 있다.

# ─────────────────────────────────────────────
# 4. 클래스 불균형: 정답이 한쪽으로 쏠리면 정확도는 속인다
# ─────────────────────────────────────────────
print_header("4. 클래스 불균형 실험")

# [용어] 클래스 불균형(class imbalance): 정답 중 한쪽이 훨씬 많은 상태. 예) 1000건 중 10건만 양성.
#        이때 '항상 음성'이라고 하면 정확도 99%지만 양성을 하나도 못 찾는다.

# 일부러 드문 사건을 만든다: '처리에 30일(720시간) 넘게 걸린 민원'
# 관측 끝 30일 이내에 접수된 미종결 민원은 30일을 넘길지 아직 모르므로, 접수 후 30일 넘게 지켜본 민원만 쓴다
old_enough = test["created"] <= test["created"].max() - pd.Timedelta(days=30)

# 대상 민원의 처리 시간. 미종결(빈 값)은 30일 넘게 열려 있으므로 무한대로 채워 '30일 초과'로 센다
h = test.loc[old_enough, "hours"].fillna(np.inf)

# 30일 초과면 1
very_slow = (h > 720).astype(int).values

# 몇 건을 뺐는지 출력 (제외 기준을 숨기지 않는다)
print(f"관측 끝 30일 이내 접수라 제외: {(~old_enough).sum():,}건 / 남은 {old_enough.sum():,}건")

# 드문 사건 비율
rate = very_slow.mean()

# 출력
print(f"30일 초과 민원 비율: {rate * 100:.1f}%")

# '항상 0'으로 찍었을 때 정확도
acc_all0 = 1 - rate

# 출력
print(f"'항상 30일 이내'라고 찍으면 정확도 {acc_all0:.3f}, 하지만 재현율 0 (하나도 못 찾음)")
# 해석: 30일 초과 민원은 9.7%뿐이라, 아무것도 안 배운 '항상 30일 이내'가 정확도 0.903을 받는다.
#       ml_09 의 진짜 모델(0.897)보다 높아 보이지만 쓸모는 0이다. 드문 사건을 찾는 문제에서는 정확도 대신 재현율·정밀도·AUC 를 봐야 한다.

# 지표 정리표를 저장한다
save_table(pd.DataFrame([
    {"항목": "24시간 초과 비율(평가)", "값": round(y_slow.mean(), 4)},
    {"항목": "AUC (24시간 초과 찾기)", "값": round(auc, 4)},
    {"항목": "30일 초과 비율(평가)", "값": round(rate, 4)},
    {"항목": "'항상 30일 이내' 정확도", "값": round(acc_all0, 4)},
]), "ml10_summary")
