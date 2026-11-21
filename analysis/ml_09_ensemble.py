"""
[ML 9단계] 앙상블: 나무를 여러 그루 쓰면 나아지나

핵심 질문 : 나무 하나 대신 여러 그루를 합치면 더 잘 맞힐까? 지금까지 만든 모델 중 무엇이 가장 나은가?
배우는 개념: 앙상블, 랜덤포레스트(배깅), 그래디언트 부스팅, 모델 비교표
사용하는 열: within_24h(타깃), problem, borough, channel, location_type, season, hour, weekday, lat, lon
실행 방법 : py -3.12 analysis/ml_09_ensemble.py   (먼저 ml_05 실행)
기존 앱 참고: 7 현대 방법(그래디언트 부스팅) 탭

[용어] 앙상블(ensemble)
  여러 모델의 답을 모아 최종 답을 정하는 방법. 한 사람보다 여러 사람의 의견을 모으면 덜 틀리는 것과 같다.

[용어] 랜덤포레스트(random forest)
  데이터를 조금씩 다르게 뽑아 깊은 나무를 수백 그루 키우고, 다수결로 답한다. 나무 하나의 과적합이 서로 상쇄된다.
  예) 나무 100그루 중 70그루가 '24시간 안 처리'라고 하면 확률 0.7.

[용어] 그래디언트 부스팅(gradient boosting)
  얕은 나무를 순서대로 키우되, 다음 나무는 '앞 나무들이 틀린 부분'을 고치도록 배운다. 오답 노트를 계속 보완하는 방식.
  표 형태 데이터에서 가장 많이 쓰이는 강력한 방법이다.

[용어] AUC (ROC 곡선 아래 넓이)
  '1인 민원'과 '0인 민원'을 하나씩 뽑았을 때, 모델이 1인 쪽에 더 높은 확률을 줄 가능성. 0.5 = 찍기, 1.0 = 완벽.
  정확도와 달리 0.5 같은 기준값을 정하지 않고 '순서를 얼마나 잘 매기나'를 본다 (ml_10 에서 자세히).
"""

# 시간 측정
import time

# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier

# 다수결 기준선
from sklearn.dummy import DummyClassifier

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

# 평가 지표
from sklearn.metrics import accuracy_score, roc_auc_score

# 전처리+모델 묶음
from sklearn.pipeline import Pipeline

# 결정나무
from sklearn.tree import DecisionTreeClassifier

# 공통 도구
from common import (FEATURES, SEED, TAB_DIR, TARGET, load_ml, make_hgb, make_onehot_preprocess, pd, plt,
                    print_header, save_fig, save_table, setup_plot)

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]

# ─────────────────────────────────────────────
# 1. 비교할 모델 목록
# ─────────────────────────────────────────────
print_header("1. 모델 학습과 비교")

# 이름 → 모델. ml_06 ~ ml_08 에서 만든 것과 같은 설정을 다시 만들어 한 표에서 공정하게 비교한다
models = {
    # 기준선: 항상 다수 쪽
    "① 다수결 (기준선)": DummyClassifier(strategy="most_frequent"),
    # 로지스틱 회귀 (ml_07 과 같은 설정)
    "② 로지스틱 회귀": Pipeline([("pre", make_onehot_preprocess(numeric=[], extra_cat=["hour", "weekday"])),
                            ("clf", LogisticRegression(max_iter=2000))]),
    # 결정나무 깊이 12 (ml_08 에서 평가 정확도가 가장 높았던 깊이)
    "③ 결정나무 (깊이 12)": Pipeline([("pre", make_onehot_preprocess()),
                               ("clf", DecisionTreeClassifier(max_depth=12, random_state=SEED))]),
    # 랜덤포레스트: 나무 100그루. min_samples_leaf=5 는 잎 하나에 최소 5건이 있어야 한다는 규칙(지나친 외우기 방지)
    # n_jobs=-1 은 컴퓨터의 모든 CPU 코어를 써서 빠르게 학습하라는 뜻
    "④ 랜덤포레스트 (100그루)": Pipeline([("pre", make_onehot_preprocess()),
                                  ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                                 n_jobs=-1, random_state=SEED))]),
    # 그래디언트 부스팅: 범주형을 원-핫 없이 그대로 다룬다
    "⑤ 그래디언트 부스팅": make_hgb(),
}

# 결과 목록
rows = []

# 모델마다 학습 → 평가
for name, model in models.items():
    # 학습 시작 시각
    t0 = time.time()
    # 학습
    model.fit(X_tr, y_tr)
    # 학습 시간
    sec = time.time() - t0
    # 평가 데이터의 '1일 확률'
    proba = model.predict_proba(X_te)[:, 1]
    # 0.5 기준으로 0/1 예측
    pred = (proba >= 0.5).astype(int)
    # 학습 정확도 (과적합 확인용)
    acc_tr = accuracy_score(y_tr, model.predict(X_tr))
    # 평가 정확도
    acc_te = accuracy_score(y_te, pred)
    # 평가 AUC. 다수결은 확률이 모두 같아 0.5 가 나온다
    auc = roc_auc_score(y_te, proba)
    # 기록
    rows.append({"모델": name, "학습 정확도": round(acc_tr, 4), "평가 정확도": round(acc_te, 4),
                 "평가 AUC": round(auc, 4), "학습 시간(초)": round(sec, 1)})
    # 진행 상황
    print(f"  {name:<20} 평가 정확도 {acc_te:.3f}  AUC {auc:.3f}  ({sec:.1f}초)")

# 표로
comp = pd.DataFrame(rows)

# ml_06 의 '민원 유형 규칙' 기준선도 표에 넣는다 (같은 표본, 같은 평가 데이터로 계산한 값)
rule = pd.read_csv(TAB_DIR / "ml06_baselines.csv", encoding="utf-8-sig")

# 규칙 기준선 행
rule_row = rule[rule["방법"].str.contains("민원 유형")].iloc[0]

# 표 두 번째 줄에 끼워 넣는다
comp = pd.concat([comp.iloc[:1],
                  pd.DataFrame([{"모델": "①' 민원 유형별 규칙 (기준선)", "평가 정확도": rule_row["평가 정확도"]}]),
                  comp.iloc[1:]], ignore_index=True)

# 출력
print(comp.to_string(index=False))
# 해석:
#   - 평가 정확도: 다수결 0.580 → 민원 유형 규칙 0.882 → 로지스틱 0.896, 나무 0.894, 랜덤포레스트 0.898, 부스팅 0.897.
#   - AUC 는 부스팅 0.960 > 랜덤포레스트 0.958 > 로지스틱 0.953 > 나무 0.936. 여러 나무를 합치면 나무 하나보다 순서를 더 잘 매긴다.
#   - 랜덤포레스트는 학습 0.913 / 평가 0.898로 차이가 가장 크다(약간의 외우기). 부스팅은 0.897 / 0.897로 거의 같다.
#   - 하지만 모델 사이 차이는 0.4%p 안쪽이고, 규칙 기준선보다 약 1.5%p 높다.
#     "복잡한 모델이 항상 훨씬 좋다"가 아니라 "핵심 정보(민원 유형)가 대부분을 설명한다"가 이 데이터의 교훈이다.
#     이럴 때는 해석하기 쉬운 로지스틱 회귀를 고르는 것도 좋은 선택이다.

# 저장
save_table(comp, "ml09_model_comparison")

# ─────────────────────────────────────────────
# 2. 비교 그림
# ─────────────────────────────────────────────

# 그림 두 칸: 정확도, AUC
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 정확도 막대 (아래→위 순서로 그리려고 뒤집는다)
c = comp.iloc[::-1]

# 정확도 가로 막대
axes[0].barh(c["모델"], c["평가 정확도"], color="#4C72B0")

# x축 범위: 차이를 잘 보이게 0.5부터
axes[0].set_xlim(0.5, 1.0)

# 제목
axes[0].set_title("평가 정확도")

# 막대 끝에 숫자 쓰기
for i, v in enumerate(c["평가 정확도"]):
    # 숫자 표시
    axes[0].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

# AUC 는 규칙 기준선 값이 없어서(NaN) 그 행을 빼고 그린다
c2 = c.dropna(subset=["평가 AUC"])

# AUC 가로 막대
axes[1].barh(c2["모델"], c2["평가 AUC"], color="#55A868")

# x축 범위
axes[1].set_xlim(0.5, 1.0)

# 제목
axes[1].set_title("평가 AUC (0.5 = 찍기)")

# 막대 끝 숫자
for i, v in enumerate(c2["평가 AUC"]):
    # 숫자 표시
    axes[1].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

# 저장
save_fig(fig, "ml09_model_comparison")
