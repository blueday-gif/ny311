"""
[ML 8단계] 결정나무와 과적합

핵심 질문 : 사람이 읽을 수 있는 '만약 ~라면' 규칙으로 맞힐 수 있을까? 나무를 깊게 키우면 무조건 좋아질까?
배우는 개념: 결정나무, 나무 깊이, 과적합(overfitting), 과소적합(underfitting)
사용하는 열: within_24h(타깃), problem, borough, channel, location_type, season, hour, weekday, lat, lon
실행 방법 : py -3.12 analysis/ml_08_tree.py   (먼저 ml_05 실행)

[용어] 결정나무(decision tree)
  "민원 유형이 불법주차인가? → 예 → 접수 시각이 22시 이후인가? → ..." 처럼 예/아니오 질문을 이어 붙여 답을 내는 모델.
  질문 순서를 데이터가 정한다. 그림으로 그리면 사람이 그대로 읽을 수 있다는 것이 큰 장점이다.

[용어] 과적합(overfitting)
  학습 데이터를 너무 꼼꼼히 '외워서', 처음 보는 데이터에서는 오히려 못 맞히는 상태.
  예) 기출문제 답을 통째로 외운 학생은 기출은 100점, 새 문제는 낮은 점수를 받는다.
  반대로 너무 단순해서 학습 데이터조차 못 맞히는 상태는 과소적합(underfitting)이라 한다.
"""

# 결정나무 모델과 그림 그리기 함수
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 정확도
from sklearn.metrics import accuracy_score

# 전처리+모델 묶음
from sklearn.pipeline import Pipeline

# 공통 도구
from common import (FEATURES, SEED, TARGET, load_ml, make_onehot_preprocess, pd, plt, print_header, save_fig,
                    save_table, setup_plot)

# 그래프 한글 설정
setup_plot()

# 표본 불러오기
train, test = load_ml(sample=True)

# 학습 입력/정답
X_tr, y_tr = train[FEATURES], train[TARGET]

# 평가 입력/정답
X_te, y_te = test[FEATURES], test[TARGET]


def make_tree(depth):
    """깊이를 정한 결정나무를 만든다. 범주형은 원-핫, 숫자(시간·요일·위경도)는 그대로 넣는다."""
    # random_state 고정: 같은 점수의 질문이 여럿일 때 고르는 순서를 고정해 결과를 재현한다
    return Pipeline([("pre", make_onehot_preprocess()),
                     ("clf", DecisionTreeClassifier(max_depth=depth, random_state=SEED))])


# ─────────────────────────────────────────────
# 1. 깊이 3짜리 작은 나무: 규칙 읽기
# ─────────────────────────────────────────────
print_header("1. 깊이 3 나무: 사람이 읽을 수 있는 규칙")

# 깊이 3 = 질문을 최대 3번까지만 한다
small = make_tree(3)

# 학습
small.fit(X_tr, y_tr)

# 원-핫 후 열 이름 (그림에 질문 내용을 쓰기 위해)
names = list(small[:-1].get_feature_names_out())

# 규칙을 글자로 출력한다
rules = export_text(small[-1], feature_names=names, show_weights=False)

# 출력
print(rules)
# 해석: "location_type_Street/Sidewalk <= 0.50" 은 "장소 유형이 Street/Sidewalk 가 아니면"이라는 뜻이다(원-핫 열은 0 또는 1).
#       맨 처음 질문이 가장 중요한 갈림길이다. 이 나무는 '장소 유형'을 먼저 묻는다. 장소 유형이 담당 기관을 알려 주기 때문이다(ml_07 참고).
#       양쪽 잎이 모두 같은 답(class: 1, class: 1)인 갈림도 있다. 답은 같아도 한쪽이 더 '확실한' 무리로 나뉘어서 나무는 그 질문을 고른다.

# 정확도
acc_small = accuracy_score(y_te, small.predict(X_te))

# 출력
print(f"깊이 3 나무 평가 정확도: {acc_small:.3f}")
# 해석: 질문 3번만으로 0.853. 다수결 기준선(0.580)보다는 훨씬 높지만, 민원 유형 규칙(0.882)보다는 낮다. 질문 3개로는 부족하다.

# 나무 그림
fig, ax = plt.subplots(figsize=(18, 8))

# plot_tree: 상자 하나 = 질문 하나. filled=True 는 정답 비율에 따라 색을 칠한다
plot_tree(small[-1], feature_names=names, class_names=["24h 초과", "24h 이내"], filled=True,
          impurity=False, proportion=True, fontsize=8, ax=ax)

# 제목
ax.set_title("깊이 3 결정나무 (상자 안 value = [24h 초과 비율, 24h 이내 비율])")

# 저장
save_fig(fig, "ml08_tree_depth3")

# 규칙 글자도 파일로 남긴다 (표 형식이 아니라 한 열짜리 표로 저장)
save_table(pd.DataFrame({"규칙": rules.splitlines()}), "ml08_tree_depth3_rules")

# ─────────────────────────────────────────────
# 2. 깊이를 바꿔 가며: 과적합 찾기
# ─────────────────────────────────────────────
print_header("2. 깊이에 따른 학습/평가 정확도")

# 시험할 깊이 목록. None = 제한 없음(끝까지 키움)
depths = [1, 2, 3, 5, 8, 12, 16, 20, 30, None]

# 결과 목록
rows = []

# 깊이마다 나무를 새로 학습한다
for d in depths:
    # 나무 만들기
    tree = make_tree(d)
    # 학습
    tree.fit(X_tr, y_tr)
    # 학습 정확도
    a_tr = accuracy_score(y_tr, tree.predict(X_tr))
    # 평가 정확도
    a_te = accuracy_score(y_te, tree.predict(X_te))
    # 실제 깊이 (None 일 때 얼마나 깊어졌는지 확인)
    real = tree[-1].get_depth()
    # 잎(마지막 상자) 개수 = 규칙 개수
    leaves = tree[-1].get_n_leaves()
    # 기록
    rows.append({"설정 깊이": "제한 없음" if d is None else d, "실제 깊이": real, "잎 개수": leaves,
                 "학습 정확도": round(a_tr, 4), "평가 정확도": round(a_te, 4), "차이(학습-평가)": round(a_tr - a_te, 4)})
    # 진행 상황 출력
    print(f"  깊이 {str(d):>4}: 학습 {a_tr:.3f} / 평가 {a_te:.3f} (잎 {leaves:,}개)")

# 표로
curve = pd.DataFrame(rows)

# 저장
save_table(curve, "ml08_depth_curve")

# 평가 정확도가 가장 높은 깊이
best = curve.loc[curve["평가 정확도"].idxmax()]

# 출력
print(f"평가 정확도가 가장 높은 설정: 깊이 {best['설정 깊이']} (평가 {best['평가 정확도']:.3f})")
# 해석: 깊이 1~2는 학습·평가 모두 낮다(과소적합). 깊이 12까지는 둘 다 오르다가 평가 0.894에서 정점을 찍는다.
#       그 뒤로는 학습 정확도만 계속 오르고(깊이 제한 없음: 0.996) 평가 정확도는 0.856으로 떨어진다(과적합).
#       제한 없는 나무는 잎이 2만 개가 넘는다. 학습 데이터를 거의 통째로 외운 것이다.

# 그림
fig, ax = plt.subplots(figsize=(8, 4.5))

# x축 위치 (None 은 숫자가 아니라서 순서 번호를 쓴다)
x = range(len(curve))

# 학습 정확도 선
ax.plot(x, curve["학습 정확도"], marker="o", label="학습 데이터 (연습 문제)")

# 평가 정확도 선
ax.plot(x, curve["평가 정확도"], marker="o", label="평가 데이터 (새 문제)")

# x축 눈금 글자
ax.set_xticks(list(x), [str(v) for v in curve["설정 깊이"]])

# x축 이름
ax.set_xlabel("나무 최대 깊이")

# y축 이름
ax.set_ylabel("정확도")

# 제목
ax.set_title("깊어질수록 학습 점수만 오른다 = 과적합")

# 범례
ax.legend()

# 저장
save_fig(fig, "ml08_depth_curve")
