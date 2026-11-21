"""
[ML 5단계] 문제 정의: 무엇을, 무엇으로 예측할 것인가

핵심 질문 : 민원이 접수되는 순간, 그 민원이 24시간 안에 처리될지 맞힐 수 있을까?
배우는 개념: 타깃(정답), 특징(feature), 데이터 누수(leakage), 학습/평가 기간 나누기
사용하는 열: created, closed, problem, borough, channel, location_type, lat, lon,
             Resolution Action Updated Date(누수 실습에만 사용)
실행 방법 : py -3.12 analysis/ml_05_problem.py
결과 파일 : data/processed/ml_within24h.parquet (ml_06 ~ ml_13 이 이 파일을 읽는다)
기존 앱 참고: 6 검정·회귀(④ 처리 시간 예측), 7 현대 방법 탭

[용어] 머신러닝(machine learning)
  정답이 있는 과거 사례를 많이 보여 주고, 컴퓨터가 '입력 → 정답' 규칙을 스스로 찾게 하는 방법.
  예) 과거 민원 수십만 건의 (유형, 자치구, 시간) → (24시간 안 처리 여부)를 보고 새 민원의 결과를 맞힌다.

[용어] 타깃(target, 정답)과 특징(feature, 입력)
  타깃 = 맞히고 싶은 값. 특징 = 맞힐 때 참고하는 값.
  예) 타깃: within_24h(1 = 24시간 안 종결, 0 = 아님). 특징: 민원 유형, 자치구, 접수 시각 등.

[용어] 데이터 누수(data leakage)
  실제로 예측하는 순간에는 알 수 없는 정보가 학습에 섞여 들어가는 것. 점수가 비정상적으로 좋아진다.
  예) '종결 시각'을 특징으로 넣으면 24시간 안 처리 여부를 그냥 계산할 수 있다. 시험 문제와 답지를 같이 주는 셈.
"""

# 시간 측정용. 학습에 걸린 시간을 보여 준다
import time

# 모델 평가 지표: ROC-AUC (ml_10 에서 자세히 배운다. 여기서는 '높을수록 잘 구분한다' 정도로만 본다)
from sklearn.metrics import roc_auc_score

# 공통 도구
from common import (CAT_FEATURES, DATA_PATH, FEATURES, ML_PATH, NUM_FEATURES, SEED, SPLIT_DATE, TARGET,
                    ExclusionLog, load_data, make_hgb, np, pd, plt, print_header, save_fig, save_table,
                    season_of, setup_plot)

# 그래프 한글 설정
setup_plot()

# 데이터 불러오기
df = load_data()

# ─────────────────────────────────────────────
# 1. 타깃 만들기: within_24h
# ─────────────────────────────────────────────
print_header("1. 타깃(within_24h) 만들기")

# 데이터의 마지막 접수 시각. 이 시각 이후는 관측하지 못했다
data_end = df["created"].max()

# 제외 로그 시작
log = ExclusionLog("ml05", len(df))

# 규칙 1: 처리 시간 음수 = 입력 오류 → 정답을 믿을 수 없어서 뺀다
bad = df["hours"] < 0

# 기록
log.add("처리 시간 < 0 (입력 오류)", bad.sum())

# 규칙 2: 아직 종결되지 않았고, 접수 후 24시간이 아직 지나지 않은 민원 → 24시간 안에 끝날지 '아직 모른다'
undecided = df["closed"].isna() & (df["created"] > data_end - pd.Timedelta(hours=24))

# 기록
log.add("미종결 + 관측 끝 24시간 이내 접수 (정답 판정 불가)", undecided.sum())

# 두 규칙에 해당하지 않는 행만 남긴다. copy() 는 원본 df 와 분리된 새 표를 만든다
ml = df.loc[~bad & ~undecided].copy()

# 로그 저장
log.save()

# 타깃: 종결됐고 처리 시간이 24시간 이하이면 1, 그 외(24시간 초과 또는 24시간 넘게 미종결)는 0
ml[TARGET] = (ml["hours"] <= 24).astype(int)
# 설명: 미종결 민원은 hours 가 NaN 이라 비교 결과가 False → 0 이 된다. 24시간 넘게 열려 있으니 0 이 맞다.

# 전체 비율 출력
print(f"분석 대상 {len(ml):,}건 중 24시간 안 처리 비율: {ml[TARGET].mean() * 100:.1f}%")
# 해석: 57.4%가 24시간 안에 처리됐다. 정답 1과 0이 반반에 가까워서 '한쪽으로만 찍기'가 잘 통하지 않는 문제다 (ml_06 기준선).

# ─────────────────────────────────────────────
# 2. 학습/평가 기간 나누기
# ─────────────────────────────────────────────
print_header("2. 학습 기간 / 평가 기간")

# [용어] 학습 데이터(train)와 평가 데이터(test)
#   학습 데이터로 규칙을 배우고, 한 번도 보지 않은 평가 데이터로 실력을 잰다. 연습 문제와 시험 문제를 나누는 것.
#   시간 순서가 있는 자료는 '과거로 배우고 미래로 시험'해야 한다. 무작위로 섞으면 미래 정보로 과거를 맞히는 셈이 된다.

# SPLIT_DATE(2026-08-01) 이전 = train, 이후 = test
ml["split"] = np.where(ml["created"] < SPLIT_DATE, "train", "test")

# 기간별 건수와 타깃 비율
split_tab = ml.groupby("split")[TARGET].agg(건수="size", 처리율_24h=lambda s: round(s.mean() * 100, 2))

# 기간 정보 추가
split_tab["기간"] = ["2026-08-01 ~ 2026-10-31", "2025-11-01 ~ 2026-07-31"]

# 출력
print(split_tab.to_string())
# 해석: 학습 9개월(약 297만 건), 평가 3개월(약 100만 건). 24시간 처리율은 57.3% vs 57.9%로 거의 같다.
#       두 기간의 비율이 크게 달랐다면 '세상이 바뀐 것(분포 변화)'이라 평가 점수가 낮아질 수 있다.

# 저장
save_table(split_tab, "ml05_split", index=True)

# ─────────────────────────────────────────────
# 3. 특징 만들기: "접수하는 순간 알 수 있는 정보"만
# ─────────────────────────────────────────────
print_header("3. 특징(feature) 만들기")

# 계절: 월 대신 쓴다 (common.season_of 설명 참고)
ml["season"] = season_of(ml["month"])

# 위치 유형의 빈 칸은 '정보 없음'이라는 하나의 범주로 둔다. 빈 것 자체가 정보일 수 있다
ml["location_type"] = ml["location_type"].cat.add_categories("(없음)").fillna("(없음)")


def keep_top(col: str, k: int) -> None:
    """학습 기간에서 많이 나온 상위 k개 범주만 남기고 나머지는 '기타'로 묶는다.

    상위 범주를 '학습 기간'에서만 고르는 이유: 평가 기간 정보를 미리 보면 그것도 누수다.
    드문 범주를 묶는 이유: 몇 건뿐인 범주는 모델이 배울 사례가 부족해 잡음만 늘린다.
    """
    # 학습 기간에서 범주별 건수를 세고 상위 k개 이름을 고른다
    top = ml.loc[ml["split"] == "train", col].value_counts().head(k).index
    # 상위 k개가 아니면 '기타'로 바꾼다
    ml[col] = ml[col].astype(str).where(ml[col].astype(str).isin(top.astype(str)), "기타").astype("category")


# 민원 유형 197개 → 상위 40개 + 기타
keep_top("problem", 40)

# 위치 유형 155개 → 상위 20개 + 기타
keep_top("location_type", 20)

# 특징 설명표. 학생이 '왜 이 열은 되고 저 열은 안 되는지' 한눈에 보도록 만든다
feat = pd.DataFrame([
    {"열": "problem", "종류": "범주", "접수 시점에 아는가": "예", "설명": "민원 유형 (상위 40개 + 기타)"},
    {"열": "borough", "종류": "범주", "접수 시점에 아는가": "예", "설명": "자치구"},
    {"열": "channel", "종류": "범주", "접수 시점에 아는가": "예", "설명": "접수 경로 (온라인/전화/모바일)"},
    {"열": "location_type", "종류": "범주", "접수 시점에 아는가": "예", "설명": "장소 유형 (상위 20개 + 기타 + 없음)"},
    {"열": "season", "종류": "범주", "접수 시점에 아는가": "예", "설명": "계절 (월에서 계산)"},
    {"열": "hour", "종류": "숫자", "접수 시점에 아는가": "예", "설명": "접수 시각(시)"},
    {"열": "weekday", "종류": "숫자", "접수 시점에 아는가": "예", "설명": "요일 번호(0=월)"},
    {"열": "lat, lon", "종류": "숫자", "접수 시점에 아는가": "예", "설명": "위도·경도 (2% 결측)"},
    {"열": "closed, status", "종류": "-", "접수 시점에 아는가": "아니오(누수)", "설명": "처리 결과 그 자체"},
    {"열": "Resolution Description", "종류": "-", "접수 시점에 아는가": "아니오(누수)", "설명": "처리 후 작성"},
    {"열": "Resolution Action Updated Date", "종류": "-", "접수 시점에 아는가": "아니오(누수)", "설명": "처리 후 기록"},
    {"열": "unique_key, Incident Zip", "종류": "-", "접수 시점에 아는가": "예", "설명": "식별자라 숫자 특징으로 쓰지 않음"},
])

# 출력
print(feat.to_string(index=False))

# 저장
save_table(feat, "ml05_features")

# 머신러닝용 데이터 저장: 특징 + 타깃 + 기간 + 처리 시간(ml_13 회귀용) + 접수 시각(정렬용)
ml_out = ml[["created", "split", TARGET, "hours"] + FEATURES].reset_index(drop=True)

# Parquet 으로 저장한다. category 자료형도 그대로 보존된다
ml_out.to_parquet(ML_PATH, index=False)

# 저장 위치 출력
print(f"  [데이터 저장] {ML_PATH.name}: {len(ml_out):,}행 × {ml_out.shape[1]}열")

# ─────────────────────────────────────────────
# 4. 누수 실습: 답지를 몰래 넣으면 어떻게 되나
# ─────────────────────────────────────────────
print_header("4. 데이터 누수 실습")

# '처리 결과가 갱신된 시각'은 처리 후에야 생긴다. 접수 시점에는 알 수 없는 정보다
upd = pd.read_parquet(DATA_PATH, columns=["Unique Key", "Resolution Action Updated Date"])

# 문자열 날짜를 시각으로 바꾼다. 형식이 안 맞으면 빈 값(NaT)
upd["upd"] = pd.to_datetime(upd["Resolution Action Updated Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# 원래 df 와 같은 순서이므로 같은 행만 골라 '갱신까지 걸린 시간'을 만든다
leak = ((upd["upd"] - df["created"]).dt.total_seconds() / 3600).loc[ml.index].reset_index(drop=True)

# 학습용 표본 10만, 평가용 표본 5만 (실습이라 작게)
tr = ml_out[ml_out["split"] == "train"].sample(100_000, random_state=SEED)

# 평가용 표본
te = ml_out[ml_out["split"] == "test"].sample(50_000, random_state=SEED)

# 결과를 모을 목록
rows = []

# (1) 정상 특징만 / (2) 정상 특징 + 누수 특징 두 경우를 비교한다
for name, use_leak in [("정상 특징만", False), ("정상 특징 + 누수 특징(갱신 시각)", True)]:
    # 학습 입력
    X_tr = tr[FEATURES].copy()
    # 평가 입력
    X_te = te[FEATURES].copy()
    # 누수 실험이면 '갱신까지 걸린 시간' 열을 추가한다
    if use_leak:
        # 학습 행에 맞는 누수 값
        X_tr["leak_hours"] = leak.loc[tr.index].values
        # 평가 행에 맞는 누수 값
        X_te["leak_hours"] = leak.loc[te.index].values
    # 모델 만들기 (그래디언트 부스팅, ml_09 에서 자세히 배운다)
    model = make_hgb()
    # 학습 시작 시각
    t0 = time.time()
    # 학습
    model.fit(X_tr, tr[TARGET])
    # 평가 데이터에서 '1일 확률' 예측
    p = model.predict_proba(X_te)[:, 1]
    # 결과 기록
    rows.append({"실험": name, "평가 AUC": round(roc_auc_score(te[TARGET], p), 4), "학습 시간(초)": round(time.time() - t0, 1)})

# 표로 만든다
leak_tab = pd.DataFrame(rows)

# 출력
print(leak_tab.to_string(index=False))
# 해석: 정상 특징만으로도 AUC 0.96으로 이미 높다. 민원 유형이 '어느 기관이 어떻게 처리하는지'를 거의 정해 주기 때문이다
#       (예: 불법 주차는 경찰이 몇 시간 안에 종결, 난방·온수는 주택국이 며칠에 걸쳐 처리).
#       누수 특징을 넣으면 0.99로 더 오른다. 틀리는 정도(1 - AUC)가 0.040 → 0.009로 약 4분의 1이 된다.
#       실력이 좋아진 것이 아니라 답을 본 것이다. 실제 접수 순간에는 이 정보가 없으므로 이 모델은 현장에서 쓸 수 없다.
#       교훈: 점수가 높다고 곧 누수는 아니고, 누수가 없다고 점수가 낮은 것도 아니다. 그래서 '접수 시점에 아는가'를 열마다 직접 점검한다.

# 저장
save_table(leak_tab, "ml05_leakage_demo")

# 막대그래프
fig, ax = plt.subplots(figsize=(7, 3))

# 가로 막대
ax.barh(leak_tab["실험"], leak_tab["평가 AUC"], color=["#4C72B0", "#C44E52"])

# x축 범위 0.5(찍기) ~ 1(만점)
ax.set_xlim(0.5, 1.0)

# x축 이름
ax.set_xlabel("평가 AUC (0.5 = 찍기, 1.0 = 완벽)")

# 제목
ax.set_title("데이터 누수: 답지를 넣으면 점수가 '너무' 좋아진다")

# 저장
save_fig(fig, "ml05_leakage_demo")
