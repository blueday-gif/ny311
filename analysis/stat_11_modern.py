"""11단계(현대적 방법): 분위수 회귀와 그래디언트 부스팅.

입력:  data/processed/nyc311_all_clean.parquet   (stat_01_clean_descriptive.py 출력)
출력:  outputs/tables/stat11_*.csv   (대시보드 7단계 탭이 읽는다)
실행:  프로젝트 루트에서  python analysis/stat_11_modern.py

======================================================================================
stat_04 의 회귀(대시보드 6단계)에서 남은 두 가지 아쉬움
======================================================================================
stat_04 에서 다중선형회귀(OLS)와 로지스틱 회귀로 처리 시간을 설명·예측했다. 두 모형은 지금도 표준이지만 한계가 있다.

  아쉬움 1. OLS 는 '평균'만 설명한다.
      처리 시간처럼 치우친 자료에서 평균은 좋은 대푯값이 아니었다(대시보드 2단계 대푯값). 게다가 정책적으로 궁금한 것은
      평균보다 "가장 오래 걸리는 10% 는 어떤가"일 때가 많다.
      -> 분위수 회귀: 평균 대신 중앙값이나 90번째 백분위수 같은 '분위수'를 직접 설명한다.

  아쉬움 2. 선형 모형은 사람이 식의 모양을 정해 줘야 한다.
      "자치구 효과 + 기관 효과 + 시간대 효과"처럼 효과를 더하는 꼴만 다룬다. 기관마다 자치구 효과가 다른 것(상호작용)이나
      휘어진 관계는 사람이 일일이 넣어 줘야 한다.
      -> 그래디언트 부스팅: 나무 수백 그루가 상호작용과 비선형을 스스로 찾아낸다.

  [통계 모형과 기계학습 모형, 무엇이 다른가]
                    회귀 (통계 모형)                      부스팅 (기계학습 모형)
      주된 목적     설명: 어떤 변수가 얼마만큼 영향을 주나    예측: 새 자료를 얼마나 잘 맞히나
      결과물        계수, 신뢰구간, p 값                   예측값. 계수가 없어 속을 들여다보기 어렵다
      유연성        사람이 정한 식의 모양 안에서만            상호작용·비선형을 자동으로
      평가          가정 점검, 유의성                       학습에 쓰지 않은 자료에서의 예측 성능
      약점          식을 잘못 세우면 틀린다                  과적합 위험, 해석이 어렵다
    둘은 경쟁이 아니라 용도가 다르다. "왜"를 묻는다면 회귀, "얼마나 잘 맞히나"를 묻는다면 부스팅이 먼저다.
    부스팅이 회귀보다 훨씬 잘 맞힌다면 회귀식이 놓친 구조(상호작용, 비선형)가 있다는 신호로도 쓸 수 있다.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from stat_common import SEED, TIME_BANDS, load_clean, save_table

SPLIT_DATE = "2024-09-01"      # 이 날짜 전 접수분으로 학습, 9월 접수분으로 평가 (stat_04 와 같은 분할)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QR_N = 40_000                  # 분위수 회귀 적합 표본 (선형계획 문제를 풀어야 해서 느리다)
TOP_AGENCIES = 6
TOP_TYPES = 100                # 부스팅에 넣을 민원 유형 수 (나머지는 '기타')
FIT_N = 300_000
EVAL_N = 200_000
PERM_N = 30_000                # 순열 중요도 계산 표본


# ============================================================================ (1) 분위수 회귀
def quantile_regression(df):
    """분위수 회귀 (quantile regression, Koenker & Bassett 1978).

    [OLS 와의 차이: 무엇을 최소화하는가]
        OLS:          잔차의 '제곱'의 합을 최소화한다      -> 조건부 **평균**을 추정한다
        중앙값 회귀:  잔차의 '절댓값'의 합을 최소화한다    -> 조건부 **중앙값**을 추정한다
        분위수 회귀:  위로 틀린 것과 아래로 틀린 것에 서로 다른 벌점을 준다 -> 원하는 **분위수**를 추정한다
      "평균은 제곱 오차의 합을, 중앙값은 절대 오차의 합을 가장 작게 하는 값"이라는 사실을 떠올리면,
      분위수 회귀는 그 생각을 회귀로 넓힌 것이다.

    [핀볼 손실(pinball loss)]  분위수 τ(타우) 를 추정할 때 쓰는 벌점.
        예측보다 실제가 크면 (실제 - 예측) x τ,  작으면 (예측 - 실제) x (1 - τ) 만큼 벌점을 준다.
        τ = 0.9 라면 '실제가 더 큰' 실수에 9배 무거운 벌점을 주므로, 예측선이 자료의 90% 를 아래에 두는 높이까지 올라간다.
        τ = 0.5 면 양쪽 벌점이 같아 절댓값 손실, 즉 중앙값 회귀가 된다.

    [계수 읽는 법]  τ = 0.9 에서 '맨해튼' 계수가 -0.3 이라면:
        "다른 조건이 같을 때, 맨해튼 민원의 90번째 백분위수(오래 걸리는 쪽 끝)가 기준 자치구보다 로그 척도로 0.3 낮다."
        같은 변수의 계수가 τ 에 따라 달라진다면, 그 변수는 분포의 '위치'만 옮기는 것이 아니라 '모양'을 바꾸는 것이다.
        예) 빨리 끝나는 민원(τ = 0.1)에서는 자치구 차이가 없는데 오래 걸리는 민원(τ = 0.9)에서만 차이가 크다면,
            "평균 처리 시간이 길다"가 아니라 "유난히 오래 끄는 민원이 많다"가 정확한 표현이다. OLS 는 이것을 구별하지 못한다.

    [장점]  극단값에 덜 흔들린다 / 오차가 정규분포라는 가정이 필요 없다 / 분포 전체를 볼 수 있다 /
            단조 변환에 대해 일관된다 (log 의 중앙값 = 중앙값의 log. 평균에서는 성립하지 않는다)
    [단점]  계산이 느리다 / 자료가 드문 분포의 양 끝(τ 가 0 이나 1 근처)에서는 추정이 불안정하다 /
            같은 값이 많이 겹친 자료(동점)에서는 계수가 0 근처에 붙거나 수렴 경고가 날 수 있다
    """
    train = df[df["date"] < SPLIT_DATE]
    top = train["agency"].value_counts().index[:TOP_AGENCIES]
    s = train.sample(min(QR_N, len(train)), random_state=SEED).copy()
    s["agency_group"] = s["agency"].where(s["agency"].isin(top), "기타")
    levels = {
        "borough": list(train["borough"].value_counts().index),
        "agency_group": list(s["agency_group"].value_counts().index),
        "time_band": [TIME_BANDS[1]] + [t for t in TIME_BANDS if t != TIME_BANDS[1]],
    }
    parts = []
    for col, lv in levels.items():
        d = pd.get_dummies(pd.Categorical(s[col], categories=lv), prefix=col, prefix_sep="=", drop_first=True, dtype=float)
        d.index = s.index
        parts.append(d)
    X = pd.concat(parts, axis=1)
    X["is_weekend"] = s["is_weekend"].astype(float)
    X = sm.add_constant(X)
    y = s["log_resolution"]

    rows = []
    ols = sm.OLS(y, X).fit(cov_type="HC1")
    ci = ols.conf_int()
    for v in X.columns:
        rows.append({"모형": "OLS (평균)", "분위수 τ": np.nan, "변수": v, "계수": ols.params[v], "CI 2.5%": ci.loc[v, 0],
                     "CI 97.5%": ci.loc[v, 1], "p": ols.pvalues[v]})
    for q in QUANTILES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # 동점이 많은 자료에서 나오는 수렴 안내
            res = sm.QuantReg(y, X).fit(q=q, max_iter=2000)
        ci = res.conf_int()
        for v in X.columns:
            rows.append({"모형": f"분위수 회귀 τ={q:.2f}", "분위수 τ": q, "변수": v, "계수": res.params[v],
                         "CI 2.5%": ci.loc[v, 0], "CI 97.5%": ci.loc[v, 1], "p": res.pvalues[v]})
        print(f"    τ = {q:.2f} 적합 완료")
    out = pd.DataFrame(rows)
    out["배율 exp(계수)"] = np.exp(out["계수"])           # (시간 + 1) 이 기준 범주의 몇 배인가
    ref = f"자치구={levels['borough'][0]}, 기관={levels['agency_group'][0]}, 시간대={levels['time_band'][0]}"

    # 계수가 분위수에 따라 얼마나 달라지는가: τ=0.1 과 τ=0.9 의 차이가 큰 변수일수록 '분포의 모양'을 바꾸는 변수다
    wide = out[out["분위수 τ"].notna()].pivot(index="변수", columns="분위수 τ", values="계수")
    summary = pd.DataFrame({
        "변수": wide.index, "τ=0.10 계수": wide[0.10].to_numpy(), "τ=0.50 계수 (중앙값)": wide[0.50].to_numpy(),
        "τ=0.90 계수": wide[0.90].to_numpy(),
        "OLS 계수 (평균)": out[out["모형"] == "OLS (평균)"].set_index("변수").loc[wide.index, "계수"].to_numpy(),
    })
    summary["τ=0.90 - τ=0.10"] = summary["τ=0.90 계수"] - summary["τ=0.10 계수"]
    summary["읽는 법"] = np.select(
        [summary["변수"] == "const", summary["τ=0.90 - τ=0.10"].abs() < 0.3, summary["τ=0.90 - τ=0.10"] > 0],
        ["기준 범주의 분위수 자체", "분포 전체를 비슷하게 옮긴다 (위치 이동)", "오래 걸리는 쪽에서 차이가 더 커진다 (꼬리를 늘인다)"],
        default="빨리 끝나는 쪽에서 차이가 더 크다 (꼬리에서는 차이가 줄어든다)")
    return out, summary, ref, len(s)


# ============================================================================ (2) 그래디언트 부스팅
def boosting(df):
    """그래디언트 부스팅 (gradient boosting, Friedman 2001)과 순열 중요도.

    [결정나무]  "기관이 NYPD 인가? -> 예: 24시간 내 처리 가능성 높음 / 아니오: 민원 유형이 난방인가? -> ..." 처럼
        예·아니오 질문을 이어 가며 자료를 나누는 모형. 이해하기 쉽지만 한 그루로는 예측이 거칠고 불안정하다.

    [부스팅]  나무를 한 그루씩 차례로 더해 간다. 새 나무는 **지금까지의 나무들이 틀린 부분**을 맞히도록 배운다.
        1) 첫 예측을 한다 (전체 평균)   2) 틀린 정도(잔차)를 계산한다   3) 그 잔차를 맞히는 작은 나무를 만든다
        4) 그 나무의 예측을 조금만(학습률) 더한다   5) 2~4 를 수백 번 반복한다
        약한 모형 여럿을 모아 강한 모형을 만드는 '앙상블'의 한 방법이다. (나무들을 따로따로 만들어 평균 내는 것은 랜덤 포레스트)
        정형 자료(표 형태의 자료)에서는 지금도 가장 잘 맞는 방법 가운데 하나로, XGBoost·LightGBM·CatBoost 가 모두 이 계열이다.
        여기서 쓰는 HistGradientBoosting 은 LightGBM 과 같은 방식(값을 구간으로 묶어 빠르게 학습)이다.

    [과적합]  모형이 학습 자료의 우연한 잡음까지 외워 버려, 새 자료에서는 오히려 못 맞히는 현상. 유연한 모형일수록 위험하다.
        막는 방법: 학습에 쓰지 않은 자료로 평가한다 / 나무를 얕게, 학습률을 작게 한다 /
                  검증 점수가 더 나아지지 않으면 나무 추가를 멈춘다(조기 종료).
        확인 방법: 학습 자료에서의 성능과 평가 자료에서의 성능을 나란히 본다. 둘의 차이가 크면 과적합이다.

    [평가 지표]
        정확도:      0.5 를 기준으로 맞힌 비율. 한쪽 범주가 훨씬 많으면 "전부 다수 범주로 찍기"만 해도 높게 나온다.
        AUC:         무작위로 고른 '24시간 내 처리' 건과 '아닌' 건 중 전자에 더 높은 확률을 줄 비율. 0.5 무작위, 1.0 완벽.
        Brier 점수:  예측 확률과 실제(0/1)의 차이를 제곱해 평균한 값. 작을수록 좋다. '확률이 얼마나 정직한가'까지 본다.
        로그 손실:   확신하고 틀릴수록 크게 벌점을 준다. 작을수록 좋다.

    [순열 중요도(permutation importance)]  부스팅에는 회귀계수가 없다. 그러면 어떤 변수가 중요한지 어떻게 알까?
        한 변수의 값을 **무작위로 뒤섞어** 그 변수와 결과의 관계를 끊은 뒤 성능이 얼마나 떨어지는지 본다.
        많이 떨어지면 모형이 그 변수에 크게 기대고 있다는 뜻이다. 모형 종류와 상관없이 쓸 수 있고(model-agnostic),
        평가 자료에서 계산하므로 '실제 예측에 도움이 되는 정도'를 잰다.
        주의 1: 서로 상관이 높은 변수들(기관과 민원 유형)은 하나를 섞어도 다른 하나가 정보를 대신 주므로 둘 다 과소평가된다.
        주의 2: 중요도는 "예측에 쓸모 있다"는 뜻이지 "원인이다"라는 뜻이 아니다.
        (SHAP 은 예측 한 건 한 건을 변수별 기여로 쪼개 주는 더 세밀한 방법이다. 생각의 뿌리는 같다.)

    [이 분석의 비교 설계]  성능 차이가 '변수를 더 넣어서'인지 '모형이 유연해서'인지 가려내려고 단계를 나눈다.
        A. 로지스틱 (stat_04 와 같은 변수: 자치구, 기관, 시간대, 주말)
        B. 로지스틱 + 민원 유형            <- A 와의 차이 = 변수(민원 유형)를 더한 효과
        C. 부스팅  (B 와 같은 변수)        <- B 와의 차이 = 모형이 유연해진 효과 (상호작용, 비선형)
        D. 부스팅 + 접수 시각·요일·좌표    <- C 와의 차이 = 세밀한 시간·위치 정보의 효과
      모든 변수는 접수 시점에 알 수 있는 정보다. 종결 시각처럼 나중에야 아는 정보를 넣으면 성능이 거짓으로 좋아진다(정보 누출).
      월(month)은 넣지 않았다. 평가 구간인 9월은 학습 구간에 없는 값이라 모형이 배울 수 없기 때문이다.
    """
    d = df.copy()
    train_all = d[d["date"] < SPLIT_DATE]
    top_types = train_all["complaint_type"].value_counts().index[:TOP_TYPES]      # 범주 묶음은 학습 구간만 보고 정한다
    d["type_group"] = d["complaint_type"].where(d["complaint_type"].isin(top_types), "기타")
    train = d[d["date"] < SPLIT_DATE].sample(min(FIT_N, len(train_all)), random_state=SEED)
    test_all = d[d["date"] >= SPLIT_DATE]
    test = test_all.sample(min(EVAL_N, len(test_all)), random_state=SEED)
    y_tr, y_te = train["within_24h"].to_numpy(), test["within_24h"].to_numpy()
    print(f"  학습 {len(train):,}건 (8월까지), 평가 {len(test):,}건 (9월), 평가 구간의 24시간 내 처리율 {y_te.mean():.1%}")

    cat_a, cat_b = ["borough", "agency", "time_band", "is_weekend"], ["borough", "agency", "time_band", "is_weekend", "type_group"]

    def onehot(cols):
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(train[cols].astype(str))
        return enc.transform(train[cols].astype(str)), enc.transform(test[cols].astype(str))

    def codes(cols):
        """범주를 정수 번호로 바꾼다. 번호는 이름표일 뿐이며, 부스팅에게 '범주형'이라고 따로 알려 준다.
        학습 구간에 없던 범주는 결측(NaN)으로 둔다. 이 모형은 결측을 스스로 처리한다."""
        a, b = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
        for c in cols:
            cats = pd.Categorical(train[c]).categories
            a[c] = pd.Categorical(train[c], categories=cats).codes.astype(float)
            b[c] = pd.Categorical(test[c], categories=cats).codes.astype(float)
            b.loc[b[c] < 0, c] = np.nan
        return a, b

    results, models = [], {}

    def evaluate(name, features, kind, p_tr, p_te):
        results.append({
            "모형": name, "종류": kind, "변수": features, "AUC (학습)": roc_auc_score(y_tr, p_tr), "AUC (평가, 9월)": roc_auc_score(y_te, p_te),
            "정확도 (평가)": accuracy_score(y_te, p_te >= 0.5), "Brier 점수 (평가)": brier_score_loss(y_te, p_te),
            "로그 손실 (평가)": log_loss(y_te, np.clip(p_te, 1e-6, 1 - 1e-6)),
        })
        print(f"    {name}: 평가 AUC {results[-1]['AUC (평가, 9월)']:.4f}, 학습 AUC {results[-1]['AUC (학습)']:.4f}")

    base = max(y_te.mean(), 1 - y_te.mean())
    p0 = np.full(len(y_te), y_tr.mean())
    results.append({"모형": "기준: 모두 같은 확률로 예측", "종류": "기준", "변수": "없음", "AUC (학습)": 0.5, "AUC (평가, 9월)": 0.5,
                    "정확도 (평가)": base, "Brier 점수 (평가)": brier_score_loss(y_te, p0), "로그 손실 (평가)": log_loss(y_te, p0)})

    for name, cols, feat in [("A. 로지스틱 회귀", cat_a, "자치구, 기관, 시간대, 주말"),
                             ("B. 로지스틱 회귀 + 민원 유형", cat_b, "A + 민원 유형")]:
        Xa, Xb = onehot(cols)
        lr = LogisticRegression(max_iter=500, C=1.0).fit(Xa, y_tr)
        evaluate(name, feat, "통계 모형", lr.predict_proba(Xa)[:, 1], lr.predict_proba(Xb)[:, 1])

    gb_params = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=15, random_state=SEED)
    Xa, Xb = codes(cat_b)
    gb_c = HistGradientBoostingClassifier(categorical_features=list(range(len(cat_b))), **gb_params).fit(Xa, y_tr)
    evaluate("C. 그래디언트 부스팅 (B와 같은 변수)", "A + 민원 유형", "기계학습", gb_c.predict_proba(Xa)[:, 1], gb_c.predict_proba(Xb)[:, 1])

    cat_d = ["borough", "agency", "type_group"]
    num_d = ["hour", "weekday", "latitude", "longitude"]
    Xa, Xb = codes(cat_d)
    for c in num_d:
        Xa[c], Xb[c] = train[c].astype(float).to_numpy(), test[c].astype(float).to_numpy()
    gb_d = HistGradientBoostingClassifier(categorical_features=[0, 1, 2], **gb_params).fit(Xa, y_tr)
    evaluate("D. 그래디언트 부스팅 + 시각·요일·좌표", "자치구, 기관, 민원 유형, 시각, 요일, 위도, 경도", "기계학습",
             gb_d.predict_proba(Xa)[:, 1], gb_d.predict_proba(Xb)[:, 1])
    res = pd.DataFrame(results)
    res["과적합 정도 (학습 AUC - 평가 AUC)"] = res["AUC (학습)"] - res["AUC (평가, 9월)"]
    res["앞 단계 대비 AUC 변화"] = res["AUC (평가, 9월)"].diff()
    res["나무 수"] = [np.nan, np.nan, np.nan, gb_c.n_iter_, gb_d.n_iter_]

    # 순열 중요도: 모형 D, 평가 자료의 표본에서 변수 하나씩 뒤섞어 AUC 하락을 잰다
    idx = np.random.default_rng(SEED).choice(len(Xb), size=min(PERM_N, len(Xb)), replace=False)
    pi = permutation_importance(gb_d, Xb.iloc[idx], y_te[idx], scoring="roc_auc", n_repeats=5, random_state=SEED)
    names = {"borough": "자치구", "agency": "담당 기관", "type_group": "민원 유형", "hour": "접수 시각", "weekday": "요일",
             "latitude": "위도", "longitude": "경도"}
    imp = pd.DataFrame({"변수": [names[c] for c in Xb.columns], "AUC 하락 (평균)": pi.importances_mean,
                        "표준편차 (5회 반복)": pi.importances_std}).sort_values("AUC 하락 (평균)", ascending=False)
    imp["뒤섞은 뒤 AUC"] = res.loc[res["모형"].str.startswith("D."), "AUC (평가, 9월)"].iloc[0] - imp["AUC 하락 (평균)"]

    # 보정(calibration): 모형이 "70% 확률"이라고 말한 건들 가운데 실제로 70% 가 24시간 안에 처리되었는가
    p = gb_d.predict_proba(Xb)[:, 1]
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    cal = (pd.DataFrame({"구간": bins, "예측": p, "실제": y_te}).groupby("구간", observed=True)
             .agg(건수=("실제", "size"), 예측확률_평균=("예측", "mean"), 실제비율=("실제", "mean")).reset_index())
    cal["구간"] = cal["구간"].astype(str)
    cal = cal.rename(columns={"예측확률_평균": "예측 확률 평균", "실제비율": "실제 비율"})
    return res, imp, cal


# ============================================================================ 실행
def main():
    df = load_clean()
    print("[1] 분위수 회귀")
    qr, qsum, ref, n_qr = quantile_regression(df)
    save_table(qr.round(5), "stat11_quantreg_coef.csv")
    save_table(qsum.round(4), "stat11_quantreg_summary.csv")
    pd.DataFrame([{"적합 표본": n_qr, "목표변수": "log(1 + 처리 시간)", "기준 범주": ref,
                   "분위수": ", ".join(f"{q:.2f}" for q in QUANTILES)}]).pipe(save_table, "stat11_quantreg_info.csv")
    print(f"  적합 {n_qr:,}건, 기준 범주: {ref}")
    print(qsum.round(3).to_string(index=False))

    print("[2] 그래디언트 부스팅")
    res, imp, cal = boosting(df)
    save_table(res.round(5), "stat11_model_compare.csv")
    save_table(imp.round(5), "stat11_permutation_importance.csv")
    save_table(cal.round(4), "stat11_calibration.csv")
    print(res[["모형", "AUC (학습)", "AUC (평가, 9월)", "정확도 (평가)", "Brier 점수 (평가)", "앞 단계 대비 AUC 변화"]].round(4).to_string(index=False))
    print(imp.round(4).to_string(index=False))
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
