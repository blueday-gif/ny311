"""6단계(추가 분석 2): 시계열 분석 - 일별 민원 접수 건수의 분해, 정상성, 기온 효과, 예측.

입력:  data/processed/nyc311_all_daily.csv   (stat_01_clean_descriptive.py 출력. 183일, 하루 한 행)
출력:  outputs/tables/stat6_*.csv, outputs/figures/stat6_*.png
실행:  프로젝트 루트에서  python analysis/stat_06_timeseries.py

======================================================================================
[개념] 시계열 자료가 특별한 이유
======================================================================================
시계열(time series)은 같은 대상을 시간 순서대로 반복 측정한 자료다. 1~4단계에서 쓴 대부분의 검정은
"관측값들이 서로 독립"이라고 가정하지만 시계열은 그렇지 않다. 오늘 민원이 많으면 내일도 많을 가능성이
높다. 이렇게 자기 자신의 과거 값과 상관되는 성질을 자기상관(autocorrelation)이라고 한다.

자기상관을 무시하면 생기는 문제:
  - 실제로는 독립된 정보가 183개보다 훨씬 적은데 183개인 것처럼 계산하게 된다.
  - 그래서 표준오차가 실제보다 작게, p 값이 실제보다 작게(더 유의하게) 나온다.
  - 4단계의 "기온과 일별 건수의 상관" p 값이 바로 이 문제를 안고 있다. 이 스크립트 (3)에서 다시 추정한다.

======================================================================================
이 스크립트가 하는 일
======================================================================================
  (1) STL 분해: 추세 + 주간 계절성 + 나머지로 나누고, 각 성분의 세기와 이상한 날을 찾는다
  (2) 정상성 검정(ADF, KPSS)과 자기상관(ACF, PACF) 확인
  (3) 기온 효과 재추정: 단순 회귀(OLS)와 '오차에 자기상관을 허용한 회귀'를 비교
  (4) 예측: 8월까지 자료로 모형을 만들고 9월 30일치를 예측해 기준 모형(계절 단순 예측)과 비교
      예측에는 기온을 쓰지 않는다. 9월의 실제 기온은 8월 31일 시점에 알 수 없는 미래 정보이기 때문이다.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from stat_common import (
    ACCENT, CATEGORICAL, DAILY, GRID, INK, MUTED, OTHER_GRAY, ROOT, SINGLE, p_text, rnd, save_fig, save_table, setup_style,
)

PERIOD = 7                      # 계절 주기: 일별 자료의 주간 반복 (월~일)
SPLIT_DATE = "2024-09-01"       # 이 날짜 전까지 학습, 이후(9월)는 평가
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
NLAGS = 28
UNUSUAL_Z = 3.5                 # 특이한 날 기준: 나머지의 robust |z|


# ============================================================================ (1) STL 분해
def decompose(y):
    """STL 분해 (Seasonal and Trend decomposition using Loess).

    [개념] 시계열을 세 성분의 합으로 본다 (가법 모형):   관측값 = 추세 + 계절성 + 나머지
      - 추세(trend):       길게 보았을 때의 오르내림 (예: 여름으로 갈수록 민원 증가)
      - 계절성(seasonal):  일정한 주기로 반복되는 무늬 (예: 주말마다 감소). 주기 = 7일
      - 나머지(remainder): 둘로 설명되지 않는 부분. 불규칙 변동과 특이한 날(공휴일, 폭풍)이 여기에 남는다
    [방법] Loess(국소 회귀)라는 매끄러운 곡선 맞춤을 반복해 성분을 분리한다. 이동평균을 쓰는 고전적 분해와
           달리 계절 무늬가 시간에 따라 조금씩 변하는 것을 허용하고, robust=True 로 두면 이상값에 덜 끌려간다.
    [가법 대 승법] 변동 폭이 수준에 비례해 커지면 승법 모형(관측값 = 추세 x 계절성 x 나머지)이 맞고,
           그럴 때는 로그를 취해 가법으로 바꾼다. 여기서는 변동 폭이 비교적 일정해 가법을 쓴다.

    [성분의 세기] Hyndman 의 지표 (0~1, 1에 가까울수록 강함)
        추세 세기   F_T = max(0, 1 - Var(나머지) / Var(추세 + 나머지))
        계절성 세기 F_S = max(0, 1 - Var(나머지) / Var(계절성 + 나머지))
      "그 성분을 빼고 나면 변동이 얼마나 줄어드는가"를 재는 값이다.
    """
    res = STL(y, period=PERIOD, robust=True).fit()
    comp = pd.DataFrame({"date": y.index, "관측값": y.to_numpy(), "추세": res.trend.to_numpy(),
                         "계절성": res.seasonal.to_numpy(), "나머지": res.resid.to_numpy()})
    f_t = max(0.0, 1 - comp["나머지"].var() / (comp["추세"] + comp["나머지"]).var())
    f_s = max(0.0, 1 - comp["나머지"].var() / (comp["계절성"] + comp["나머지"]).var())
    strength = pd.DataFrame([
        {"지표": "추세 세기 F_T", "값": f_t, "읽는 법": "1에 가까울수록 추세가 뚜렷함"},
        {"지표": "계절성(주간) 세기 F_S", "값": f_s, "읽는 법": "1에 가까울수록 요일 무늬가 뚜렷함"},
        {"지표": "추세 범위 (최대 - 최소, 건)", "값": comp["추세"].max() - comp["추세"].min(), "읽는 법": ""},
        {"지표": "계절성 진폭 (최대 - 최소, 건)", "값": comp["계절성"].max() - comp["계절성"].min(), "읽는 법": ""},
        {"지표": "나머지 표준편차 (건)", "값": comp["나머지"].std(), "읽는 법": ""},
    ])

    # 요일 효과: 계절 성분을 요일별로 평균하면 "평균적인 날에 비해 그 요일에 몇 건 더 들어오는가"가 된다
    comp["요일"] = [WEEKDAYS[d] for d in y.index.day_of_week]
    weekday = (comp.groupby("요일")[["관측값", "계절성"]].mean().reindex(WEEKDAYS)
                   .rename(columns={"관측값": "평균 접수 건수", "계절성": "요일 효과(건)"}).reset_index())

    # 특이한 날: 나머지가 유난히 큰 날. 표준편차 대신 MAD(중앙값 절대편차)로 척도를 잡는다.
    # 표준편차는 이상값 자체에 의해 부풀려지므로, 이상값을 찾는 기준으로는 MAD 가 더 안전하다.
    # 1.4826 을 곱하면 정규분포에서 표준편차와 같은 크기가 된다.
    mad = 1.4826 * (comp["나머지"] - comp["나머지"].median()).abs().median()
    comp["robust_z"] = (comp["나머지"] - comp["나머지"].median()) / mad
    # 기준 3.5 는 Iglewicz-Hoaglin 의 수정 z 점수 권고값이다. robust=True 인 STL 은 이상값을 추세·계절성에
    # 섞지 않고 나머지로 몰아 주므로 나머지의 꼬리가 두껍다. 그래서 정규분포에서 기대되는 것보다 많은 날이 걸린다.
    # 표는 |z| 가 큰 순서로 정렬한다. 위쪽 몇 개(공휴일, 폭풍, 시스템 문제 등)만 실제로 들여다볼 가치가 있다.
    unusual = comp.loc[comp["robust_z"].abs() > UNUSUAL_Z, ["date", "요일", "관측값", "추세", "계절성", "나머지", "robust_z"]]
    return comp, strength, weekday, unusual.sort_values("robust_z", key=np.abs, ascending=False)


# ============================================================================ (2) 정상성, 자기상관
def stationarity(y):
    """정상성(stationarity) 검정.

    [개념] 정상 시계열: 평균, 분산, 자기상관 구조가 시간이 지나도 변하지 않는 시계열.
           ARIMA 같은 모형은 정상성을 전제로 하므로, 추세나 계절성이 있으면 먼저 없애야 한다.
    [차분] 차분(differencing)은 인접한 값의 차이를 구하는 것이다.
             1차 차분    y_t - y_{t-1}: 추세를 없앤다
             계절 차분   y_t - y_{t-7}: 주간 반복을 없앤다 ("지난주 같은 요일보다 얼마나 변했나")
    [두 검정은 귀무가설이 반대다]
      ADF  (Augmented Dickey-Fuller): H0 = 단위근이 있다(비정상).  p < 0.05 이면 정상이라는 증거
      KPSS:                           H0 = 정상이다.               p < 0.05 이면 비정상이라는 증거
      둘 다 정상 쪽을 가리킬 때(ADF 기각, KPSS 기각 못함) 정상이라고 판단하는 것이 안전하다.
      KPSS 의 p 값은 표로 보간하므로 0.01~0.1 범위 밖은 '0.01 이하' 또는 '0.1 이상'으로만 알 수 있다.
    """
    series = {
        "원자료": y,
        "1차 차분": y.diff().dropna(),
        f"계절 차분 (lag {PERIOD})": y.diff(PERIOD).dropna(),
    }
    rows = []
    for name, s in series.items():
        adf_stat, adf_p = adfuller(s, autolag="AIC", result_object=False)[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # p 값이 표 범위 밖일 때의 안내 경고
            kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        adf_ok, kpss_ok = adf_p < 0.05, kpss_p >= 0.05
        verdict = "정상" if adf_ok and kpss_ok else ("비정상" if not adf_ok and not kpss_ok else "판단 엇갈림")
        rows.append({"시계열": name, "n": len(s), "ADF 통계량": adf_stat, "ADF p": adf_p,
                     "KPSS 통계량": kpss_stat, "KPSS p": kpss_p, "판정": verdict})
    return pd.DataFrame(rows)


def autocorr_table(y):
    """ACF 와 PACF.

    [ACF]  자기상관함수: y_t 와 k일 전 값 y_{t-k} 의 상관계수를 시차 k 별로 구한 것.
           시차 7, 14, 21 에서 봉우리가 솟으면 주간 계절성이 있다는 뜻이다.
    [PACF] 편자기상관함수: 그 사이 시차(1 ~ k-1)의 영향을 걷어낸 뒤 남는 y_t 와 y_{t-k} 의 직접 상관.
    [용도] ARIMA(p, d, q) 의 차수를 고르는 단서가 된다.
           AR(p) 과정이면 PACF 가 시차 p 뒤에서 끊기고, MA(q) 과정이면 ACF 가 시차 q 뒤에서 끊긴다.
    [유의 한계] 백색잡음(자기상관이 전혀 없는 계열)이라면 표본 자기상관은 대략 ±1.96/sqrt(n) 안에 들어온다.
    """
    a = acf(y, nlags=NLAGS, fft=True)
    p = pacf(y, nlags=NLAGS, method="ywm")
    bound = 1.96 / np.sqrt(len(y))
    out = pd.DataFrame({"시차(일)": range(NLAGS + 1), "ACF": a, "PACF": p})
    out["유의 한계(±)"] = bound
    return out


# ============================================================================ (3) 기온 효과
def temperature_effect(daily):
    """기온이 1°C 오르면 민원이 몇 건 늘어나는가: 두 가지 추정을 비교한다.

    [모형 A] 단순 회귀(OLS):  건수 = a + b*기온 + c*주말 + 오차,  오차는 서로 독립이라고 가정
    [모형 B] ARMA 오차 회귀:  같은 식이지만 오차가 자기상관을 가진다고 본다 (오차 ~ AR(1) + 주간 AR(1))

    [진단 도구]
      Durbin-Watson 통계량: 잔차의 1차 자기상관을 잰다. 2 근처면 자기상관 없음, 0 쪽이면 양의 자기상관.
      Ljung-Box 검정: H0 = 잔차에 (지정한 시차까지) 자기상관이 없다. p < 0.05 이면 자기상관이 남아 있다.
    모형 A 의 잔차에 자기상관이 남아 있으면 A 의 표준오차와 p 값은 믿을 수 없고, B 의 값을 봐야 한다.
    보통 계수 자체는 비슷하지만 B 의 표준오차가 더 크다(더 정직하다).

    [주의] 기온은 계절 추세와 함께 움직인다. "봄에서 여름으로 가며 민원이 는 것"과 "더운 날 민원이 는 것"이
           섞여 있을 수 있으므로, 이 계수를 기온의 순수한 인과 효과로 읽으면 안 된다.
    """
    if "temp_mean_c" not in daily.columns or daily["temp_mean_c"].isna().any():
        print("  기온 자료가 없어 이 단계를 건너뛴다.")
        return None
    d = daily.set_index("date").asfreq("D")
    X = sm.add_constant(d[["temp_mean_c", "is_weekend"]].astype(float))
    y = d["n_complaints"].astype(float)

    ols = sm.OLS(y, X).fit()
    dw = durbin_watson(ols.resid)
    lb_ols = acorr_ljungbox(ols.resid, lags=[14])["lb_pvalue"].iloc[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arma = SARIMAX(y, exog=X, order=(1, 0, 0), seasonal_order=(1, 0, 0, PERIOD)).fit(disp=False)
    lb_arma = acorr_ljungbox(arma.resid[PERIOD:], lags=[14])["lb_pvalue"].iloc[0]

    rows = []
    for name, res, lb, note in [("A. 단순 회귀 (OLS)", ols, lb_ols, f"Durbin-Watson {dw:.2f}"),
                                ("B. ARMA 오차 회귀", arma, lb_arma, "오차: AR(1) + 주간 AR(1)")]:
        ci = res.conf_int().loc["temp_mean_c"]
        rows.append({"모형": name, "기온 계수 (건/°C)": res.params["temp_mean_c"], "표준오차": res.bse["temp_mean_c"],
                     "p": res.pvalues["temp_mean_c"], "CI 2.5%": ci[0], "CI 97.5%": ci[1],
                     "주말 계수 (건)": res.params["is_weekend"], "잔차 Ljung-Box p (시차 14)": lb,
                     "잔차 자기상관": "남아 있음" if lb < 0.05 else "없음", "비고": note})
    return pd.DataFrame(rows)


# ============================================================================ (4) 예측
def metrics(actual, pred):
    """예측 오차 지표.
      MAE  (평균 절대 오차): |실제 - 예측| 의 평균. 단위가 '건'이라 해석이 쉽다.
      RMSE (평균 제곱근 오차): 오차를 제곱해 평균한 뒤 제곱근. 큰 오차에 더 큰 벌점을 준다.
      MAPE (평균 절대 백분율 오차): |오차| / 실제값 의 평균(%). 규모가 다른 계열끼리 비교할 때 쓴다.
    """
    err = actual - pred
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "MAPE(%)": (err.abs() / actual).mean() * 100}


def forecast(y):
    """8월 31일까지로 모형을 만들고 9월 30일치를 한 번에 예측한다 (다단계 예측, 중간에 다시 적합하지 않음).

    [왜 시간 순으로 나누는가] 무작위로 섞어 나누면 미래 자료로 과거를 맞히는 셈이 되어 성능이 부풀려진다.
        시계열은 반드시 "과거로 학습, 미래로 평가"해야 한다.

    [기준 모형] 계절 단순 예측(seasonal naive): "다음 주 월요일은 지난주 월요일과 같다."
        복잡한 모형은 최소한 이 기준을 이겨야 쓸 가치가 있다. 기준 없이 오차만 보면 좋은지 나쁜지 알 수 없다.

    [Holt-Winters] 지수평활법. 최근 관측값에 더 큰 가중치를 주며 수준·추세·계절성을 각각 갱신한다.

    [SARIMA(p,d,q)(P,D,Q)_7] 계절 ARIMA.
        AR(p): 자기회귀. 과거 p개 값으로 현재를 설명      I(d): d번 차분해 정상으로 만듦
        MA(q): 이동평균. 과거 q개 예측오차로 현재를 설명   (P,D,Q)_7: 같은 구조를 7일 간격으로 적용
        후보 차수 가운데 AIC 가 가장 작은 것을 고른다.
    [AIC] 아카이케 정보 기준 = -2·로그가능도 + 2·모수 개수. 자료를 잘 맞힐수록 작아지고 모수가 많을수록 커진다.
        '잘 맞으면서 단순한' 모형을 고르는 기준이며, 작을수록 좋다. 같은 자료·같은 차분 차수끼리만 비교할 수 있어,
        여기서는 d 와 D 를 고정한 후보끼리 비교한다.

    [예측구간] 95% 예측구간은 "미래 관측값이 95% 확률로 들어갈 범위"다. 계수의 불확실성을 나타내는 신뢰구간과
        다르며, 먼 미래일수록 넓어진다. 실제값이 구간에 든 비율(포함률)이 95% 근처인지 확인한다.
    """
    train, test = y[y.index < SPLIT_DATE], y[y.index >= SPLIT_DATE]
    h = len(test)

    # 기준: 학습 구간 마지막 7일을 요일에 맞춰 반복
    last_week = train.iloc[-PERIOD:].to_numpy()
    naive = pd.Series(np.tile(last_week, h // PERIOD + 1)[:h], index=test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hw = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add",
                                  seasonal_periods=PERIOD).fit()
        hw_pred = hw.forecast(h)

        # d=0, D=1 고정: 계절 차분으로 주간 무늬와 완만한 추세를 함께 없앤다
        cands = []
        for order in [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (2, 0, 1)]:
            for seas in [(0, 1, 1, PERIOD), (1, 1, 0, PERIOD), (1, 1, 1, PERIOD)]:
                try:
                    m = SARIMAX(train, order=order, seasonal_order=seas).fit(disp=False)
                    cands.append({"order (p,d,q)": str(order), "seasonal (P,D,Q,s)": str(seas), "AIC": m.aic,
                                  "BIC": m.bic, "모수 개수": len(m.params), "_model": m})
                except Exception as e:                   # 수렴 실패한 후보는 건너뛴다
                    print(f"    후보 {order}x{seas} 적합 실패: {e}")
    sel = pd.DataFrame(cands).sort_values("AIC").reset_index(drop=True)
    best = sel.loc[0, "_model"]
    sel["선택"] = ["선택"] + [""] * (len(sel) - 1)
    f = best.get_forecast(h)
    pi = f.conf_int(alpha=0.05)
    sarima_name = f"SARIMA{sel.loc[0, 'order (p,d,q)']}{sel.loc[0, 'seasonal (P,D,Q,s)']}"

    # 선택한 모형의 잔차 진단: 잔차가 백색잡음이면 모형이 자기상관 구조를 다 잡아낸 것이다
    lb = acorr_ljungbox(best.resid[PERIOD:], lags=[14], model_df=len(best.params) - 1)["lb_pvalue"].iloc[0]

    out = pd.DataFrame({"date": test.index, "실제": test.to_numpy(), "계절 단순 예측": naive.to_numpy(),
                        "Holt-Winters": hw_pred.to_numpy(), "SARIMA": f.predicted_mean.to_numpy(),
                        "SARIMA 95% 하한": pi.iloc[:, 0].to_numpy(), "SARIMA 95% 상한": pi.iloc[:, 1].to_numpy()})
    rows = []
    for name, col in [("계절 단순 예측 (기준)", "계절 단순 예측"), ("Holt-Winters (가법, 감쇠 추세)", "Holt-Winters"),
                      (sarima_name, "SARIMA")]:
        rows.append({"모형": name, **metrics(out["실제"], out[col])})
    met = pd.DataFrame(rows)
    met["기준 대비 MAE 변화(%)"] = (met["MAE"] / met.loc[0, "MAE"] - 1) * 100
    coverage = ((out["실제"] >= out["SARIMA 95% 하한"]) & (out["실제"] <= out["SARIMA 95% 상한"])).mean() * 100
    met["95% 예측구간 포함률(%)"] = [np.nan, np.nan, coverage]
    met["잔차 Ljung-Box p (시차 14)"] = [np.nan, np.nan, lb]
    print(f"  선택 모형 {sarima_name} (AIC {sel.loc[0, 'AIC']:.1f}), 잔차 Ljung-Box p {p_text(lb)} "
          f"({'자기상관 없음' if lb >= 0.05 else '자기상관 남음'}), 예측구간 포함률 {coverage:.0f}%")
    return out, met, sel.drop(columns="_model"), len(train)


# ============================================================================ 그래프
def figures(plt, comp, weekday, acf_t, fc, unusual):
    # STL 분해 4단 그림
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.6), sharex=True)
    for ax, col, color in zip(axes, ["관측값", "추세", "계절성", "나머지"], [INK, SINGLE, CATEGORICAL[2], MUTED]):
        if col == "나머지":
            ax.vlines(comp["date"], 0, comp[col], color=color, lw=1)
            hit = comp[comp["date"].isin(unusual["date"])]
            ax.scatter(hit["date"], hit[col], color=ACCENT, s=20, zorder=3, label=f"특이한 날 (robust |z| > {UNUSUAL_Z})")
            ax.axhline(0, color=OTHER_GRAY, lw=0.8)
            if len(hit):
                ax.legend(loc="upper center")
        else:
            ax.plot(comp["date"], comp[col], color=color, lw=1.4)
        ax.set_title(col, fontsize=11)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    fig.suptitle("STL 분해: 일별 접수 건수 = 추세 + 주간 계절성 + 나머지", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "stat6_01_stl_decomposition.png")

    # 요일 효과
    fig, ax = plt.subplots(figsize=(7, 3.6))
    v = weekday["요일 효과(건)"]
    ax.bar(weekday["요일"], v, color=[ACCENT if x < 0 else SINGLE for x in v], width=0.62)
    for x, val in enumerate(v):
        ax.text(x, val, f"{val:+,.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color=MUTED)
    ax.axhline(0, color=OTHER_GRAY, lw=0.8)
    ax.set_title("요일 효과: 평균적인 날에 비해 몇 건 더 접수되는가")
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.margins(y=0.18)
    save_fig(fig, "stat6_02_weekday_effect.png")

    # ACF / PACF
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, col in zip(axes, ["ACF", "PACF"]):
        t = acf_t.iloc[1:]
        ax.vlines(t["시차(일)"], 0, t[col], color=SINGLE, lw=2.2)
        b = acf_t["유의 한계(±)"].iloc[0]
        ax.axhspan(-b, b, color=OTHER_GRAY, alpha=0.3, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(0, NLAGS + 1, 7))
        ax.set_xlabel("시차 (일)")
        ax.set_title(f"{col} (회색 띠 안이면 0과 구별되지 않음)")
    save_fig(fig, "stat6_03_acf_pacf.png")

    # 예측
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    hist = comp[comp["date"] >= pd.Timestamp(SPLIT_DATE) - pd.Timedelta(days=42)]
    hist = hist[hist["date"] < SPLIT_DATE]
    ax.plot(hist["date"], hist["관측값"], color=MUTED, lw=1.3, label="학습 구간 (마지막 6주)")
    ax.fill_between(fc["date"], fc["SARIMA 95% 하한"], fc["SARIMA 95% 상한"], color=SINGLE, alpha=0.15, lw=0,
                    label="SARIMA 95% 예측구간")
    ax.plot(fc["date"], fc["실제"], color=INK, lw=1.6, label="실제")
    ax.plot(fc["date"], fc["SARIMA"], color=SINGLE, lw=1.6, label="SARIMA 예측")
    ax.plot(fc["date"], fc["계절 단순 예측"], color=ACCENT, lw=1.2, ls="--", label="계절 단순 예측 (기준)")
    ax.axvline(pd.Timestamp(SPLIT_DATE), color=OTHER_GRAY, lw=1, ls=":")
    ax.set_title("9월 일별 접수 건수 예측 (8월 31일까지의 자료로 30일 앞을 예측)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(loc="lower left", ncol=3)
    save_fig(fig, "stat6_04_forecast.png")


# ============================================================================ 실행
def main():
    if not DAILY.exists():
        raise SystemExit(f"{DAILY.relative_to(ROOT)} 가 없습니다. 먼저 analysis/stat_01_clean_descriptive.py 를 실행하세요.")
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    # asfreq("D"): 날짜 간격이 정확히 하루임을 명시한다. 빠진 날이 있으면 NaN 이 생겨 바로 드러난다.
    y = daily.set_index("date")["n_complaints"].astype(float).asfreq("D")
    if y.isna().any():
        raise SystemExit("일별 자료에 빠진 날짜가 있습니다.")

    print("[1] STL 분해")
    comp, strength, weekday, unusual = decompose(y)
    save_table(rnd(comp.drop(columns="요일"), 2), "stat6_stl_components.csv")
    save_table(strength.round(3), "stat6_stl_strength.csv")
    save_table(weekday.round(1), "stat6_weekday_effect.csv")
    save_table(rnd(unusual, 2), "stat6_unusual_days.csv")
    print(strength.round(3).to_string(index=False))
    print(f"  특이한 날 {len(unusual)}일: " + ", ".join(f"{d:%m-%d}({z:+.1f})" for d, z in zip(unusual["date"], unusual["robust_z"])))

    print("[2] 정상성 검정과 자기상관")
    st = stationarity(y)
    save_table(st.round(4), "stat6_stationarity.csv")
    print(st.round(3).to_string(index=False))
    acf_t = autocorr_table(y)
    save_table(acf_t.round(4), "stat6_acf_pacf.csv")
    print(f"  ACF: 시차 1 = {acf_t.loc[1, 'ACF']:.2f}, 시차 7 = {acf_t.loc[7, 'ACF']:.2f}, 시차 14 = {acf_t.loc[14, 'ACF']:.2f} "
          f"(유의 한계 ±{acf_t.loc[0, '유의 한계(±)']:.2f})")

    print("[3] 기온 효과 재추정")
    temp = temperature_effect(daily)
    if temp is not None:
        save_table(temp.round(4), "stat6_temperature_effect.csv")
        print(temp[["모형", "기온 계수 (건/°C)", "표준오차", "p", "잔차 자기상관"]].round(3).to_string(index=False))

    print("[4] 예측")
    fc, met, sel, n_train = forecast(y)
    save_table(rnd(fc, 1), "stat6_forecast.csv")
    save_table(met.round(3), "stat6_forecast_metrics.csv")
    save_table(sel.round(2), "stat6_model_selection.csv")
    print(f"  학습 {n_train}일, 평가 {len(fc)}일")
    print(met[["모형", "MAE", "RMSE", "MAPE(%)", "기준 대비 MAE 변화(%)"]].round(2).to_string(index=False))

    print("[5] 그래프")
    figures(setup_style(), comp, weekday, acf_t, fc, unusual)


if __name__ == "__main__":
    main()
