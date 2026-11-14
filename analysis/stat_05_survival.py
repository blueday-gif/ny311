"""5단계(추가 분석 1): 생존분석 - 민원이 종결되기까지 걸리는 시간.

입력:  data/raw/nyc311_all_2024_apr_sep.csv   (미종결 민원이 필요해 정제 전 원본을 다시 읽는다. 수정하지 않음)
출력:  outputs/tables/stat5_*.csv, outputs/figures/stat5_*.png
실행:  프로젝트 루트에서  python analysis/stat_05_survival.py

======================================================================================
[개념] 왜 생존분석인가
======================================================================================
1~4단계에서는 "종결된 민원"만 남기고 처리 시간의 평균·중앙값을 구했다. 여기에는 함정이 있다.
분석 시점에 아직 처리 중인 민원은 '오래 걸리는 민원'일 가능성이 높은데, 그것을 통째로 버리면
남은 자료는 빨리 끝난 민원 쪽으로 치우친다. 그 결과 처리 시간이 실제보다 짧게 추정된다.

생존분석(survival analysis)은 "어떤 사건이 일어날 때까지 걸린 시간"을 다루는 통계 방법이다.
원래 의학에서 환자의 생존 기간을 분석하려고 만들어져 이름이 '생존'이지만, 사건은 무엇이든 된다.
  - 이 분석에서 '사건(event)'    = 민원 종결
  - '생존(survival)'            = 아직 종결되지 않고 열려 있는 상태
  - '생존 시간(survival time)'  = 접수부터 종결까지의 시간 (= Resolution_Hours)
주의: 보통은 사건이 나쁜 일(사망, 고장)이지만 여기서는 좋은 일(종결)이다. 그래서 해석 방향이
      반대다. 생존곡선이 빨리 떨어질수록, 위험(hazard)이 클수록 "민원을 빨리 처리한다"는 뜻이다.

======================================================================================
[개념] 중도절단 (censoring)
======================================================================================
관측을 끝낸 시점까지 사건이 일어나지 않은 자료를 '중도절단되었다(censored)'고 한다.
예: 9월 28일에 접수되어 관측 종료일(10월 1일)까지 열려 있는 민원.
    이 민원의 처리 시간은 모르지만 "적어도 3일보다 길다"는 정보는 있다.
생존분석의 핵심은 이 부분 정보를 버리지 않는 것이다. 중도절단 자료는 "그 시점까지는 사건 없이
버텼다"는 사실만큼만 계산에 기여한다.
  - 우측 중도절단(right censoring): 실제 사건 시간이 관측값보다 '오른쪽(더 뒤)'에 있다. 이 분석의 경우.
  - 가정: 중도절단은 사건 시간과 무관해야 한다(비정보적 중도절단). 여기서는 관측 종료일이라는
          달력상의 이유로 잘리는 것(행정적 중도절단)이므로 이 가정이 대체로 성립한다.

이 분석의 설정: 관측 종료 시점을 2024-10-01 00:00 으로 둔다 ("10월 1일 아침에 분석한다"고 가정).
  - 그 전에 종결된 민원              -> 사건 발생(event=1), 시간 = 종결 - 접수
  - 그때까지 열려 있던 민원           -> 중도절단(event=0), 시간 = 관측 종료 - 접수
    (원본은 2년 뒤에 내려받았으므로 실제로는 그 뒤 종결일이 적혀 있는 민원도 있다. 하지만 10월 1일
     시점에는 알 수 없던 정보이므로 쓰지 않는다. 이렇게 해야 '분석 시점에 아는 정보만 쓴다'는 원칙을 지킨다.)

======================================================================================
이 스크립트가 하는 일
======================================================================================
  (1) Kaplan-Meier 생존곡선을 직접 구현해 그리고, 라이브러리(statsmodels) 결과와 일치하는지 확인
  (2) 종결 건만 쓴 단순 중앙값과 Kaplan-Meier 중앙값을 비교해 치우침의 크기를 확인
  (3) 자치구별·기관별 생존곡선과 로그순위 검정(log-rank test)
  (4) Cox 비례위험 모형으로 여러 변수를 동시에 고려한 위험비(hazard ratio) 추정
  (5) 비례위험 가정 점검(log-log 그림)과 시간 순 분할 평가(C-index)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from stat_01_clean_descriptive import load_and_clean
from stat_common import (
    ACCENT, BOROUGH_COLOR, BOROUGH_KO, BOROUGHS, CATEGORICAL, END, GRID, INK, MUTED, OTHER_GRAY, SEED, SINGLE,
    TIME_BANDS, hour_axis, p_text, save_fig, save_table, setup_style,
)

CUTOFF = pd.Timestamp(END)          # 관측 종료 시점 (2024-10-01 00:00)
SPLIT_DATE = "2024-09-01"           # Cox 모형: 이 날짜 전 접수분으로 적합, 이후 접수분으로 평가
TOP_AGENCIES = 6
LOGRANK_N = 200_000                 # 로그순위 검정에 쓰는 무작위 표본 크기
COX_N = 60_000                      # Cox 적합 표본 크기 (전체로 적합하면 매우 느리다)
CINDEX_N = 3_000                    # C-index 계산 표본 (모든 쌍을 비교하므로 n^2 에 비례해 느려진다)
CHECK_HOURS = [1, 6, 24, 72, 168, 720]   # 생존확률을 읽을 시점: 1시간, 6시간, 1일, 3일, 1주, 30일


# ============================================================================ 자료 준비
def build_survival_data():
    """정제 기준은 1단계와 같되 미종결을 남기고, (duration, event) 두 열을 만든다.

    생존분석 자료는 항상 이 두 열이 한 쌍이다.
      duration: 관측된 시간. 사건이 났으면 사건까지의 시간, 아니면 관측이 끝난 시점까지의 시간.
      event:    1 = 사건(종결)이 관측됨, 0 = 중도절단.
    """
    df, _, _ = load_and_clean(keep_open=True)
    closed_in_window = df["closed_dt"].notna() & (df["closed_dt"] <= CUTOFF)
    end_time = df["closed_dt"].where(closed_in_window, CUTOFF)      # 종결 못 봤으면 관측 종료 시점에서 자른다
    df["duration"] = (end_time - df["created_dt"]).dt.total_seconds() / 3600
    df["event"] = closed_in_window.astype(int)

    n_never = int(df["closed_dt"].isna().sum())
    n_late = int((df["closed_dt"] > CUTOFF).sum())
    log = pd.DataFrame([
        ("생존분석 대상 행 수", len(df), "1단계 정제 기준 적용, 미종결은 유지"),
        ("사건 관측 (관측 종료 전 종결)", int(df["event"].sum()), "event = 1"),
        ("중도절단: 원본에도 종결일 없음", n_never, "event = 0, 관측 종료 시점에서 절단"),
        ("중도절단: 관측 종료 후에 종결됨", n_late, "event = 0, 이후 종결일은 쓰지 않음"),
        ("중도절단 비율(%)", round((1 - df["event"].mean()) * 100, 2), ""),
    ], columns=["항목", "값", "비고"])
    return df, log


# ============================================================================ Kaplan-Meier
def kaplan_meier(duration, event):
    """Kaplan-Meier 추정량을 직접 계산한다 (공부용 구현. 결과는 statsmodels 와 대조한다).

    [개념] 생존함수 S(t) = P(T > t): 시점 t 까지 사건이 일어나지 않았을 확률.
           여기서는 "접수 후 t시간이 지나도 아직 열려 있을 확률".

    [아이디어] t시간을 버티려면 그 전의 모든 순간을 차례로 버텨야 한다. 그래서 S(t)는
    사건이 일어난 시점마다의 '그 순간을 버틸 조건부 확률'을 전부 곱한 값이다 (곱-극한 추정량).

        S(t) = 곱_{t_i <= t} ( 1 - d_i / n_i )

        t_i: 사건이 한 번이라도 일어난 시점
        d_i: 시점 t_i 에 일어난 사건 수
        n_i: 시점 t_i 직전까지 살아남아 관측 중이던 대상 수 = '위험집합(risk set)'의 크기

    중도절단된 대상은 사건 수 d_i 에는 들어가지 않지만, 잘리기 전까지는 위험집합 n_i 에 들어 있다.
    이것이 중도절단 자료의 정보를 '버틴 만큼만' 쓰는 방법이다.

    [분산] Greenwood 공식: Var[S(t)] ≈ S(t)^2 * 합_{t_i <= t} d_i / ( n_i (n_i - d_i) )
    [신뢰구간] S 는 0~1 사이라 그냥 S ± 1.96·SE 로 만들면 범위를 벗어날 수 있다. 그래서 log(-log S) 로
              바꾼 척도에서 구간을 만든 뒤 되돌린다 (log-log 변환 신뢰구간).
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=float)
    times, inv = np.unique(duration, return_inverse=True)       # 서로 다른 관측 시점 (오름차순)
    d = np.bincount(inv, weights=event)                         # 시점별 사건 수
    leaving = np.bincount(inv)                                  # 시점별로 위험집합을 떠나는 수 (사건 + 중도절단)
    at_risk = len(duration) - np.concatenate([[0], np.cumsum(leaving)[:-1]])    # 시점 직전의 위험집합 크기

    surv = np.cumprod(1 - d / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        greenwood = np.cumsum(np.where(at_risk > d, d / (at_risk * (at_risk - d)), 0.0))
        se_loglog = np.sqrt(greenwood) / np.abs(np.log(surv))   # log(-log S) 척도의 표준오차
        lo = surv ** np.exp(1.96 * se_loglog)
        hi = surv ** np.exp(-1.96 * se_loglog)
    lo = np.where(np.isfinite(lo), lo, surv)                    # S 가 정확히 0 이나 1 인 곳은 구간을 만들 수 없다
    hi = np.where(np.isfinite(hi), hi, surv)
    return pd.DataFrame({"time": times, "at_risk": at_risk, "events": d, "survival": surv, "ci_low": lo, "ci_high": hi})


def surv_at(km, hours):
    """S(t) 는 계단함수다: 다음 사건이 일어날 때까지 값이 유지된다. t 이하의 마지막 시점 값을 읽는다."""
    idx = np.searchsorted(km["time"].to_numpy(), hours, side="right") - 1
    s = km["survival"].to_numpy()
    return np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)


def km_median(km):
    """중앙 생존시간: S(t) 가 처음으로 0.5 이하가 되는 시점 = 절반이 종결되는 데 걸리는 시간.

    중도절단이 많아 곡선이 0.5 아래로 내려가지 않으면 중앙값은 '추정 불가'다 (NaN 으로 둔다).
    평균은 꼬리의 중도절단 때문에 구하기 어렵기 때문에 생존분석에서는 평균보다 중앙값을 쓴다.
    """
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else np.nan


def summarize(km, label, n, n_event):
    row = {"집단": label, "n": n, "사건 수": n_event, "중도절단(%)": round((1 - n_event / n) * 100, 2),
           "KM 중앙값(h)": km_median(km)}
    for h, s in zip(CHECK_HOURS, surv_at(km, CHECK_HOURS)):
        # 1 - S(t) = t 시간 안에 종결될 확률 (누적 발생률)
        row[f"{h}h 내 종결률(%)"] = round((1 - s) * 100, 2)
    return row


def curve_on_grid(km, group_type, group):
    """대시보드용: 계단 곡선을 로그 간격 격자점에서 읽어 작게 저장한다."""
    grid = np.unique(np.concatenate([[0], np.expm1(np.linspace(0, np.log1p(km["time"].max()), 160))]))
    idx = np.clip(np.searchsorted(km["time"].to_numpy(), grid, side="right") - 1, 0, None)
    first = km["time"].iloc[0]
    out = pd.DataFrame({"구분": group_type, "집단": group, "시간(h)": grid,
                        "생존확률": np.where(grid >= first, km["survival"].to_numpy()[idx], 1.0),
                        "CI 하한": np.where(grid >= first, km["ci_low"].to_numpy()[idx], 1.0),
                        "CI 상한": np.where(grid >= first, km["ci_high"].to_numpy()[idx], 1.0)})
    return out.round(5)


# ============================================================================ 로그순위 검정
def logrank(df, col, label):
    """로그순위 검정 (log-rank test): 두 개 이상 집단의 생존곡선이 같은지 검정한다.

    [가설] H0: 모든 집단의 생존함수가 같다.   H1: 적어도 한 집단이 다르다.
    [원리] 사건이 일어난 시점마다 "집단 구분이 의미 없다면 각 집단에서 기대되는 사건 수"를 계산한다.
           (그 시점의 전체 사건 수를 위험집합 크기 비율대로 나눈 값)
           관측 사건 수(O)와 기대 사건 수(E)의 차이를 모든 시점에 걸쳐 더해 카이제곱 통계량을 만든다.
           자유도 = 집단 수 - 1.
    [특징] 4단계의 Kruskal-Wallis 와 목적은 비슷하지만 중도절단 자료를 그대로 쓸 수 있다.
           곡선이 서로 교차하지 않을 때(비례위험일 때) 검정력이 가장 좋다.
    [주의] 표본이 크면 아주 작은 차이도 유의하게 나온다. p 값보다 곡선과 중앙값 차이를 본다.
    """
    sub = df.sample(min(LOGRANK_N, len(df)), random_state=SEED)
    chi2, p = survdiff(sub["duration"].to_numpy(), sub["event"].to_numpy(), sub[col].to_numpy())
    k = sub[col].nunique()
    return {"검정": f"로그순위 검정: {label}", "표본 n": len(sub), "집단 수": k, "카이제곱": chi2, "자유도": k - 1,
            "p": p, "판정 (α=0.05)": "생존곡선이 다름" if p < 0.05 else "차이 유의하지 않음"}


# ============================================================================ Cox 비례위험 모형
def dummies(d, col, levels):
    """범주형 변수를 0/1 가변수로 바꾼다. levels 의 첫 범주가 기준(reference)이 되어 열에서 빠진다.
    계수는 "기준 범주에 비해" 어떤지를 뜻하므로, 기준을 무엇으로 잡았는지 항상 함께 적어야 한다."""
    cat = pd.Categorical(d[col], categories=levels)
    out = pd.get_dummies(cat, prefix=col, prefix_sep="=", drop_first=True, dtype=float)
    out.index = d.index
    return out


def concordance_index(duration, event, risk):
    """Harrell 의 C-index (일치 지수): 모형이 "누가 먼저 종결될지"의 순서를 얼마나 잘 맞히는가.

    두 민원을 짝지었을 때, 실제로 먼저 종결된 쪽에 모형이 더 높은 위험 점수를 줬으면 '일치'다.
      C = 0.5: 동전 던지기와 같음,  C = 1.0: 순서를 완벽하게 맞힘.  (로지스틱 회귀의 AUC 와 같은 뜻)
    비교 가능한 쌍: 시간이 더 짧은 쪽에서 사건이 실제로 관측된 쌍만 쓴다.
      (짧은 쪽이 중도절단이면 둘 중 누가 먼저 종결됐는지 알 수 없기 때문이다.)
    """
    t, e, r = np.asarray(duration), np.asarray(event), np.asarray(risk)
    comparable = (t[:, None] < t[None, :]) & (e[:, None] == 1)      # i 가 j 보다 먼저, i 는 사건 관측
    concordant = (r[:, None] > r[None, :]) & comparable
    tied = (r[:, None] == r[None, :]) & comparable
    return (concordant.sum() + 0.5 * tied.sum()) / comparable.sum()


def cox_model(df):
    """Cox 비례위험 모형 (Cox proportional hazards model).

    [개념] 위험함수(hazard) h(t): 시점 t 까지 사건이 없었던 대상이 '바로 그 순간' 사건을 겪는 순간 비율.
           여기서는 "아직 열려 있는 민원이 지금 이 순간 종결될 속도".
    [모형] h(t | x) = h0(t) * exp(b1*x1 + b2*x2 + ...)
           h0(t): 기저 위험. 모든 설명변수가 기준값일 때의 위험. 모양을 가정하지 않는다(그래서 준모수적 모형).
           exp(b): 위험비(hazard ratio, HR). 그 변수가 1 늘 때 위험이 몇 배가 되는지.
    [해석] 이 분석의 사건은 '종결'이므로
           HR > 1: 기준 범주보다 종결 속도가 빠르다 (민원이 빨리 처리된다)
           HR < 1: 기준 범주보다 종결 속도가 느리다
           예) HR = 2 는 "어느 시점에서든 종결될 순간 속도가 2배"라는 뜻이지, 처리 시간이 절반이라는 뜻은 아니다.
    [가정] 비례위험(proportional hazards): 위험비가 시간이 지나도 일정하다. 즉 두 집단의 위험곡선이
           서로 일정한 배율을 유지한다. 가정 점검은 아래 log-log 그림으로 한다.
    [추정] 부분가능도(partial likelihood)를 최대화한다. 사건이 일어난 시점마다 "위험집합 안에서 하필 이
           대상이 사건을 겪을 확률"만 따지므로 h0(t) 를 몰라도 계수를 추정할 수 있다.
    [동점] 같은 시각에 종결된 민원이 여러 건이면 순서를 알 수 없다. 여기서는 Breslow 근사를 쓴다.

    설명변수는 4단계 회귀와 같다(모두 접수 시점에 아는 정보). 4단계와 달리 미종결 민원도 포함된다.
    """
    df = df.copy()
    df["date"] = df["created_dt"].dt.normalize()
    train = df[df["date"] < SPLIT_DATE]
    top = train["agency"].value_counts().index[:TOP_AGENCIES]      # 범주 묶음은 학습 구간만 보고 정한다
    df["agency_group"] = df["agency"].where(df["agency"].isin(top), "기타")
    train, test = df[df["date"] < SPLIT_DATE], df[df["date"] >= SPLIT_DATE]

    levels = {
        "borough": list(train["borough"].value_counts().index),
        "agency_group": list(train["agency_group"].value_counts().index),
        "time_band": [TIME_BANDS[1]] + [t for t in TIME_BANDS if t != TIME_BANDS[1]],
    }

    def design(d):
        X = pd.concat([dummies(d, c, lv) for c, lv in levels.items()], axis=1)
        X["is_weekend"] = d["is_weekend"].astype(float)
        return X                                    # Cox 모형에는 절편이 없다 (h0(t) 가 그 역할을 한다)

    fit = train.sample(min(COX_N, len(train)), random_state=SEED)
    X = design(fit)
    res = sm.PHReg(fit["duration"], X, status=fit["event"], ties="breslow").fit()

    ci = res.conf_int()
    table = pd.DataFrame({
        "변수": X.columns, "계수(log HR)": res.params, "표준오차": res.bse, "z": res.tvalues, "p": res.pvalues,
        "위험비(HR)": np.exp(res.params), "HR CI 2.5%": np.exp(ci[:, 0]), "HR CI 97.5%": np.exp(ci[:, 1]),
        "유의 (α=0.05)": np.where(res.pvalues < 0.05, "예", "아니오"),
        "해석": np.where(res.pvalues >= 0.05, "기준과 차이 유의하지 않음",
                       np.where(res.params > 0, "기준보다 빨리 종결", "기준보다 늦게 종결")),
    })

    # 시간 순 분할 평가: 9월 접수분에서 위험 점수(X·b)가 실제 종결 순서와 얼마나 일치하는가
    ev = test.sample(min(CINDEX_N, len(test)), random_state=SEED)
    tr = fit.sample(min(CINDEX_N, len(fit)), random_state=SEED)
    c_test = concordance_index(ev["duration"], ev["event"], design(ev).to_numpy() @ res.params)
    c_train = concordance_index(tr["duration"], tr["event"], design(tr).to_numpy() @ res.params)
    summary = pd.DataFrame([{
        "모델": "Cox 비례위험 (Breslow)", "적합 n": len(fit), "적합 구간 사건 수": int(fit["event"].sum()),
        "설명변수 수": X.shape[1], "C-index (학습 표본)": c_train, "C-index (평가, 9월 접수)": c_test,
        "기준 범주": f"자치구={levels['borough'][0]}, 기관={levels['agency_group'][0]}, 시간대={levels['time_band'][0]}",
    }])
    print(f"  적합 {len(fit):,}건 (사건 {int(fit['event'].sum()):,}건), C-index 학습 {c_train:.3f} / 평가 {c_test:.3f}")
    print(f"  기준 범주: {summary['기준 범주'].iloc[0]}")
    return table, summary


# ============================================================================ 그래프
def step(ax, km, color, label, band=False, lw=1.8):
    """생존곡선은 계단 모양으로 그린다(사건이 일어난 순간에만 떨어진다). x 는 log1p(시간)."""
    # 점이 수십만 개면 그리기가 느리므로 격자에서 읽은 값으로 그린다
    g = curve_on_grid(km, "", "")
    x = np.log1p(g["시간(h)"])
    ax.step(x, g["생존확률"], where="post", color=color, lw=lw, label=label)
    if band:
        ax.fill_between(x, g["CI 하한"], g["CI 상한"], step="post", color=color, alpha=0.18, lw=0)


def finish(ax, title, max_hours):
    hour_axis(ax, max_hours)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.grid(True, color=GRID)
    ax.axhline(0.5, color=OTHER_GRAY, lw=1, ls="--")       # 이 선과 곡선이 만나는 곳이 중앙 생존시간
    ax.set_xlabel("접수 후 경과 시간 (로그 축)")
    ax.set_ylabel("아직 열려 있는 비율  S(t)")
    ax.set_title(title)


def figures(plt, df, km_all, km_by, naive_median):
    max_h = float(df["duration"].max())

    # (1) 전체 곡선 + 종결 건만 쓴 경험분포와 비교
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    step(ax, km_all, SINGLE, "Kaplan-Meier (미종결 포함)", band=True)
    closed = np.sort(df.loc[df["event"] == 1, "duration"].to_numpy())
    grid = np.expm1(np.linspace(0, np.log1p(max_h), 200))
    naive = 1 - np.searchsorted(closed, grid, side="right") / len(closed)     # 종결 건만의 1 - 경험누적분포
    ax.plot(np.log1p(grid), naive, color=ACCENT, lw=1.5, ls="--", label="종결 건만 사용 (미종결 제외)")
    med = km_median(km_all)
    if np.isfinite(med):
        ax.annotate(f"KM 중앙값 {med:,.1f}h\n종결 건만 {naive_median:,.1f}h", (np.log1p(med), 0.5),
                    xytext=(12, 14), textcoords="offset points", fontsize=9, color=INK)
    finish(ax, "민원 생존곡선: 접수 후 아직 열려 있는 비율", max_h)
    ax.legend(loc="upper right")
    save_fig(fig, "stat5_01_km_overall.png")

    # (2) 자치구별, (3) 기관별
    for key, name, colors, labels in [
        ("borough", "stat5_02_km_borough.png", BOROUGH_COLOR, BOROUGH_KO),
        ("agency", "stat5_03_km_agency.png", None, None),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for i, (g, km) in enumerate(km_by[key].items()):
            step(ax, km, colors[g] if colors else CATEGORICAL[i % len(CATEGORICAL)], labels[g] if labels else g, lw=1.6)
        finish(ax, f"{'자치구' if key == 'borough' else '담당 기관'}별 생존곡선 (빨리 내려갈수록 빨리 처리)", max_h)
        ax.legend(loc="upper right", ncol=2)
        save_fig(fig, name)

    # (4) 비례위험 가정 점검: log(-log S(t)) 대 log(t)
    #     Cox 모형이 맞다면 S_A(t) = S_B(t)^HR 이므로 log(-log S) 곡선들이 세로로 일정한 간격(= log HR)을
    #     유지하며 평행해야 한다. 곡선이 교차하거나 간격이 크게 변하면 그 변수는 비례위험 가정에 어긋난다.
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for g, km in km_by["agency"].items():
        k = km[(km["survival"] > 0.001) & (km["survival"] < 0.999) & (km["time"] > 0)]
        if len(k) > 400:
            k = k.iloc[np.linspace(0, len(k) - 1, 400).astype(int)]
        ax.plot(np.log(k["time"]), np.log(-np.log(k["survival"])), lw=1.5, label=g,
                color=CATEGORICAL[list(km_by["agency"]).index(g) % len(CATEGORICAL)])
    ticks = [0.1, 1, 6, 24, 168, 720]
    ax.set_xticks(np.log(ticks), ["6분", "1h", "6h", "1일", "1주", "30일"])
    ax.set_xlabel("경과 시간 (로그 축)")
    ax.set_ylabel("log( -log S(t) )")
    ax.set_title("비례위험 가정 점검: 곡선이 평행하면 가정 성립")
    ax.xaxis.grid(True, color=GRID)
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, "stat5_04_ph_check_loglog.png")


def forest(plt, cox):
    t = cox.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(t) + 1.4))
    y = np.arange(len(t))
    colors = [SINGLE if s == "예" else OTHER_GRAY for s in t["유의 (α=0.05)"]]
    ax.hlines(y, t["HR CI 2.5%"], t["HR CI 97.5%"], color=colors, lw=1.6)
    ax.scatter(t["위험비(HR)"], y, color=colors, s=26, zorder=3)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.set_yticks(y, t["변수"])
    ax.set_xlabel("위험비 HR (로그 축) · 1보다 크면 기준 범주보다 빨리 종결 · 회색은 유의하지 않음")
    ax.set_title("Cox 비례위험 모형: 위험비와 95% 신뢰구간")
    save_fig(fig, "stat5_05_cox_hazard_ratio.png")


# ============================================================================ 실행
def main():
    print("[1] 생존분석 자료 (duration, event)")
    df, log = build_survival_data()
    save_table(log, "stat5_data_log.csv")
    print(log.to_string(index=False))

    print("[2] Kaplan-Meier: 직접 구현과 statsmodels 대조")
    km_all = kaplan_meier(df["duration"], df["event"])
    check = df.sample(min(50_000, len(df)), random_state=SEED)
    mine = kaplan_meier(check["duration"], check["event"])
    ref = SurvfuncRight(check["duration"].to_numpy(), check["event"].to_numpy())
    # statsmodels 는 사건이 있었던 시점만 돌려주므로 같은 시점끼리 맞춰 비교한다
    diff = np.abs(surv_at(mine, ref.surv_times) - ref.surv_prob).max()
    print(f"  표본 {len(check):,}건에서 두 구현의 생존확률 최대 차이: {diff:.2e}  ({'일치' if diff < 1e-8 else '불일치 - 확인 필요'})")

    print("[3] 미종결을 버리면 얼마나 치우치는가")
    naive_median = float(df.loc[df["event"] == 1, "duration"].median())
    km_med = km_median(km_all)
    s24 = float(surv_at(km_all, [24])[0])
    naive24 = float((df.loc[df["event"] == 1, "duration"] <= 24).mean())
    bias = pd.DataFrame([
        {"지표": "처리시간 중앙값(h)", "종결 건만 사용": naive_median, "Kaplan-Meier (미종결 포함)": km_med},
        {"지표": "24시간 내 종결률(%)", "종결 건만 사용": naive24 * 100, "Kaplan-Meier (미종결 포함)": (1 - s24) * 100},
    ])
    bias["차이"] = bias["Kaplan-Meier (미종결 포함)"] - bias["종결 건만 사용"]
    save_table(bias.round(3), "stat5_naive_vs_km.csv")
    print(bias.round(2).to_string(index=False))

    print("[4] 집단별 생존곡선")
    top_ag = df["agency"].value_counts().index[:TOP_AGENCIES]
    km_by = {"borough": {}, "agency": {}}
    rows = [summarize(km_all, "전체", len(df), int(df["event"].sum()))]
    curves = [curve_on_grid(km_all, "전체", "전체")]
    for key, col, groups, name in [("borough", "borough", BOROUGHS, "자치구"), ("agency", "agency", top_ag, "기관")]:
        for g in groups:
            sub = df[df[col] == g]
            km = kaplan_meier(sub["duration"], sub["event"])
            km_by[key][g] = km
            label = BOROUGH_KO[g] if key == "borough" else g
            rows.append(summarize(km, f"{name}: {label}", len(sub), int(sub["event"].sum())))
            curves.append(curve_on_grid(km, name, label))
    summary = pd.DataFrame(rows)
    save_table(summary.round(3), "stat5_km_summary.csv")
    save_table(pd.concat(curves), "stat5_km_curves.csv")
    print(summary[["집단", "n", "중도절단(%)", "KM 중앙값(h)", "24h 내 종결률(%)", "168h 내 종결률(%)"]].round(2).to_string(index=False))

    print("[5] 로그순위 검정")
    lr = pd.DataFrame([logrank(df, "borough", "자치구"),
                       logrank(df[df["agency"].isin(top_ag)], "agency", f"기관 상위 {TOP_AGENCIES}")])
    save_table(lr.round(4), "stat5_logrank.csv")
    for _, r in lr.iterrows():
        print(f"  {r['검정']}: χ²={r['카이제곱']:,.1f}, df={r['자유도']}, p {p_text(r['p'])} -> {r['판정 (α=0.05)']}")

    print("[6] Cox 비례위험 모형")
    cox, cox_summary = cox_model(df)
    save_table(cox.round(5), "stat5_cox_coef.csv")
    save_table(cox_summary.round(4), "stat5_cox_summary.csv")
    print(cox[["변수", "위험비(HR)", "HR CI 2.5%", "HR CI 97.5%", "p", "해석"]].round(3).to_string(index=False))

    print("[7] 그래프")
    plt = setup_style()
    figures(plt, df, km_all, km_by, naive_median)
    forest(plt, cox)


if __name__ == "__main__":
    main()
