"""10단계(기초 개념): 데이터 사전, 대푯값, 퍼짐, 표본과 불확실성.

입력:  data/raw/nyc311_all_2024_apr_sep.csv        (결측 건수 확인용. 수정하지 않음)
       data/processed/nyc311_all_clean.parquet, data/processed/nyc311_all_daily.csv
출력:  outputs/tables/stat10_*.csv   (대시보드의 0, 2, 3, 5단계 탭이 읽는다)
실행:  프로젝트 루트에서  python analysis/stat_10_concepts.py

======================================================================================
이 스크립트의 자리
======================================================================================
번호는 10이지만 내용은 가장 기초다. stat_01~09 의 분석이 당연하게 쓰고 넘어간 개념들을 뒤늦게 채운다.
대시보드에서는 학습 순서에 맞춰 맨 앞쪽(0, 2, 3, 5단계)에 놓인다.

  (1) 데이터 사전   이 열은 어떤 종류의 값인가. 종류에 따라 할 수 있는 계산이 정해진다.
  (2) 대푯값        자료 전체를 숫자 하나로 요약한다면. 평균에도 여러 종류가 있고 답이 크게 다를 수 있다.
  (3) 퍼짐          대푯값 주위로 얼마나 흩어져 있는가. 대푯값만 보면 절반만 본 것이다.
  (4) 불확실성      우리가 가진 것은 '표본'이다. 표본에서 구한 값은 얼마나 믿을 수 있는가.

이 네 가지는 순서대로 쌓인다. 값의 종류를 알아야 대푯값을 고를 수 있고, 대푯값과 퍼짐을 알아야
"표본 평균이 얼마나 흔들리는가(불확실성)"를 말할 수 있으며, 그것이 뒤 단계의 가설검정으로 이어진다.
"""

import numpy as np
import pandas as pd
from scipy import stats

from stat_common import BOROUGH_KO, BOROUGHS, DAILY, RAW_311, SEED, load_clean, save_table

TRIM = 0.10                   # 절사평균: 양 끝에서 잘라내는 비율
SHIFTS = [0.01, 0.1, 1, 10]   # 0시간 처리: 더하는 상수 후보 (시간)
CLT_SIZES = [5, 30, 200, 2000]
CLT_REPS = 3000
BOOT_N = 20_000               # 부트스트랩 시연용 표본 크기
BOOT_B = 2000                 # 부트스트랩 반복 수
COVER_N = 50                  # 포함률 실험: 작은 표본의 크기
COVER_REPS = 400
COVER_B = 300
TOP_AGENCIES = 6


# ============================================================================ (1) 데이터 사전
def data_dictionary(df):
    """열마다 '값의 종류(척도)'를 적는다. 통계 분석의 첫 단계는 계산이 아니라 이 분류다.

    [측정 척도 4가지] 아래로 갈수록 할 수 있는 계산이 늘어난다.
        명목(nominal):  이름표일 뿐 순서가 없다. 예: 자치구, 기관.       가능: 개수 세기, 최빈값, 비율
        순서(ordinal):  순서는 있지만 간격이 같다는 보장이 없다. 예: 만족도. 가능: + 중앙값, 순위
        구간(interval): 간격이 일정하지만 '0'이 없음을 뜻하지 않는다. 예: 섭씨 온도, 날짜. 가능: + 평균, 차이
        비율(ratio):    진짜 0이 있어 '몇 배'를 말할 수 있다. 예: 처리 시간, 건수. 가능: + 비, 기하·조화평균
      섭씨 20도는 10도의 '두 배 덥다'고 할 수 없지만(구간), 10시간은 5시간의 두 배다(비율).
      기하평균과 조화평균은 '몇 배'라는 말이 성립하는 비율 척도에서만 뜻이 있다.

    [숫자처럼 생겼지만 숫자가 아닌 것] 가장 흔한 실수다.
        식별자(unique_key): 숫자로 적혀 있어도 이름표다. 평균을 내면 아무 뜻이 없다.
        우편번호, 전화번호도 마찬가지다.
        시각(hour): 0~23은 원처럼 이어져 있다(23시 다음이 0시). 그냥 평균을 내면 틀린 답이 나온다 ((2)에서 확인).
    """
    raw_missing = pd.read_csv(RAW_311, dtype=str).isna().sum() if RAW_311.exists() else pd.Series(dtype=int)
    spec = [
        # 열, 원본 열, 뜻, 종류, 척도, 할 수 있는 계산, 주의
        ("unique_key", "unique_key", "민원 1건의 고유 번호", "식별자", "명목",
         "개수 세기, 중복 확인", "숫자처럼 보여도 이름표다. 평균·합계·상관은 뜻이 없다"),
        ("created_dt", "created_date", "접수 일시", "날짜·시간", "구간",
         "차이(기간), 기간별 집계", "날짜 자체의 '몇 배'는 뜻이 없다. 두 날짜의 차이는 비율 척도가 된다"),
        ("closed_dt", "closed_date", "종결 일시", "날짜·시간", "구간",
         "차이(기간)", "미종결이면 비어 있다. 결측 자체가 정보다 (10단계 생존분석)"),
        ("agency", "agency", "담당 기관", "범주", "명목", "빈도, 비율, 최빈값", "순서가 없다. 번호를 붙여 평균을 내면 안 된다"),
        ("complaint_type", "complaint_type", "민원 유형", "범주", "명목", "빈도, 비율, 최빈값", "범주가 184개로 많다. 묶어서 써야 할 때가 많다"),
        ("borough", "borough", "자치구", "범주", "명목", "빈도, 비율, 최빈값", "집단 비교의 기준이 되는 변수"),
        ("latitude", "latitude", "위도", "수치(연속)", "구간", "지도 표시, 거리 계산(변환 후)", "각도다. 거리로 쓰려면 km 로 바꿔야 한다 (11단계)"),
        ("longitude", "longitude", "경도", "수치(연속)", "구간", "지도 표시, 거리 계산(변환 후)", "경도 1도의 길이는 위도에 따라 달라진다"),
        ("resolution_hours", None, "처리 시간 = 종결 - 접수 (시간)", "수치(연속)", "비율",
         "모든 평균, 분산, 비, 로그", "0 이상이고 오른쪽 꼬리가 매우 길다. 이 프로젝트의 중심 변수"),
        ("log_resolution", None, "log(1 + 처리 시간)", "수치(연속, 파생)", "구간",
         "평균, 분산, 회귀", "로그를 취하면 '몇 배'가 '차이'로 바뀐다. 평균을 되돌리면 기하평균이 된다"),
        ("within_24h", None, "24시간 안에 처리되었는가", "이진(0/1)", "명목",
         "비율 (0/1 의 평균이 곧 비율이다)", "평균이 뜻을 갖는 유일한 명목 변수 꼴"),
        ("date", None, "접수 날짜", "날짜", "구간", "일별 집계", ""),
        ("hour", None, "접수 시각 (0~23)", "수치(이산, 순환)", "구간(순환)",
         "빈도, 최빈값, 원형 평균", "23시와 0시는 이웃이다. 산술평균을 내면 틀린다"),
        ("weekday", None, "요일 (0=월 ... 6=일)", "범주(순환)", "순서(순환)", "빈도, 최빈값", "숫자는 이름표다. 평균 요일 2.7 같은 값은 뜻이 없다"),
        ("is_weekend", None, "주말 여부", "이진(0/1)", "명목", "비율", ""),
        ("month", None, "접수 월", "수치(이산)", "순서", "빈도, 월별 집계", ""),
        ("time_band", None, "접수 시간대 (4구간)", "범주", "순서", "빈도, 비율", "hour 를 구간으로 묶은 것"),
    ]
    rows = []
    for col, raw_col, meaning, kind, scale, ops, note in spec:
        s = df[col]
        if kind.startswith("수치") or kind.startswith("이진"):
            example = f"최소 {s.min():,.4g} · 중앙 {s.median():,.4g} · 최대 {s.max():,.4g}"
        elif kind.startswith("날짜"):
            example = f"{s.min():%Y-%m-%d} ~ {s.max():%Y-%m-%d}"
        else:
            example = ", ".join(map(str, s.value_counts().index[:3]))
        rows.append({
            "열": col, "뜻": meaning, "종류": kind, "척도": scale, "서로 다른 값": int(s.nunique()),
            "원본 결측 건수": int(raw_missing[raw_col]) if raw_col in raw_missing.index else None,
            "값의 예": example, "할 수 있는 계산": ops, "주의": note,
        })
    return pd.DataFrame(rows)


# ============================================================================ (2) 대푯값
def three_means(x, shift=0.0):
    """세 가지 평균. shift 는 0 을 피하려고 더했다가 다시 빼는 상수다.

    [산술평균]  (x1 + x2 + ... + xn) / n
        '더해서 똑같이 나누면' 얼마씩인가. 값들을 모두 이 숫자로 바꿔도 **합**이 유지된다.
        맞는 상황: 더하는 것이 뜻이 있는 양 (건수, 금액, 길이).  약점: 극단값 하나에 크게 끌려간다.

    [기하평균]  (x1 * x2 * ... * xn) 의 n제곱근  =  exp( log x 들의 산술평균 )
        값들을 모두 이 숫자로 바꿔도 **곱**이 유지된다.
        맞는 상황: 곱으로 변하는 양 (성장률, 수익률, 배율), 그리고 오른쪽 꼬리가 긴 양수 자료.
        "로그를 취해 평균을 내고 되돌린 값"이라서 stat_02 의 로그 변환(대시보드 4단계)과 같은 이야기다.
        로그정규분포를 따르는 자료에서는 기하평균이 중앙값과 같다.  약점: 0 이나 음수가 있으면 계산할 수 없다.

    [조화평균]  n / (1/x1 + 1/x2 + ... + 1/xn)  =  '역수의 산술평균'의 역수
        맞는 상황: 속도·비율처럼 "분자가 고정되고 분모가 변하는" 양.
        같은 거리를 시속 60km 로 가고 시속 40km 로 돌아오면 평균 속력은 50 이 아니라 48(조화평균)이다.
        느린 구간에서 시간을 더 많이 쓰기 때문이다.  약점: 0 이 있으면 계산할 수 없고, 0 에 가까운 값 하나가 결과를 지배한다.

    [항상 성립하는 순서]  산술평균 >= 기하평균 >= 조화평균  (모든 값이 양수일 때)
        세 값이 같아지는 것은 모든 값이 똑같을 때뿐이다. 값들이 고르지 않을수록 세 평균의 간격이 벌어지므로,
        산술평균 / 기하평균 의 비는 그 자체로 '자료가 얼마나 고르지 않은가'를 재는 지표가 된다.
    """
    v = np.asarray(x, dtype=float) + shift
    am = v.mean() - shift
    gm = np.exp(np.log(v).mean()) - shift
    hm = len(v) / (1 / v).sum() - shift
    return am, gm, hm


def circular_mean_hour(hours):
    """원형 평균: 시각처럼 끝과 처음이 이어진 값의 평균.

    23시와 1시의 산술평균은 12시(정오)지만, 상식적인 답은 0시(자정)다. 0~23 은 직선이 아니라 원이기 때문이다.
    해결: 각 시각을 시계 위의 한 점(각도)으로 보고, 그 점들의 x, y 좌표를 각각 평균한 뒤 다시 각도로 되돌린다.
    """
    ang = np.asarray(hours) / 24 * 2 * np.pi
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * 24 % 24)


def means_analysis(df, daily):
    h = df["resolution_hours"].to_numpy()
    pos = h[h > 0]
    n_zero = int((h == 0).sum())

    def row(name, column, x, shift, how, note):
        am, gm, hm = three_means(x, shift)
        return {"변수": name, "열": column, "0 처리": how, "n": len(x), "산술평균": am, "기하평균": gm, "조화평균": hm,
                "중앙값": float(np.median(x)), f"절사평균({TRIM:.0%})": stats.trim_mean(x, TRIM),
                "산술/기하 비": am / gm if gm > 0 else np.nan, "비고": note}

    rows = [
        row("처리 시간 (시간)", "resolution_hours", h, 1.0, "0 포함 (1을 더해 계산 후 뺌)",
            "오른쪽 꼬리가 매우 길다. 산술평균이 중앙값의 수십 배"),
        row("처리 시간 (시간)", "resolution_hours", pos, 0.0, "0 제외",
            f"0시간 {n_zero:,}건({n_zero / len(h):.1%})을 뺀 결과. 조화평균이 0 근처 값에 끌려간다"),
        row("일별 접수 건수", "n_complaints (일별 집계)", daily["n_complaints"].to_numpy(dtype=float), 0.0, "0 없음",
            "값이 고르다. 세 평균이 거의 같다"),
    ]
    for a in df["agency"].value_counts().index[:TOP_AGENCIES]:
        rows.append(row(f"처리 시간: {a}", "resolution_hours (기관별)", df.loc[df["agency"] == a, "resolution_hours"].to_numpy(),
                        1.0, "0 포함 (1을 더해 계산 후 뺌)", ""))
    means = pd.DataFrame(rows)

    # 0 처리 민감도: 더하는 상수를 바꾸면 기하·조화평균이 얼마나 달라지는가
    #   기하평균과 조화평균은 0 이 하나라도 있으면 계산할 수 없다. 흔한 대처는 (a) 0 을 빼거나 (b) 작은 상수를 더하는 것인데,
    #   어느 쪽도 '중립적'이지 않다. 더하는 상수가 작을수록 0 근처 값들이 결과를 더 강하게 끌어내린다.
    #   그래서 이런 평균을 보고할 때는 0 을 어떻게 처리했는지를 반드시 함께 적어야 한다.
    sens = [{"0 처리": "0 제외", "더한 상수(시간)": 0.0, "n": len(pos), **dict(zip(["산술평균", "기하평균", "조화평균"], three_means(pos)))}]
    for c in SHIFTS:
        sens.append({"0 처리": f"0 포함, {c:g}시간을 더해 계산", "더한 상수(시간)": c, "n": len(h),
                     **dict(zip(["산술평균", "기하평균", "조화평균"], three_means(h, c)))})
    sens = pd.DataFrame(sens)

    # 조화평균이 실제로 뜻을 갖는 곳: 처리 속도
    #   민원 1건을 처리하는 데 걸린 시간이 h 라면 처리 속도는 1/h (건/시간)이다.
    #   질문 A "민원 하나하나의 속도를 평균하면?"          -> 속도의 산술평균 = 1 / (시간의 조화평균)
    #   질문 B "전체 민원을 전체 시간으로 나눈 처리량은?"  -> 총 건수 / 총 시간 = 1 / (시간의 산술평균) = 속도의 조화평균
    #   두 답은 수백 배 차이가 난다. A 는 몇 분 만에 끝난 민원들이 값을 끌어올리고, B 는 오래 걸린 민원들이 시간을 다 잡아먹는다.
    #   "평균 속도"라는 같은 말이 질문에 따라 전혀 다른 숫자가 된다는 것이 조화평균이 가르쳐 주는 점이다.
    speed = 1 / pos
    am_h, gm_h, hm_h = three_means(pos)
    speed_t = pd.DataFrame([
        {"질문": "A. 민원 하나하나의 처리 속도를 평균하면", "계산": "속도(1/시간)의 산술평균", "값 (건/시간)": speed.mean(),
         "같은 값": f"1 / 처리시간의 조화평균 = 1 / {hm_h:.3f}", "누가 좌우하나": "몇 분 만에 끝난 민원"},
        {"질문": "B. 전체 민원 수를 전체 처리 시간으로 나누면", "계산": "속도의 조화평균 = 총 건수 / 총 시간", "값 (건/시간)": len(pos) / pos.sum(),
         "같은 값": f"1 / 처리시간의 산술평균 = 1 / {am_h:.1f}", "누가 좌우하나": "몇 달씩 걸린 민원"},
        {"질문": "C. 전형적인 민원 하나의 속도는", "계산": "속도의 기하평균 (= 속도의 중앙값에 가까움)", "값 (건/시간)": 1 / gm_h,
         "같은 값": f"1 / 처리시간의 기하평균 = 1 / {gm_h:.2f}", "누가 좌우하나": "가운데에 있는 민원"},
    ])

    # 가중평균: 집단별 비율을 합칠 때
    #   자치구별 24시간 내 처리율 5개를 그냥 평균하면 "자치구 하나하나를 똑같이 본 평균"이 되고,
    #   건수로 가중하면 "민원 하나하나를 똑같이 본 평균", 즉 시 전체의 실제 비율이 된다.
    #   규모가 다른 집단의 평균을 다시 평균할 때는 무엇을 한 표로 칠 것인지 먼저 정해야 한다.
    g = df.groupby("borough").agg(건수=("within_24h", "size"), 처리율=("within_24h", "mean")).reindex(BOROUGHS)
    weighted = pd.DataFrame({"자치구": [BOROUGH_KO[b] for b in g.index], "민원 건수": g["건수"].to_numpy(),
                             "24시간 내 처리율(%)": g["처리율"].to_numpy() * 100,
                             "가중치(건수 비중 %)": (g["건수"] / g["건수"].sum()).to_numpy() * 100})
    summary_w = pd.DataFrame([
        {"계산": "단순평균 (자치구 5개를 똑같이)", "값(%)": g["처리율"].mean() * 100, "뜻": "평균적인 자치구의 처리율"},
        {"계산": "가중평균 (건수로 가중)", "값(%)": np.average(g["처리율"], weights=g["건수"]) * 100, "뜻": "시 전체 민원의 실제 처리율"},
    ])

    # 범주형·순환형 변수의 대푯값
    hour = df["hour"].to_numpy()
    mode_rows = []
    for col, label in [("complaint_type", "민원 유형"), ("agency", "담당 기관"), ("borough", "자치구"), ("hour", "접수 시각"),
                       ("weekday", "요일 (0=월)")]:
        vc = df[col].value_counts()
        mode_rows.append({"변수": label, "열": col, "최빈값": str(vc.index[0]), "최빈값의 비중(%)": vc.iloc[0] / len(df) * 100,
                          "범주 수": len(vc), "평균을 내도 되나": "안 된다 (이름표)" if col != "hour" else "원형 평균을 써야 한다"})
    modes = pd.DataFrame(mode_rows)
    hour_t = pd.DataFrame([
        {"계산": "산술평균 (틀린 방법)", "값(시)": hour.mean(), "설명": "0~23 을 직선으로 본 값. 밤 시간대가 양 끝으로 갈라져 한낮 쪽으로 끌려간다"},
        {"계산": "원형 평균 (맞는 방법)", "값(시)": circular_mean_hour(hour), "설명": "시계 위의 점으로 보고 평균한 값"},
        {"계산": "최빈값", "값(시)": float(pd.Series(hour).mode().iloc[0]), "설명": "민원이 가장 많이 들어오는 시각"},
    ])
    return means, sens, speed_t, weighted, summary_w, modes, hour_t


# ============================================================================ (3) 퍼짐
def spread_analysis(df, daily):
    """퍼짐(산포)의 측도. 대푯값이 같아도 퍼짐이 다르면 전혀 다른 자료다.

    [범위]        최댓값 - 최솟값. 가장 단순하지만 극단값 두 개로만 정해진다.
    [분산]        '평균에서 떨어진 거리의 제곱'의 평균. 단위가 제곱이라 직접 읽기 어렵다.
                  표본분산은 n 이 아니라 n-1 로 나눈다 (평균을 자료에서 추정하느라 자유도를 하나 썼기 때문).
    [표준편차]    분산의 제곱근. 원래 단위로 돌아온다. "평균에서 보통 이만큼 떨어져 있다".
    [사분위범위 IQR]  Q3 - Q1. 가운데 50% 가 차지하는 폭. 극단값의 영향을 받지 않는다.
    [MAD]         중앙값 절대편차 = '중앙값에서 떨어진 거리'의 중앙값. 가장 튼튼한(robust) 퍼짐 측도.
                  1.4826 을 곱하면 정규분포에서 표준편차와 같은 크기가 된다.
    [변동계수 CV] 표준편차 / 평균. 단위가 없어져서 단위나 규모가 다른 변수끼리 퍼짐을 비교할 수 있다.
                  예: 일별 건수(평균 9천)와 처리 시간(평균 360)의 표준편차를 직접 비교하면 뜻이 없지만 CV 는 비교된다.

    [짝을 맞춰 쓴다]  대칭이고 극단값이 없으면: 평균 + 표준편차.  치우쳤거나 극단값이 있으면: 중앙값 + IQR (또는 MAD).
        치우친 자료에 평균과 표준편차를 쓰면 "평균 364시간, 표준편차 1,387시간"처럼 평균에서 표준편차를 빼면
        음수가 되는, 있을 수 없는 범위를 말하게 된다.

    [경험 법칙 68-95-99.7]  정규분포라면 평균 ±1, ±2, ±3 표준편차 안에 각각 68%, 95%, 99.7% 가 들어온다.
        이 법칙이 맞는지 직접 세어 보면 자료가 정규분포와 얼마나 다른지 알 수 있다.
    [이상치 규칙 세 가지]  같은 자료라도 규칙에 따라 이상치 개수가 크게 다르다.
        Z 규칙:    |x - 평균| > 3 표준편차.  이상치 자신이 평균과 표준편차를 부풀려 스스로를 가린다(마스킹).
        IQR 규칙:  Q1 - 1.5 IQR 보다 작거나 Q3 + 1.5 IQR 보다 큼 (상자그림의 수염 밖). 치우친 자료에서는 너무 많이 잡힌다.
        MAD 규칙:  |x - 중앙값| / (1.4826 MAD) > 3.5.  가장 튼튼하다.
    """
    series = {
        "처리 시간 (시간)": df["resolution_hours"].to_numpy(),
        "log(1 + 처리 시간)": df["log_resolution"].to_numpy(),
        "일별 접수 건수": daily["n_complaints"].to_numpy(dtype=float),
    }
    rows, rule_rows = [], []
    for name, x in series.items():
        mean, sd, med = x.mean(), x.std(ddof=1), np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        mad = 1.4826 * np.median(np.abs(x - med))
        rows.append({
            "변수": name, "n": len(x), "평균": mean, "중앙값": med, "범위": x.max() - x.min(), "분산": x.var(ddof=1), "표준편차": sd,
            "IQR": iqr, "MAD (x1.4826)": mad, "변동계수 CV(%)": sd / mean * 100, "P5": np.percentile(x, 5), "P95": np.percentile(x, 95),
            "평균 - 표준편차": mean - sd, "왜도": stats.skew(x),
            "어울리는 요약": "중앙값 + IQR" if abs(stats.skew(x)) > 1 else "평균 + 표준편차",
        })
        z = np.abs(x - mean) / sd
        within = [float((z <= k).mean() * 100) for k in (1, 2, 3)]
        n_z = int((z > 3).sum())
        n_iqr = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        n_mad = int((np.abs(x - med) / mad > 3.5).sum()) if mad > 0 else 0
        rule_rows.append({
            "변수": name, "±1 표준편차 안(%)": within[0], "±2 표준편차 안(%)": within[1], "±3 표준편차 안(%)": within[2],
            "정규분포라면": "68.3 / 95.4 / 99.7",
            "이상치: Z 규칙": n_z, "이상치: IQR 규칙": n_iqr, "이상치: MAD 규칙": n_mad,
            "IQR 규칙 이상치 비율(%)": n_iqr / len(x) * 100,
        })
    # 기관별 변동계수: 평균이 수백 배 다른 집단끼리도 '상대적인 고르지 않음'을 비교할 수 있다
    cv_rows = []
    for a in df["agency"].value_counts().index[:TOP_AGENCIES]:
        x = df.loc[df["agency"] == a, "resolution_hours"].to_numpy()
        q1, q3 = np.percentile(x, [25, 75])
        cv_rows.append({"기관": a, "n": len(x), "평균(h)": x.mean(), "표준편차(h)": x.std(ddof=1), "변동계수 CV(%)": x.std(ddof=1) / x.mean() * 100,
                        "중앙값(h)": np.median(x), "IQR(h)": q3 - q1, "IQR / 중앙값": (q3 - q1) / np.median(x) if np.median(x) > 0 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(rule_rows), pd.DataFrame(cv_rows)


# ============================================================================ (4) 표본과 불확실성
def clt_demo(df, rng):
    """표본분포와 중심극한정리(CLT).

    [모집단과 표본]  알고 싶은 것은 모집단(관심 대상 전체)의 값(모수)이지만, 볼 수 있는 것은 그 일부인 표본뿐이다.
        표본에서 구한 값(통계량)은 표본을 다시 뽑을 때마다 조금씩 달라진다.
        이 실습에서는 171만 건 전체를 '모집단'으로 삼고 거기서 작은 표본을 반복해 뽑아, 평소에는 볼 수 없는
        "표본을 다시 뽑으면 답이 얼마나 달라지는가"를 직접 눈으로 확인한다.

    [표본분포]  같은 크기의 표본을 수없이 뽑아 매번 평균을 구했을 때, 그 평균들이 이루는 분포.
        원자료의 분포(민원 하나하나의 처리 시간)와는 다른 것이다. 헷갈리기 쉬우니 주의.
    [표준오차 SE]  표본분포의 표준편차. "표본평균이 표본마다 이 정도 흔들린다"는 뜻이다.
        이론값은 σ / sqrt(n).  표본을 4배로 늘려야 오차가 절반이 된다 (100배 늘려야 1/10).
        표준편차(자료 하나하나의 퍼짐)와 표준오차(평균이라는 추정값의 흔들림)를 구별하는 것이 핵심이다.

    [중심극한정리]  원자료가 어떤 분포든, 표본이 충분히 크면 표본평균의 분포는 정규분포에 가까워진다.
        t 검정, 신뢰구간 등 수많은 방법이 이 정리에 기대고 있다.
    [흔한 오해 "n >= 30 이면 된다"]  교과서의 30 은 '적당히 얌전한' 분포에서의 경험칙일 뿐이다.
        처리 시간처럼 꼬리가 극단적으로 긴 자료에서는 n = 200 에서도 표본평균의 분포가 여전히 치우쳐 있다.
        아래 표의 '표본평균의 왜도'가 0 에 가까워지는 속도를 원자료와 로그 변환값에서 비교해 보자.
        로그를 취한 쪽은 훨씬 작은 n 에서 정규분포에 가까워진다. stat_02 의 변환(대시보드 4단계)이 뒤의 추론을 쉽게 만드는 이유다.
    """
    pops = {"처리 시간 (시간)": df["resolution_hours"].to_numpy(), "log(1 + 처리 시간)": df["log_resolution"].to_numpy()}
    rows, samples = [], []
    for name, x in pops.items():
        mu, sigma = x.mean(), x.std()
        for n in CLT_SIZES:
            means = x[rng.integers(0, len(x), size=(CLT_REPS, n))].mean(axis=1)
            rows.append({
                "변수": name, "표본 크기 n": n, "모집단 평균": mu, "표본평균들의 평균": means.mean(),
                "표준오차 (이론 σ/√n)": sigma / np.sqrt(n), "표준오차 (시뮬레이션)": means.std(ddof=1),
                "표본평균의 왜도": stats.skew(means),
                "모집단 평균의 ±10% 안에 든 비율(%)": float((np.abs(means - mu) <= 0.1 * abs(mu)).mean() * 100),
                "정규분포에 가까운가": "가깝다" if abs(stats.skew(means)) < 0.3 else "아직 치우침",
            })
            samples.append(pd.DataFrame({"변수": name, "표본 크기 n": n, "표본평균": np.round(means, 4)}))
    return pd.DataFrame(rows), pd.concat(samples)


def bootstrap_demo(df, rng):
    """부트스트랩(bootstrap): 공식 없이 컴퓨터로 불확실성을 재는 현대적 방법 (Efron, 1979).

    [문제]  평균의 표준오차는 σ/√n 이라는 공식이 있다. 그런데 중앙값, 기하평균, 절사평균, 90번째 백분위수의
        표준오차는? 공식이 없거나 매우 복잡하다. 평균조차도 꼬리가 긴 자료에서는 공식이 잘 맞지 않는다.

    [아이디어]  표본분포를 보려면 모집단에서 표본을 여러 번 뽑아야 하지만 우리에게는 표본이 하나뿐이다.
        그래서 **가진 표본을 모집단인 척 삼고**, 거기서 같은 크기의 표본을 **복원추출**(뽑은 것을 다시 넣고 뽑기)로
        수천 번 다시 뽑는다. 그때마다 통계량을 계산하면 그 값들의 흩어짐이 곧 불확실성의 추정이 된다.
        "신발 끈(bootstrap)을 당겨 스스로를 들어 올린다"는 말에서 온 이름이다.

    [백분위 신뢰구간]  부트스트랩으로 얻은 통계량 2,000개의 2.5번째와 97.5번째 백분위수를 95% 신뢰구간으로 쓴다.

    [신뢰구간의 뜻]  "이 구간에 참값이 있을 확률이 95%"가 아니다 (참값은 고정된 수라서 확률을 말할 수 없다).
        "같은 방법으로 표본을 뽑아 구간을 만드는 일을 반복하면, 그 구간들의 95% 가 참값을 포함한다"가 정확한 뜻이다.
        아래 포함률 실험이 바로 이 반복을 실제로 해 보는 것이다.

    [한계]  만능이 아니다. 표본이 아주 작거나 꼬리가 매우 길면 부트스트랩도 참값을 놓친다 (표본에 없는 극단값은
        다시 뽑아도 나오지 않으므로). 최댓값 같은 극단 통계량에도 맞지 않는다. 포함률 실험에서 직접 확인한다.
    """
    h = df["resolution_hours"].to_numpy()
    funcs = {
        "산술평균": lambda v: v.mean(axis=-1),
        "중앙값": lambda v: np.median(v, axis=-1),
        "기하평균 (1을 더해 계산)": lambda v: np.expm1(np.log1p(v).mean(axis=-1)),
        f"절사평균({TRIM:.0%})": lambda v: stats.trim_mean(v, TRIM, axis=-1),
        "90번째 백분위수": lambda v: np.percentile(v, 90, axis=-1),
    }
    sample = rng.choice(h, size=BOOT_N, replace=False)
    boot = {k: [] for k in funcs}
    for start in range(0, BOOT_B, 100):                          # 메모리를 아끼려고 100번씩 나눠 뽑는다
        re = sample[rng.integers(0, BOOT_N, size=(100, BOOT_N))]
        for k, f in funcs.items():
            boot[k].append(f(re))
    rows = []
    for k, f in funcs.items():
        b = np.concatenate(boot[k])
        est, truth = float(f(sample)), float(f(h))
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({"통계량": k, "표본 추정값": est, "부트스트랩 표준오차": b.std(ddof=1), "95% CI 하한": lo, "95% CI 상한": hi,
                     "구간 폭 / 추정값(%)": (hi - lo) / est * 100, "전체 자료의 값(참값)": truth,
                     "참값 포함": "예" if lo <= truth <= hi else "아니오"})
    ci = pd.DataFrame(rows)
    # 비교용: 평균의 고전적인 t 신뢰구간 = 평균 ± t x (s / sqrt(n))
    se = sample.std(ddof=1) / np.sqrt(BOOT_N)
    tcrit = stats.t.ppf(0.975, BOOT_N - 1)
    classic = {"통계량": "산술평균 (고전적 t 구간)", "표본 추정값": sample.mean(), "부트스트랩 표준오차": se,
               "95% CI 하한": sample.mean() - tcrit * se, "95% CI 상한": sample.mean() + tcrit * se,
               "구간 폭 / 추정값(%)": 2 * tcrit * se / sample.mean() * 100, "전체 자료의 값(참값)": h.mean(),
               "참값 포함": "예" if abs(sample.mean() - h.mean()) <= tcrit * se else "아니오"}
    ci = pd.concat([ci, pd.DataFrame([classic])], ignore_index=True)

    # 포함률 실험: 95% 신뢰구간은 정말 95% 를 맞히는가
    #   크기 50 인 표본을 400번 뽑고, 그때마다 구간을 만들어 참값(전체 자료의 값)이 들어 있는지 센다.
    #   이름값을 하는 방법이라면 포함률이 95% 근처여야 한다.
    cover = []
    log_h = np.log1p(h)
    targets = [("처리 시간의 산술평균", h, lambda v: v.mean(axis=-1)), ("처리 시간의 중앙값", h, lambda v: np.median(v, axis=-1)),
               ("log(1 + 처리 시간)의 산술평균", log_h, lambda v: v.mean(axis=-1))]
    for label, pop, f in targets:
        truth = float(f(pop))
        hit_t = hit_b = 0
        w_t, w_b = [], []
        for _ in range(COVER_REPS):
            s = pop[rng.integers(0, len(pop), COVER_N)]
            se_s = s.std(ddof=1) / np.sqrt(COVER_N)
            tc = stats.t.ppf(0.975, COVER_N - 1)
            hit_t += abs(s.mean() - truth) <= tc * se_s
            w_t.append(2 * tc * se_s)
            b = f(s[rng.integers(0, COVER_N, size=(COVER_B, COVER_N))])
            lo, hi = np.percentile(b, [2.5, 97.5])
            hit_b += lo <= truth <= hi
            w_b.append(hi - lo)
        is_mean = "산술평균" in label
        cover.append({"대상": label, "표본 크기": COVER_N, "참값": truth,
                      "t 구간 포함률(%)": hit_t / COVER_REPS * 100 if is_mean else np.nan,
                      "부트스트랩 구간 포함률(%)": hit_b / COVER_REPS * 100,
                      "t 구간 평균 폭": float(np.mean(w_t)) if is_mean else np.nan, "부트스트랩 구간 평균 폭": float(np.mean(w_b)),
                      "목표": 95.0})
    return ci, pd.DataFrame(cover)


def proportion_ci(df):
    """비율의 신뢰구간: 자치구별 24시간 내 처리율.

    [Wald 구간]   p ± 1.96 x sqrt( p(1-p)/n ).  가장 흔히 배우는 공식. n 이 작거나 p 가 0, 1 에 가까우면 잘 맞지 않는다.
    [Wilson 구간] 같은 생각을 더 정확하게 푼 것. 표본이 작아도 잘 맞아서 요즘은 이쪽을 권한다.
    여기서는 자치구마다 수만~수십만 건이라 두 구간이 거의 같고 폭도 매우 좁다 (소수 둘째 자리 수준).
    표본이 이렇게 크면 '우연에 의한 오차'는 거의 사라진다. 그렇다고 숫자가 옳다는 뜻은 아니다. 남는 것은
    자료 자체의 치우침(누가 신고하는가, 종결 시각을 어떻게 기록하는가)이며, 신뢰구간은 그것을 알려 주지 못한다.
    비교를 위해 "표본이 100건뿐이었다면" 구간이 얼마나 넓어지는지도 함께 계산한다.
    """
    z = 1.96
    rows = []
    for b in BOROUGHS:
        x = df.loc[df["borough"] == b, "within_24h"]
        n, p = len(x), x.mean()
        for label, m in [("실제 건수", n), ("100건뿐이었다면", 100)]:
            wald = z * np.sqrt(p * (1 - p) / m)
            center = (p + z * z / (2 * m)) / (1 + z * z / m)
            half = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
            rows.append({"자치구": BOROUGH_KO[b], "표본": label, "n": m, "24시간 내 처리율(%)": p * 100,
                         "Wald 하한": (p - wald) * 100, "Wald 상한": (p + wald) * 100,
                         "Wilson 하한": (center - half) * 100, "Wilson 상한": (center + half) * 100,
                         "구간 폭(%p)": 2 * half * 100})
    return pd.DataFrame(rows)


# ============================================================================ 실행
def main():
    df = load_clean()
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    rng = np.random.default_rng(SEED)

    print("[1] 데이터 사전")
    dd = data_dictionary(df)
    save_table(dd, "stat10_data_dictionary.csv")
    print(dd[["열", "종류", "척도", "서로 다른 값", "원본 결측 건수"]].to_string(index=False))

    print("[2] 대푯값")
    means, sens, speed_t, weighted, summary_w, modes, hour_t = means_analysis(df, daily)
    save_table(means.round(4), "stat10_means.csv")
    save_table(sens.round(4), "stat10_zero_sensitivity.csv")
    save_table(speed_t.round(5), "stat10_speed_means.csv")
    save_table(weighted.round(3), "stat10_weighted_groups.csv")
    save_table(summary_w.round(3), "stat10_weighted_summary.csv")
    save_table(modes.round(2), "stat10_modes.csv")
    save_table(hour_t.round(2), "stat10_hour_means.csv")
    print(means[["변수", "0 처리", "산술평균", "기하평균", "조화평균", "중앙값", "산술/기하 비"]].round(2).to_string(index=False))
    print(sens.round(3).to_string(index=False))
    print(speed_t[["질문", "값 (건/시간)"]].round(4).to_string(index=False))
    print(summary_w.round(2).to_string(index=False))
    print(hour_t.round(2).to_string(index=False))
    ok = (means["산술평균"] >= means["기하평균"] - 1e-9) & (means["기하평균"] >= means["조화평균"] - 1e-9)
    print(f"  산술 >= 기하 >= 조화: {int(ok.sum())}/{len(means)}개 행에서 성립")

    print("[3] 퍼짐")
    spread, rules, cv = spread_analysis(df, daily)
    save_table(spread.round(4), "stat10_spread.csv")
    save_table(rules.round(3), "stat10_spread_rules.csv")
    save_table(cv.round(3), "stat10_cv_by_agency.csv")
    print(spread[["변수", "평균", "표준편차", "IQR", "MAD (x1.4826)", "변동계수 CV(%)", "평균 - 표준편차", "어울리는 요약"]].round(2).to_string(index=False))
    print(rules.round(2).to_string(index=False))

    print("[4] 표본분포와 중심극한정리")
    clt, clt_samples = clt_demo(df, rng)
    save_table(clt.round(4), "stat10_clt_summary.csv")
    save_table(clt_samples, "stat10_clt_samples.csv")
    print(clt[["변수", "표본 크기 n", "표준오차 (이론 σ/√n)", "표준오차 (시뮬레이션)", "표본평균의 왜도", "정규분포에 가까운가"]].round(3).to_string(index=False))

    print("[5] 부트스트랩과 신뢰구간")
    ci, cover = bootstrap_demo(df, rng)
    save_table(ci.round(4), "stat10_bootstrap_ci.csv")
    save_table(cover.round(3), "stat10_ci_coverage.csv")
    print(ci.round(3).to_string(index=False))
    print(cover.round(2).to_string(index=False))
    pc = proportion_ci(df)
    save_table(pc.round(3), "stat10_proportion_ci.csv")
    print(pc[pc["표본"] == "실제 건수"][["자치구", "n", "24시간 내 처리율(%)", "Wilson 하한", "Wilson 상한"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
