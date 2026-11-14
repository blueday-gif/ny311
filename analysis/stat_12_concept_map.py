"""12단계(학습 안내): 학습 지도, 개념 지도, 용어집.

입력:  outputs/tables/stat1_* ~ stat11_*   (앞 단계 스크립트들의 결과 표. 그래서 맨 마지막에 실행한다)
출력:  outputs/tables/stat12_learning_path.csv   대시보드 13단계의 순서, 핵심 질문, 배우는 개념
       outputs/tables/stat12_concept_map.csv     단계마다 "개념 - 적용한 열 - 현재 값 - 이 데이터에서의 뜻"
       outputs/tables/stat12_glossary.csv        용어집
실행:  프로젝트 루트에서  python analysis/stat_12_concept_map.py

개념 지도의 '현재 값'은 손으로 적지 않고 결과 표에서 읽어 온다. 분석을 다시 돌려 값이 바뀌면 이 스크립트만
다시 실행하면 된다. 어떤 표가 없으면 그 행만 건너뛰고 무엇을 건너뛰었는지 알려 준다.
"""

import pandas as pd

from stat_common import BOROUGH_KO, ROOT, TABLES, p_text, save_table

# ============================================================================ 학습 지도
# 대시보드의 탭 순서 = 공부 순서. 스크립트 번호(분석을 만든 순서)와 다르다는 점에 주의.
#   1부: 데이터와 친해지기 - 무엇이 들어 있는지, 숫자 하나로 요약하면, 흩어진 정도는, 모양은
#   2부: 불확실성과 추론 - 표본의 한계를 알고, 차이와 관계를 검정하고, 더 나은 모형으로
#   3부: 비교와 시간 - 요인이 여럿일 때, 시간 순서가 있을 때, 끝나지 않은 사건이 있을 때
#   4부: 위치와 유형 - 지도 위의 패턴, 닮은 것끼리 묶기
LEARNING_PATH = [
    ("s0", 0, "1부 데이터와 친해지기", "시작하기: 데이터 사전", "이 표에는 무엇이 들어 있고, 각 열로 무엇을 계산할 수 있나",
     "관측 단위, 변수, 측정 척도(명목·순서·구간·비율), 식별자, 결측", "모든 열", "없음", "stat_10"),
    ("s1", 1, "1부 데이터와 친해지기", "둘러보기: 지도와 필터", "언제, 어디서, 어떤 민원이 들어오나",
     "빈도, 비율, 집계, 필터링, 표본 지도", "borough, complaint_type, date, latitude, longitude", "0단계", "stat_01"),
    ("s2", 2, "1부 데이터와 친해지기", "대푯값: 숫자 하나로 요약하기", "자료 전체를 숫자 하나로 말한다면 무엇이어야 하나",
     "산술·기하·조화 평균, 중앙값, 최빈값, 절사평균, 가중평균, 원형 평균", "resolution_hours, n_complaints, hour, within_24h", "0~1단계", "stat_10"),
    ("s3", 3, "1부 데이터와 친해지기", "퍼짐과 기술통계", "값들이 대푯값 주위로 얼마나 흩어져 있나",
     "범위, 분산, 표준편차, IQR, MAD, 변동계수, 경험 법칙, 이상치 규칙, 빈도표", "resolution_hours, n_complaints, agency", "2단계", "stat_01, stat_10"),
    ("s4", 4, "1부 데이터와 친해지기", "분포의 모양, 변환, 표준화", "분포는 어떤 모양이고, 단위가 다른 값은 어떻게 맞춰 비교하나",
     "히스토그램, 왜도, 첨도, Q-Q plot, 정규성 검정, 로그·Box-Cox 변환, Z-score, Min-Max", "resolution_hours, 자치구 지표", "3단계", "stat_02, stat_03"),
    ("s5", 5, "2부 불확실성과 추론", "표본과 불확실성", "표본에서 구한 값을 얼마나 믿을 수 있나",
     "모집단과 표본, 표본분포, 표준오차, 중심극한정리, 신뢰구간, 부트스트랩, 포함률", "resolution_hours, within_24h", "2~4단계", "stat_10"),
    ("s6", 6, "2부 불확실성과 추론", "가설검정과 회귀", "차이나 관계가 우연인가, 그렇다면 얼마나 큰가",
     "귀무가설, p 값, 효과크기, ANOVA, Kruskal-Wallis, 사후검정, 카이제곱, 상관, 선형·로지스틱 회귀", "resolution_hours, borough, complaint_type, temp_mean_c", "5단계", "stat_04"),
    ("s7", 7, "2부 불확실성과 추론", "현대적 방법: 분위수 회귀와 부스팅", "평균 말고 분포 전체를 설명하려면? 더 잘 맞히려면?",
     "분위수 회귀, 핀볼 손실, 그래디언트 부스팅, 과적합, AUC, 보정, 순열 중요도", "resolution_hours, within_24h, complaint_type", "6단계", "stat_11"),
    ("s8", 8, "3부 비교와 시간", "기관 비교: 요인이 둘 이상일 때", "자치구 차이는 처리 속도 차이인가, 민원 구성 차이인가",
     "이원배치 ANOVA, 상호작용, 단순 주효과, 다중비교, 혼합효과 모형, ICC, 교란, 심슨의 역설", "resolution_hours, agency, borough, complaint_type", "6단계", "stat_07"),
    ("s9", 9, "3부 비교와 시간", "시계열: 시간 순서가 있는 자료", "시간에 따라 어떻게 변하고, 앞으로는 어떨까",
     "추세, 계절성, STL 분해, 자기상관, 정상성, ARIMA, 예측 평가", "n_complaints, temp_mean_c", "6단계", "stat_06"),
    ("s10", 10, "3부 비교와 시간", "생존분석: 끝나지 않은 사건", "아직 처리 중인 민원까지 고려하면 처리 시간은 얼마인가",
     "중도절단, 생존함수, Kaplan-Meier, 로그순위 검정, Cox 모형, 위험비, C-index", "resolution_hours, closed_dt", "5·6단계", "stat_05"),
    ("s11", 11, "4부 위치와 유형", "공간 분석: 지도 위의 패턴", "어디에 몰려 있고, 가까운 곳끼리 비슷한가",
     "KDE, 대역폭, 입지계수, DBSCAN, HDBSCAN, 공간 가중치, Moran's I, LISA, MAUP", "latitude, longitude, complaint_type", "3·6단계", "stat_08"),
    ("s12", 12, "4부 위치와 유형", "군집과 차원 축소", "동네를 몇 가지 유형으로 묶을 수 있나",
     "PCA, 적재값, 설명 분산, K-means, 실루엣, 부트스트랩 안정성, ARI", "1km 격자 칸의 민원 구성", "4·11단계", "stat_09"),
]

# ============================================================================ 용어집
# (용어, 영어, 처음 나오는 단계, 쉬운 설명)
GLOSSARY = [
    ("관측 단위", "unit of observation", 0, "표의 한 행이 무엇인가. 이 데이터에서는 민원 1건이다. 일별 집계에서는 하루, 군집분석에서는 1km 격자 칸이다."),
    ("변수", "variable", 0, "관측 단위마다 측정한 값. 표의 열 하나."),
    ("측정 척도", "scale of measurement", 0, "값의 종류. 명목 < 순서 < 구간 < 비율 순으로 할 수 있는 계산이 늘어난다."),
    ("명목 척도", "nominal", 0, "이름표일 뿐 순서가 없는 값. 자치구, 기관. 개수와 비율, 최빈값만 뜻이 있다."),
    ("비율 척도", "ratio", 0, "진짜 0 이 있어 '몇 배'를 말할 수 있는 값. 처리 시간, 건수. 기하·조화평균은 여기서만 뜻이 있다."),
    ("식별자", "identifier", 0, "행을 구별하려고 붙인 번호. 숫자처럼 보여도 평균이나 합계를 내면 안 된다."),
    ("결측", "missing value", 0, "비어 있는 값. 왜 비었는지가 중요하다. 종결일이 빈 것은 '아직 처리 중'이라는 정보다."),
    ("빈도", "frequency", 1, "어떤 값이 몇 번 나왔는가. 범주형 변수 요약의 기본."),
    ("산술평균", "arithmetic mean", 2, "모두 더해 개수로 나눈 값. 합을 유지한다. 극단값에 끌려간다."),
    ("기하평균", "geometric mean", 2, "모두 곱해 n제곱근을 취한 값 = 로그의 평균을 되돌린 값. 곱(배율)을 유지한다. 0 이 있으면 계산할 수 없다."),
    ("조화평균", "harmonic mean", 2, "역수의 평균의 역수. 속도·비율처럼 분자가 고정된 양의 평균에 쓴다."),
    ("중앙값", "median", 2, "크기순으로 줄 세웠을 때 한가운데 값. 극단값에 흔들리지 않는다."),
    ("최빈값", "mode", 2, "가장 많이 나온 값. 범주형 변수의 대푯값."),
    ("절사평균", "trimmed mean", 2, "양 끝에서 일정 비율을 잘라내고 구한 산술평균. 평균과 중앙값의 중간 성격."),
    ("가중평균", "weighted mean", 2, "값마다 다른 무게를 주고 구한 평균. 크기가 다른 집단의 비율을 합칠 때 쓴다."),
    ("원형 평균", "circular mean", 2, "시각·방향처럼 끝과 처음이 이어진 값의 평균. 각도로 바꿔 계산한다."),
    ("분산", "variance", 3, "평균에서 떨어진 거리의 제곱의 평균. 단위가 제곱이다."),
    ("표준편차", "standard deviation", 3, "분산의 제곱근. 자료 하나하나가 평균에서 보통 얼마나 떨어져 있나."),
    ("사분위범위", "IQR", 3, "Q3 - Q1. 가운데 50% 가 차지하는 폭. 극단값에 강하다."),
    ("MAD", "median absolute deviation", 3, "중앙값에서 떨어진 거리의 중앙값. 가장 튼튼한 퍼짐 측도."),
    ("변동계수", "coefficient of variation", 3, "표준편차 / 평균. 단위가 사라져 규모가 다른 변수끼리 퍼짐을 비교할 수 있다."),
    ("이상치", "outlier", 3, "다른 값들과 동떨어진 값. 어떤 규칙(Z, IQR, MAD)으로 정하느냐에 따라 개수가 크게 달라진다."),
    ("왜도", "skewness", 4, "분포가 한쪽으로 치우친 정도. 양수면 오른쪽 꼬리가 길다."),
    ("첨도", "kurtosis", 4, "꼬리가 얼마나 두꺼운가. 초과첨도 0 이 정규분포 기준."),
    ("Q-Q plot", "quantile-quantile plot", 4, "자료의 분위수를 정규분포의 분위수와 나란히 찍은 그림. 직선에 가까울수록 정규분포에 가깝다."),
    ("로그 변환", "log transform", 4, "값에 로그를 취하는 것. '몇 배'를 '차이'로 바꾸고 오른쪽 꼬리를 줄인다."),
    ("표준화 (Z-score)", "standardization", 4, "(값 - 평균) / 표준편차. 평균 0, 표준편차 1 로 맞춘다."),
    ("정규화 (Min-Max)", "normalization", 4, "(값 - 최솟값) / (최댓값 - 최솟값). 0~1 로 맞춘다."),
    ("모집단과 표본", "population, sample", 5, "알고 싶은 전체와 실제로 관찰한 일부."),
    ("표본분포", "sampling distribution", 5, "표본을 반복해서 뽑을 때 통계량(예: 평균)이 이루는 분포."),
    ("표준오차", "standard error", 5, "표본분포의 표준편차. 추정값이 표본마다 얼마나 흔들리나. 표준편차와 다르다."),
    ("중심극한정리", "central limit theorem", 5, "표본이 크면 표본평균의 분포가 정규분포에 가까워진다. 꼬리가 긴 자료에서는 '크다'의 기준이 훨씬 높다."),
    ("신뢰구간", "confidence interval", 5, "같은 방법을 반복하면 그 구간들의 95% 가 참값을 포함하도록 만든 범위."),
    ("부트스트랩", "bootstrap", 5, "가진 표본에서 복원추출을 반복해 불확실성을 재는 방법. 공식이 없는 통계량에도 쓸 수 있다."),
    ("귀무가설", "null hypothesis", 6, "'차이가 없다', '관계가 없다'는 기본 가정. 검정은 이것을 기각할 증거가 있는지 본다."),
    ("p 값", "p-value", 6, "귀무가설이 맞다면 지금만큼 극단적인 결과가 나올 확률. 효과의 크기를 말해 주지 않는다."),
    ("효과크기", "effect size", 6, "차이나 관계가 얼마나 큰가 (η², ε², Cramér's V 등). 표본이 크면 p 값보다 이것이 중요하다."),
    ("상관계수", "correlation", 6, "두 변수가 함께 움직이는 정도. -1~1. 인과를 뜻하지 않는다."),
    ("회귀계수", "regression coefficient", 6, "다른 변수를 고정했을 때 그 변수가 한 단위 변하면 결과가 얼마나 변하나."),
    ("분위수 회귀", "quantile regression", 7, "평균 대신 중앙값, 90번째 백분위수 같은 분위수를 설명하는 회귀."),
    ("그래디언트 부스팅", "gradient boosting", 7, "앞 나무들이 틀린 부분을 다음 나무가 맞히도록 나무를 차례로 더해 가는 예측 모형."),
    ("과적합", "overfitting", 7, "학습 자료의 잡음까지 외워 새 자료에서 오히려 못 맞히는 현상."),
    ("AUC", "area under the ROC curve", 7, "양성 하나와 음성 하나를 골랐을 때 양성에 더 높은 점수를 줄 확률. 0.5 무작위, 1 완벽."),
    ("순열 중요도", "permutation importance", 7, "변수 하나를 뒤섞었을 때 성능이 얼마나 떨어지나. 모형 종류와 상관없이 쓸 수 있다."),
    ("상호작용", "interaction", 8, "한 요인의 효과가 다른 요인의 수준에 따라 달라지는 것."),
    ("교란", "confounding", 8, "두 변수의 관계를 제3의 변수가 만들어 내거나 왜곡하는 것."),
    ("심슨의 역설", "Simpson's paradox", 8, "전체로 본 관계와 집단 안에서 본 관계의 방향이 반대가 되는 현상."),
    ("ICC", "intraclass correlation", 8, "전체 분산 가운데 묶음(민원 유형) 차이로 생기는 비율."),
    ("자기상관", "autocorrelation", 9, "자기 자신의 과거 값과의 상관. 시계열에서 독립 가정을 깨뜨린다."),
    ("정상성", "stationarity", 9, "평균과 분산이 시간에 따라 변하지 않는 성질. ARIMA 의 전제."),
    ("중도절단", "censoring", 10, "관측이 끝날 때까지 사건이 일어나지 않은 자료. 사건 시간은 모르지만 '적어도 그만큼'은 안다."),
    ("위험비", "hazard ratio", 10, "기준 집단에 비해 사건이 일어나는 순간 속도가 몇 배인가."),
    ("커널 밀도 추정", "KDE", 11, "점마다 작은 언덕을 쌓아 매끄러운 밀도 지도를 만드는 방법."),
    ("DBSCAN", "density-based clustering", 11, "반경 안에 이웃이 충분한 점들을 이어 군집을 만들고 나머지는 잡음으로 남기는 방법."),
    ("Moran's I", "Moran's I", 11, "내 값과 이웃들의 값의 상관. 가까운 곳끼리 비슷한 정도."),
    ("MAUP", "modifiable areal unit problem", 11, "구역을 어떤 크기·모양으로 나누느냐에 따라 통계가 달라지는 문제."),
    ("주성분 분석", "PCA", 12, "서로 얽힌 여러 변수를 퍼짐이 큰 방향 순으로 새 축 몇 개로 요약하는 방법."),
    ("K-means", "K-means", 12, "중심점 k개를 두고 가까운 것끼리 묶기를 반복하는 군집화 방법."),
    ("실루엣 계수", "silhouette", 12, "자기 군집과는 가깝고 다른 군집과는 먼 정도. -1~1."),
    ("ARI", "adjusted Rand index", 12, "두 군집 결과가 얼마나 같은가. 1 완전 일치, 0 무작위 수준."),
]

_missing = []


def tb(name):
    path = TABLES / name
    if not path.exists():
        _missing.append(name)
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def build_concept_map():
    rows = []

    def add(key, concept, plain, column, value, meaning):
        rows.append({"key": key, "개념": concept, "한 줄 정의": plain, "적용한 열": column, "현재 값": value, "이 데이터에서의 뜻": meaning})

    def safe(fn):
        try:
            fn()
        except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
            print(f"  건너뜀 ({fn.__name__}): {type(e).__name__} {e}")

    # ---------------- 0 시작하기
    def s0():
        dd = tb("stat10_data_dictionary.csv")
        sc = dd["척도"].str.replace(r"\(.*\)", "", regex=True).value_counts()
        add("s0", "측정 척도", "값의 종류에 따라 할 수 있는 계산이 정해진다", "모든 열",
            ", ".join(f"{k} {v}개" for k, v in sc.items()),
            "비율 척도는 처리 시간과 건수뿐이다. 기하·조화평균, '몇 배' 비교는 이 열들에서만 뜻이 있다")
        uk = dd.set_index("열").loc["unique_key"]
        add("s0", "식별자", "숫자처럼 보여도 이름표다", "unique_key", f"서로 다른 값 {uk['서로 다른 값']:,.0f}개 = 행 수",
            "중복 확인과 개수 세기에만 쓴다. 평균을 내면 아무 뜻이 없다")
        cl = dd.set_index("열")
        add("s0", "결측", "비어 있는 값. 왜 비었는지가 중요하다", "closed_date, latitude",
            f"종결일 {cl.loc['closed_dt', '원본 결측 건수']:,.0f}건, 좌표 {cl.loc['latitude', '원본 결측 건수']:,.0f}건",
            "종결일 결측은 '아직 처리 중'이라는 정보라서 10단계 생존분석이 살려 쓴다. 좌표 결측은 지도에서만 뺀다")
        log = tb("stat1_exclusion_log.csv").set_index("항목")["건수"]
        add("s0", "관측 단위와 정제", "한 행이 무엇이고, 무엇을 뺐는가", "전체 행",
            f"원본 {log['원본 행 수']:,.0f}건 → 분석 {log['최종 분석 대상 행 수']:,.0f}건",
            "뺀 이유와 건수를 기록해 두어야 결과를 재현하고 의심할 수 있다")

    # ---------------- 1 둘러보기
    def s1():
        fb = tb("stat1_freq_borough.csv")
        top = fb.iloc[0]
        add("s1", "빈도와 비율", "몇 번 나왔고 전체의 몇 %인가", "borough", f"{BOROUGH_KO.get(top['borough'], top['borough'])} {top['비율(%)']:.1f}% (1위)",
            "건수가 많다고 문제가 심하다는 뜻은 아니다. 인구가 다르기 때문이다 (4단계의 인구 대비 비율)")
        ft = tb("stat1_freq_complaint_type.csv")
        add("s1", "범주가 많을 때", "상위 몇 개만 보고 나머지는 묶는다", "complaint_type",
            f"184개 유형 중 상위 3개가 {ft['비율(%)'].iloc[:3].sum():.1f}%", "범주 대부분은 드물다. 그래서 분석마다 상위 N개 + 기타로 묶었다")
        ns = tb("stat1_numeric_summary.csv").set_index("변수")
        add("s1", "집계", "행 여러 개를 하나로 요약한 새 관측 단위", "date → 일별 접수 건수",
            f"하루 평균 {ns.loc['일별 민원 접수 건수', '평균']:,.0f}건 (183일)", "민원 1건이 아니라 '하루'가 관측 단위가 된다. 9단계 시계열의 재료")

    # ---------------- 2 대푯값
    def s2():
        m = tb("stat10_means.csv")
        r = m[(m["열"] == "resolution_hours") & m["0 처리"].str.startswith("0 포함")].iloc[0]
        add("s2", "산술평균", "모두 더해 개수로 나눈 값", "resolution_hours", f"{r['산술평균']:,.1f}시간",
            f"중앙값 {r['중앙값']:.1f}시간의 약 {r['산술평균'] / r['중앙값']:.0f}배. 몇 달씩 걸린 소수의 민원이 끌어올렸다")
        add("s2", "기하평균", "로그의 평균을 되돌린 값", "resolution_hours (+1)", f"{r['기하평균']:,.1f}시간",
            "4단계에서 배울 로그 변환과 같은 이야기다 (로그의 평균을 되돌린 값). 전형적인 민원의 처리 시간에 더 가깝다")
        d = m[m["변수"] == "일별 접수 건수"].iloc[0]
        add("s2", "세 평균의 간격", "산술 ≥ 기하 ≥ 조화. 고르지 않을수록 벌어진다", "resolution_hours vs n_complaints",
            f"산술/기하 비: 처리 시간 {r['산술/기하 비']:.1f}, 일별 건수 {d['산술/기하 비']:.2f}",
            "일별 건수는 고르게 퍼져 세 평균이 거의 같고, 처리 시간은 극단적으로 고르지 않다")
        sp = tb("stat10_speed_means.csv")
        add("s2", "조화평균", "역수의 평균의 역수. 속도·비율에 쓴다", "1 / resolution_hours (처리 속도)",
            f"하나하나의 속도 평균 {sp['값 (건/시간)'].iloc[0]:.2f}건/h vs 전체 처리량 {sp['값 (건/시간)'].iloc[1]:.4f}건/h",
            "'평균 속도'라는 같은 말이 질문에 따라 수백 배 다른 숫자가 된다")
        w = tb("stat10_weighted_summary.csv")
        add("s2", "가중평균", "값마다 무게를 달리 준 평균", "within_24h (자치구별)",
            f"단순 {w['값(%)'].iloc[0]:.1f}% vs 가중 {w['값(%)'].iloc[1]:.1f}%", "자치구 규모가 달라서 '평균적인 자치구'와 '시 전체'의 답이 다르다")
        hm = tb("stat10_hour_means.csv")
        add("s2", "원형 평균", "끝과 처음이 이어진 값의 평균", "hour",
            f"산술 {hm['값(시)'].iloc[0]:.1f}시 vs 원형 {hm['값(시)'].iloc[1]:.1f}시", "밤 민원(23시, 0시)이 양 끝으로 갈라져 산술평균이 틀어진다")

    # ---------------- 3 퍼짐
    def s3():
        s = tb("stat10_spread.csv").set_index("변수")
        h = s.loc["처리 시간 (시간)"]
        add("s3", "표준편차와 IQR", "평균 기준의 퍼짐과 가운데 50% 의 폭", "resolution_hours",
            f"표준편차 {h['표준편차']:,.0f}h, IQR {h['IQR']:,.0f}h",
            f"평균 - 표준편차 = {h['평균 - 표준편차']:,.0f}시간으로 음수가 된다. 치우친 자료에는 중앙값 + IQR 이 맞다")
        dly = s.loc["일별 접수 건수"]
        add("s3", "변동계수", "표준편차 / 평균. 단위 없는 상대적 퍼짐", "n_complaints vs resolution_hours",
            f"일별 건수 {dly['변동계수 CV(%)']:.0f}%, 처리 시간 {h['변동계수 CV(%)']:.0f}%", "처리 시간이 상대적으로 30배 더 고르지 않다")
        ru = tb("stat10_spread_rules.csv").set_index("변수")
        r = ru.loc["처리 시간 (시간)"]
        add("s3", "경험 법칙 68-95-99.7", "정규분포라면 ±1, 2, 3 표준편차 안에 드는 비율", "resolution_hours",
            f"±1 표준편차 안 {r['±1 표준편차 안(%)']:.0f}% (정규라면 68%)", "표준편차가 극단값 때문에 부풀어서 거의 모든 값이 ±1 안에 들어간다")
        add("s3", "이상치 규칙", "Z, IQR, MAD 규칙", "resolution_hours",
            f"Z {r['이상치: Z 규칙']:,.0f}건 / IQR {r['이상치: IQR 규칙']:,.0f}건 / MAD {r['이상치: MAD 규칙']:,.0f}건",
            "같은 자료인데 규칙에 따라 10배 넘게 다르다. 치우친 자료에서는 '이상치'가 아니라 '긴 꼬리'일 수 있다")

    # ---------------- 4 분포
    def s4():
        n = tb("stat2_normality_tests.csv")
        add("s4", "왜도와 변환", "치우친 정도. 로그·Box-Cox 로 줄인다", "resolution_hours",
            f"원자료 {n['왜도'].iloc[0]:.2f} → log1p {n['왜도'].iloc[1]:.2f} → Box-Cox {n['왜도'].iloc[2]:.2f}",
            "변환하면 대칭에 가까워져 평균·표준편차·t 검정이 다시 뜻을 갖는다")
        p_max = n["Shapiro-Wilk p"].max()
        add("s4", "정규성 검정", "정규분포인지 검정한다", "resolution_hours (세 가지 변환)",
            "세 변환 모두 Shapiro-Wilk p < 0.001" if p_max < 0.001 else f"Shapiro-Wilk p 값 최대 {p_max:.3f}","셋 다 기각. 표본이 크면 작은 이탈도 기각되므로 Q-Q 그림과 왜도로 판단한다")
        o = tb("stat3_outlier_borough_days.csv")
        add("s4", "Z-score 와 이상치", "평균에서 표준편차 몇 배 떨어졌나", "자치구 x 날짜의 일별 건수",
            f"|Z| > 3 인 자치구-날짜 {len(o)}건 / 915", "자치구가 5개뿐이면 |Z| 가 2 를 넘을 수 없어 날짜 단위로 내려가 찾았다")
        e = tb("stat3_efficiency_index.csv").sort_values("순위")
        add("s4", "Min-Max 정규화", "최소 0, 최대 1 로 맞춘다", "자치구 지표 3개 -> 효율성 지수",
            f"1위 {BOROUGH_KO[e['borough'].iloc[0]]} {e['효율성 지수(0-100)'].iloc[0]:.0f}점, "
            f"5위 {BOROUGH_KO[e['borough'].iloc[-1]]} {e['효율성 지수(0-100)'].iloc[-1]:.0f}점",
            "5개 중 최고·최저가 항상 100·0 이 되므로 작은 차이도 크게 보인다. 민원 유형도 보정하지 않았다 (8단계에서 다시 본다)")

    # ---------------- 5 불확실성
    def s5():
        c = tb("stat10_clt_summary.csv")
        r30 = c[(c["변수"] == "처리 시간 (시간)") & (c["표본 크기 n"] == 30)].iloc[0]
        l30 = c[(c["변수"] != "처리 시간 (시간)") & (c["표본 크기 n"] == 30)].iloc[0]
        add("s5", "표준오차", "표본평균이 표본마다 흔들리는 정도 = σ/√n", "resolution_hours, n=30",
            f"{r30['표준오차 (이론 σ/√n)']:,.0f}시간", "표본 30건으로 평균을 내면 ±250시간쯤 흔들린다. 표본을 4배로 늘려야 절반이 된다")
        add("s5", "중심극한정리", "표본이 크면 표본평균이 정규분포에 가까워진다", "resolution_hours vs log",
            f"n=30 표본평균의 왜도: 원자료 {r30['표본평균의 왜도']:.2f}, 로그 {l30['표본평균의 왜도']:.2f}",
            "'n ≥ 30 이면 된다'는 꼬리가 긴 자료에서는 틀린다. 로그를 취하면 훨씬 빨리 정규에 가까워진다")
        b = tb("stat10_bootstrap_ci.csv").set_index("통계량")
        md = b.loc["중앙값"]
        add("s5", "부트스트랩 신뢰구간", "복원추출을 반복해 만든 구간", "resolution_hours 의 중앙값 (표본 2만 건)",
            f"{md['표본 추정값']:.2f}h, 95% CI {md['95% CI 하한']:.2f} ~ {md['95% CI 상한']:.2f}", "중앙값은 간단한 표준오차 공식이 없지만(분포의 밀도를 알아야 한다) 부트스트랩으로는 바로 구간을 만든다")
        cv = tb("stat10_ci_coverage.csv").set_index("대상")
        add("s5", "포함률", "95% 신뢰구간이 실제로 참값을 포함하는 비율", "resolution_hours 의 평균, n=50",
            f"t 구간 {cv.loc['처리 시간의 산술평균', 't 구간 포함률(%)']:.0f}%, 로그 평균 {cv.loc['log(1 + 처리 시간)의 산술평균', 't 구간 포함률(%)']:.0f}%",
            "꼬리가 긴 자료에 공식을 그대로 쓰면 '95%' 구간이 실제로는 80% 대만 맞힌다")
        pc = tb("stat10_proportion_ci.csv")
        real, small = pc[pc["표본"] == "실제 건수"], pc[pc["표본"] != "실제 건수"]
        add("s5", "비율의 신뢰구간", "Wilson 구간", "within_24h (자치구별)",
            f"폭 {real['구간 폭(%p)'].mean():.2f}%p (실제) vs {small['구간 폭(%p)'].mean():.0f}%p (100건이라면)",
            "표본이 크면 우연 오차는 거의 없다. 남는 문제는 자료 자체의 치우침이며 신뢰구간은 그것을 알려 주지 않는다")

    # ---------------- 6 검정·회귀
    def s6():
        g = tb("stat4_group_tests.csv")
        kw = g[g["검정"].str.startswith("Kruskal")].iloc[0]
        add("s6", "p 값과 효과크기", "우연인가와 얼마나 큰가는 다른 질문", "resolution_hours ~ borough",
            f"Kruskal-Wallis p {p_text(kw['p'])}, {kw['효과크기']}",
            "유의하지만 자치구가 설명하는 몫은 0.2% 도 안 된다. 표본이 크면 p 값만 보면 안 된다")
        chi = tb("stat4_chi2_summary.csv").iloc[0]
        add("s6", "카이제곱과 Cramér's V", "범주형 두 변수의 독립성과 연관 강도", "borough x complaint_type",
            f"Cramér's V = {chi[chr(67) + 'ramér' + chr(39) + 's V']:.3f}", "자치구마다 민원 구성이 다르다. 8단계에서 이것이 교란 변수가 된다")
        cr = tb("stat4_correlation.csv").set_index("변수")
        t = cr.loc["일평균 기온(°C)"]
        add("s6", "상관계수", "함께 움직이는 정도. 인과가 아니다", "temp_mean_c ~ n_complaints",
            f"Pearson r = {t['Pearson r']:.2f}, Spearman ρ = {t['Spearman ρ']:.2f}", "약한 양의 상관. 9단계에서 자기상관을 고려하면 근거가 더 약해진다")
        ms = tb("stat4_model_summary.csv")
        add("s6", "회귀와 평가", "설명변수로 결과를 설명·예측한다", "log(resolution_hours), within_24h",
            f"선형 R² {ms['R² (평가, 9월)'].iloc[0]:.2f}, 로지스틱 AUC {ms['AUC (평가)'].iloc[1]:.3f} (9월 평가)",
            "8월까지로 배우고 9월로 평가했다. 미래 정보를 쓰지 않도록 시간 순으로 나눈 것이 핵심이다")

    # ---------------- 7 현대 방법
    def s7():
        q = tb("stat11_quantreg_summary.csv").set_index("변수")
        d = q.loc["agency_group=DPR"]
        add("s7", "분위수 회귀", "평균 대신 분위수를 설명한다", "log(resolution_hours) ~ 기관",
            f"공원국(DPR) 계수: τ=0.1 {d['τ=0.10 계수']:.2f} → τ=0.9 {d['τ=0.90 계수']:.2f}",
            "기관 차이는 빨리 끝나는 민원보다 오래 끄는 민원에서 훨씬 크다. OLS 의 계수 하나로는 보이지 않는다")
        mc = tb("stat11_model_compare.csv").set_index("모형")
        a = mc.loc[[i for i in mc.index if i.startswith("A.")][0], "AUC (평가, 9월)"]
        b = mc.loc[[i for i in mc.index if i.startswith("B.")][0], "AUC (평가, 9월)"]
        dd = mc.loc[[i for i in mc.index if i.startswith("D.")][0], "AUC (평가, 9월)"]
        add("s7", "그래디언트 부스팅", "앞 나무의 실수를 다음 나무가 고친다", "within_24h",
            f"AUC: 로지스틱 {a:.3f} → +민원 유형 {b:.3f} → 부스팅 {dd:.3f}",
            "성능 향상의 대부분은 모형이 아니라 변수(민원 유형)를 더한 데서 왔다. 좋은 변수가 좋은 알고리즘보다 먼저다")
        im = tb("stat11_permutation_importance.csv")
        add("s7", "순열 중요도", "변수를 뒤섞었을 때 성능이 얼마나 떨어지나", "within_24h 예측 모형",
            f"1위 {im['변수'].iloc[0]} (AUC -{im['AUC 하락 (평균)'].iloc[0]:.3f}), 2위 {im['변수'].iloc[1]} (-{im['AUC 하락 (평균)'].iloc[1]:.3f})",
            "기관은 민원 유형과 겹치는 정보라 과소평가된다. 중요도는 '예측에 쓸모'이지 '원인'이 아니다")

    # ---------------- 8 기관
    def s8():
        an = tb("stat7_anova.csv").set_index("변동 요인")
        add("s8", "이원배치 ANOVA", "요인 둘의 효과와 상호작용을 함께 본다", "log(resolution_hours) ~ 자치구 x 기관",
            f"편 η²: 기관 {an.loc['주효과: 기관', '편 η²']:.2f}, 자치구 {an.loc['주효과: 자치구', '편 η²']:.3f}, 상호작용 {an.loc['상호작용: 자치구 x 기관', '편 η²']:.3f}",
            "처리 시간은 거의 기관이 결정한다. 자치구 효과는 작고 기관마다 조금씩 다르다")
        v = tb("stat7_mixed_variance.csv")
        add("s8", "ICC", "분산 중 묶음 차이의 비율", "log(resolution_hours), 민원 유형",
            f"{v['ICC'].iloc[0]:.2f}", "처리 시간 변동의 대부분이 민원 유형 차이다. 유형을 무시한 비교는 위험하다")
        fx = tb("stat7_mixed_fixed.csv")
        flip = fx[(fx["A. OLS 계수 (유형 무시)"] * fx["B. 혼합모형 계수 (유형 통제)"]) < 0]
        add("s8", "심슨의 역설", "전체와 집단 안의 방향이 반대", "자치구 계수 (기준 브루클린)",
            f"부호가 바뀐 자치구 {len(flip)}개: {', '.join(flip['자치구'])}", "4단계 효율성 지수와 6단계 검정이 말한 '어느 자치구가 빠른가'는 민원 구성을 보정하면 답이 바뀐다")

    # ---------------- 9 시계열
    def s9():
        s = tb("stat6_stl_strength.csv").set_index("지표")["값"]
        add("s9", "STL 분해", "추세 + 계절성 + 나머지", "n_complaints",
            f"추세 세기 {s['추세 세기 F_T']:.2f}, 주간 계절성 세기 {s['계절성(주간) 세기 F_S']:.2f}", "둘 다 약하다. 특정한 날의 급증(폭풍, 행사)이 더 크다")
        ac = tb("stat6_acf_pacf.csv").set_index("시차(일)")
        add("s9", "자기상관", "어제·지난주 값과의 상관", "n_complaints",
            f"시차 1일 {ac.loc[1, 'ACF']:.2f}, 시차 7일 {ac.loc[7, 'ACF']:.2f}", "오늘 값이 독립이 아니므로 보통의 검정을 그대로 쓰면 p 값이 너무 작게 나온다")
        te = tb("stat6_temperature_effect.csv")
        add("s9", "자기상관을 무시하면", "표준오차가 과소평가된다", "temp_mean_c -> n_complaints",
            f"기온 계수의 표준오차 {te['표준오차'].iloc[0]:.1f} → {te['표준오차'].iloc[1]:.1f}", "자기상관을 허용하면 오차가 두 배가 되고 p 값이 0.05 경계로 올라간다")
        fm = tb("stat6_forecast_metrics.csv")
        add("s9", "예측과 기준 모형", "단순한 기준을 이겨야 쓸모가 있다", "n_complaints (9월)",
            f"SARIMA MAE 기준 대비 {fm['기준 대비 MAE 변화(%)'].iloc[-1]:+.1f}%", "복잡한 모형이 '지난주 같은 요일' 예측보다 조금 나을 뿐이다")

    # ---------------- 10 생존
    def s10():
        lg = tb("stat5_data_log.csv").set_index("항목")["값"]
        add("s10", "중도절단", "관측이 끝날 때까지 사건이 없음", "closed_date",
            f"{float(lg['중도절단 비율(%)']):.1f}%", "이들을 버리면 오래 걸리는 민원이 빠져 처리 시간이 짧게 보인다")
        nk = tb("stat5_naive_vs_km.csv").set_index("지표")
        add("s10", "Kaplan-Meier", "중도절단을 살려 쓴 생존곡선", "resolution_hours",
            f"중앙값: 종결 건만 {nk.loc['처리시간 중앙값(h)', '종결 건만 사용']:.1f}h vs KM {nk.loc['처리시간 중앙값(h)', 'Kaplan-Meier (미종결 포함)']:.1f}h",
            "미종결을 버리면 중앙값이 약 2시간 짧게 나온다")
        km = tb("stat5_km_summary.csv").set_index("집단")
        add("s10", "집단별 생존곡선", "곡선이 빨리 내려갈수록 빨리 처리", "resolution_hours ~ agency",
            f"KM 중앙값: NYPD {km.loc['기관: NYPD', 'KM 중앙값(h)']:.1f}h, DPR {km.loc['기관: DPR', 'KM 중앙값(h)']:,.0f}h",
            "공원국(DPR) 민원은 절반이 두 달 가까이 열려 있다")
        cx = tb("stat5_cox_summary.csv").iloc[0]
        add("s10", "Cox 모형과 C-index", "위험비로 비교하고 순서 맞히기로 평가", "resolution_hours, event",
            f"C-index {cx['C-index (평가, 9월 접수)']:.3f} (9월 평가)", "어느 민원이 먼저 끝날지 77% 의 쌍에서 맞힌다")

    # ---------------- 11 공간
    def s11():
        k = tb("stat8_kde_concentration.csv").set_index("분류")
        add("s11", "KDE 와 집중도", "밀도 지도와 지니계수", "latitude, longitude",
            f"지니: 거리·상업 소음 {k.loc['거리·상업 소음', '지니계수']:.2f}, 주차·차량 {k.loc['주차·차량', '지니계수']:.2f}",
            "소음은 좁은 곳에 몰리고 주차는 도시 전체에 퍼진다")
        c = tb("stat8_cluster_method_compare.csv")
        db = c[c["방법"].str.startswith("DBSCAN")].iloc[0]
        add("s11", "DBSCAN", "밀집 덩어리 + 잡음", "상업시설 소음 신고 위치",
            f"군집 {db['군집 수']:.0f}개, 잡음 {db['잡음 위치 비율(%)']:.0f}%", "상업 소음 신고의 절반 이상은 유흥가가 아닌 흩어진 업소에서 나온다")
        mg = tb("stat8_moran_global.csv").set_index("변수")
        add("s11", "Moran's I", "가까운 곳끼리 비슷한가", "1km 격자",
            f"유형 보정 처리시간 I = {mg.loc['유형 보정 로그 처리시간', chr(77) + 'oran' + chr(39) + 's I']:.2f}",
            "민원 유형을 보정해도 '빨리 처리되는 동네'와 '늦게 처리되는 동네'가 덩어리를 이룬다")

    # ---------------- 12 군집
    def s12():
        pv = tb("stat9_pca_variance.csv")
        add("s12", "PCA 설명 분산", "새 축이 담은 정보의 비율", "격자 칸의 변수 11개",
            f"PC1 {pv['설명 분산 비율(%)'].iloc[0]:.0f}%, PC1+PC2 {pv['누적(%)'].iloc[1]:.0f}%", "두 축으로 절반도 요약되지 않는다. 동네의 민원 구성은 여러 방향으로 다르다")
        ks = tb("stat9_k_selection.csv")
        sel = ks[ks["선택"] == "선택"].iloc[0]
        add("s12", "실루엣", "군집이 얼마나 뚜렷이 갈렸나", "K-means",
            f"k = {sel['k']:.0f}, 실루엣 {sel['실루엣']:.2f}", "낮다. 동네들은 뚜렷한 무리가 아니라 연속적인 스펙트럼에 가깝다")
        mc = tb("stat9_method_compare.csv")
        ari = "기준 K-means 와의 일치도(ARI)"
        add("s12", "ARI 로 방법 비교", "다른 방법도 같은 군집을 내는가", "K-means vs PCA, Ward",
            f"PCA 후 {mc[mc['방법'].str.startswith('PCA')][ari].iloc[0]:.2f}, Ward {mc[mc['방법'].str.startswith('Ward')][ari].iloc[0]:.2f}",
            "큰 줄기는 같지만 경계의 칸들은 방법에 따라 다른 군집으로 간다")
        co = tb("stat9_spatial_coherence.csv").iloc[0]
        add("s12", "위치 없이 묶어도 붙어 있다", "닮은 동네는 이웃해 있는가", "군집 결과 + 격자 인접",
            f"같은 군집인 이웃 쌍 {co['같은 군집인 쌍 비율(%)']:.0f}% (무작위 {co['무작위일 때 기대 비율(%)']:.0f}%)", "11단계 Moran's I 와 같은 이야기: 동네의 성격은 지리적으로 이어진다")

    for fn in (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12):
        safe(fn)
    return pd.DataFrame(rows)


def main():
    lp = pd.DataFrame(LEARNING_PATH, columns=["key", "단계", "파트", "제목", "핵심 질문", "배우는 개념", "쓰는 열", "먼저 볼 단계", "만든 스크립트"])
    save_table(lp, "stat12_learning_path.csv")
    cm = build_concept_map()
    cm.insert(1, "단계", cm["key"].map(lp.set_index("key")["단계"]))
    save_table(cm, "stat12_concept_map.csv")
    gl = pd.DataFrame(GLOSSARY, columns=["용어", "영어", "처음 나오는 단계", "쉬운 설명"])
    save_table(gl, "stat12_glossary.csv")
    print(cm.groupby("단계").size().rename("개념 수").to_string())
    print(f"용어 {len(gl)}개")
    if _missing:
        print(f"없는 표: {sorted(set(_missing))} -> 해당 스크립트를 먼저 실행하세요 (폴더: {TABLES.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
