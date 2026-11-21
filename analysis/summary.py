"""
[요약] 발견 5가지, 한계, 대시보드 연결표

핵심 질문 : 지금까지 15단계에서 무엇을 알게 됐고, 무엇은 아직 모르나?
배우는 개념: 결과 요약, 한계 기록(표본 편향, 신고 편향, 관측 기간), 결과와 화면 연결
사용하는 열: 없음 (outputs/tables/ 의 결과 표를 읽는다)
실행 방법 : py -3.12 analysis/summary.py   (eda_01 ~ ml_15 를 먼저 모두 실행)

숫자를 코드에 직접 적지 않고 결과 표에서 읽어 온다. 데이터가 바뀌어 다시 실행해도 요약이 자동으로 맞춰진다.
"""

# 공통 도구
from common import TAB_DIR, pd, print_header, save_table


def read(name: str) -> pd.DataFrame:
    """outputs/tables/{name}.csv 를 읽는다."""
    # 엑셀 호환 인코딩으로 저장했으므로 같은 인코딩으로 읽는다
    return pd.read_csv(TAB_DIR / f"{name}.csv", encoding="utf-8-sig")


# ─────────────────────────────────────────────
# 1. 결과 표 읽기
# ─────────────────────────────────────────────

# EDA 결과
overview = read("eda01_overview").set_index("항목")["값"]

# 처리 시간 요약
hours = read("eda02_hours_summary").set_index("통계량")["값"]

# 왜도
skew = read("eda03_skewness").set_index("값")["왜도"]

# 민원 유형 누적 비율
problem = read("eda04_problem_counts")

# 시간대별 건수
by_hour = read("eda04_by_hour").set_index("hour")["건수"]

# 기준선
base = read("ml06_baselines").set_index("방법")["평가 정확도"]

# 모델 비교
comp = read("ml09_model_comparison")

# 순열 중요도
imp = read("ml11_permutation_importance")

# 예측 결과
fc = read("ml14_forecast_results").set_index("방법")

# 군집 K 선택
ks = read("ml15_k_selection")

# ─────────────────────────────────────────────
# 2. 발견 5가지
# ─────────────────────────────────────────────
print_header("발견 5가지")

# 가장 좋은 모델 (평가 정확도 기준, 기준선 제외)
models = comp.dropna(subset=["평가 AUC"])

# 다수결 기준선 행은 뺀다
models = models[~models["모델"].str.contains("기준선")]

# 1등 모델
best = models.loc[models["평가 정확도"].idxmax()]

# 민원 유형 규칙 정확도
rule_acc = base[[i for i in base.index if "민원 유형" in i][0]]

# 다수결 정확도
dummy_acc = base[[i for i in base.index if "다수결" in i][0]]

# 시계열: 계절 단순 예측과 가장 좋은 모델 MAPE
mape = fc["MAPE(%)"]

# 가장 낮은 MAPE 방법
best_fc = mape.idxmin()

# 발견 목록
findings = [
    f"처리 시간은 크게 치우쳐 있다: 중앙값 {hours['중앙값']:.1f}시간, 평균 {hours['평균']:.1f}시간(약 {hours['평균 / 중앙값 (배)']:.0f}배), "
    f"왜도 {skew.iloc[0]:.1f} → log1p 후 {skew.iloc[1]:.2f}. 요약은 중앙값, 모델링은 로그 변환으로 한다.",
    f"민원은 소수 유형에 몰린다: 197개 유형 중 상위 15개가 전체의 {problem['누적 비율(%)'].iloc[14]:.1f}%.",
    f"하루 주기가 뚜렷하다: {by_hour.idxmax()}시에 가장 많고({by_hour.max():,}건) {by_hour.idxmin()}시에 가장 적다({by_hour.min():,}건). 주말은 평일보다 적다.",
    f"24시간 안 처리 여부는 '민원 유형'이 대부분 결정한다: 다수결 {dummy_acc:.3f} → 민원 유형 규칙 {rule_acc:.3f} → "
    f"최고 모델({best['모델']}) {best['평가 정확도']:.3f}, AUC {best['평가 AUC']:.3f}. "
    f"순열 중요도 1위는 '{imp.iloc[0]['열']}'.",
    f"하루 접수 건수 예측은 '{best_fc}'가 가장 정확했다(MAPE {mape.min():.1f}%). "
    f"단순한 기준선과 복잡한 모델의 차이를 반드시 함께 보고해야 한다.",
]

# 번호를 붙여 출력
for i, f in enumerate(findings, start=1):
    # 한 줄씩
    print(f"{i}. {f}")

# 저장
save_table(pd.DataFrame({"번호": range(1, 6), "발견": findings}), "summary_findings")

# ─────────────────────────────────────────────
# 3. 한계
# ─────────────────────────────────────────────
print_header("한계")

# 한계 목록
limits = [
    ("관측 기간", f"{overview['접수 시작'][:10]} ~ {overview['접수 끝'][:10]} 약 1년뿐이다. 연도별 추세, 같은 달의 해마다 차이는 알 수 없다. "
             "파일 이름의 '2020년부터'와 실제 기간이 다르다."),
    ("신고 편향", "건수는 '발생한 문제의 양'이 아니라 '신고된 양'이다. 311 을 잘 아는 주민, 온라인 접근성이 좋은 지역에서 더 많이 신고된다."),
    ("표본 편향", "머신러닝은 학습 20만·평가 10만 건 표본으로 했다(seed 고정). 드문 민원 유형은 표본에 몇 건 없어 결과가 불안정하다. "
             "자치구 비교는 인구 자료가 없어 1인당 비율로 바꾸지 못했다."),
    ("미종결 민원", f"처리 시간 분석에서 미종결 민원(약 4%)을 뺐다. 오래 걸리는 민원이 덜 들어가 처리 시간을 짧게 볼 수 있다(중도절단). "
               "제대로 다루려면 생존분석(기존 앱 10단계)이 필요하다."),
    ("인과 아님", "상관·중요도·오즈비는 모두 '연관'이다. 장소 유형이 중요하게 나온 것은 담당 기관을 대신 알려 주기 때문이지, 장소가 처리 속도를 바꾸는 원인이라는 뜻이 아니다."),
    ("데이터 품질", "장소 유형에 같은 뜻의 다른 표기(RESIDENTIAL BUILDING / Residential Building)가 섞여 있다. 같은 시각·유형·위치 중복 신고 후보가 "
               f"{int(overview['같은 시각·유형·위치 중복']):,}건 있다(지우지 않음)."),
]

# 출력
for name, text in limits:
    # 한 줄씩
    print(f"- [{name}] {text}")

# 저장
save_table(pd.DataFrame(limits, columns=["한계", "내용"]), "summary_limitations")

# ─────────────────────────────────────────────
# 4. 대시보드 연결표
# ─────────────────────────────────────────────
print_header("스크립트 ↔ 대시보드 탭 연결표")

# 각 스크립트가 기존 앱의 어느 탭과 이어지는지, 새 대시보드에서 어느 탭이 될지 정리한다
mapping = pd.DataFrame([
    ("eda_01_check.py", "데이터 점검", "0 시작하기", "1부-① 데이터 점검", "eda01_overview, eda01_missing"),
    ("eda_02_summary.py", "기술통계", "2 대푯값, 3 퍼짐·기술통계", "1부-② 숫자로 요약", "eda02_hours_summary"),
    ("eda_03_distribution.py", "분포·이상치", "4 분포·변환·표준화", "1부-③ 분포와 이상치", "eda03_skewness, eda03_outliers_iqr"),
    ("eda_04_patterns.py", "범주·시간·관계·집단 비교", "1 둘러보기, 6 검정·회귀, 8 기관 비교, 9 시계열", "1부-④ 패턴과 비교", "eda04_*"),
    ("ml_05_problem.py", "문제 정의·누수", "6 검정·회귀(④)", "2부-① 무엇을 예측하나", "ml05_split, ml05_leakage_demo"),
    ("ml_06_split_baseline.py", "기준선", "(없음, 새 내용)", "2부-② 기준선", "ml06_baselines, ml06_monthly_rate"),
    ("ml_07_logistic.py", "로지스틱 회귀", "6 검정·회귀", "2부-③ 첫 모델", "ml07_accuracy, ml07_odds_ratio"),
    ("ml_08_tree.py", "결정나무·과적합", "(없음, 새 내용)", "2부-④ 나무와 과적합", "ml08_depth_curve"),
    ("ml_09_ensemble.py", "앙상블·모델 비교", "7 현대 방법", "2부-⑤ 모델 비교", "ml09_model_comparison"),
    ("ml_10_evaluation.py", "평가 지표", "7 현대 방법(AUC, 보정)", "2부-⑥ 평가 지표", "ml10_threshold_table"),
    ("ml_11_interpret.py", "해석", "7 현대 방법(순열 중요도)", "2부-⑦ 모델 해석", "ml11_permutation_importance"),
    ("ml_12_tuning.py", "교차검증·튜닝", "(없음, 새 내용)", "2부-⑧ 튜닝", "ml12_grid_results, ml12_final_test"),
    ("ml_13_regression.py", "회귀", "6 검정·회귀, 10 생존분석", "3부-① 처리 시간 예측", "ml13_regression_results"),
    ("ml_14_forecast.py", "시계열 예측", "9 시계열", "3부-② 내일 건수 예측", "ml14_forecast_results"),
    ("ml_15_clustering.py", "군집", "12 군집·PCA", "3부-③ 동네 유형", "ml15_k_selection, ml15_cluster_profile"),
    ("summary.py", "요약·한계", "(각 탭 꼬리말 정리표)", "마무리", "summary_findings, summary_limitations"),
], columns=["스크립트", "주제", "기존 앱 탭", "새 대시보드 탭(제안)", "대시보드가 읽을 표"])

# 출력
print(mapping.to_string(index=False))

# 저장
save_table(mapping, "summary_dashboard_mapping")

# 군집 K 출력 (참고)
print(f"\n참고: 군집 실루엣 최고 K = {int(ks.loc[ks['실루엣 점수'].idxmax(), 'K'])}")
