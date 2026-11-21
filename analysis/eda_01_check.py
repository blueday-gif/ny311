"""
[EDA 1단계] 데이터 점검

핵심 질문 : 이 표에는 무엇이 들어 있고, 믿고 써도 되는 상태인가?
배우는 개념: 행과 열, 자료형, 결측치(결측률), 관측 기간, 중복 행
사용하는 열: 44개 열 전체(결측률), created(기간), unique_key·created·problem·lat·lon(중복)
실행 방법 : py -3.12 analysis/eda_01_check.py
기존 앱 참고: 0 시작하기 탭

[용어] 결측치(missing value)
  값이 비어 있는 칸. "왜 비었는지"가 중요하다.
  예) closed(종결 시각)가 비어 있으면 '아직 처리 중'이라는 뜻일 수 있다. 실수로 빠진 값과는 다르다.

[용어] 자료형(data type)
  열에 들어 있는 값의 종류. 숫자(int, float), 글자(string), 날짜(timestamp) 등이 있다.
  예) unique_key 는 숫자형이지만 민원 번호(이름표)라서 평균을 내면 뜻이 없다.
"""

# pyarrow.parquet: Parquet 파일의 정보(행 수, 열 목록)를 데이터 전체를 읽지 않고 확인할 때 쓴다
import pyarrow.parquet as pq

# common.py 에 모아 둔 공통 도구를 가져온다
from common import DATA_PATH, ExclusionLog, load_data, pd, plt, print_header, save_fig, save_table, setup_plot

# 그래프 한글 글꼴을 설정한다
setup_plot()

# ─────────────────────────────────────────────
# 1. 행/열 수와 자료형
# ─────────────────────────────────────────────
print_header("1. 행/열 수와 자료형")

# ParquetFile: 파일을 열기만 하고 내용은 아직 읽지 않는다. 메타데이터(요약 정보)만 빠르게 본다
pf = pq.ParquetFile(DATA_PATH)

# 전체 행 수. 메타데이터에 이미 적혀 있어서 바로 알 수 있다
n_rows = pf.metadata.num_rows

# 열 이름 목록
col_names = pf.schema_arrow.names

# 화면에 출력한다. :, 는 천 단위 쉼표를 붙여 읽기 쉽게 한다
print(f"행 수: {n_rows:,} / 열 수: {len(col_names)}")
# 해석: 한 행 = 민원 1건이다. 약 396만 건, 44개 항목이 기록되어 있다.

# 열마다 자료형을 표로 만든다. 자료형을 알아야 어떤 계산이 가능한지 판단할 수 있다
dtypes = pd.DataFrame({
    "열": col_names,
    "자료형": [str(t) for t in pf.schema_arrow.types],
})

# 자료형별로 열이 몇 개인지 센다
print(dtypes["자료형"].value_counts().to_string())
# 해석: 대부분 string(글자)이다. 숫자로 계산할 수 있는 열은 날짜 2개와 위도·경도 정도뿐이다.

# ─────────────────────────────────────────────
# 2. 결측률: 열마다 비어 있는 칸의 비율
# ─────────────────────────────────────────────
print_header("2. 결측률")

# 열 전체를 Arrow 형식으로 읽는다. pandas로 바꾸지 않아서 메모리를 덜 쓴다
table = pf.read()

# 열마다 결측 건수를 센다. null_count 는 Arrow가 미리 계산해 둔 값이라 빠르다
missing = pd.DataFrame({
    "열": col_names,
    "결측 건수": [table[c].null_count for c in col_names],
})

# 결측률(%) = 결측 건수 / 전체 행 수 × 100
missing["결측률(%)"] = (missing["결측 건수"] / n_rows * 100).round(2)

# 결측률이 높은 순서로 정렬한다. 문제가 큰 열을 먼저 보기 위해서다
missing = missing.sort_values("결측률(%)", ascending=False)

# 자료형 정보도 붙인다
missing = missing.merge(dtypes, on="열")

# 표를 저장한다
save_table(missing, "eda01_missing")

# 결측률 구간별로 열이 몇 개인지 요약한다. pd.cut 은 숫자를 구간으로 나누는 함수
bins = pd.cut(missing["결측률(%)"], [-0.1, 0, 5, 50, 90, 100],
              labels=["0%", "0~5%", "5~50%", "50~90%", "90% 초과"])

# 구간별 개수를 출력한다
print(bins.value_counts().sort_index().to_string())
# 해석: 90% 넘게 비어 있는 열(택시·다리·도로 관련)은 특정 민원에만 쓰는 칸이라 일반 분석에서 뺀다.

# 결측률 막대그래프. 상위 25개 열만 그린다 (44개 전부는 글자가 겹쳐 읽기 어렵다)
fig, ax = plt.subplots(figsize=(8, 8))

# 그릴 데이터: 결측률이 높은 25개 열
top = missing.head(25).iloc[::-1]

# 가로 막대그래프. 열 이름이 길어서 세로 막대보다 가로가 읽기 쉽다
ax.barh(top["열"], top["결측률(%)"], color="#4C72B0")

# x축 이름
ax.set_xlabel("결측률 (%)")

# 그림 제목
ax.set_title("열별 결측률 (상위 25개)")

# 그림 저장
save_fig(fig, "eda01_missing_bar")

# ─────────────────────────────────────────────
# 3. 관측 기간
# ─────────────────────────────────────────────
print_header("3. 관측 기간")

# 분석용 열만 불러온다 (common.load_data 가 파생 열 hours, hour 등을 만들어 준다)
df = load_data()

# 가장 이른 접수 시각
start = df["created"].min()

# 가장 늦은 접수 시각
end = df["created"].max()

# 기간을 출력한다
print(f"접수 기간: {start} ~ {end}  ({(end - start).days}일)")
# 해석: 파일 이름은 '2020년부터'지만 실제로는 최근 약 1년(2025-11-01 ~ 2026-10-31)만 들어 있다. 연도별 추세는 볼 수 없다.

# 첫날과 마지막 날은 하루가 다 차지 않았는지 확인한다 (시작이 21시이면 그날은 3시간만 들어 있다)
print(f"첫날 시작 시각: {start.time()} / 마지막 날 끝 시각: {end.time()}")
# 해석: 첫날·마지막 날은 '부분 일자'다. 일별 건수를 셀 때 이 두 날을 빼야 평균이 왜곡되지 않는다.

# ─────────────────────────────────────────────
# 4. 중복 행
# ─────────────────────────────────────────────
print_header("4. 중복 행")

# unique_key(민원 번호)가 겹치는 행 수. 0이어야 정상이다
dup_key = df["unique_key"].duplicated().sum()

# 출력
print(f"민원 번호 중복: {dup_key:,}건")
# 해석: 0건이면 같은 민원이 두 번 저장되지는 않았다는 뜻이다.

# "같은 시각·같은 유형·같은 위치"에 들어온 민원. 번호는 다르지만 같은 일을 여러 사람이 신고했을 수 있다
same_event = df.duplicated(subset=["created", "problem", "lat", "lon"], keep="first")

# 그런 행이 몇 건인지 센다
n_same = int(same_event.sum())

# 비율과 함께 출력한다
print(f"같은 시각·유형·위치 중복: {n_same:,}건 ({n_same / len(df) * 100:.2f}%)")
# 해석: 이 행들은 '오류'가 아니라 '여러 명의 신고'일 수 있다. 지우지 않고 개수만 기록한다.

# 제외 로그: 이 단계는 아무것도 지우지 않았다는 사실을 기록한다
log = ExclusionLog("eda01", len(df))

# 중복 후보는 '표시만 하고 유지'했다고 남긴다
log.add("같은 접수 시각·민원 유형·위도·경도 (중복 신고 후보)", n_same, "유지(기록만)")

# 로그 저장
log.save()

# ─────────────────────────────────────────────
# 5. 요약 표 저장
# ─────────────────────────────────────────────

# 이 단계의 핵심 숫자를 한 표로 모은다. 대시보드가 이 표를 읽어서 보여 줄 수 있다
overview = pd.DataFrame([
    {"항목": "행 수", "값": n_rows},
    {"항목": "열 수", "값": len(col_names)},
    {"항목": "접수 시작", "값": str(start)},
    {"항목": "접수 끝", "값": str(end)},
    {"항목": "기간(일)", "값": (end - start).days},
    {"항목": "민원 번호 중복", "값": int(dup_key)},
    {"항목": "같은 시각·유형·위치 중복", "값": n_same},
    {"항목": "결측률 90% 초과 열 수", "값": int((missing["결측률(%)"] > 90).sum())},
])

# 저장
save_table(overview, "eda01_overview")
