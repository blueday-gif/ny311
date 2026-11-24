"""책 작업 스크립트가 함께 쓰는 경로. 저장소 어디서 실행해도 같은 경로가 된다."""

from pathlib import Path

# book/src/ 의 부모의 부모 = 저장소 루트
ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "raw" / "311_Service_Requests_from_2020_to_Present_20261112.csv"

# 책 작업의 중간 결과 (git이 무시하는 data/ 아래)
INTERIM = ROOT / "data" / "interim" / "nyc311_book"
STORY_PQ = INTERIM / "requests_story.parquet"
SLIM_PQ = INTERIM / "requests_slim.parquet"
WEATHER = INTERIM / "weather_nyc.json"

# 결과물 (커밋하는 쪽)
REPORTS = ROOT / "book" / "reports"
FIGURES = REPORTS / "figures"

# DuckDB SQL 안에 넣을 경로 문자열 (역슬래시 대신 /)
STORY_PQ_SQL = STORY_PQ.as_posix()
