# NYC 311 책 작업

같은 311 데이터(2025-11-01 ~ 2026-10-31)로 만든 읽을거리 세 가지와, 그 숫자를 계산한 코드입니다.
저장소의 EDA·머신러닝 강좌(`analysis/`, `dashboard/`)와는 따로 돌아갑니다.

| 결과물 | 파일 | 만든 코드 |
| --- | --- | --- |
| 뉴욕 311, 질문 100개 (70쪽) | `reports/nyc311_hundred_questions.pdf` | **없음** (코드가 이 저장소에 없음) |
| 신고는 일기다 (37쪽, 27문항) | `reports/nyc311_question_book.html`, `.pdf` | `src/build_book.py` |
| 7.9초마다 기록하는 도시의 일기 (웹북) | `reports/7_9_311.html` | 없음 (손으로 만든 HTML) |

## 폴더

| 경로 | 내용 |
| --- | --- |
| `src/paths.py` | 모든 스크립트가 함께 쓰는 경로 |
| `src/story_stats.py` | 신고는 일기다의 기본 통계 → `story_stats.json` |
| `src/_new_questions.py` | 새 질문(7월 4일, 추수감사절, 한파 등) → `book_new_stats.json` |
| `src/_politics_check.py` | 선거·분모 장 → `book_politics_stats.json` |
| `src/_season_zips.py` | 계절·우편번호 확인용 (출력만 함) |
| `src/build_book.py` | 위 JSON으로 `reports/nyc311_question_book.html` 생성 |
| `src/eda_stages.py`, `src/stages_09_14.py` | 초기 EDA·모델 실험 (원본 CSV → `requests_slim.parquet`) |
| `notes/` | 숫자 검토 메모 |

데이터는 git에 올리지 않고 `data/interim/nyc311_book/`에 둡니다.

## 실행

```powershell
.venv\Scripts\python.exe book\src\story_stats.py
.venv\Scripts\python.exe book\src\_new_questions.py
.venv\Scripts\python.exe book\src\_politics_check.py
.venv\Scripts\python.exe book\src\build_book.py
```

필요한 패키지: pandas, pyarrow, duckdb. `eda_stages.py`와 `stages_09_14.py`는 matplotlib, scipy, scikit-learn, lightgbm도 필요합니다.

옮긴 뒤 `build_book.py`로 다시 만든 HTML은 이전 것과 바이트 단위로 같았습니다.
통계 JSON 세 개도 숫자는 같았습니다. 행 순서와 소수점 15번째 자리만 달랐는데, SQL에 정렬 기준이 없어서 생기는 차이입니다.

## 아직 없는 것

- `requests_story.parquet`를 만든 코드. 파일만 있습니다.
- `weather_nyc.json`(Open-Meteo 일별 기상)을 받은 코드. 파일만 있습니다.
- HTML을 PDF로 바꾼 방법.
- 질문 100개 책의 계산 코드. 이 책의 기준 건수(중복 제거 3,962,167건)가 신고는 일기다(3,963,959건)와 다른 이유도 이 코드가 있어야 확인할 수 있습니다.
