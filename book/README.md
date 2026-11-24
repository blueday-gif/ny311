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
| `src/numbers_registry.py` | 세 결과물의 핵심 숫자를 기준별로 다시 계산 → `numbers.csv` |
| `src/eda_stages.py`, `src/stages_09_14.py` | 초기 EDA·모델 실험 (원본 CSV → `requests_slim.parquet`) |
| `notes/` | 숫자 검토 메모 |

데이터는 git에 올리지 않고 `data/interim/nyc311_book/`에 둡니다.

## 실행

```powershell
.venv\Scripts\python.exe book\src\story_stats.py
.venv\Scripts\python.exe book\src\_new_questions.py
.venv\Scripts\python.exe book\src\_politics_check.py
.venv\Scripts\python.exe book\src\build_book.py
.venv\Scripts\python.exe book\src\numbers_registry.py
```

PDF는 Edge(크로미움)로 HTML을 인쇄해 만듭니다. 원래 PDF와 같은 엔진(Skia/PDF)이고 쪽수(37쪽)도 같습니다.

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="D:\ny311\book\reports\nyc311_question_book.pdf" "file:///D:/ny311/book/reports/nyc311_question_book.html"
```

필요한 패키지: pandas, pyarrow, duckdb. `eda_stages.py`와 `stages_09_14.py`는 matplotlib, scipy, scikit-learn, lightgbm도 필요합니다.

옮긴 뒤 `build_book.py`로 다시 만든 HTML은 이전 것과 바이트 단위로 같았습니다.
통계 JSON 세 개도 숫자는 같았습니다. 행 순서와 소수점 15번째 자리만 달랐는데, SQL에 정렬 기준이 없어서 생기는 차이입니다.

## 숫자 기준 (`numbers.csv`)

같은 질문인데 결과물마다 숫자가 달랐던 이유는 계산이 틀려서가 아니라 기준이 달라서였습니다.

| 기준 | 건수 | 쓰는 곳 |
| --- | --- | --- |
| 전체 | 3,963,959 | 신고는 일기다 |
| 중복 제거 | 3,962,167 | 질문 100개. 고유번호만 다르고 나머지 43개 열이 모두 같은 1,792줄을 지움 |
| 한 주소 제외 | 3,915,062 | 질문 100개의 소음·경찰·경로 질문, 웹북 경찰 장. 중복 제거 뒤 브롱크스 북부 한 주소 47,105건을 뺌 |

| 숫자 | 전체 | 한 주소 제외 | 왜 다른가 |
| --- | --- | --- | --- |
| 딱지(“issued a summons”) | 13.1% (224,738 / 1,720,638) | 13.4% (224,728 / 1,673,589) | 그 주소의 경찰 결과 4.7만 건이 분모에서 빠짐 |
| 체포(“made an arrest”) | 1,169건 | 1,163건 | 그 주소의 체포 6건 |
| 이웃 소음 진입 불가 | 11.4% | 12.7% (질문 100개는 13%로 반올림) | 같음 |

| 숫자 | 쓰는 곳 | 정의 |
| --- | --- | --- |
| 자정 마감 110,698건 | 신고는 일기다, 웹북 | 종료 시각 00:00:00, 상태 Closed, 종료 ≥ 접수 (처리 시간 계산 대상 안에서) |
| 자정 마감 11.9만 건 | 질문 100개 | 종료 시각 00:00:00인 모든 기록 (118,903건, 중복 제거 기준) |
| 앱 비중 불법주차 46%, 구걸 91% | 웹북 | 해당 민원 중 접수 경로 MOBILE. 출처가 없던 숫자였는데 계산으로 확인됨 |

이번에 고친 곳:
- 웹북 경찰 장의 딱지 문장이 전체 기준(13.1%)이었는데, 같은 문단의 체포 1,163건과 진입 불가 12.7%, 첫 화면의 13.4%는 한 주소 제외 기준이었습니다. 문장을 한 주소 제외 기준(13.4%, 224,728 / 1,673,589)으로 바꿔 한 기준으로 맞췄습니다.
- 신고는 일기다의 ‘백 개 질문집에 대해’ 쪽에 두 책 건수와 체포 수가 왜 다른지 정확한 이유를 적고 PDF를 다시 만들었습니다.
- 질문 100개 PDF는 만든 코드가 없어 고치지 않았습니다. 그 책의 숫자는 위 기준으로 모두 재현됩니다.

## 아직 없는 것

- `requests_story.parquet`를 만든 코드. 파일만 있습니다.
- `weather_nyc.json`(Open-Meteo 일별 기상)을 받은 코드. 파일만 있습니다.
- 질문 100개 책의 계산 코드. 기준 건수 차이는 `numbers.csv`로 설명되지만, 다른 질문의 숫자를 다시 만들려면 이 코드가 필요합니다.
