# project-ny311

뉴욕시 311 민원 약 396만 건(2025-11-01 ~ 2026-10-31, 전체 민원 유형)으로
데이터 분석(EDA)과 머신러닝을 처음부터 차례대로 익히는 학습용 프로젝트입니다.

- 분석 스크립트 16개: 한 줄씩 한국어 주석, 통계 용어 설명, 결과 해석 주석 포함
- Streamlit 대시보드 17페이지: 분석 결과(outputs/)를 읽어 단계별로 보여 줌

## 폴더 구조

```
project-ny311/
├── data/                       # git 에 올리지 않음 (.gitignore)
│   ├── raw/                    # 원본 zip (읽기만 함)
│   └── processed/              # Parquet 변환본, 머신러닝용 데이터
├── scripts/
│   └── convert_csv_to_parquet.py   # 원본 zip → Parquet
├── analysis/                   # 분석 스크립트 (번호 순서대로 실행)
│   ├── common.py               # 공통 도구 (경로, seed, 저장, 제외 로그)
│   ├── eda_01 ~ eda_04         # 1부 데이터 알기
│   ├── ml_05 ~ ml_12           # 2부 머신러닝 기본기
│   ├── ml_13 ~ ml_15           # 3부 응용
│   ├── summary.py              # 발견 5가지, 한계, 대시보드 연결표
│   └── run_all.py              # 전체 실행
├── outputs/
│   ├── figures/                # 그래프 (PNG)
│   └── tables/                 # 요약 표·제외 기준 로그 (CSV)
└── dashboard/                  # Streamlit 앱
    ├── app.py                  # 진입점 (메뉴)
    ├── dash_common.py
    └── app_pages/              # 페이지 17개
```

## 실행

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install scipy scikit-learn matplotlib pyarrow
py -3.12 scripts/convert_csv_to_parquet.py
py -3.12 analysis/run_all.py
py -3.12 -m streamlit run dashboard/app.py
```

- `run_all.py` 는 약 6분 걸립니다. seed 가 고정되어 있어 다시 실행해도 같은 결과가 나옵니다.
- 대시보드는 원본 데이터 없이 `outputs/` 만 있으면 실행됩니다.

## 배포 (Streamlit Community Cloud)

- Main file path: `dashboard/app.py` (기존 배포 설정 그대로)
- 패키지는 루트의 `requirements.txt` 가 쓰입니다. 대시보드에 필요한 것만 설치되고, 분석용 패키지는 주석으로 적어 두었습니다.
- `main` 브랜치에 푸시하면 앱이 자동으로 다시 배포됩니다.

## 작업 규칙

- `data/raw/` 원본은 읽기만 하고 수정하지 않습니다.
- 그래프는 `outputs/figures/`, 표는 `outputs/tables/` 에 저장합니다.
- 제외하거나 대체한 데이터는 기준과 건수를 `*_exclusion_log.csv` 로 남깁니다.
- 식별자(unique_key, 우편번호)는 숫자 분석 변수로 쓰지 않습니다.
