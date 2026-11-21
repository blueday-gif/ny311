"""
전체 분석을 번호 순서대로 한 번에 실행한다.

실행 방법: py -3.12 analysis/run_all.py
  - 하나라도 실패하면 거기서 멈추고 어느 단계에서 실패했는지 알려 준다.
  - 처음부터 끝까지 다시 실행해도 같은 결과가 나온다 (seed 고정, 결과 파일은 덮어쓴다).
  - 전체 실행에 약 10~20분 걸린다 (컴퓨터 성능에 따라 다름).
"""

# subprocess: 다른 파이썬 파일을 별도 프로그램처럼 실행하는 도구
import subprocess

# sys.executable: 지금 이 파일을 실행 중인 파이썬(py -3.12)의 경로. 같은 파이썬으로 모든 단계를 돌리기 위해 쓴다
import sys

# 시간 측정
import time

# 경로
from pathlib import Path

# 이 파일이 있는 폴더 (analysis/)
HERE = Path(__file__).resolve().parent

# 실행 순서. ml_05 가 머신러닝용 데이터를 만들기 때문에 ml_06 ~ ml_13 보다 먼저 와야 한다
STEPS = [
    "eda_01_check.py",
    "eda_02_summary.py",
    "eda_03_distribution.py",
    "eda_04_patterns.py",
    "ml_05_problem.py",
    "ml_06_split_baseline.py",
    "ml_07_logistic.py",
    "ml_08_tree.py",
    "ml_09_ensemble.py",
    "ml_10_evaluation.py",
    "ml_11_interpret.py",
    "ml_12_tuning.py",
    "ml_13_regression.py",
    "ml_14_forecast.py",
    "ml_15_clustering.py",
    "summary.py",
]

# 전체 시작 시각
start = time.time()

# 단계별로 실행
for step in STEPS:
    # 무엇을 실행하는지 알려 준다
    print(f"\n>>> {step} 실행 중...", flush=True)
    # 단계 시작 시각
    t0 = time.time()
    # 실행. cwd=HERE 로 analysis 폴더에서 실행해야 'from common import ...' 가 동작한다
    result = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE)
    # 실패(종료 코드가 0이 아님)하면 멈춘다
    if result.returncode != 0:
        # 실패 단계 출력
        print(f"\n!!! {step} 에서 실패했습니다. 위 오류 메시지를 확인하세요.")
        # 실패 코드로 종료
        sys.exit(result.returncode)
    # 걸린 시간
    print(f"<<< {step} 완료 ({time.time() - t0:.0f}초)", flush=True)

# 전체 걸린 시간
print(f"\n전체 완료: {(time.time() - start) / 60:.1f}분. 결과는 outputs/figures/, outputs/tables/ 에 있습니다.")
