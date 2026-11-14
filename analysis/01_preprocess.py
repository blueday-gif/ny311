"""NYC 311 맨해튼 소음 민원 전처리.

입력:  data/raw/nyc311_manhattan_noise_2024_apr_sep.csv (수정하지 않음)
출력:  data/processed/nyc311_noise_clean.csv        전체 행 + 검증 플래그 + 파생변수
       data/processed/nyc311_noise_zip_analysis.csv 군집분석 대상 (zip 유효 행)
       data/processed/nyc311_zip_features.csv       zip 단위 집계 변수
       outputs/tables/preprocess_exclusion_log.csv  제외 기준과 건수

실행: 프로젝트 루트에서  python analysis/01_preprocess.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nyc311_manhattan_noise_2024_apr_sep.csv"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

MIN_ZIP_COUNT = 30  # zip 집계의 최소 건수 기준 (미만이면 is_stable = False)
INVALID_ZIPS = {"10000", "12345"}  # 실제 우편번호가 아닌 기본값/임시값
MANHATTAN_ZIP_PREFIXES = ("100", "101", "102")

# 구형 라벨 정리: complaint_type = 'Noise' 행의 descriptor 기준
DESCRIPTOR_MAP = {
    "Jackhammer": "Noise: Jack Hammering (NC2)",
    "Construction": "Noise: Construction Before/After Hours (NM1)",
}

COMPLAINT_SHORT = {
    "Noise - Street/Sidewalk": "street",
    "Noise - Residential": "residential",
    "Noise - Commercial": "commercial",
    "Noise": "generic",
    "Noise - Helicopter": "helicopter",
    "Noise - Vehicle": "vehicle",
    "Noise - Park": "park",
    "Noise - House of Worship": "worship",
}


def clean_text(s: pd.Series) -> pd.Series:
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # 모든 열을 문자열로 읽어 zip 앞자리 0과 빈 값을 보존
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    n_raw = len(df)
    log = [("원본 행 수", n_raw, "")]

    # 1. 텍스트 정리
    for col in ["complaint_type", "descriptor", "incident_zip", "borough", "created_date"]:
        df[col] = clean_text(df[col])

    # 2. 날짜 파싱
    df["created_dt"] = pd.to_datetime(df["created_date"], format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")
    n_bad_date = int(df["created_dt"].isna().sum())
    log.append(("날짜 파싱 실패 (제외 아님, 확인용)", n_bad_date, "created_date 형식 위반"))

    # 3. 중복 검사
    n_dup_key = int(df["unique_key"].duplicated().sum())
    log.append(("unique_key 중복 행 (확인용)", n_dup_key, "중복이면 첫 행만 유지"))
    if n_dup_key:
        df = df.drop_duplicates(subset="unique_key", keep="first").copy()

    # 4. 라벨 정리
    mask_generic = df["complaint_type"].eq("Noise")
    df["complaint_std"] = df["complaint_type"]
    n_relabel = int((mask_generic & df["descriptor"].isin(DESCRIPTOR_MAP.keys())).sum())
    df.loc[mask_generic, "descriptor"] = df.loc[mask_generic, "descriptor"].replace(DESCRIPTOR_MAP)
    log.append(("구형 라벨 표준화 대상 (Jackhammer, Construction)", n_relabel, "descriptor 값 치환"))

    # 5. zip 검증 플래그
    zip_missing = df["incident_zip"].eq("")
    zip_invalid_value = df["incident_zip"].isin(INVALID_ZIPS)
    zip_not_manhattan = ~df["incident_zip"].str.startswith(MANHATTAN_ZIP_PREFIXES) & ~zip_missing & ~zip_invalid_value
    df["zip_missing"] = zip_missing
    df["zip_invalid"] = zip_invalid_value
    df["zip_not_manhattan"] = zip_not_manhattan
    df["zip_valid"] = ~(zip_missing | zip_invalid_value | zip_not_manhattan)

    log.append(("zip 결측 (빈 값)", int(zip_missing.sum()), "제외 (zip 기반 분석)"))
    log.append(("zip 기본값 (10000, 12345)", int(zip_invalid_value.sum()), "제외 (zip 기반 분석)"))
    log.append(("zip이 맨해튼 범위(100xx-102xx) 밖", int(zip_not_manhattan.sum()), "제외 (zip 기반 분석)"))
    n_zip_invalid_total = int((~df["zip_valid"]).sum())
    log.append(("zip 기반 분석 제외 합계", n_zip_invalid_total, "중복 없이 집계"))

    # 6. 파생 변수 (식별자는 수치형으로 쓰지 않음)
    dt = df["created_dt"]
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.day_of_week  # 0=월요일
    df["month"] = dt.dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["time_band"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["night", "morning", "afternoon", "evening", "late_night"],
        ordered=False,
    ).astype(str)
    df.loc[df["hour"].isin([22, 23]), "time_band"] = "late_night"
    df.loc[df["hour"].between(0, 5), "time_band"] = "night"
    df["complaint_code"] = df["complaint_std"].map(COMPLAINT_SHORT).fillna("other")

    # 7. 저장: 전체 행 (플래그 포함)
    keep_cols = [
        "unique_key", "created_date", "created_dt", "complaint_std", "descriptor",
        "incident_zip", "borough", "zip_missing", "zip_invalid", "zip_not_manhattan",
        "zip_valid", "hour", "weekday", "month", "is_weekend", "is_night",
        "time_band", "complaint_code",
    ]
    df_out = df[keep_cols].sort_values(["created_dt", "unique_key"]).reset_index(drop=True)
    df_out.to_csv(PROC / "nyc311_noise_clean.csv", index=False, encoding="utf-8")

    # 8. zip 분석 대상
    df_zip = df_out[df_out["zip_valid"]].copy()
    df_zip.to_csv(PROC / "nyc311_noise_zip_analysis.csv", index=False, encoding="utf-8")
    log.append(("zip 기반 분석 대상 행 수", len(df_zip), "군집분석 입력 기준"))

    # 9. zip 단위 집계 변수
    g = df_zip.groupby("incident_zip")
    feat = pd.DataFrame({
        "n_complaints": g.size(),
        "night_share": g["is_night"].mean(),
        "weekend_share": g["is_weekend"].mean(),
    })
    type_share = (
        df_zip.groupby(["incident_zip", "complaint_code"]).size()
        .unstack(fill_value=0)
    )
    type_share = type_share.div(type_share.sum(axis=1), axis=0).add_prefix("share_")
    feat = feat.join(type_share, how="left").fillna(0)
    feat["log_n_complaints"] = np.log1p(feat["n_complaints"])
    feat["is_stable"] = feat["n_complaints"] >= MIN_ZIP_COUNT
    feat = feat.reset_index().rename(columns={"incident_zip": "zip"})
    feat.sort_values("n_complaints", ascending=False).to_csv(
        PROC / "nyc311_zip_features.csv", index=False, encoding="utf-8"
    )
    n_unstable = int((~feat["is_stable"]).sum())
    log.append((f"zip 수 (전체)", len(feat), ""))
    log.append((f"건수 {MIN_ZIP_COUNT}건 미만 zip (is_stable=False, 제외 아님)", n_unstable, "군집 전 판단"))

    log_df = pd.DataFrame(log, columns=["항목", "건수", "처리"])
    log_df.to_csv(TABLES / "preprocess_exclusion_log.csv", index=False, encoding="utf-8-sig")

    print(log_df.to_string(index=False))
    print(f"\n저장 완료: {PROC}")


if __name__ == "__main__":
    main()
