"""
311 민원 데이터 전처리: 분석용 파일을 만든다.

- 입력: data/processed/311_requests_2020_present.parquet (convert_csv_to_parquet.py 결과, 읽기만 함)
- 출력 (data/processed/):
    311_requests_clean.parquet        메인 테이블 (요청 1건 = 1행)
    311_requests_specialized.parquet  일부 민원 유형에만 값이 있는 컬럼 10개 (unique_key 로 연결)
    agency_lookup.parquet             기관 코드 → 기관 이름
    311_value_changes.parquet         컬럼별 결측 규칙으로 NULL 이 된 셀의 원래 값
- 요약: reports/preprocessing_summary.json (단계별 건수)

원칙: 행을 삭제하지 않는다. 표기를 통일하고, 계산 컬럼을 더하고, 문제가 있는 행에는 플래그를 붙인다.

실행: py -3.12 scripts/preprocess.py  (약 1~2분, 메모리 약 6GB)
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "311_requests_2020_present.parquet"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

# 데이터 추출일. 이 날짜 이후의 날짜는 미래 날짜로 표시한다
EXTRACT_DATE = pd.Timestamp("2026-11-12")
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# 원본 열 이름 → snake_case 이름
RENAME = {
    "Unique Key": "unique_key",
    "Created Date": "created_date",
    "Closed Date": "closed_date",
    "Agency": "agency",
    "Agency Name": "agency_name",
    "Problem (formerly Complaint Type)": "problem",
    "Problem Detail (formerly Descriptor)": "problem_detail",
    "Additional Details": "additional_details",
    "Location Type": "location_type",
    "Incident Zip": "incident_zip",
    "Incident Address": "incident_address",
    "Street Name": "street_name",
    "Cross Street 1": "cross_street_1",
    "Cross Street 2": "cross_street_2",
    "Intersection Street 1": "intersection_street_1",
    "Intersection Street 2": "intersection_street_2",
    "Address Type": "address_type",
    "City": "city",
    "Landmark": "landmark",
    "Facility Type": "facility_type",
    "Status": "status",
    "Due Date": "due_date",
    "Resolution Description": "resolution_description",
    "Resolution Action Updated Date": "resolution_action_updated_date",
    "Community Board": "community_board",
    "Council District": "council_district",
    "Police Precinct": "police_precinct",
    "BBL": "bbl",
    "Borough": "borough",
    "X Coordinate (State Plane)": "x_coordinate",
    "Y Coordinate (State Plane)": "y_coordinate",
    "Open Data Channel Type": "open_data_channel_type",
    "Park Facility Name": "park_facility_name",
    "Park Borough": "park_borough",
    "Vehicle Type": "vehicle_type",
    "Taxi Company Borough": "taxi_company_borough",
    "Taxi Pick Up Location": "taxi_pick_up_location",
    "Bridge Highway Name": "bridge_highway_name",
    "Bridge Highway Direction": "bridge_highway_direction",
    "Road Ramp": "road_ramp",
    "Bridge Highway Segment": "bridge_highway_segment",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Location": "location",
}

# 모든 텍스트 컬럼에 적용하는 결측 표기 (값 전체가 일치할 때만, 대소문자 무시)
# 실제로 걸리는 값: Unknown, UNKNOWN, unknown, Unspecified, None(street_name 1건)
MISSING_MARKERS = {"N/A", "UNKNOWN", "UNSPECIFIED", "NONE"}
# 커뮤니티 보드의 결측 표기: "Unspecified QUEENS", "0 Unspecified" 등
COMMUNITY_BOARD_MISSING = re.compile(r"^(0 )?Unspecified( [A-Z ]+)?$")

# 대소문자만 다른 표기를 합칠 범주형 컬럼 (주소 계열·긴 문장 컬럼은 제외)
CASE_MERGE_COLUMNS = [
    "agency", "problem", "problem_detail", "additional_details", "location_type",
    "address_type", "facility_type", "status", "open_data_channel_type",
    "vehicle_type", "taxi_company_borough", "bridge_highway_name",
    "bridge_highway_direction", "road_ramp", "bridge_highway_segment",
]

# 비교용 키를 만들 컬럼
KEY_COLUMNS = {
    "incident_address": "address_key",
    "street_name": "street_name_key",
    "cross_street_1": "cross_street_1_key",
    "cross_street_2": "cross_street_2_key",
    "intersection_street_1": "intersection_street_1_key",
    "intersection_street_2": "intersection_street_2_key",
    "landmark": "landmark_key",
}

# 별도 테이블로 분리할 컬럼 (특정 민원에만 값이 있음)
SPECIALIZED_COLUMNS = [
    "due_date", "facility_type", "park_facility_name", "vehicle_type",
    "taxi_company_borough", "taxi_pick_up_location",
    "bridge_highway_name", "bridge_highway_direction", "road_ramp", "bridge_highway_segment",
]

BOROUGH_IMPUTE_SHARE = 0.95

summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def null_where(arr: pa.ChunkedArray, mask) -> pa.ChunkedArray:
    """mask 가 true 인 위치를 NULL 로 바꾼다."""
    return pc.if_else(mask, pa.scalar(None, arr.type), arr)


def count_true(mask) -> int:
    return int(pc.sum(pc.fill_null(mask, False)).as_py() or 0)


# ─────────────────────────────────────────────
# 1단계: 값 표준화
# ─────────────────────────────────────────────

def standardize(cols: dict[str, pa.ChunkedArray], n: int) -> tuple[pd.Series, pd.DataFrame]:
    text_cols = [c for c, a in cols.items() if pa.types.is_string(a.type)]
    s1: dict = {}

    # 앞뒤 공백 제거
    trimmed = {}
    for c in text_cols:
        new = pc.utf8_trim_whitespace(cols[c])
        changed = count_true(pc.not_equal(new, cols[c]))
        if changed:
            trimmed[c] = changed
        cols[c] = new
    s1["trim_whitespace_cells"] = trimmed

    # 결측 표기 → NULL, 바뀐 컬럼 이름은 missing_markers 에 남긴다
    markers = pa.array(sorted(MISSING_MARKERS))
    marker_counts = {}
    marker_names = np.full(n, "", dtype=object)
    for c in text_cols:
        if c == "community_board":
            mask = pc.match_substring_regex(cols[c], COMMUNITY_BOARD_MISSING.pattern)
        else:
            mask = pc.is_in(pc.utf8_upper(cols[c]), value_set=markers)
        k = count_true(mask)
        if not k:
            continue
        marker_counts[c] = k
        idx = np.flatnonzero(pc.fill_null(mask, False).to_numpy(zero_copy_only=False))
        marker_names[idx] = marker_names[idx] + ("," + c)
        cols[c] = null_where(cols[c], mask)
    s1["missing_marker_cells"] = marker_counts
    missing_markers = pd.Series(marker_names).str.lstrip(",").replace("", None)
    s1["rows_with_missing_markers"] = int(missing_markers.notna().sum())

    # 도시 대문자 통일 건수는 컬럼별 규칙 적용 전에 센다 (Not provided → NOT PROVIDED 포함)
    s1["city_uppercased_cells"] = count_true(pc.not_equal(pc.utf8_upper(cols["city"]), cols["city"]))

    # 컬럼별 결측 코드 → NULL (원래 값을 기록)
    question_only = r"^\?+$"
    column_rules = [
        ("incident_zip", "zip_all_zero", pc.equal(cols["incident_zip"], "00000")),
        ("bbl", "bbl_all_zero", pc.equal(cols["bbl"], "0000000000")),
        ("city", "city_not_provided", pc.equal(pc.utf8_upper(cols["city"]), "NOT PROVIDED")),
    ] + [
        (c, "question_marks_only", pc.match_substring_regex(cols[c], question_only))
        for c in ["cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2"]
    ]
    keys = cols["unique_key"].to_numpy()
    changes = []
    rule_counts = {}
    for c, rule, mask in column_rules:
        m = pc.fill_null(mask, False).to_numpy(zero_copy_only=False)
        rule_counts[c] = int(m.sum())
        if m.any():
            changes.append(pd.DataFrame({
                "unique_key": keys[m],
                "column": c,
                "original_value": cols[c].filter(pa.array(m)).to_pylist(),
                "rule": rule,
            }))
            cols[c] = null_where(cols[c], mask)
    value_changes = pd.concat(changes, ignore_index=True)
    s1["column_rule_cells"] = rule_counts
    s1["column_rule_total_cells"] = len(value_changes)
    s1["column_rule_rows"] = int(value_changes["unique_key"].nunique())

    # 잘못된 우편번호(5자리 숫자가 아님) → NULL
    bad_zip = pc.invert(pc.match_substring_regex(cols["incident_zip"], r"^\d{5}$"))
    s1["invalid_zip_values"] = pc.unique(cols["incident_zip"].filter(pc.fill_null(bad_zip, False))).to_pylist()
    s1["invalid_zip_cells"] = count_true(bad_zip)
    cols["incident_zip"] = null_where(cols["incident_zip"], bad_zip)

    # 도시 이름을 대문자로
    cols["city"] = pc.utf8_upper(cols["city"])

    # 대소문자만 다른 범주 통합: 더 많이 쓰인 표기를 따른다
    case_merged = {}
    for c in CASE_MERGE_COLUMNS:
        vc = pd.DataFrame(pc.value_counts(cols[c]).to_pylist(), columns=["values", "counts"])
        vc = vc.dropna(subset=["values"])
        vc["upper"] = vc["values"].str.upper()
        vc = vc.sort_values("counts", ascending=False)
        canon = vc.groupby("upper")["values"].first()
        vc["canon"] = vc["upper"].map(canon)
        remap = vc[vc["values"] != vc["canon"]]
        if remap.empty:
            continue
        case_merged[c] = {
            "cells": int(remap["counts"].sum()),
            "mapping": dict(zip(remap["values"], remap["canon"])),
        }
        s = cols[c].to_pandas()
        cols[c] = pa.chunked_array([pa.array(s.replace(dict(zip(remap["values"], remap["canon"]))), type=pa.string())])
    s1["case_merged"] = case_merged

    # 날짜 문자열·좌표 문자열을 숫자/날짜형으로
    for c in ["due_date", "resolution_action_updated_date"]:
        raw = cols[c].to_pandas()
        parsed = pd.to_datetime(raw, format=DATE_FORMAT, errors="coerce")
        s1[f"{c}_parse_failed"] = int((raw.notna() & parsed.isna()).sum())
        cols[c] = pa.chunked_array([pa.array(parsed.astype("datetime64[ms]"), from_pandas=True)])
    for c in ["x_coordinate", "y_coordinate"]:
        cols[c] = pc.cast(pc.replace_substring(cols[c], ",", ""), pa.float64())

    # 우편번호로 Borough 채우기: 같은 우편번호의 95% 이상이 한 Borough 일 때만
    zb = pd.DataFrame({"zip": cols["incident_zip"].to_pandas(), "borough": cols["borough"].to_pandas()})
    known = zb.dropna()
    share = known.groupby(["zip", "borough"]).size().rename("n").reset_index()
    share["share"] = share["n"] / share.groupby("zip")["n"].transform("sum")
    top = share.sort_values("share", ascending=False).drop_duplicates("zip")
    zip_to_borough = top[top["share"] >= BOROUGH_IMPUTE_SHARE].set_index("zip")["borough"]
    fill = zb["borough"].isna() & zb["zip"].map(zip_to_borough).notna()
    zb.loc[fill, "borough"] = zb.loc[fill, "zip"].map(zip_to_borough)
    cols["borough"] = pa.chunked_array([pa.array(zb["borough"], type=pa.string())])
    cols["borough_imputed"] = pa.chunked_array([pa.array(fill.to_numpy())])
    s1["borough_missing_before_impute"] = int(zb["borough"].isna().sum() + fill.sum())
    s1["borough_imputed_rows"] = int(fill.sum())
    s1["borough_still_missing"] = int(zb["borough"].isna().sum())

    # 비교용 키: 앞뒤 공백 제거 + 연속 공백을 한 칸으로 + 대문자
    keys_summary = {}
    for src, dst in KEY_COLUMNS.items():
        key = pc.utf8_upper(pc.utf8_trim_whitespace(pc.replace_substring_regex(cols[src], r"\s+", " ")))
        cols[dst] = key
        keys_summary[dst] = {
            "source_unique": len(pc.unique(cols[src].drop_null())),
            "key_unique": len(pc.unique(key.drop_null())),
        }
    s1["comparison_keys"] = keys_summary

    summary["step1_standardize"] = s1
    return missing_markers, value_changes


# ─────────────────────────────────────────────
# 2단계: 컬럼 정리 (제외 전 검사)
# ─────────────────────────────────────────────

def check_dropped_columns(raw: pa.Table) -> None:
    s2 = {}
    # agency 와 agency_name 이 1:1 인지
    pairs = pd.DataFrame({"a": raw["Agency"].to_pandas(), "n": raw["Agency Name"].to_pandas()}).drop_duplicates()
    assert pairs["a"].is_unique and pairs["n"].is_unique, "agency ↔ agency_name 이 1:1 이 아님"
    s2["agency_name_one_to_one"] = True
    # park_borough 가 borough 와 모든 행이 같은지 (원본 기준)
    same = pc.equal(raw["Park Borough"], raw["Borough"])
    both_null = pc.and_(pc.is_null(raw["Park Borough"]), pc.is_null(raw["Borough"]))
    diff = raw.num_rows - count_true(pc.or_(same, both_null))
    assert diff == 0, f"park_borough 와 borough 가 다른 행 {diff}"
    s2["park_borough_equals_borough"] = True
    # location 이 POINT (경도 위도) 와 같은 값인지
    loc = raw["Location"].to_pandas()
    pts = loc.str.extract(r"POINT \((-?[\d.]+) (-?[\d.]+)\)").astype(float)
    lat, lon = raw["Latitude"].to_numpy(zero_copy_only=False), raw["Longitude"].to_numpy(zero_copy_only=False)
    assert (loc.isna().to_numpy() == (np.isnan(lat) | np.isnan(lon))).all(), "location 결측 위치가 위도·경도와 다름"
    ok = loc.notna().to_numpy()
    max_diff = float(max(np.abs(pts[0].to_numpy()[ok] - lon[ok]).max(), np.abs(pts[1].to_numpy()[ok] - lat[ok]).max()))
    assert max_diff < 1e-6, f"location 과 위도·경도 차이 {max_diff}"
    s2["location_equals_lat_lon"] = {"max_abs_diff": max_diff}
    s2["dropped_from_main"] = ["agency_name", "park_borough", "location"]
    s2["moved_to_specialized"] = SPECIALIZED_COLUMNS
    summary["step2_columns"] = s2


# ─────────────────────────────────────────────
# 5단계용: 중복 의심 (unique_key 외 43개 컬럼이 모두 같은 행)
# ─────────────────────────────────────────────

def system_duplicates(raw: pa.Table) -> np.ndarray:
    other = [c for c in raw.column_names if c != "Unique Key"]
    # 컬럼마다 해시를 섞어 행 해시를 만들고, 해시가 겹치는 행만 실제 값으로 다시 비교한다
    h = np.zeros(raw.num_rows, dtype=np.uint64)
    for c in other:
        hc = pd.util.hash_pandas_object(raw[c].to_pandas(), index=False).to_numpy()
        h = h * np.uint64(1_000_003) ^ hc
    hs = pd.Series(h)
    cand = np.flatnonzero(hs.duplicated(keep=False).to_numpy())
    sub = raw.take(pa.array(cand)).to_pandas()
    sub = sub.sort_values("Unique Key")
    grp = sub.groupby(other, dropna=False, sort=False)["Unique Key"]
    first = grp.transform("min")
    size = grp.transform("size")
    dup_keys = set(sub.loc[(size > 1) & (sub["Unique Key"] != first), "Unique Key"])
    n_groups = int(sub.loc[size > 1].groupby(other, dropna=False).ngroups)
    keys = raw["Unique Key"].to_numpy()
    flag = np.isin(keys, list(dup_keys))
    summary["step5_duplicates"] = {
        "flag_system_duplicate": int(flag.sum()),
        "system_duplicate_groups": n_groups,
    }
    return flag


def main() -> None:
    t0 = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("읽기")
    raw = pq.read_table(IN_PATH)
    n = raw.num_rows
    summary["input"] = {"file": IN_PATH.name, "rows": n, "columns": raw.num_columns}

    log("2단계 검사 · 중복 의심 판정 (원본 값 기준)")
    check_dropped_columns(raw)
    dup_flag = system_duplicates(raw)

    cols = {RENAME[c]: raw[c] for c in raw.column_names}
    agency_lookup = (
        pd.DataFrame({"agency": raw["Agency"].to_pandas(), "agency_name": raw["Agency Name"].to_pandas()})
        .drop_duplicates().sort_values("agency").reset_index(drop=True)
    )
    del raw

    log("1단계 값 표준화")
    missing_markers, value_changes = standardize(cols, n)

    log("3단계 계산 컬럼 · 4단계 플래그")
    created = cols["created_date"].to_pandas()
    closed = cols["closed_date"].to_pandas()
    updated = cols["resolution_action_updated_date"].to_pandas()
    status = cols["status"].to_pandas()

    hours = (closed - created).dt.total_seconds() / 3600
    is_closed = (status == "Closed").fillna(False).astype(bool)
    calc = {
        "resolution_hours": hours,
        "is_closed": is_closed,
        "created_year": created.dt.year.astype("int16"),
        "created_month": created.dt.to_period("M").dt.to_timestamp().astype("datetime64[ms]"),
        "created_dow": (created.dt.day_of_week + 1).astype("int8"),
        "created_hour": created.dt.hour.astype("int8"),
    }
    after_extract = EXTRACT_DATE + pd.Timedelta(days=1)  # 추출일 다음 날 0시 이후 = 추출일 이후
    flags = {
        "flag_negative_duration": (closed < created).to_numpy(),
        "flag_zero_duration": (closed == created).to_numpy(),
        "flag_duration_over_365d": (hours > 365 * 24).to_numpy(),
        "flag_midnight_close": (closed.notna() & (closed == closed.dt.normalize())).to_numpy(),
        "flag_status_date_mismatch": ((is_closed & closed.isna()) | (~is_closed & closed.notna())).to_numpy(),
        "flag_updated_before_created": (updated < created).to_numpy(),
        "flag_future_date": ((created >= after_extract) | (closed >= after_extract) | (updated >= after_extract)).to_numpy(),
        "flag_no_location": (cols["latitude"].to_pandas().isna() | cols["longitude"].to_pandas().isna()).to_numpy(),
    }
    summary["step4_flags"] = {k: int(v.sum()) for k, v in flags.items()}
    summary["step4_flags"]["borough_imputed"] = int(pc.sum(cols["borough_imputed"]).as_py())

    log("5단계 중복 표시 · 동일 조건 신고 그룹")
    agency_dup = cols["resolution_description"].to_pandas().str.contains("duplicate", case=False, na=False).to_numpy()
    summary["step5_duplicates"]["flag_agency_duplicate"] = int(agency_dup.sum())

    keys = cols["unique_key"].to_numpy()
    g = pd.DataFrame({
        "unique_key": keys,
        "problem": cols["problem"].to_pandas(),
        "problem_detail": cols["problem_detail"].to_pandas(),
        "borough": cols["borough"].to_pandas(),
        "address_key": cols["address_key"].to_pandas(),
        "day": created.dt.normalize(),
    })
    groupable = g["borough"].notna() & g["address_key"].notna()
    group_id = pd.Series(keys, dtype="int64")
    gg = g[groupable]
    group_id[groupable.to_numpy()] = gg.groupby(
        ["problem", "problem_detail", "borough", "address_key", "day"], dropna=False, sort=False
    )["unique_key"].transform("min").to_numpy()
    group_size = group_id.map(group_id.value_counts()).astype("int32")
    sizes = group_id.value_counts()
    summary["step5_duplicates"].update({
        "incident_groups": int(sizes.size),
        "repeated_groups": int((sizes > 1).sum()),
        "rows_in_repeated_groups": int(sizes[sizes > 1].sum()),
        "singleton_no_address_or_borough_rows_all": int((~groupable).sum()),
        "singleton_no_address_or_borough_rows_base": int(((~groupable).to_numpy() & ~dup_flag).sum()),
    })

    # 이전 방식(원문 주소 문자열만, 자치구 없음)과 비교
    old = g[["unique_key", "problem", "problem_detail", "day"]].copy()
    old["addr"] = cols["incident_address"].to_pandas()
    old_ok = old["addr"].notna()
    old_id = pd.Series(keys, dtype="int64")
    old_id[old_ok.to_numpy()] = old[old_ok].groupby(
        ["problem", "problem_detail", "addr", "day"], dropna=False, sort=False
    )["unique_key"].transform("min").to_numpy()
    pair = pd.DataFrame({"old": old_id, "new": group_id, "borough": g["borough"]})
    new_per_old = pair.groupby("old")["new"].nunique()
    split_old = new_per_old[new_per_old > 1].index
    old_per_new = pair.groupby("new")["old"].nunique()
    merged_new = old_per_new[old_per_new > 1].index
    boro_per_old = pair.groupby("old")["borough"].nunique()
    change_check = {
        "groups_before": int(old_id.nunique()),
        "groups_after": int(group_id.nunique()),
        "old_groups_with_mixed_borough": int((boro_per_old > 1).sum()),
        "rows_in_old_mixed_borough_groups": int(pair["old"].isin(boro_per_old[boro_per_old > 1].index).sum()),
        "old_groups_split": int(len(split_old)),
        "rows_in_split_old_groups": int(pair["old"].isin(split_old).sum()),
        "new_groups_merging_multiple_old": int(len(merged_new)),
        "rows_in_merged_new_groups": int(pair["new"].isin(merged_new).sum()),
        "new_groups_with_mixed_borough": int((pair.groupby("new")["borough"].nunique() > 1).sum()),
    }
    (REPORT_DIR / "preprocessing_change_check.json").write_text(
        json.dumps(change_check, ensure_ascii=False, indent=2), encoding="utf-8")
    del g, old, pair

    log("저장")
    special = pa.table({"unique_key": cols["unique_key"], **{c: cols[c] for c in SPECIALIZED_COLUMNS}})
    any_value = None
    for c in SPECIALIZED_COLUMNS:
        m = pc.is_valid(special[c])
        any_value = m if any_value is None else pc.or_(any_value, m)
    special = special.filter(any_value)

    drop = set(SPECIALIZED_COLUMNS) | {"agency_name", "park_borough", "location"}
    main_cols = {c: a for c, a in cols.items() if c not in drop}
    for k, v in calc.items():
        main_cols[k] = pa.array(v, from_pandas=True)
    for k, v in flags.items():
        main_cols[k] = pa.array(v)
    main_cols["flag_system_duplicate"] = pa.array(dup_flag)
    main_cols["flag_agency_duplicate"] = pa.array(agency_dup)
    main_cols["incident_group_id"] = pa.array(group_id.to_numpy())
    main_cols["incident_group_size"] = pa.array(group_size.to_numpy())
    main_cols["missing_markers"] = pa.array(missing_markers, type=pa.string())
    clean = pa.table(main_cols)

    pq.write_table(clean, OUT_DIR / "311_requests_clean.parquet", compression="zstd")
    pq.write_table(special, OUT_DIR / "311_requests_specialized.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(agency_lookup, preserve_index=False), OUT_DIR / "agency_lookup.parquet")
    pq.write_table(pa.Table.from_pandas(value_changes, preserve_index=False), OUT_DIR / "311_value_changes.parquet")

    summary["outputs"] = {
        "311_requests_clean.parquet": {"rows": clean.num_rows, "columns": clean.num_columns},
        "311_requests_specialized.parquet": {"rows": special.num_rows, "columns": special.num_columns},
        "agency_lookup.parquet": {"rows": len(agency_lookup)},
        "311_value_changes.parquet": {"rows": len(value_changes)},
    }
    assert clean.num_rows == n, "행 수가 달라짐"
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    (REPORT_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"완료 ({summary['elapsed_seconds']}초)")


if __name__ == "__main__":
    main()
