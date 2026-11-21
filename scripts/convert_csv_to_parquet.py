"""
원본 zip 안의 CSV를 읽어 Parquet으로 변환한다.

- 입력: data/raw/311_Service_Requests_from_2020_to_Present_20261112.csv.zip (읽기만 함)
- 출력: data/processed/311_requests_2020_present.parquet
- 로그: outputs/tables/conversion_log.csv

실행: py -3.12 scripts/convert_csv_to_parquet.py
"""

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

# 프로젝트 루트 기준 상대경로 (이 파일 위치 scripts/ 의 부모 폴더)
ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = ROOT / "data" / "raw" / "311_Service_Requests_from_2020_to_Present_20261112.csv.zip"
OUT_PARQUET = ROOT / "data" / "processed" / "311_requests_2020_present.parquet"
OUT_LOG = ROOT / "outputs" / "tables" / "conversion_log.csv"

# 날짜 형식 예: "11/12/2026 03:15:00 PM"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
DATE_COLUMNS = ["Created Date", "Closed Date"]

# 숫자로 다룰 열만 지정하고 나머지는 문자열로 읽는다 (우편번호 같은 식별자는 문자열 유지)
# X/Y 좌표는 "1,014,655"처럼 천 단위 쉼표가 있어 숫자 변환이 실패하므로 문자열로 둔다 (EDA 단계에서 정리)
FLOAT_COLUMNS = ["Latitude", "Longitude"]
INT_COLUMNS = ["Unique Key"]

BATCH_BYTES = 64 * 1024 * 1024  # 한 번에 읽을 덩어리 크기(64MB): 메모리 부족을 막음


def read_header(zf: zipfile.ZipFile, member: str) -> list[str]:
    """CSV의 첫 줄(열 이름)만 읽어 열 목록을 얻는다."""
    with zf.open(member) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        return next(csv.reader(text))


def build_column_types(header: list[str]) -> dict:
    """열마다 자료형을 지정한다. 지정하지 않으면 덩어리마다 타입이 달라질 수 있다."""
    types = {}
    for col in header:
        if col in INT_COLUMNS:
            types[col] = pa.int64()
        elif col in FLOAT_COLUMNS:
            types[col] = pa.float64()
        else:
            types[col] = pa.string()
    return types


def parse_dates(table: pa.Table, log: dict) -> pa.Table:
    """문자열 날짜를 timestamp로 바꾼다. 형식이 안 맞는 값은 null로 두고 개수를 기록한다.

    pyarrow의 strptime은 AM/PM(%p) 형식을 읽지 못해 pandas로 파싱한다.
    """
    for col in DATE_COLUMNS:
        original = table[col].to_pandas()
        parsed = pd.to_datetime(original, format=DATE_FORMAT, errors="coerce")
        # 원래 값이 있었는데 파싱 후 비어 있게 된 행 수 = 형식 불일치 건수
        bad = int((original.notna() & parsed.isna()).sum())
        log[f"{col}_parse_failed"] = log.get(f"{col}_parse_failed", 0) + bad
        new_col = pa.array(parsed, from_pandas=True).cast(pa.timestamp("s"))
        table = table.set_column(table.schema.get_field_index(col), col, new_col)
    return table


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    log = {"input_file": RAW_ZIP.name, "rows": 0}

    with zipfile.ZipFile(RAW_ZIP) as zf:
        # __MACOSX 폴더의 숨김 파일(._로 시작)은 제외하고 실제 CSV만 고른다
        member = next(
            n for n in zf.namelist()
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
        )
        log["csv_member"] = member

        header = read_header(zf, member)
        log["columns"] = len(header)

        read_opts = pa_csv.ReadOptions(block_size=BATCH_BYTES, column_names=header, skip_rows=1)
        conv_opts = pa_csv.ConvertOptions(column_types=build_column_types(header), strings_can_be_null=True)

        writer = None
        with zf.open(member) as raw:
            reader = pa_csv.open_csv(raw, read_options=read_opts, convert_options=conv_opts)
            for batch in reader:
                table = pa.Table.from_batches([batch])
                table = parse_dates(table, log)
                log["rows"] += table.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(OUT_PARQUET, table.schema, compression="snappy")
                writer.write_table(table)
        if writer is not None:
            writer.close()

    # 변환 기록을 표로 남긴다 (항목, 값)
    with open(OUT_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        for k, v in log.items():
            w.writerow([k, v])

    print(f"완료: {log['rows']:,}행, {log['columns']}열 -> {OUT_PARQUET}")
    for k, v in log.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
