"""세 결과물에 나오는 핵심 숫자를 한 번에 다시 계산해 book/numbers.csv 로 남긴다.

같은 질문이라도 결과물마다 기준이 달라 숫자가 다르다. 기준은 세 가지다.
- 전체: 원본 3,963,959건 그대로 (신고는 일기다)
- 중복 제거: 고유번호만 다르고 나머지 43개 열이 모두 같은 줄을 하나만 남김 (질문 100개)
- 한 주소 제외: 중복 제거 뒤, 브롱크스 북부 한 주소(띄어쓰기 정리 후 비교)의 47,105건을 뺌
  (질문 100개의 소음·경찰·신고 경로 질문)

실행: .venv\\Scripts\\python.exe book\\src\\numbers_registry.py
"""

import csv

import duckdb

from paths import ROOT, STORY_PQ_SQL

FULL_PQ = (ROOT / "data" / "processed" / "311_requests_2020_present.parquet").as_posix()
OUT = ROOT / "book" / "numbers.csv"

# 주소는 개인 거주지일 수 있어 결과 파일에는 적지 않는다. 비교에만 쓴다.
TOP_ADDRESS = "655 EAST 230 STREET"


def build(con: duckdb.DuckDBPyConnection) -> None:
    cols = [c[0] for c in con.sql(f"DESCRIBE SELECT * FROM '{FULL_PQ}'").fetchall()]
    others = ", ".join(f'"{c}"' for c in cols if c != "Unique Key")
    # 같은 묶음에서 고유번호가 가장 작은 줄만 남긴다
    con.sql(f"""CREATE TABLE keep AS SELECT min("Unique Key") k FROM '{FULL_PQ}' GROUP BY {others}""")
    con.sql(f"""
        CREATE TABLE t AS
        SELECT s.*,
               k.k IS NOT NULL AS dedup,
               regexp_replace(trim(upper(coalesce(s.address, ''))), '\\s+', ' ', 'g') = '{TOP_ADDRESS}' AS top_addr
        FROM read_parquet('{STORY_PQ_SQL}') s
        LEFT JOIN keep k ON s.unique_key = k.k
    """)


BASES = {
    "전체": "TRUE",
    "중복 제거": "dedup",
    "한 주소 제외": "dedup AND NOT top_addr",
}


def one(con, sql: str):
    return con.sql(sql).fetchall()[0]


def main() -> None:
    con = duckdb.connect()
    build(con)
    rows = []

    def add(id_, base, value, display, definition, denominator, used_in, note=""):
        rows.append({
            "id": id_, "기준": base, "값": value, "표시": display, "정의": definition,
            "분모": denominator, "쓰인 곳": used_in, "비고": note,
        })

    # 1. 기준 건수
    for base, w in BASES.items():
        n = one(con, f"SELECT count(*) FROM t WHERE {w}")[0]
        add("base_count", base, n, f"{n:,}건", "1년 접수 건수", "", {
            "전체": "신고는 일기다, 웹북",
            "중복 제거": "질문 100개 (방법 메모)",
            "한 주소 제외": "질문 100개 (소음·경찰·경로 질문)",
        }[base])

    # 2~3. 경찰 결과 문장: 딱지, 체포
    for base, w in BASES.items():
        n, summons, arrest = one(con, f"""
            SELECT count(*),
                   count(*) FILTER (WHERE resolution ILIKE '%issued a summons%'),
                   count(*) FILTER (WHERE resolution ILIKE '%made an arrest%')
            FROM t WHERE agency = 'NYPD' AND resolution IS NOT NULL AND {w}""")
        used_s = {"전체": "신고는 일기다", "중복 제거": "", "한 주소 제외": "웹북 첫 화면·3장"}[base]
        used_a = {"전체": "신고는 일기다", "중복 제거": "", "한 주소 제외": "질문 100개 Q080, 웹북 3장"}[base]
        add("nypd_summons", base, summons, f"{summons / n:.1%} ({summons:,}건)",
            "경찰 결과 문장에 'issued a summons'가 있는 건", f"결과 문장이 있는 경찰 민원 {n:,}건", used_s)
        add("nypd_arrest", base, arrest, f"{arrest:,}건 ({arrest / n:.2%})",
            "경찰 결과 문장에 'made an arrest'가 있는 건 ('effect an arrest' 상투 문구는 제외)",
            f"결과 문장이 있는 경찰 민원 {n:,}건", used_a)

    # 4. 이웃 소음 진입 불가
    for base in ("전체", "한 주소 제외"):
        w = BASES[base]
        n, ne = one(con, f"""
            SELECT count(*), count(*) FILTER (WHERE resolution ILIKE '%unable to gain entry%')
            FROM t WHERE agency = 'NYPD' AND problem = 'Noise - Residential' AND resolution IS NOT NULL AND {w}""")
        add("noise_no_entry", base, ne, f"{ne / n:.1%}", "이웃 소음 경찰 결과 중 'unable to gain entry'",
            f"이웃 소음 경찰 결과 {n:,}건", {"전체": "", "한 주소 제외": "질문 100개 Q081 (13%), 웹북 3장 (12.7%)"}[base])

    # 5. 자정 마감: 정의가 두 가지다
    for base in ("전체", "중복 제거"):
        w = BASES[base]
        strict, loose = one(con, f"""
            SELECT count(*) FILTER (WHERE status = 'Closed' AND closed >= created AND strftime(closed, '%H:%M:%S') = '00:00:00'),
                   count(*) FILTER (WHERE strftime(closed, '%H:%M:%S') = '00:00:00')
            FROM t WHERE {w}""")
        add("midnight_closed", base, strict, f"{strict:,}건",
            "종료 시각이 정확히 00:00:00 이면서 상태 Closed, 종료 ≥ 접수", "처리 시간 계산 대상",
            "신고는 일기다, 웹북" if base == "전체" else "")
        add("midnight_any", base, loose, f"{loose / 10000:.1f}만 건",
            "종료 시각이 정확히 00:00:00 인 모든 기록 (상태 무관)", "",
            "질문 100개 (Q084 주의, 방법 메모)" if base == "중복 제거" else "")

    # 6. 앱 비중 (웹북 심슨의 역설 문장)
    for problem, label in (("Illegal Parking", "불법주차"), ("Panhandling", "구걸")):
        w = BASES["전체"]
        n, mob = one(con, f"SELECT count(*), count(*) FILTER (WHERE channel = 'MOBILE') FROM t WHERE problem = '{problem}' AND {w}")
        add(f"mobile_share_{problem.lower().replace(' ', '_')}", "전체", mob, f"{mob / n:.0%}",
            f"{label} 민원 중 접수 경로가 MOBILE(앱)인 비율", f"{label} 민원 {n:,}건", "웹북 심슨의 역설")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['id']:<32} {r['기준']:<8} {r['표시']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
