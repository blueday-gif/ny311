"""기초통계 포트폴리오용 원본 내려받기 (NYC 311 전 유형 + 일별 기온).

기존 data/raw/nyc311_manhattan_noise_2024_apr_sep.csv 에는 closed_date, agency, latitude,
longitude 가 없고 맨해튼 소음 민원만 들어 있어 이번 분석에 쓸 수 없다. 같은 기간(2024-04-01 ~
2024-09-30)의 5개 자치구 전 유형 민원을 필요한 열만 골라 새 파일로 받는다.

출력:  data/raw/nyc311_all_2024_apr_sep.csv          NYC Open Data 311 (erm2-nwe9)
       data/raw/nyc_daily_weather_2024_apr_sep.csv   Open-Meteo 일별 기온·강수 (센트럴파크 좌표)
       data/raw/SOURCE_all.txt                       출처, 쿼리, 행 수, 내려받은 날짜

이미 있는 파일은 덮어쓰지 않는다(원본 보호). 다시 받으려면 직접 지운 뒤 실행한다.
실행: 프로젝트 루트에서  python analysis/stat_00_download.py
"""

import csv
import io
import time
from datetime import date, datetime, timedelta

import requests

from stat_common import END, RAW, RAW_311, RAW_WEATHER, ROOT, START

API_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
COLUMNS = ["unique_key", "created_date", "closed_date", "agency", "complaint_type", "borough", "latitude", "longitude"]
PAGE = 50_000
API_WEATHER = "https://archive-api.open-meteo.com/v1/archive"
CENTRAL_PARK = (40.7794, -73.9692)


def get(url, params, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r
            msg = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            msg = repr(e)
        wait = 5 * (attempt + 1)
        print(f"    재시도 {attempt + 1}/{tries} ({msg}) - {wait}초 대기")
        time.sleep(wait)
    raise SystemExit(f"요청 실패: {url}")


def download_311():
    if RAW_311.exists():
        print(f"건너뜀: {RAW_311.relative_to(ROOT)} 가 이미 있습니다.")
        return None
    part = RAW_311.with_suffix(".csv.part")
    start, end = date.fromisoformat(START), date.fromisoformat(END)
    total = 0
    # 하루 단위로 나눠 받는다: 깊은 offset 을 피하고, 실패해도 그날만 다시 요청하면 된다.
    with open(part, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        day = start
        while day < end:
            nxt = day + timedelta(days=1)
            offset, n_day = 0, 0
            while True:
                r = get(API_311, {
                    "$select": ",".join(COLUMNS),
                    "$where": f"created_date >= '{day}T00:00:00' AND created_date < '{nxt}T00:00:00'",
                    "$order": "unique_key",
                    "$limit": PAGE,
                    "$offset": offset,
                })
                rows = list(csv.reader(io.StringIO(r.text)))
                if rows and rows[0] != COLUMNS:
                    raise SystemExit(f"예상과 다른 열 구성: {rows[0]}")
                body = rows[1:]
                writer.writerows(body)
                n_day += len(body)
                if len(body) < PAGE:
                    break
                offset += PAGE
            total += n_day
            print(f"  {day}  {n_day:>7,}행  (누적 {total:,})")
            day = nxt
    part.rename(RAW_311)
    print(f"저장: {RAW_311.relative_to(ROOT)}  {total:,}행")
    return total


def download_weather():
    if RAW_WEATHER.exists():
        print(f"건너뜀: {RAW_WEATHER.relative_to(ROOT)} 가 이미 있습니다.")
        return None
    last = date.fromisoformat(END) - timedelta(days=1)
    r = get(API_WEATHER, {
        "latitude": CENTRAL_PARK[0],
        "longitude": CENTRAL_PARK[1],
        "start_date": START,
        "end_date": str(last),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York",
    })
    d = r.json()["daily"]
    with open(RAW_WEATHER, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "temp_mean_c", "temp_max_c", "temp_min_c", "precip_mm"])
        w.writerows(zip(d["time"], d["temperature_2m_mean"], d["temperature_2m_max"],
                        d["temperature_2m_min"], d["precipitation_sum"]))
    print(f"저장: {RAW_WEATHER.relative_to(ROOT)}  {len(d['time'])}행")
    return len(d["time"])


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    n311 = download_311()
    nw = download_weather()
    if n311 is None and nw is None:
        return
    lines = [f"[{datetime.now():%Y-%m-%d %H:%M} 내려받음, analysis/stat_00_download.py]"]
    if n311 is not None:
        lines += [
            "NYC Open Data | 311 Service Requests from 2020 to Present",
            f"API: {API_311}",
            f"SELECT {','.join(COLUMNS)}",
            f"WHERE created_date >= '{START}T00:00:00' AND created_date < '{END}T00:00:00' (하루 단위 분할 요청, ORDER BY unique_key)",
            f"Rows: {n311}. 한 행 = 311 민원 1건. created_date/closed_date 는 뉴욕 현지 시각. unique_key 는 식별자.",
        ]
    if nw is not None:
        lines += [
            "Open-Meteo Historical Weather API (ERA5 재분석 기반, 관측소 실측값 아님)",
            f"API: {API_WEATHER}  좌표 {CENTRAL_PARK} (센트럴파크), timezone America/New_York",
            f"Rows: {nw}. 한 행 = 하루. 기온 섭씨, 강수 mm.",
        ]
    with open(RAW / "SOURCE_all.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"출처 기록: {(RAW / 'SOURCE_all.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
