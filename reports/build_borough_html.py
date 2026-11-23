"""
자치구 성격 대시보드를 한 개짜리 HTML 파일로 만든다 (서버 없이 어디서나 열리고, 정적 호스팅에 그대로 올릴 수 있다).

입력 : outputs/tables/edac10_*.csv  (먼저 py -3.12 analysis/eda_borough_character.py 실행)
       reports/borough_character_template.html
출력 : reports/borough_character.html
실행 : py -3.12 reports/build_borough_html.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "outputs" / "tables"
HERE = Path(__file__).resolve().parent


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv", encoding="utf-8-sig")


sig = read("edac10_signature")
theme = read("edac10_theme")
monthly = read("edac10_theme_monthly")
sim = read("edac10_similarity")
dist = read("edac10_distance")
members = read("edac10_theme_members")

data = {
    "signature": [
        {"b": r.borough, "dir": r.방향, "p": r.problem, "theme": r.theme, "n": int(r.건수),
         "me": round(r._7, 3), "rest": round(r._8, 3), "ratio": round(r.배율, 3),
         "gratio": None if pd.isna(r._10) else round(r._10, 3),
         "cbHit": int(r._11), "cbN": int(r._12), "topCb": None if pd.isna(r._14) else r._14,
         "topShare": None if pd.isna(r._15) else round(r._15, 1)}
        for r in sig.itertuples()
    ],
    "theme": [
        {"b": r.borough, "theme": r.theme, "share": round(r._3, 2), "ratio": round(r.배율, 3),
         "pc": round(r._5, 1), "pcRatio": round(r._6, 3)}
        for r in theme.itertuples()
    ],
    "monthly": [{"b": r.borough, "ym": r.ym, "theme": r.theme, "share": round(r._5, 2)} for r in monthly.itertuples()],
    "similarity": {"labels": list(sim["자치구"]), "matrix": sim.drop(columns="자치구").round(3).values.tolist()},
    "distance": [
        {"b": r.borough, "js": round(r._4, 3), "jsCity": round(r._3, 3), "half": int(r._5), "eff": round(r._6, 1),
         "n": int(r.신고수), "cityShare": round(r._8, 1)}
        for r in dist.itertuples()
    ],
    "members": members.groupby("theme")["problem"].apply(list).to_dict(),
}

# 열 순서가 바뀌면 위의 _숫자 위치가 틀어지므로 확인한다
assert list(sig.columns[6:8]) == ["자치구 비중(%)", "나머지 비중(%)"], sig.columns
assert list(theme.columns[2:6]) == ["비중(%)", "배율", "1만명당", "1만명당 시 평균 대비"], theme.columns
assert list(dist.columns[2:8]) == ["시 전체와의 JS 거리", "나머지 자치구와의 JS 거리", "50% 도달 유형 수", "유효 유형 수",
                                   "신고수", "시 전체 신고 중 비중(%)"], dist.columns

payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
html = (HERE / "borough_character_template.html").read_text(encoding="utf-8")
assert "/*__DATA__*/null" in html
body = html.replace("/*__DATA__*/null", payload)

# 1) 단독 배포용: 완전한 HTML 문서 (GitHub Pages·Netlify 등 정적 호스팅에 그대로 올린다)
HEAD = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""
split = body.index("</style>") + len("</style>")  # 제목·글꼴·스타일까지가 head, 나머지가 body
standalone = HEAD + body[:split] + "\n</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n"
out = HERE / "borough_character.html"
out.write_text(standalone, encoding="utf-8")
print(f"저장: {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")

# 2) 조각본: 문서 틀(doctype·head·body)이 없는 버전. 틀을 자동으로 씌워 주는 곳(Claude 아티팩트 등)에 올릴 때 쓴다
frag = HERE / "borough_character.fragment.html"
frag.write_text(body, encoding="utf-8")
print(f"저장: {frag.relative_to(ROOT)}")
