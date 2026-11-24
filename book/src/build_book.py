"""Question book. Numbers come from recomputed JSON, not from memory.

Pages that repeat the older hundred-question PDF are not rebuilt.
Those gaps are named once, on the closing comment page.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import duckdb
import pandas as pd
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

ROOT = Path(r"D:\nyc311")
NEW = json.loads((INTERIM / "book_new_stats.json").read_text(encoding="utf-8"))
POL = json.loads((INTERIM / "book_politics_stats.json").read_text(encoding="utf-8"))
OUT = REPORTS / "nyc311_question_book.html"

BORO = {
    "BRONX": "브롱크스",
    "MANHATTAN": "맨해튼",
    "BROOKLYN": "브루클린",
    "QUEENS": "퀸스",
    "STATEN ISLAND": "스태튼아일랜드",
}
AGENCY = [
    ("hpd", "주택"),
    ("nypd", "경찰"),
    ("dsny", "청소"),
    ("dot", "교통"),
    ("dpr", "공원"),
    ("dep", "환경"),
]
SEASON_ZIPS = {
    "astoria": (18357, 2465, 22460, 5615),
    "bayside": (5694, 272, 6292, 1294),
    "rest": (29203, 4385, 32679, 6038),
}

CSS = r"""
@page { size: A4; margin: 15mm 15mm 16mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; color: #1c1917; }
.sheet { page-break-after: always; break-after: page; position: relative; }
@media screen {
  body { background: #e7e5e4; }
  .sheet {
    width: 210mm; min-height: 297mm; margin: 8mm auto; padding: 15mm 15mm 16mm;
    background: #fafaf9;
  }
}
@media print {
  html, body { background: #fff; }
  .sheet { width: auto; min-height: 266mm; margin: 0; padding: 0; background: #fff; }
}
header { display: flex; justify-content: space-between; align-items: baseline;
  height: 8mm; border-bottom: 0.3pt solid #d6d3d1; font-size: 9pt; letter-spacing: .14em; }
.kicker { color: #c2410c; }
.qid, .folio { color: #78716c; font-variant-numeric: tabular-nums; }
h1 { font-size: 36pt; line-height: 1.15; margin: 4mm 0; font-weight: 700; }
h2.q { font-size: 22pt; line-height: 1.25; font-weight: 700; margin: 3mm 0 2mm; max-height: 28mm; overflow: hidden; }
.lead { font-size: 11.5pt; line-height: 1.45; color: #44403c; margin: 0 0 4mm; max-height: 14mm; overflow: hidden; }
.fig { height: 138mm; margin: 0 0 4mm; display: flex; flex-direction: column; }
.figcap { height: 8mm; margin: 0; font-size: 9pt; color: #78716c; }
.plot { flex: 1; min-height: 0; }
.plot svg { width: 100%; height: 100%; display: block; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm;
  border-top: 0.4pt solid #d6d3d1; padding-top: 3mm; margin: 0 0 4mm; }
.facts b { display: block; font-size: 16pt; color: #9a3412; font-weight: 700; font-variant-numeric: tabular-nums; }
.facts span { display: block; margin-top: 1mm; font-size: 8pt; line-height: 1.35; color: #57534e; }
.band { list-style: none; margin: 0 0 4mm; padding: 3mm 4mm; background: #f5f5f4; }
.band li { font-size: 9.5pt; line-height: 1.45; }
.notes { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
.note { margin: 0; font-size: 8.5pt; line-height: 1.4; color: #57534e; }
.tag { display: block; margin-bottom: 1mm; font-size: 8pt; letter-spacing: .12em; color: #c2410c; }
.folio { position: absolute; right: 15mm; bottom: 6mm; font-size: 8.5pt; }
@media print { .folio { right: 0; bottom: 0; } }
.cols { display: flex; align-items: flex-end; height: 100%; gap: 1.2mm; }
.cols .c { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; min-width: 0; }
.cols .shaft { width: 68%; background: #a8a29e; }
.cols .shaft.hot { background: #c2410c; }
.cols .shaft.mark { box-shadow: inset 0 0 0 0.8pt #c2410c; }
.cols .lab { height: 10mm; font-size: 7.5pt; color: #57534e; text-align: center; line-height: 1.2; }
.cols .val { font-size: 8pt; color: #44403c; font-variant-numeric: tabular-nums; margin-bottom: 1mm; }
.stacks { display: flex; flex-direction: column; justify-content: space-evenly; height: 78%; }
.srow { display: grid; grid-template-columns: 28mm 1fr; gap: 3mm; align-items: center; }
.sname { font-size: 10pt; color: #44403c; }
.sbar { display: flex; height: 8mm; background: #e7e5e4; }
.sbar i { display: block; height: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: 2mm 4mm; margin-top: 3mm; font-size: 8.5pt; color: #44403c; }
.legend b { display: inline-block; width: 3.5mm; height: 3.5mm; margin-right: 1mm; vertical-align: -0.4mm; }
.bign { height: 100%; display: flex; flex-direction: column; justify-content: flex-end; }
.bign .n { font-size: 64pt; line-height: .9; color: #9a3412; font-weight: 700; margin: 0; }
.bign .cmp { display: flex; justify-content: space-between; border-top: 0.4pt solid #d6d3d1; margin-top: 6mm; padding-top: 3mm; font-size: 11pt; }
.opener { min-height: 250mm; display: flex; flex-direction: column; }
.opener .rule { width: 28mm; height: 1.2mm; background: #c2410c; margin: 8mm 0 4mm; }
.opener .ch { font-size: 9pt; letter-spacing: .18em; color: #c2410c; margin: 0; }
.opener .gap { flex: 1; }
.opener .num { font-size: 72pt; line-height: .85; color: #c2410c; margin: 0; font-weight: 700; }
.opener h2 { font-size: 32pt; line-height: 1.15; margin: 4mm 0; max-width: 150mm; }
.dek { font-size: 12pt; line-height: 1.45; max-width: 165mm; margin: 0 0 2mm; }
.scene { font-size: 11pt; color: #44403c; margin: 0 0 6mm; }
.previews { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6mm; border-top: 0.4pt solid #d6d3d1; padding-top: 4mm; }
.previews b { display: block; font-size: 18pt; color: #9a3412; }
.previews span { display: block; margin-top: 1mm; font-size: 8.5pt; color: #57534e; }
.cover { min-height: 250mm; display: flex; flex-direction: column; }
.cover .hero { margin-top: auto; font-size: 84pt; line-height: .85; color: #9a3412; font-weight: 700; }
.meta { font-size: 8.5pt; color: #57534e; margin-top: 6mm; }
.wire { height: 100%; display: grid; grid-template-rows: 16mm 1fr 18mm 16mm; gap: 3mm; }
.wire div { border: 0.4pt solid #d6d3d1; display: flex; align-items: center; padding: 0 4mm; font-size: 12pt; color: #44403c; }
.wire .tall { background: #f5f5f4; align-items: flex-end; padding-bottom: 4mm; }
.toc { columns: 2; column-gap: 8mm; }
.entry { break-inside: avoid; display: grid; grid-template-columns: 12mm 1fr 28mm; gap: 2mm; padding: 2.6mm 0; border-bottom: 0.3pt solid #e7e5e4; font-size: 9pt; }
.entry b { color: #9a3412; text-align: right; font-weight: 700; }
.close li { font-size: 13.5pt; line-height: 1.45; margin: 0; padding: 7mm 0; border-bottom: 0.3pt solid #e7e5e4; }
.sources { list-style: none; margin: 6mm 0 0; padding: 0; }
.sources li { font-size: 10.5pt; line-height: 1.5; padding: 3.5mm 0; border-bottom: 0.3pt solid #e7e5e4; color: #44403c; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8mm; height: 100%; }
.pair .panel { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.pair h3 { margin: 0 0 2mm; font-size: 11pt; font-weight: 700; }
.pair .cols { flex: 1; height: auto; }
.pair .panel + .panel { border-left: 0.4pt solid #d6d3d1; padding-left: 5mm; }
"""


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def pct(n: float, d: int = 1) -> str:
    return f"{n:.{d}f}%"


def page(kicker, qid, title, lead, chart, facts, band, news, stat) -> str:
    facts_html = "".join(f"<p><b>{esc(v)}</b><span>{esc(lab)}</span></p>" for v, lab in facts)
    band_html = "".join(f"<li>{esc(x)}</li>" for x in band)
    return f"""
<section class="sheet">
  <header><span class="kicker">{esc(kicker)}</span><span class="qid">{esc(qid)}</span></header>
  <h2 class="q">{esc(title)}</h2>
  <p class="lead">{esc(lead)}</p>
  {chart}
  <div class="facts">{facts_html}</div>
  <ul class="band">{band_html}</ul>
  <div class="notes">
    <p class="note"><span class="tag">기사</span>{esc(news)}</p>
    <p class="note"><span class="tag">통계</span>{esc(stat)}</p>
  </div>
  <p class="folio"></p>
</section>
"""


def opener(num, title, dek, scene, previews) -> str:
    cells = "".join(f"<p><b>{esc(v)}</b><span>{esc(lab)}</span></p>" for v, lab in previews)
    return f"""
<section class="sheet">
  <div class="opener">
    <div class="rule"></div>
    <p class="ch">CHAPTER {esc(num)}</p>
    <div class="gap"></div>
    <p class="num">{esc(num)}</p>
    <h2>{esc(title)}</h2>
    <p class="dek">{esc(dek)}</p>
    <p class="scene">{esc(scene)}</p>
    <div class="previews">{cells}</div>
  </div>
  <p class="folio"></p>
</section>
"""


def figure(cap: str, body: str) -> str:
    return f'<figure class="fig"><p class="figcap">{esc(cap)}</p><div class="plot">{body}</div></figure>'


def chart_columns(bins: list[tuple[str, float]], hot: int, mark: int | None = None, cap: str = "") -> str:
    peak = max((v for _, v in bins), default=1) or 1
    cols = []
    for i, (name, value) in enumerate(bins):
        klass = "shaft"
        if i == hot:
            klass += " hot"
        elif i == mark:
            klass += " mark"
        height = max(0.8, value / peak * 88)
        shown = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.1f}"
        cols.append(
            f'<div class="c"><span class="val">{shown}</span>'
            f'<span class="{klass}" style="height:{height:.1f}%"></span>'
            f'<span class="lab">{esc(name)}</span></div>'
        )
    return figure(cap, f'<div class="cols">{"".join(cols)}</div>')


def chart_pair(panels: list[tuple[str, list[tuple[str, float]], int]], cap: str) -> str:
    blocks = []
    for title, bins, hot in panels:
        inner = chart_columns(bins, hot=hot, cap="").split('<div class="plot">', 1)[1].rsplit("</div></figure>", 1)[0]
        blocks.append(f'<div class="panel"><h3>{esc(title)}</h3>{inner}</div>')
    return figure(cap, f'<div class="pair">{"".join(blocks)}</div>')


def chart_dumbbell(rows: list[tuple[str, float, float]], left: str, right: str, cap: str, unit: str = "") -> str:
    vals = [v for _, a, b in rows for v in (a, b)]
    lo = min(0, min(vals))
    hi = max(vals) * 1.08 if max(vals) > 0 else 1
    span = hi - lo or 1

    def x(v: float) -> float:
        return 46 + (v - lo) / span * 124

    n = len(rows)
    parts = [
        f'<text x="46" y="8" font-size="8" fill="#78716c">{esc(left)}</text>',
        f'<text x="170" y="8" font-size="8" fill="#78716c" text-anchor="end">{esc(right)}</text>',
    ]
    if lo < 0 < hi:
        zx = x(0)
        parts.append(f'<line x1="{zx:.1f}" y1="12" x2="{zx:.1f}" y2="96" stroke="#d6d3d1" stroke-width="0.6"/>')
    for i, (name, a, b) in enumerate(rows):
        y = 18 + (i + 0.5) * (78 / n)
        parts.append(f'<text x="2" y="{y + 1.5:.1f}" font-size="8" fill="#44403c">{esc(name)}</text>')
        parts.append(
            f'<line x1="{x(a):.1f}" y1="{y:.1f}" x2="{x(b):.1f}" y2="{y:.1f}" stroke="#a8a29e" stroke-width="1.4"/>'
        )
        parts.append(f'<circle cx="{x(a):.1f}" cy="{y:.1f}" r="2.6" fill="#57534e"/>')
        hot = "#c2410c" if i == 0 else "#57534e"
        parts.append(f'<circle cx="{x(b):.1f}" cy="{y:.1f}" r="2.6" fill="{hot}"/>')
        parts.append(
            f'<text x="{x(a):.1f}" y="{y - 3.2:.1f}" font-size="7" fill="#44403c" text-anchor="middle">{a:.1f}{esc(unit)}</text>'
        )
        parts.append(
            f'<text x="{x(b):.1f}" y="{y + 6.2:.1f}" font-size="7" fill="#44403c" text-anchor="middle">{b:.1f}{esc(unit)}</text>'
        )
    return figure(cap, f'<svg viewBox="0 0 180 100">{"".join(parts)}</svg>')


def chart_stacks(rows: list[tuple[str, list[tuple[str, float, bool]]]], cap: str) -> str:
    colors = ["#c2410c", "#44403c", "#57534e", "#78716c", "#a8a29e", "#d6d3d1", "#e7e5e4"]
    legend_done: list[tuple[str, str]] = []
    body = []
    bar_h = 16 if len(rows) <= 3 else 8
    for name, parts in rows:
        segs = []
        for i, (lab, value, hot) in enumerate(parts):
            color = "#c2410c" if hot else colors[min(i + 1, len(colors) - 1)]
            segs.append(f'<i style="width:{value:.2f}%;background:{color}"></i>')
            if lab and lab not in [item[0] for item in legend_done]:
                legend_done.append((lab, color))
        body.append(
            f'<div class="srow"><span class="sname">{esc(name)}</span>'
            f'<span class="sbar" style="height:{bar_h}mm">{"".join(segs)}</span></div>'
        )
    legend = "".join(f'<span><b style="background:{c}"></b>{esc(lab)}</span>' for lab, c in legend_done)
    return figure(cap, f'<div class="stacks">{"".join(body)}</div><div class="legend">{legend}</div>')


def chart_stack_one(parts: list[tuple[str, float, bool]], cap: str) -> str:
    return chart_stacks([("전체", parts)], cap)


def chart_scatter(points: list[tuple[str, float, float, bool]], cap: str) -> str:
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    xmin, xmax = 0, max(xs) * 1.15
    ymin, ymax = 0, max(ys) * 1.25

    def sx(v: float) -> float:
        return 22 + (v - xmin) / (xmax - xmin) * 148

    def sy(v: float) -> float:
        return 88 - (v - ymin) / (ymax - ymin) * 72

    parts = [
        f'<line x1="22" y1="88" x2="174" y2="88" stroke="#d6d3d1"/>',
        f'<line x1="22" y1="88" x2="22" y2="12" stroke="#d6d3d1"/>',
        '<text x="96" y="98" font-size="8" fill="#78716c" text-anchor="middle">가로 · 주차 몫 %</text>',
        '<text x="28" y="10" font-size="8" fill="#78716c">세로 · 난방 몫 %</text>',
    ]
    for lab, x, y, hot in points:
        color = "#c2410c" if hot else "#57534e"
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{color}"/>')
        if hot:
            parts.append(
                f'<text x="{sx(x) + 3:.1f}" y="{sy(y) - 2:.1f}" font-size="7" fill="#9a3412">{esc(lab)}</text>'
            )
    return figure(cap, f'<svg viewBox="0 0 180 100">{"".join(parts)}</svg>')


def chart_bignum(value: str, unit: str, other: str, other_v: str, cap: str) -> str:
    body = (
        f'<div class="bign"><p class="n">{esc(value)}</p>'
        f'<p class="cmp"><span>{esc(unit)}</span><span>{esc(other)} {esc(other_v)}</span></p></div>'
    )
    return figure(cap, body)


def chart_groups(measures: list[str], series: list[tuple[str, list[float], str]], cap: str) -> str:
    flat = [v for _, vals, _ in series for v in vals]
    hi = max(flat) * 1.15 or 1
    n = len(measures)
    parts = []
    for lab, _vals, color in series:
        parts.append(f'<text x="{150 if color == "#c2410c" else 40}" y="8" font-size="8" fill="{color}">{esc(lab)}</text>')
    # legend properly
    parts = []
    lx = 36
    for lab, _vals, color in series:
        parts.append(f'<circle cx="{lx}" cy="6" r="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 4}" y="8" font-size="8" fill="#44403c">{esc(lab)}</text>')
        lx += 48
    for i, name in enumerate(measures):
        y = 20 + (i + 0.5) * (74 / n)
        parts.append(f'<text x="2" y="{y + 1:.1f}" font-size="8" fill="#44403c">{esc(name)}</text>')
        parts.append(f'<line x1="36" y1="{y:.1f}" x2="172" y2="{y:.1f}" stroke="#e7e5e4" stroke-width="0.5"/>')
        for _lab, vals, color in series:
            x = 36 + vals[i] / hi * 136
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{y - 3.4:.1f}" font-size="6.5" fill="#44403c" text-anchor="middle">{vals[i]:.1f}</text>'
            )
    return figure(cap, f'<svg viewBox="0 0 180 100">{"".join(parts)}</svg>')


def cold_days() -> list[tuple[str, float]]:
    weather = json.loads((WEATHER).read_text(encoding="utf-8"))["daily"]
    wdf = pd.DataFrame({"day": pd.to_datetime(weather["time"]), "temp": weather["temperature_2m_mean"]})
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('" + STORY_PQ_SQL + "')")
    con.register("weather", wdf)
    df = con.execute(
        """
        WITH heat AS (
          SELECT CAST(created AS DATE) d, count(*) n
          FROM t WHERE problem = 'HEAT/HOT WATER' GROUP BY 1
        )
        SELECT CAST(w.day AS DATE) AS d, coalesce(h.n, 0) AS n
        FROM weather w
        LEFT JOIN heat h ON h.d = CAST(w.day AS DATE)
        WHERE w.temp < 0
          AND CAST(w.day AS DATE) NOT IN (DATE '2025-11-01', DATE '2026-10-31')
        ORDER BY 1
        """
    ).fetchdf()
    out = []
    for row in df.itertuples(index=False):
        label = "2/8" if str(row.d).startswith("2026-02-08") else ""
        out.append((label, float(row.n)))
    return out


def agency_parts(row: dict, hot: str = "hpd") -> list[tuple[str, float, bool]]:
    n = row["n"]
    parts = []
    used = 0
    for key, lab in AGENCY:
        share = 100 * row[key] / n
        used += row[key]
        parts.append((lab, share, key == hot))
    parts.append(("그 밖", 100 * (n - used) / n, False))
    return parts


def mix_row(block: dict) -> dict:
    return block


def chapters(cold: list[tuple[str, float]]) -> list[str]:
    j = NEW["july4"]
    fw = NEW["fireworks_july4_evening_to_july5_morning"]
    th = NEW["thanksgiving"]
    coldest = NEW["coldest"]
    snow = NEW["snow_parking"]
    heli = NEW["helicopter"]
    boro = {r["borough"]: r for r in POL["borough_agency"]}
    season = {r["borough"]: r for r in POL["season"]}
    enc = {r["borough"]: r for r in POL["encampment"]}
    phone = {r["borough"]: r for r in POL["phone"]}
    ast = POL["astoria"]
    bay = POL["bayside"]
    rest = POL["aoc_rest"]
    aoc_n = ast["n"] + rest["n"]
    aoc = {k: ast[k] + rest[k] for k in ("n", "nypd", "hpd", "dsny", "dot", "dpr", "dep")}

    def sh(block, key):
        return 100 * block[key] / block["n"]

    fw_bins = []
    for row in fw:
        name = f"{row['hr']}" if row["day"] == "2026-07-04" else f"+{row['hr']}"
        if row["hr"] in (21, 0, 1) or row["hr"] == 18:
            name = ("21시" if row["day"] == "2026-07-04" and row["hr"] == 21 else name)
            if row["day"] == "2026-07-05" and row["hr"] == 0:
                name = "0시"
            if row["day"] == "2026-07-05" and row["hr"] == 1:
                name = "1시"
            if row["day"] == "2026-07-04" and row["hr"] == 18:
                name = "18시"
        else:
            name = ""
        fw_bins.append((name, row["n"]))
    hot_fw = next(i for i, row in enumerate(fw) if row["hr"] == 21 and row["day"] == "2026-07-04")
    mark_fw = next(i for i, row in enumerate(fw) if row["hr"] == 1 and row["day"] == "2026-07-05")

    cold_hot = next(i for i, (lab, _v) in enumerate(cold) if lab == "2/8")
    heli_bins = [(str(h) if h % 3 == 0 else "", n) for h, n in enumerate(heli["by_hour"])]

    order = ["BRONX", "MANHATTAN", "BROOKLYN", "QUEENS", "STATEN ISLAND"]
    stacks = [(BORO[k], agency_parts(boro[k])) for k in order]
    place_stacks = [
        ("아스토리아", agency_parts(ast)),
        ("베이사이드", agency_parts(bay)),
    ]
    three = [
        ("브롱크스", agency_parts(boro["BRONX"])),
        ("퀸스 10개", agency_parts(aoc)),
        ("스태튼아일랜드", agency_parts(boro["STATEN ISLAND"])),
    ]

    def season_share(key, num, den):
        return 100 * season[key][num] / season[key][den]

    winter_summer = [
        (
            BORO[k],
            season_share(k, "winter_heat", "winter_n"),
            season_share(k, "summer_park", "summer_n"),
        )
        for k in order
    ]
    # Highlight Queens, the borough whose order flips, by placing it first.
    winter_summer = [winter_summer[3]] + winter_summer[:3] + winter_summer[4:]

    enc_rows = []
    for k in order:
        r = enc[k]
        enc_rows.append((BORO[k], 100 * r["before_nypd"] / r["before_n"], 100 * r["after_nypd"] / r["after_n"]))

    phone_rows = []
    for k in ["STATEN ISLAND", "BRONX", "BROOKLYN", "QUEENS", "MANHATTAN"]:
        r = phone[k]
        phone_rows.append((BORO[k], 100 * r["phone"] / r["n"], 100 * r["phone"] / r["known"]))

    zips = []
    for r in POL["aoc_zips"]:
        n = r["n"]
        heat = 100 * r["heat"] / n
        park = 100 * r["parking"] / n
        hot = r["zip"] in ("11372", "11103", "11356")
        zips.append((r["zip"], park, heat, hot))

    az_w, az_h, az_s, az_p = SEASON_ZIPS["astoria"]
    bz_w, bz_h, bz_s, bz_p = SEASON_ZIPS["bayside"]
    rz_w, rz_h, rz_s, rz_p = SEASON_ZIPS["rest"]

    sat = j["saturday_mean"]
    pages = [
        opener(
            "01", "건너뛴 날",
            "기존 질문집이 75배와 크리스마스로 채운 자리에, 그 책이 건너뛴 날만 둔다.",
            "7월 4일의 나머지는 토요일과 비슷하고, 가장 추운 하루의 난방은 다른 영하 날과 다르다.",
            [("1.11배", "불꽃을 뺀 7월 4일"), ("−14°C", "난방 6,118건"), ("49%", "눈 온 날의 주차")],
        ),
        page(
            "01 · 건너뛴 날", "Q01",
            "7월 4일, 불꽃을 빼면 평소 토요일보다 바빴나?",
            "불꽃을 빼면 1.11배다. 도시 전체가 들뜬 날이라기보다 불꽃 칸이 바쁜 날이다.",
            chart_columns(
                [("토요일", sat), ("7월 4일", j["all"]), ("불꽃 제외", j["without_fireworks"])],
                hot=2,
                cap="평소 토요일 50일과 2026-07-04. 축은 하루 신고 건수.",
            ),
            [("1.34배", "불꽃 포함"), ("1.11배", "불꽃 제외"), ("2.5배", "길 소음"), ("66%", "주차")],
            [
                "공식 불꽃은 맨해튼 강변 중계로 보인다.",
                "311의 불꽃 칸은 집 앞의 불법 불꽃이다.",
                "길 소음은 평소 토요일의 2.5배, 집 안 소음은 88%다.",
                "주차는 평소 토요일보다 적었다.",
            ],
            "AP 유튜브, 7월 4일 직전 브리핑. 시장은 공식 행사와 폭염 때 311 전화를 말했다.",
            "비교는 다른 토요일 50일의 평균. 첫날, 마지막 날, 7월 4일은 평균에서 뺐다.",
        ),
        page(
            "01 · 건너뛴 날", "Q02",
            "자정을 넘긴 불꽃은 어느 날 장부에 남나?",
            "7월 5일 0시부터 3시까지 965건이다. 전날 밤 9시의 739건보다 많다.",
            chart_columns(fw_bins, hot=hot_fw, mark=mark_fw, cap="7월 4일 18시부터 7월 5일 8시. 주황은 21시, 테두리는 다음 날 1시."),
            [("739건", "4일 21시"), ("965건", "5일 0–3시"), ("1,688건", "5일 하루"), ("2,257건", "4일 하루")],
            [
                "방송은 7월 4일 밤에 끝난다.",
                "장부의 날짜는 자정에 넘어간다.",
                "하루짜리 배수에는 다음 날 새벽이 빠진다.",
                "주황 테두리는 다음 날 1시의 442건이다.",
            ],
            "같은 AP 브리핑은 공식 쇼를 말한다. 이 칸은 그 시각 전후의 불법 불꽃 신고다.",
            "막대는 그 시각에 접수된 불법 불꽃 건수. 하루 합계와 한 축에 두지 않았다.",
        ),
        page(
            "01 · 건너뛴 날", "Q03",
            "7월 4일 불꽃 신고는 맨해튼 하늘만의 일인가?",
            "건수는 퀸스 618, 브루클린 573, 맨해튼 353이다. 화면의 장소와 민원의 장소가 다르다.",
            chart_columns(
                [("퀸스", 618), ("브루클린", 573), ("브롱크스", 432), ("맨해튼", 353), ("스태튼", 281)],
                hot=0,
                cap="2026-07-04 불법 불꽃 2,257건. 인구로 나누지 않은 건수.",
            ),
            [("618", "퀸스"), ("573", "브루클린"), ("432", "브롱크스"), ("353", "맨해튼")],
            [
                "방송 화면은 강변과 다리 쪽 하늘이다.",
                "신고는 집 앞에서 접수된다.",
                "퀸스가 인구가 많아 건수가 많을 수 있다.",
                "이 쪽은 일부러 건수만 보여 준다.",
            ],
            "시의 독립기념일 중계는 맨해튼 강변이다. 민원 지도는 그 화면과 다른 자치구를 가리킨다.",
            "분모는 그날 불법 불꽃 2,257건. 1인당 비율로 바꾸지 않았다.",
        ),
        page(
            "01 · 건너뛴 날", "Q04",
            "더운 날의 불꽃이 22배라고 말해도 되나?",
            "그 더운 날 평균 119건 안에 7월 4일 2,257건이 들어 있다. 기온의 배수가 아니다.",
            chart_columns(
                [("7월 4일", 2257), ("더운 날 평균", 119), ("영하인 날", 5.5)],
                hot=0,
                cap="더운 날은 일평균 25도 이상. 영하인 날과 축을 공유한다.",
            ),
            [("119건", "더운 날 평균"), ("2,257건", "그중 하루"), ("5.5건", "영하인 날"), ("39일", "25도 이상")],
            [
                "7월 4일의 기온은 29.5도로 그 구간에 들어간다.",
                "명절 하루가 구간 평균을 차지한다.",
                "이 책은 그 22배를 더위의 효과로 쓰지 않는다.",
                "큰 하루를 빼 보면 평균이 누구의 것인지 보인다.",
            ],
            "날씨 코너가 더위와 불꽃을 한 문장에 넣을 때, 독립기념일이 그 안에 있는지 먼저 본다.",
            "더운 날 39일, 영하인 날 59일. 7월 4일은 더운 날 쪽에 포함된 채 평균에 들어 있다.",
        ),
        page(
            "01 · 건너뛴 날", "Q05",
            "추수감사절의 집 안 소음은 명절의 숫자인가?",
            "한 주소를 빼면 1,121건, 다른 겨울 목요일의 1.46배다. 주차는 오히려 71%다.",
            chart_dumbbell(
                [("집 안 소음", 770, 1121), ("주차", 1696, 1202)],
                "다른 목요일",
                "11월 27일",
                "왼쪽은 주소를 뺀 겨울 목요일 10일의 평균. 오른쪽은 추수감사절.",
                unit="",
            ),
            [("2,522", "소음 전체"), ("1,401", "한 주소"), ("1.46배", "주소 제외"), ("71%", "주차")],
            [
                "추수감사절은 2025년 11월 27일 목요일이다.",
                "집이 모이는 날이라 층간 소음이 기사 소재가 된다.",
                "그날 소음 2,522건 중 1,401건은 한 주소다.",
                "도로명은 적지 않는다. 우편번호 10466의 그 주소다.",
            ],
            "명절 식탁 기사는 집 안을 보여주고, 주차 기사는 한산한 길을 보여 준다. 장부에서는 둘의 부호가 반대다.",
            "비교 목요일은 11월 6일부터 1월 29일, 11월 27일·크리스마스·1월 1일 제외. 그 주소도 뺐다.",
        ),
        page(
            "01 · 건너뛴 날", "Q06",
            "가장 추운 날, 난방 신고는 다른 영하 날의 몇 배인가?",
            "2월 8일, 영하 14도, 난방 6,118건. 다른 영하 58일의 2.36배다.",
            chart_columns(cold, hot=cold_hot, cap="영하인 날의 난방 건수. 주황은 2026-02-08. 첫날과 마지막 날은 뺐다."),
            [("6,118", "2월 8일"), ("2,591", "다른 영하 날"), ("149", "20도 이상"), ("−14°C", "일평균기온")],
            [
                "난방 민원은 겨울 기사마다 나오지만 영하인 날이 다 같지는 않다.",
                "이날 도시 전체 신고는 14,514건이다.",
                "20도 이상 125일의 난방은 하루 149건이다.",
                "한 날짜의 배수이지, 추위 정책의 성적은 아니다.",
            ],
            "한파 브리핑은 가장 추운 아침을 보여 준다. 이 칸은 그 하루를 다른 영하 날 옆에 놓는다.",
            "기온은 Open-Meteo 일평균. 평균 2,591건은 2월 8일을 뺀 영하 58일이다.",
        ),
        page(
            "01 · 건너뛴 날", "Q07",
            "눈이 내리던 날, 불법주차는 같이 늘었나?",
            "2월 23일 주차는 평소 월요일의 49%였다. 다음 날 눈 민원은 더 늘고 주차는 94%로 돌아왔다.",
            chart_pair(
                [
                    ("눈·얼음 민원", [("23일", 6524), ("24일", 11370)], 1),
                    ("불법주차", [("월요일", 1743), ("23일", 848), ("화요일", 1782), ("24일", 1669)], 1),
                ],
                "왼쪽은 눈 민원 건수. 오른쪽은 주차. 축을 나누었다. 주황은 눈 민원이 더 많은 24일과, 주차가 줄어든 23일.",
            ),
            [("49%", "23일 주차"), ("94%", "24일 주차"), ("6,524", "23일 눈"), ("11,370", "24일 눈")],
            [
                "23일 적설은 12.9cm, 24일 적설은 0cm다.",
                "눈 민원은 24일에 더 많다. 눈이 그친 뒤의 신고다.",
                "주차는 차를 빼는 사람의 신고라 시계가 다르다.",
                "민원 총량을 제설 성적표로 읽지 않는다.",
            ],
            "폭설 다음 날 브리핑은 민원 건수를 성적처럼 말한다. 주차와 눈은 같은 방향으로 움직이지 않았다.",
            "주차 비교는 다른 월요일 48일, 다른 화요일 50일. 공휴일로 쉬는 월·화는 평균에서 뺐다.",
        ),
        page(
            "01 · 건너뛴 날", "Q08",
            "헬기 소음 신고는 한밤중인가, 해 질 무렵인가?",
            "1년 10,971건 중 15–19시가 4,509건, 41%다. 한 시각의 정점은 16시다.",
            chart_columns(heli_bins, hot=16, cap="0시부터 23시. 숫자는 3시간마다. 주황은 16시, 1,097건."),
            [("41%", "15–19시"), ("213건", "0–5시"), ("1,097", "16시"), ("10,971", "1년")],
            [
                "허드슨 강과 맨해튼 남단 헬기는 오래된 지역 뉴스다.",
                "이 장부에는 운항 횟수가 없고 신고만 있다.",
                "한밤중 0–5시는 213건이다.",
                "16시의 1,097건은 365일에 걸친 그 시각의 합이다.",
            ],
            "지역 뉴스는 낮의 강변 헬기를 찍는다. 신고의 시각도 해 질 무렵에 모인다.",
            "문제 유형은 Noise - Helicopter. 장소로 나누지 않고 시각만 세었다.",
        ),
        page(
            "01 · 건너뛴 날", "Q09",
            "난방 신고는 건물 전체인가, 그 집만인가?",
            "352,278건 중 건물 전체가 65.6%다. 겨울 기사가 보일러 하나를 찍는 장면과 같은 칸이다.",
            chart_stack_one(
                [("건물 전체", 65.6, True), ("그 세대만", 34.4, False)],
                "HEAT/HOT WATER. 두 조각의 합이 100이다.",
            ),
            [("231,208", "건물 전체"), ("121,070", "그 세대만"), ("65.6%", "건물 몫"), ("352,278", "분모")],
            [
                "세입자가 겨울에 주택국 장부에 남기는 기록이다.",
                "접수 칸이 건물 전체인지 그 세대만인지를 적는다.",
                "66%는 건물 수가 아니라 신고 건수의 몫이다.",
                "자치구 순위로 다시 그리지 않는다.",
            ],
            "난방이 끊긴 건물 꼭지는 현관과 복도를 보여 준다. 장부에서도 건물 전체 칸이 더 크다.",
            "세부 유형 ENTIRE BUILDING과 APARTMENT ONLY. 다른 칸은 이 둘뿐이었다.",
        ),
        opener(
            "02", "신고 다음",
            "경찰이 남긴 문장과, 2026년 9월 22일에 바뀐 텐트 신고의 접수처.",
            "건수는 거의 그대로다. 바뀐 것은 어느 기관의 장부에 적히는지다.",
            [("13.1%", "딱지 문장"), ("73%→3%", "텐트의 경찰 몫"), ("다섯 곳", "같은 방향")],
        ),
        page(
            "02 · 신고 다음", "Q10",
            "경찰이 딱지를 발부했다고 적은 비율은?",
            "결과 문장 1,720,638건 중 13.1%에 “issued a summons”가 있다.",
            chart_stack_one(
                [("딱지 문장", 13.1, True), ("그 문구 없음", 86.9, False)],
                "NYPD 결과 문장. 소환장이라는 낱말만 센 비율이 아니다.",
            ),
            [("224,738", "딱지 문장"), ("13.1%", "몫"), ("172만", "결과 문장"), ("0건", "증거없음과 겹침")],
            [
                "소환장은 체포가 아니라 현장 벌금 쪽의 딱지다.",
                "문장에는 발부할 필요가 없었다가 많이 들어 있다.",
                "낱말만 세면 딱지가 부풀어 오른다.",
                "이웃 소음만의 딱지 확률은 이 13.1%가 아니다.",
            ],
            "현장 정리 기사 속의 딱지는 한 장면이다. 이 쪽은 경찰 문장 전체의 정의만 보여 준다.",
            "문자열은 issued a summons. 증거 없음 문구와의 교집합은 0건이다.",
        ),
        page(
            "02 · 신고 다음", "Q11",
            "9월 22일, 텐트 신고는 어디로 갔나?",
            "이전 325일은 하루 130건, 이후 40일도 하루 129건이다. 경찰 몫만 73%에서 3%로 옮겼다.",
            chart_dumbbell(
                [("경찰 몫", 73.4, 3.1)],
                "9월 22일 전",
                "9월 22일 후",
                "시 전체 텐트 신고. 하루 건수는 130에서 129라 그림에 넣지 않았다.",
                unit="%",
            ),
            [("42,222", "이전"), ("5,147", "이후"), ("130건", "이전 하루"), ("129건", "이후 하루")],
            [
                "DHS는 노숙인 서비스국이다.",
                "아웃리치는 경찰 대신 그 기관의 현장팀이 접수한다는 뜻이다.",
                "이 날짜는 민원 창의 마지막 40일이라 계절을 가리지 못한다.",
                "텐트가 줄었다는 문장은 쓰지 않는다.",
            ],
            "amNewYork, 2026-09-25. 시는 9월 22일부터 텐트 311을 DHS 아웃리치로 보낸다고 확인했다.",
            "전후 비교는 그 날짜에 보인 접수 기관의 차이다. 정책이 텐트를 줄였다는 뜻이 아니다.",
        ),
        page(
            "02 · 신고 다음", "Q12",
            "접수처 변경은 다섯 자치구에서 같이 일어났나?",
            "다섯 곳 모두 경찰 몫이 5% 아래로 떨어졌다. 이후 건수의 64%는 맨해튼이다.",
            chart_dumbbell(enc_rows, "9/22 전", "9/22 후", "자치구별 텐트 신고 중 NYPD 몫.", unit="%"),
            [("64%", "이후의 맨해튼"), ("3,305", "맨해튼 이후"), ("5,147", "이후 합"), ("5% 아래", "다섯 곳")],
            [
                "방향은 다섯 자치구가 같다.",
                "크기는 맨해튼에 몰려 있다.",
                "이전 자치구 합은 시 전체보다 9건 적다. 몫은 자치구 안에서 계산했다.",
                "선거 이야기로 잇지 않는다.",
            ],
            "같은 amNewYork 확인은 시의 접수 규칙이다. 자치구 의회가 각각 정한 규칙이 아니다.",
            "이후는 2026-09-22 이상. 스태튼아일랜드 이후는 43건이라 비율의 폭이 넓다.",
        ),
        opener(
            "03", "표가 있는 곳",
            "2025년 11월 4일 시장 선거와 311을 한 지도에 놓되, 한 사람의 표와 한 사람의 신고로 옮기지 않는다.",
            "이긴 네 자치구는 한 기관이 아니고, 퀸스 안에서도 주차는 닮고 주택 민원은 갈린다.",
            [("두 줄", "맘다니 50.8%"), ("36%→12%", "이긴 곳의 주택국"), ("24.8·23.0", "주차는 붙어 있다")],
        ),
        page(
            "03 · 표가 있는 곳", "Q13",
            "맘다니 50.8%는 몇 줄인가?",
            "민주당 944,950표는 쿠오모보다 이미 38,336표 많다. 노동가족당 169,234표는 역전표가 아니라 둘째 줄이다.",
            chart_stacks(
                [
                    ("맘다니", [("민주당", 944950 / 11141.84, True), ("노동가족당", 169234 / 11141.84, False)]),
                    ("쿠오모", [("다른 후보", 906614 / 11141.84, False)]),
                    ("슬리와", [("다른 후보", 153749 / 11141.84, False)]),
                ],
                "막대 길이는 표수. 100%가 아니다. 맘다니 합계 1,114,184표를 기준으로 길이를 맞췄다.",
            ),
            [("944,950", "민주당"), ("169,234", "노동가족당"), ("1,114,184", "후보 합계"), ("906,614", "쿠오모")],
            [
                "조란 맘다니는 아스토리아 주의회 의원 출신으로 민주당과 노동가족당에 함께 올랐다.",
                "앤드루 쿠오모는 전 뉴욕주 지사로, 이 선거에서는 여러 줄의 합이 906,614표다.",
                "커티스 슬리와는 공화당 후보, 153,749표.",
                "뉴욕은 한 후보가 둘 이상의 정당 줄에 오를 수 있다.",
            ],
            "선거관리위원회 2025-11-04 시장 선거 확정 개표. 50.8%는 후보를 합친 몫이다.",
            "민주당 한 줄만으로도 쿠오모의 줄 합보다 크다. 나머지 0.9포인트의 표수는 만들지 않았다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q14",
            "이긴 네 자치구의 민원은 같은 기관으로 가나?",
            "주택국 몫은 브롱크스 36%, 맨해튼 27%, 브루클린 22%, 퀸스 12%다. 한 줄이 아니다.",
            chart_stacks(stacks, "주황은 주택국. 타임은 맘다니가 네 자치구에서, 쿠오모가 스태튼아일랜드에서 이겼다고 썼다."),
            [("36%", "브롱크스 주택"), ("12%", "퀸스 주택"), ("22%", "스태튼 청소"), ("32%", "스태튼 경찰")],
            [
                "타임 2025-11-05의 승패는 자치구 단위다. 자치구 득표율은 적지 않는다.",
                "스태튼아일랜드의 둘째 기관은 주택국이 아니라 청소국 22%다.",
                "그 자치구의 경찰 몫 32%는 다섯 곳 중 가장 낮다.",
                "승패의 순서와 기관의 순서는 같지 않다.",
            ],
            "타임, 2025-11-05. 페이지에는 2026-10-19 수정 표시가 있다. 게재일로 인용한다.",
            "몫은 그 자치구 민원 안의 기관 구성. 한 사람의 표와 한 사람의 신고가 아니다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q15",
            "보도가 가른 두 퀸스는 어느 기관에서 갈리나?",
            "주차는 24.8%와 23.0%로 붙어 있다. 주택국은 17.5%와 4.3%로 갈린다.",
            chart_dumbbell(
                [
                    ("주택국", sh(ast, "hpd"), sh(bay, "hpd")),
                    ("경찰", sh(ast, "nypd"), sh(bay, "nypd")),
                    ("청소", sh(ast, "dsny"), sh(bay, "dsny")),
                    ("교통", sh(ast, "dot"), sh(bay, "dot")),
                    ("공원", sh(ast, "dpr"), sh(bay, "dpr")),
                ],
                "아스토리아",
                "베이사이드",
                "왼쪽 80,648건, 오른쪽 24,211건. 첫 줄 주택국만 주황 끝점.",
                unit="%",
            ),
            [("24.8%", "아스토리아 주차"), ("23.0%", "베이사이드 주차"), ("17.5%", "아스토리아 주택"), ("4.3%", "베이사이드 주택")],
            [
                "아스토리아는 퀸스 서부, 베이사이드는 퀸스 북동부다.",
                "타임이 적은 약 66%와 25포인트는 동네 기사이지 이 우편번호의 득표율이 아니다.",
                "퀸스 전체는 그 보도에서 이긴 쪽에 있다. 진 자치구와의 비교가 아니다.",
                "우편번호는 선거구 경계가 아니다.",
            ],
            "타임 2025-11-05, QNS 2025-11-05. 서부 퀸스와 동부 퀸스가 갈렸다는 동네 보도다.",
            "주차 몫은 거의 누운 기준이다. 이 그림의 기울기는 기관 몫이다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q16",
            "NY-14 퀸스 우편번호 열 개는 한 곳인가?",
            "열 개의 평균은 난방 5.8%, 주차 21%다. 11372는 난방 12.6%이고 11356은 1.1%다.",
            chart_scatter(zips, "가로 주차, 세로 난방. 주황은 11372, 11103, 11356."),
            [("12.6%", "11372 난방"), ("32.7%", "11103 주차"), ("1.1%", "11356 난방"), ("17.6%", "11368 진입로")],
            [
                "알렉산드리아 오카시오코르테스는 연방 하원 NY-14 의원이다.",
                "의원실 안내는 이 지역구가 퀸스 북부와 브롱크스 동부를 함께 넣는다고 적는다.",
                "이 열 개는 그 경계가 아니라 퀸스 쪽 근사다.",
                "11368은 진입로 17.6%가 주차 16.5%보다 크다.",
            ],
            "지역구 안내는 두 자치구를 한 구역으로 부른다. 민원 구성은 그 호칭 안에서 갈린다.",
            "점 열 개는 우편번호별 민원 안의 몫. 동네 이름은 더 붙이지 않는다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q17",
            "브롱크스를 섞으면 누구의 평균이 되나?",
            "퀸스 열 개의 주택국은 15%, 브롱크스는 36%다. 한 평균으로 넣으면 서로의 장부를 지운다.",
            chart_stacks(three, "주황은 주택국. 퀸스 10개는 아스토리아 네 개와 나머지 여섯 개의 합, 206,103건."),
            [("36%", "브롱크스 주택"), ("15%", "퀸스 10 주택"), ("52%", "퀸스 10 경찰"), ("22%", "스태튼 청소")],
            [
                "NY-14의 브롱크스 쪽을 이 열 개와 섞지 않는 이유다.",
                "스태튼아일랜드는 그 반대 극이 아니다.",
                "주택국은 낮지만 둘째 조각은 청소국이다.",
                "지역구 평균은 어느 쪽의 장부도 아니다.",
            ],
            "의원실이 두 자치구를 한 지역구로 적는 것과, 민원을 한 평균으로 적는 것은 다른 일이다.",
            "세 막대는 각자 자기 민원 안의 기관 몫. 인구당이 아니다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q18",
            "겨울의 몫과 여름의 몫은 같은 순서인가?",
            "겨울 난방 1위는 브롱크스 28.8%다. 여름 주차 1위는 퀸스 19.3%다. 순서가 바뀐다.",
            chart_dumbbell(
                winter_summer,
                "겨울 난방",
                "여름 주차",
                "그 계절 민원 안의 몫. 첫 줄 퀸스만 오른쪽 점이 주황.",
                unit="%",
            ),
            [("28.8%", "브롱크스 겨울"), ("19.3%", "퀸스 여름"), ("13.4%", "아스토리아 겨울"), ("4.8%", "베이사이드 겨울")],
            [
                "겨울은 12·1·2월, 여름은 6·7·8월이다.",
                "이 겨울은 선거 다음 겨울이고 이 여름은 2026년 여름이다.",
                "아스토리아 여름 주차는 25.0%, 베이사이드는 20.6%다.",
                "브롱크스 겨울 분모에서 10466의 한 주소를 빼도 난방 1위는 그대로다.",
            ],
            "계절 기사는 겨울의 난방과 여름의 주차를 따로 쓴다. 같은 자치구의 순서가 바뀌는지는 잘 안 적는다.",
            "10466은 한 주소, 47,105건, 주거지 소음 47,013건, 시 전체 주거지 소음의 11.3%. 도로명은 적지 않는다.",
        ),
        page(
            "03 · 표가 있는 곳", "Q19",
            "같은 퀸스에서 길의 민원은 무엇인가?",
            "베이사이드는 가로 파손 9.1%, 나무 4.0%, 버려진 차 4.8%다. 아스토리아의 그 셋은 3% 안팎이다.",
            chart_groups(
                ["가로 파손", "나무 피해", "버려진 차", "진입로"],
                [
                    ("아스토리아", [sh(ast, "street"), sh(ast, "tree"), sh(ast, "car"), sh(ast, "driveway")], "#57534e"),
                    ("베이사이드", [sh(bay, "street"), sh(bay, "tree"), sh(bay, "car"), sh(bay, "driveway")], "#c2410c"),
                    ("나머지 여섯", [sh(rest, "street"), sh(rest, "tree"), sh(rest, "car"), sh(rest, "driveway")], "#a8a29e"),
                ],
                "몫 %. 주황은 베이사이드. 주차 24.8과 23.0은 이 축에 넣지 않았다.",
            ),
            [("9.1%", "베이사이드 가로"), ("7.3%", "아스토리아 진입로"), ("12.0%", "나머지 진입로"), ("7.6%", "아스토리아 집 안 소음")],
            [
                "신문이 득표로 가른 두 퀸스는 주차 몫에서는 닮는다.",
                "집 앞의 다른 민원에서는 다르다.",
                "나머지 여섯 개의 둘째 민원은 진입로 15,040건이다.",
                "66%와 25포인트를 이 우편번호에 붙이지 않는다.",
            ],
            "QNS가 적은 서부와 동부의 갈림은 표 이야기다. 이 그림은 그 우편번호 장부의 길거리 민원이다.",
            "가로 파손은 Street Condition, 나무는 Damaged Tree, 차는 Abandoned Vehicle.",
        ),
        page(
            "03 · 표가 있는 곳", "Q20",
            "이 장에서 쓰지 않는 문장은 무엇인가?",
            "주차 몫은 거의 눕고, 주택국과 겨울 난방은 기운다. 그 기울기를 표의 이유로 잇지 않는다.",
            chart_groups(
                ["주차 몫", "주택국 몫", "겨울 난방"],
                [
                    ("아스토리아", [sh(ast, "parking"), sh(ast, "hpd"), 100 * az_h / az_w], "#57534e"),
                    ("베이사이드", [sh(bay, "parking"), sh(bay, "hpd"), 100 * bz_h / bz_w], "#c2410c"),
                ],
                "세 지표는 분모가 다르다. 한 축에 놓은 것은 기울기를 나란히 보기 위해서다.",
            ),
            [("24.8→23.0", "주차"), ("17.5→4.3", "주택국"), ("13.4→4.8", "겨울 난방"), ("쓰지 않음", "표가 민원을 만들었다")],
            [
                "진보 득표가 높아서 민원이 많다.",
                "민원이 많아서 그가 이겼다.",
                "임대료를 얼리면 난방 신고가 줄고 표가 오른다.",
                "이 우편번호의 득표율이 66%이거나 25포인트다.",
            ],
            "없는 것은 사람 단위로 붙은 표와 민원, 그리고 정책을 무작위로 배정한 비교다.",
            "민원 창은 선거 사흘 전부터 이듬해 10월까지다. 캠페인 기간도, 취임 첫해의 성적표도 아니다.",
        ),
        opener(
            "04", "분모",
            "같은 파일에서 1위와 8시간이 바뀌는 자리만 모은다.",
            "빈 칸, 자정에 닫힌 기록, 기관마다 다른 시계. 평균은 그 시계를 한 도시에 섞은 결과다.",
            [("8시간", "가운데 민원"), ("30%", "빈 칸을 빼면"), ("−0.7p", "맞춘 뒤의 채널")],
        ),
        page(
            "04 · 분모", "Q21",
            "채널이 비어 있는 33만 건은 어느 기관의 기록인가?",
            "빈 칸 332,035건의 54%는 교통국, 35%는 건물국이다. 그 칸을 분모에 두면 전화 1위가 바뀐다.",
            chart_dumbbell(
                phone_rows,
                "빈 칸 포함",
                "빈 칸 제외",
                "전화 몫 %. 첫 줄 스태튼아일랜드의 오른쪽 점이 주황. 빈 칸을 빼면 그 자치구가 30%로 앞선다.",
                unit="%",
            ),
            [("33.2만", "UNKNOWN"), ("54%", "교통국"), ("35%", "건물국"), ("375건", "경찰의 빈 칸")],
            [
                "빈 칸을 넣으면 전화 1위는 브롱크스 26.1%다.",
                "빼면 스태튼아일랜드 30.1%, 브롱크스 27.6%다.",
                "10466의 한 주소가 이 순서를 가른 것은 아니다.",
                "교통국과 건물국 기록이 채널을 비워 둔 탓이 크다.",
            ],
            "전화 민원을 자치구 성격처럼 말하는 기사는 빈 칸을 분모에 뒀는지 적지 않는다.",
            "UNKNOWN 332,035건. 알려진 채널만 분모로 둔 전화 몫이 오른쪽 점이다.",
        ),
        page(
            "04 · 분모", "Q22",
            "도시의 8시간은 어느 기관의 한가운데인가?",
            "경찰의 가운데는 1.4시간이다. 평균 6.6일을 끌어올리는 꼬리는 주택국과 공원국 쪽에 있다.",
            chart_pair(
                [
                    ("중앙값 · 시간", [("경찰", 1.4), ("환경", 22.1), ("청소", 39.9), ("교통", 54.0), ("주택", 100), ("공원", 116)], 0),
                    ("평균 · 일", [("경찰", 0.11), ("환경", 4.1), ("청소", 3.2), ("교통", 8.3), ("주택", 14.1), ("공원", 20.5)], 5),
                ],
                "두 그림의 축은 다르다. 왼쪽 주황은 경찰의 한가운데, 오른쪽 주황은 공원국 평균.",
            ),
            [("1.4시간", "경찰 중앙값"), ("4.2일", "주택 중앙값"), ("20.5일", "공원 평균"), ("130건", "경찰 7일 초과")],
            [
                "도시의 8시간은 이 여섯 시계를 한 파일에 섞은 가운데다.",
                "난방·온수만의 중앙값은 1.8일, 평균은 2.0일로 붙어 있다.",
                "긴 꼬리는 난방 종류가 아니라 주택국 전체와 공원국이다.",
                "경찰이 7일을 넘긴 건은 130건뿐이다.",
            ],
            "처리 속도 기사는 평균 일수를 쓴다. 기관을 나누지 않으면 경찰의 빠른 시계가 평균을 숨긴다.",
            "닫힘이고 소요 시간이 0 이상인 민원. 경찰 7일 초과는 1,720,469건 중 130건.",
        ),
        page(
            "04 · 분모", "Q23",
            "자정에 닫힌 기록을 빼면 8시간은 남나?",
            "110,698건, 2.9%를 빼면 중앙값은 8.0시간에서 6.9시간으로 준다. 가운데는 여전히 하루 안이다.",
            chart_columns(
                [("빼기 전", 8.0), ("자정 제외", 6.9)],
                hot=1,
                cap="중앙값, 시간. 평균은 아래 숫자. 6.62일에서 5.98일로 준다.",
            ),
            [("110,698", "00:00:00"), ("2.9%", "닫힌 민원 중"), ("6.62일", "빼기 전 평균"), ("5.98일", "뺀 평균")],
            [
                "시각이 정확히 자정인 마감은 일괄 처리의 자국일 수 있다.",
                "빼기 전 중앙값은 0.33일, 약 8시간이다.",
                "평균은 6.62일에서 5.98일로 준다.",
                "8시간이라는 문장은 자정을 빼지 않아도 남는다.",
            ],
            "처리 시간 보도는 마감 시각이 어떻게 찍히는지 적지 않을 때가 많다.",
            "닫힌 시각이 접수 이후인 3,784,880건. 상태만 Closed인 건수와는 991건 차이.",
        ),
        page(
            "04 · 분모", "Q24",
            "처리시간의 평균과 가운데는 왜 다른가?",
            "가운데는 8시간, 평균은 6.6일이다. 절반은 하루 안에 닫히고 일부는 몇 달이다.",
            chart_columns(
                [("중앙값", 8), ("평균", 159)],
                hot=0,
                cap="시간으로 환산. 평균 6.62일은 약 159시간.",
            ),
            [("0.33일", "중앙값"), ("6.62일", "평균"), ("378만", "분모"), ("기관별", "앞쪽")],
            [
                "평균과 중앙값은 다른 질문이다.",
                "오른쪽 꼬리가 길면 평균이 가운데보다 커진다.",
                "어느 쪽이 틀린 계산이 아니다.",
                "꼬리가 어느 기관인지는 앞 쪽이 보여 준다.",
            ],
            "시가 평균 처리시간을 발표하면 긴 꼬리가 그 숫자를 가져간다.",
            "상태는 Closed이고 닫힌 시각이 접수 이후인 민원. 자정 일괄 마감은 이 쪽에서 빼지 않았다.",
        ),
        page(
            "04 · 분모", "Q25",
            "200건만 보면 그 8시간은 얼마나 흔들리나?",
            "브루클린 닫힌 민원 200건의 중앙값 구간 너비는 16.7시간이다. 2만 건이면 1.3시간이다.",
            chart_columns(
                [("200건", 16.7), ("2천 건", 8.6), ("2만 건", 1.3)],
                hot=0,
                cap="95% 구간 너비, 시간. 복원추출.",
            ),
            [("16.7시간", "200건"), ("8.6시간", "2,000건"), ("1.3시간", "2만 건"), ("브루클린", "뽑은 곳")],
            [
                "부트스트랩은 그 표본을 다시 뽑아 추정량이 얼마나 움직이는지 본다.",
                "여론조사의 오차범위와 구조만 같다.",
                "내년 뉴욕의 처리 속도에 대한 오차범위는 아니다.",
                "파일 전체의 가운데는 거의 움직이지 않는다.",
            ],
            "작은 표본으로 도시의 속도를 말하는 문장은 이 너비 안에 있다.",
            "브루클린의 닫힌 민원에서 200, 2,000, 2만 건을 복원추출한 중앙값의 구간 너비.",
        ),
        page(
            "04 · 분모", "Q26",
            "채널이 모바일이면 더 빨리 닫히나?",
            "겉으로는 7일 안에 닫힌 비율이 26.8포인트 높았다. 자치구·기관·종류·달을 맞추면 −0.7포인트다.",
            chart_dumbbell(
                [("모바일−전화", 26.8, -0.7)],
                "맞추기 전",
                "맞춘 뒤",
                "퍼센트포인트. 0에 세로선. 오른쪽 점이 이 질문의 답.",
                unit="",
            ),
            [("+26.8p", "맞추기 전"), ("−0.7p", "맞춘 뒤"), ("7일", "기준"), ("구성", "채널이 아니라")],
            [
                "앱을 쓰면 행정이 빨라진다는 문장이 아니다.",
                "앱으로 들어오는 민원의 구성이 달랐다.",
                "−0.7이 이 파일 안에서 0과 떨어져 있어도 효과로 옮기지 않는다.",
                "채널을 바꾼 실험이 이 장부에는 없다.",
            ],
            "디지털 민원 홍보는 채널을 성과처럼 말한다. 겉차이는 거의 전부 민원 구성이었다.",
            "조정은 두 겹으로 나눠, 한쪽에서 맞춘 예측을 다른 쪽에 썼다.",
        ),
        page(
            "04 · 분모", "Q27",
            "신고는 결국 닫히나?",
            "95.5%는 상태 칸이 Closed라는 뜻이다. 문제가 해결됐다는 뜻이 아니다.",
            chart_stack_one(
                [("Closed", 95.5, True), ("In Progress", 2.6, False), ("Open", 1.4, False), ("그 밖", 0.5, False)],
                "3,963,959건의 상태 코드.",
            ),
            [("95.5%", "Closed"), ("3,785,871", "닫힘"), ("178,088", "그 밖"), ("문장", "해결이 아님")],
            [
                "이 파일은 진행 중인 현재가 아니라 이미 끝난 신고가 대부분이다.",
                "딱지 문장과 텐트 접수처가 그 차이를 보여 준다.",
                "닫힘과 해결은 다른 변수다.",
                "처리율이라는 홍보 문구와 상태 코드는 다르다.",
            ],
            "시정 홍보의 처리율은 이 상태 칸을 쓸 때가 있다. 칸의 이름은 행정이 붙인다.",
            "Closed 3,785,871건. 상태 코드의 구성이지 결과 문장의 구성이 아니다.",
        ),
    ]
    return pages


def front(index_rows: list[tuple[str, str, str]]) -> list[str]:
    entries = "".join(
        f'<div class="entry"><span>{esc(qid)}</span><span>{esc(title)}</span><b>{esc(val)}</b></div>'
        for qid, title, val in index_rows
    )
    return [
        f"""
<section class="sheet">
  <div class="cover">
    <p class="kicker">NYC 311 · 2025년 11월 – 2026년 10월</p>
    <p class="hero">396만</p>
    <h1>신고는 일기다</h1>
    <p class="dek">기존 백 개 질문집이 이미 센 질문은 다시 만들지 않았다. 그 책이 건너뛴 날, 자정을 넘는 시각, 그리고 2025년 시장 선거의 지도만 펼친다.</p>
    <p class="meta">자료  NYC Open Data 311 · 기온  Open-Meteo · 인구  2020년 센서스 · 득표  뉴욕시 선거관리위원회 확정 개표</p>
  </div>
  <p class="folio"></p>
</section>
""",
        f"""
<section class="sheet">
  <header><span class="kicker">읽는 법</span><span class="qid">00</span></header>
  <h2 class="q">가운데 그림이 답이다</h2>
  <p class="lead">글은 그림의 캡션이다. 주황은 그 질문이 가리키는 값 하나에만 쓴다.</p>
  {figure("한 쪽의 뼈대", '<div class="wire"><div>질문 한 줄과, 답 한 문장</div><div class="tall">도표. 이쪽이 쪽의 가운데다</div><div>네 개의 숫자</div><div>기사 한 줄 · 통계 한 줄</div></div>')}
  <ul class="band">
    <li>기사 칸은 그 주제가 신문이나 영상에 어떻게 나왔는지다.</li>
    <li>숫자가 다르면 기간이나 종류가 다른 것이다.</li>
    <li>통계 칸은 분모와 비교 기준이다.</li>
    <li>민원이 표를 만들었거나, 정책이 민원을 줄였다는 문장은 쓰지 않는다.</li>
  </ul>
  <p class="folio"></p>
</section>
""",
        f"""
<section class="sheet">
  <header><span class="kicker">차례</span><span class="qid"></span></header>
  <h2 class="q">네 묶음, 스물일곱 질문</h2>
  <div class="toc">{entries}</div>
  <p class="folio"></p>
</section>
""",
    ]


def back() -> list[str]:
    return [
        """
<section class="sheet">
  <header><span class="kicker">맺음</span><span class="qid"></span></header>
  <h2 class="q">다섯 줄</h2>
  <ol class="close">
    <li>7월 4일은 불꽃을 빼면 평소 토요일의 1.11배다. 다음 날 새벽 세 시간이 전날 밤 9시보다 많다.</li>
    <li>가장 추운 날의 난방은 다른 영하 날의 2.36배다. 눈 온 날의 주차는 평소 월요일의 49%였다.</li>
    <li>텐트 신고는 다섯 자치구에서 접수처만 바뀌었다. 하루 건수는 130에서 129다.</li>
    <li>이긴 네 자치구의 주택국 몫은 36%에서 12%까지 갈린다. 아스토리아와 베이사이드의 주차는 24.8%와 23.0%로 붙어 있다.</li>
    <li>민주당 한 줄이 이미 쿠오모보다 38,336표 많다. 분모가 바뀌면 전화의 1위도 바뀐다.</li>
  </ol>
  <p class="folio"></p>
</section>
""",
        """
<section class="sheet">
  <header><span class="kicker">기존 자료</span><span class="qid"></span></header>
  <h2 class="q">백 개 질문집에 대해</h2>
  <p class="lead">그 책의 질문을 더 정밀한 막대로 다시 내지 않았다. 여기서는 그 책이 비워 둔 자리만 적는다.</p>
  <ul class="band">
    <li>1년 건수, 몇 초, 불법주차 15.6%, 경찰 43%, 상위 10종, 요일, 크리스마스, 7월 4일 75배와 밤 9시는 그 책에 있다.</li>
    <li>난방의 자치구 배수, 진입로 12배, 소화전, 헬기의 장소, 택시 93 대 1, 한 주소 47,105건, 딱지가 아닌 체포 문장도 그 책에 있다.</li>
    <li>그 책의 바닥은 고유번호만 다르고 나머지 열이 모두 같은 1,792줄을 지운 3,962,167건이고, 이 책은 지우지 않은 3,963,959건이다. 같은 질문을 다시 세지 않았다.</li>
    <li>그 책은 한 주소를 빼지 않으면 겨울 소음이 뒤집힌다고 이미 적었다. 그 배수를 교정판으로 되살리지 않았다.</li>
    <li>없는 것은 2025년 11월 4일 선거, 한 후보의 두 정당 줄, 퀸스 서부와 동부의 기관 구성, 7월 5일 새벽, 추수감사절, 가장 추운 하루, 눈 온 날의 주차, 헬기의 시각이다.</li>
    <li>그 책은 자치구에 성격을 하나씩 붙인다. 한 자치구 안의 상반된 구성과, 승패와 어긋나는 기관은 보여 주지 않는다.</li>
    <li>체포 문장은 그 책 1,163건, 이 파일 1,169건이다. 차이 6건은 그 책이 뺀 한 주소의 체포다. “arrest”가 들어간 42만 건은 상투 문구라 체포가 아니다.</li>
  </ul>
  <p class="note">이 파일의 배수는 접수된 신고의 배율이고, 2025년 11월 4일의 표는 투표소의 기록이다. 둘을 한 지도에 놓는 이유는 구성이 어디서 닮고 어디서 갈리는지 보기 위해서다.</p>
  <p class="folio"></p>
</section>
""",
        """
<section class="sheet">
  <header><span class="kicker">만든 법</span><span class="qid"></span></header>
  <h2 class="q">숫자를 다시 만들 수 있게</h2>
  <ul class="band">
    <li>고유번호가 있는 3,963,959건. 중복 키는 0. 접수 2025-11-01 21:09:28부터 2026-10-31 21:09:02.</li>
    <li>기온과 적설은 Open-Meteo 일별값, 격자 위도 40.81 경도 −74.02. 인구는 2020년 센서스 자치구 인구. 이 책의 새 비교는 인구당을 쓰지 않았다.</li>
    <li>딱지는 결과 문장의 “issued a summons”만. 체포로 세지 않은 42만 건은 “effect an arrest”가 들어간 상투 문구다.</li>
    <li>아스토리아는 11102·11103·11105·11106. 베이사이드는 11360·11361·11364. NY-14 퀸스 근사는 그 넷에 11354·11356·11368·11369·11370·11372를 더한 열 개다.</li>
  </ul>
  <ul class="sources">
    <li>뉴욕시 선거관리위원회, 2025-11-04 시장 선거 확정 보고서. 민주당 944,950, 노동가족당 169,234, 합계 1,114,184.</li>
    <li>타임, 2025-11-05. 페이지에 2026-10-19 수정 표시. 아스토리아 약 66%와 베이사이드 25포인트는 동네 보도다.</li>
    <li>QNS, 2025-11-05. 퀸스 서부와 동부. amNewYork, 2026-09-25. 텐트 접수처.</li>
    <li>AP 유튜브, “LIVE: Mamdani speaks on NYC security measures ahead of 4th of July”.</li>
  </ul>
  <p class="folio"></p>
</section>
""",
    ]


def main() -> None:
    cold = cold_days()
    body = chapters(cold)
    # Contents uses the question titles in order. Pulled from the generated tags.
    import re
    rows = []
    for block in body:
        found = re.search(r'class="qid">([^<]*)</span>[\s\S]*?<h2 class="q">([^<]+)</h2>', block)
        if not found:
            continue
        qid, title = found.group(1), found.group(2)
        if not qid.startswith("Q"):
            continue
        val = re.search(r'<div class="facts"><p><b>([^<]+)</b>', block)
        rows.append((qid, title, val.group(1) if val else ""))
    parts = ["<!DOCTYPE html>", "<html lang=\"ko\">", "<head>", "<meta charset=\"utf-8\">",
             "<title>신고는 일기다 · NYC 311</title>", f"<style>{CSS}</style>", "</head>", "<body>"]
    parts.extend(front(rows))
    parts.extend(body)
    parts.extend(back())
    parts.append("</body></html>")
    html_out = "\n".join(parts)
    n = 0

    def repl(_m: re.Match) -> str:
        nonlocal n
        n += 1
        return f'<p class="folio">{n:02d}</p>'

    html_out = re.sub(r'<p class="folio"></p>', repl, html_out)
    OUT.write_text(html_out, encoding="utf-8")
    print("sheets", html_out.count('class="sheet"'), "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
