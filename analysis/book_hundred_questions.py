"""
[읽을거리] 뉴욕 311, 질문 100개 — 책(PDF)용 계산과 그림

질문 설계 : docs/hundred/questions.json (Q001~Q100), docs/hundred/plan.md (공통 정의)
입력      : data/processed/311_requests_clean.parquet
            data/external/nyc_daily_temp.json (Open-Meteo, 맨해튼 부근 일평균 기온)
            docs/borough_context.csv (자치구 인구, 2020 센서스)
출력      : outputs/figures/h100_Q001.png ~ h100_Q100.png (질문마다 그림 1개)
            reports/book/hundred_numbers.json (질문별 핵심 수치 + check_values 대조 결과)
실행 방법 : D:/dev/python/python.exe analysis/book_hundred_questions.py

공통 정의 (plan.md)
- [B]   : flag_system_duplicate == False 인 행 전체. 모든 질문의 출발점
- SA    : 슈퍼 신고 주소 address_key == '655 EAST 230 STREET'. 소음·NYPD·채널·날씨 질문은 이 주소를 빼고 계산
          (incident_address 원문은 공백이 두 칸이라 반드시 address_key 로 거른다)
- FULL  : 완전한 날 2025-11-02 ~ 2026-10-30 (363일). 일 단위 평균/최대/최소는 이 기간만, 0건인 날은 0으로 채운다
- POP   : 2020 센서스 인구. '1인당' = 건수 / 인구 x 10만
- NYPD 결과 분류 : resolution_description 문장으로 분류. 딱지는 'issued a summons' 구절로만 찾는다
- 처리시간 : is_closed & resolution_hours > 0 & ~flag_status_date_mismatch & ~flag_midnight_close
- category 형 열의 value_counts 는 0건 범주가 섞이므로 0보다 큰 것만 쓴다
"""

import json
import re

from common import ROOT, np, pd, plt, setup_plot

import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

setup_plot()
# ── 책(PDF) 인쇄용 스타일: 그림을 인쇄 폭(인치) 그대로 잡아 책 전체에서 글자 크기가 같게 나오게 한다 ──
plt.rcParams.update({
    "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
    "axes.labelcolor": "#333", "xtick.color": "#444", "ytick.color": "#333",
    "grid.color": "#e2e2e2", "grid.linewidth": .6, "lines.linewidth": 1.5,
    "figure.facecolor": "white", "axes.titlepad": 4,
})
FW, FH = 5.0, 3.2  # 모든 그림 같은 크기 (책에서 폭 127mm 칸에 그대로 들어간다)
ACC, GREY, INK, MUTED = "#C2410C", "#BEC4CD", "#1d2433", "#6b7280"
HEAT_CMAP = LinearSegmentedColormap.from_list("h100", ["#F7F4EE", "#F2C48D", "#E07A3F", "#A3341A", "#3B1410"])
THOU = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "reports" / "book"
FIG.mkdir(parents=True, exist_ok=True)

SA = "655 EAST 230 STREET"
BORO = ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]
KB = {"MANHATTAN": "맨해튼", "BROOKLYN": "브루클린", "QUEENS": "퀸스", "BRONX": "브롱크스", "STATEN ISLAND": "스태튼아일랜드"}
ctx = pd.read_csv(ROOT / "docs" / "borough_context.csv")
pop = ctx.set_index(ctx.columns[0])["population_census2020"]
pop.index = pop.index.str.upper()
pop = pop.reindex(BORO).astype(float)

# 그림 라벨용 한글 이름 (없는 이름은 영어 그대로 쓰고, 실행 끝에 목록을 출력한다)
KO = {
    "Illegal Parking": "불법주차", "Noise - Residential": "이웃 소음", "HEAT/HOT WATER": "난방·온수",
    "Blocked Driveway": "차고 앞 막힘", "Noise - Street/Sidewalk": "길거리 소음", "UNSANITARY CONDITION": "집 안 위생",
    "Street Condition": "도로 파손", "PLUMBING": "배관", "Abandoned Vehicle": "버려진 차", "Dirty Condition": "더러운 거리",
    "PAINT/PLASTER": "페인트·벽", "Water System": "상수도", "Snow or Ice": "눈·얼음", "Noise": "환경국 소음(공사 등)",
    "Noise - Commercial": "가게 소음", "Traffic Signal Condition": "신호등 고장", "DOOR/WINDOW": "문·창문",
    "Noise - Vehicle": "차량 소음", "WATER LEAK": "누수", "Encampment": "노숙 텐트", "GENERAL": "일반 수리",
    "Damaged Tree": "나무 피해", "Street Light Condition": "가로등", "Illegal Dumping": "무단 투기", "ELECTRIC": "전기",
    "Derelict Vehicles": "방치 차량", "Missed Collection": "쓰레기 미수거", "Homeless Person Assistance": "노숙인 도움 요청",
    "FLOORING/STAIRS": "바닥·계단", "Elevator": "엘리베이터", "APPLIANCE": "가전", "Rodent": "쥐", "SAFETY": "안전",
    "Non-Emergency Police Matter": "비긴급 경찰 사안", "Vendor Enforcement": "노점 단속", "Overgrown Tree/Branches": "웃자란 나무",
    "Sewer": "하수도", "Water Maintenance": "수도 관리", "Traffic": "교통", "Panhandling": "구걸", "Illegal Fireworks": "불꽃놀이",
    "Drug Activity": "마약", "Noise - Park": "공원 소음", "Drinking": "길거리 음주", "Noise - House of Worship": "종교시설 소음",
    "Noise - Helicopter": "헬기 소음", "Animal-Abuse": "동물 학대", "Sidewalk Condition": "보도 파손",
    "Street Sign - Damaged": "도로 표지 파손", "Dead Animal": "동물 사체", "Lost Property": "분실물",
    "Water Conservation": "물 낭비", "Taxi Complaint": "택시 불만", "For Hire Vehicle Complaint": "콜택시 불만",
    "Root/Sewer/Sidewalk Condition": "나무뿌리·보도", "Lot Condition": "공터 상태", "Construction Lead Dust": "공사 납 먼지",
    "Dead/Dying Tree": "죽어 가는 나무", "Building/Use": "건물 용도 위반", "Unleashed Dog": "목줄 없는 개",
    "Illegal Tree Damage": "불법 나무 훼손", "General Construction/Plumbing": "공사·배관(건물국)", "Maintenance or Facility": "공원 시설",
    "Bike/Roller/Skate": "자전거·롤러", "Urinating in Public": "노상 방뇨", "Graffiti": "낙서", "Commercial Disposal Complaint": "상가 쓰레기",
    "Food Establishment": "음식점 위생", "Taxi Licensee Complaint": "택시 면허 불만", "Tanning": "태닝 업소", "Calorie Labeling": "칼로리 표시",
    "Oil or Gas Spill": "기름 유출", "Dry Weather Discharge": "맑은 날 배수", "Tunnel Condition": "터널 상태",
    "DEP Sidewalk Condition": "보도(환경국)", "Public Payphone Complaint": "공중전화", "Squeegee": "차창 닦기 강매",
    "DEP Highway Condition": "도로(환경국)", "FHV Licensee Complaint": "콜택시 면허 불만", "Leaning Bar": "기대는 바(정류장)",
    "DSNY Internal": "청소국 내부", "Building Condition": "건물 상태", "Construction Safety Enforcement": "공사 안전",
    "Building Marshals office": "건물 보안관실", "Trapping Pigeon": "비둘기 포획", "Public Toilet": "공중화장실", "Lifeguard": "인명구조원",
    "Radioactive Material": "방사성 물질", "Non-Residential Heat": "상가 난방", "School Maintenance": "학교 관리", "Street Sweeping Complaint": "도로 청소",
    "Curb Condition": "연석 파손", "Highway Condition": "고속도로 상태", "Water Quality": "수질", "Emergency Response Team (ERT)": "긴급대응팀",
    "Indoor Air Quality": "실내 공기", "Air Quality": "대기질", "Mosquitoes": "모기", "Harboring Bees/Wasps": "벌·말벌", "Smoking or Vaping": "흡연",
    "Bus Stop Shelter Complaint": "버스 정류장", "Plant": "환경국 시설", "Snow Removal": "제설",
    "Obstruction": "통행 방해", "Uprooted Stump": "뽑힌 그루터기",
    # 기관
    "NYPD": "경찰", "HPD": "주택보존개발국", "DSNY": "청소국", "DOT": "교통국", "DEP": "환경보호국", "DPR": "공원국",
    "DOB": "건물국", "DOHMH": "보건국", "DHS": "노숙인서비스국", "TLC": "택시리무진위원회", "DCWP": "소비자보호국", "EDC": "경제개발공사",
}
MISSING = set()


def ko(x):
    if x in KO:
        return KO[x]
    MISSING.add(x)
    return x


NYPD_KO = {"noevidence": "가 보니 흔적 없음", "corrected": "현장 정리", "notnecessary": "경찰 조치 불필요", "summons": "소환장(딱지)",
           "noentry": "들어가지 못함", "notjuris": "관할 아님", "enc_none": "텐트 없음", "other": "기타", "enc_found": "텐트 확인",
           "report": "보고서 작성", "arrest": "체포", "blank": "빈 값"}
MONTHS = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MLAB = [f"{m}월" for m in MONTHS]
BIN_LAB = ["영하", "0~10도", "10~20도", "20~25도", "25도 이상"]

V: dict = {}


# ═════════════════════════════════════════════
# 그림 도우미
# ═════════════════════════════════════════════
def new():
    return plt.subplots(figsize=(FW, FH))


def save(fig, q):
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG / f"h100_{q}.png")  # bbox 를 자르지 않아 모든 그림이 같은 크기(5.0 x 3.2 인치)
    plt.close(fig)


def f0(v):
    return f"{v:,.0f}"


def f1(v):
    return f"{v:,.1f}"


def pct(v):
    return f"{v:.1f}%"


def hbar(q, labels, vals, hi=(), fmt=f0, xlabel=None, ref=None, reflab=None, xmax=None, title=None):
    """가로 막대. labels/vals 는 위에서 아래 순서. hi 는 강조할 위치(정수) 또는 라벨."""
    fig, ax = new()
    n = len(vals)
    y = np.arange(n)[::-1]
    on = [(i in hi) or (labels[i] in hi) for i in range(n)]
    ax.barh(y, vals, color=[ACC if o else GREY for o in on], height=.66, zorder=2)
    ax.set_yticks(y, labels)
    m = max(vals) if max(vals) > 0 else 1
    for yi, v, o in zip(y, vals, on):
        ax.text(v + m * .012, yi, fmt(v), va="center", fontsize=7.2, color=ACC if o else "#444",
                fontweight="bold" if o else "normal")
    for t, o in zip(ax.get_yticklabels(), on):
        if o:
            t.set_color(ACC); t.set_fontweight("bold")
    ax.set_xlim(0, xmax or m * 1.16)
    ax.set_ylim(-.6, n - .4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(THOU if m >= 100 else mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="x", zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ref is not None:
        ax.axvline(ref, color=INK, ls=(0, (3, 2)), lw=.9, zorder=3)
        ax.text(ref, n - .45, f" {reflab}", fontsize=7.2, color=INK, va="bottom")
    if title:
        ax.set_title(title, loc="left", color="#333")
    save(fig, q)


def vbar(q, labels, vals, hi=(), fmt=f0, ylabel=None, show=True, rot=0, ref=None, reflab=None, every=1, xnotes=()):
    fig, ax = new()
    x = np.arange(len(vals))
    on = [(i in hi) or (labels[i] in hi) for i in range(len(vals))]
    ax.bar(x, vals, color=[ACC if o else GREY for o in on], width=.7, zorder=2)
    ax.set_xticks(x[::every], [labels[i] for i in range(0, len(labels), every)], rotation=rot)
    m = max(vals) if max(vals) > 0 else 1
    if show:
        for xi, v, o in zip(x, vals, on):
            ax.text(xi, v + m * .015, fmt(v), ha="center", va="bottom", fontsize=7.2,
                    color=ACC if o else "#555", fontweight="bold" if o else "normal")
    ax.set_ylim(0, m * 1.17)
    ax.yaxis.set_major_formatter(THOU if m >= 100 else mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="x", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ref is not None:
        ax.axhline(ref, color=INK, ls=(0, (3, 2)), lw=.9, zorder=3)
        ax.text(len(vals) - .5, ref, f"{reflab} ", fontsize=7.2, color=INK, ha="right", va="center",
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=4)
    for xi, tx in xnotes:
        ax.text(xi - .35, -.13, tx, transform=ax.get_xaxis_transform(), fontsize=7.2, color=MUTED, ha="left", va="top")
        ax.axvline(xi - .5, ymin=-.12, ymax=0, color=MUTED, lw=.7, clip_on=False)
    save(fig, q)
    return


TILE_POS = {"BRONX": (1.55, 0), "MANHATTAN": (1, 1.05), "QUEENS": (2.1, 1.05), "BROOKLYN": (1.55, 2.1), "STATEN ISLAND": (.35, 2.75)}


def boro(q, per, hi, xlabel="인구 10만 명당 신고 수", fmt=f0, kind="bar", cnt=None):
    """자치구 비교. kind = bar(막대+뉴욕 평균 점선) / dot(점 그림) / tile(구 위치를 본뜬 타일 지도)."""
    s = pd.Series(per).sort_values(ascending=False)
    avg = sum(cnt.values()) / pop.sum() * 1e5 if cnt else None
    if kind == "bar":
        hbar(q, [KB[b] for b in s.index], list(s.values), hi=[KB[h] for h in hi], fmt=fmt, xlabel=xlabel,
             ref=avg, reflab=f"뉴욕 평균 {avg:,.0f}" if avg else None)
        return
    fig, ax = new()
    if kind == "dot":
        y = np.arange(5)[::-1]
        on = [b in hi for b in s.index]
        ax.hlines(y, 0, s.values, color=GREY, lw=2.2, zorder=1)
        ax.scatter(s.values, y, s=[90 if o else 50 for o in on], color=[ACC if o else INK for o in on], zorder=3)
        for yi, v, o in zip(y, s.values, on):
            ax.text(v + s.max() * .03, yi, fmt(v), va="center", fontsize=7.6, color=ACC if o else "#444", fontweight="bold" if o else "normal")
        ax.set_yticks(y, [KB[b] for b in s.index])
        for t, o in zip(ax.get_yticklabels(), on):
            if o:
                t.set_color(ACC); t.set_fontweight("bold")
        if avg:
            ax.axvline(avg, color=INK, ls=(0, (3, 2)), lw=.9)
            ax.text(avg, 4.45, f" 뉴욕 평균 {avg:,.0f}", fontsize=7.2, color=INK, va="bottom")
        ax.set_xlim(0, s.max() * 1.18); ax.set_ylim(-.6, 4.9)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(THOU); ax.grid(axis="x"); ax.set_xlabel(xlabel)
    else:
        vmax = s.max()
        for b, (cx, cy) in TILE_POS.items():
            v = per[b]
            col = HEAT_CMAP(.08 + .8 * v / vmax)
            on = b in hi
            ax.add_patch(plt.Rectangle((cx, -cy), .98, .9, color=col, ec=ACC if on else "white", lw=2.2 if on else 1))
            dark = v / vmax > .55
            ax.text(cx + .49, -cy + .58, KB[b], ha="center", va="center", fontsize=8, color="white" if dark else "#333", fontweight="bold")
            ax.text(cx + .49, -cy + .28, fmt(v), ha="center", va="center", fontsize=10.5, color="white" if dark else ACC if on else "#333", fontweight="bold")
        ax.text(3.25, -2.2, xlabel + (f"\n뉴욕 평균 {avg:,.0f}" if avg else ""), fontsize=7.6, color=MUTED, ha="left", va="center")
        ax.set_xlim(.2, 4.9); ax.set_ylim(-2.95, 1.0); ax.set_aspect("equal"); ax.axis("off")
    save(fig, q)


def month_bar(q, by_month, hi=(), ylabel="신고 수", fmt=f0):
    vals = [by_month.get(m, 0) for m in MONTHS]
    vbar(q, MLAB, vals, hi=[MONTHS.index(h) for h in hi], ylabel=ylabel, fmt=fmt)


def bins_bar(q, by_bin, hi, ylabel="하루 평균 신고 수"):
    vals = [by_bin[k] for k in ["<0", "0-10", "10-20", "20-25", "25+"]]
    vbar(q, [f"{l}\n({d}일)" for l, d in zip(BIN_LAB, BIN_DAYS)], vals, hi=hi, fmt=f1, ylabel=ylabel)


def daily_line(q, s, hi_date, lab, ylabel="하루 신고 수", ma=False, xytext=(-110, -8)):
    fig, ax = new()
    ax.plot(s.index, s.values, color=INK if not ma else GREY, lw=.8)
    d = pd.Timestamp(hi_date)
    ax.scatter([d], [s[d]], color=ACC, s=18, zorder=4)
    ax.annotate(lab, (d, s[d]), textcoords="offset points", xytext=xytext, fontsize=7.6, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=.7))
    ax.set_ylim(0, s.max() * 1.12)
    ax.yaxis.set_major_formatter(THOU)
    ax.set_ylabel(ylabel)
    month_axis(ax)
    ax.grid(axis="y")
    save(fig, q)


def month_axis(ax):
    ticks = pd.date_range("2025-11-01", "2026-10-01", freq="MS")
    ax.set_xticks(ticks, [f"{t.month}월" if t.month != 1 else "26년\n1월" for t in ticks])
    ax.set_xlim(pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31"))
    ax.tick_params(axis="x", length=2)


def calendar(q, s, hi_date=None, label="하루 신고 수"):
    """달력 히트맵: 열 = 주, 행 = 요일."""
    s = s.reindex(DAYS, fill_value=0)
    start = DAYS[0] - pd.Timedelta(days=DAYS[0].day_of_week)
    wk = ((s.index - start).days // 7).values
    dw = s.index.day_of_week.values
    grid = np.full((7, wk.max() + 1), np.nan)
    grid[dw, wk] = s.values
    fig, ax = new()
    im = ax.imshow(grid, aspect="auto", cmap=HEAT_CMAP, interpolation="nearest")
    ax.set_yticks(range(7), list("월화수목금토일"), fontsize=7.2)
    ms = pd.date_range("2025-12-01", "2026-10-01", freq="MS")
    ax.set_xticks([((m - start).days // 7) for m in ms], [f"{m.month}월" for m in ms])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    if hi_date is not None:
        d = pd.Timestamp(hi_date)
        w, r = (d - start).days // 7, d.day_of_week
        ax.add_patch(plt.Rectangle((w - .5, r - .5), 1, 1, fill=False, ec=INK, lw=1.3))
    cb = fig.colorbar(im, ax=ax, shrink=.9, pad=.012, aspect=14)
    cb.set_label(label, fontsize=7.2); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=0)
    cb.ax.yaxis.set_major_formatter(THOU)
    save(fig, q)


def concentration(q, counts, unit, cuts):
    """몇 % 의 건물·주소가 신고의 몇 % 를 내는지: '개수' 막대와 '신고' 막대를 같은 구간으로 나눠 나란히 놓는다."""
    a = np.sort(np.asarray(counts))[::-1]
    edges = [0] + [max(int(a.size * c / 100), 1) for c in cuts] + [a.size]
    share_n = [(edges[i + 1] - edges[i]) / a.size * 100 for i in range(len(edges) - 1)]
    share_r = [a[edges[i]:edges[i + 1]].sum() / a.sum() * 100 for i in range(len(edges) - 1)]
    labs = [f"상위 {cuts[0]:g}%"] + [f"다음 {cuts[i + 1] - cuts[i]:g}%" for i in range(len(cuts) - 1)] + [f"나머지 {100 - cuts[-1]:g}%"]
    cols = [ACC, "#EBA57C", "#DADDE2"] if len(labs) == 3 else [ACC, "#E07A3F", "#F2C48D", "#DADDE2"]
    fig, ax = new()
    for yy, vals in [(1, share_n), (0, share_r)]:
        left = 0
        for v_, c_ in zip(vals, cols):
            ax.barh(yy, v_, left=left, color=c_, height=.5, edgecolor="white", lw=.8)
            if v_ >= 7:
                ax.text(left + v_ / 2, yy, f"{v_:.0f}%", ha="center", va="center", fontsize=7.2,
                        color="white" if c_ in (ACC, "#E07A3F") else "#333", fontweight="bold")
            left += v_
    for c_, lb in zip(cols, labs):
        ax.barh([0], [0], color=c_, label=f"{lb} {unit}")
    ax.annotate("", xy=(share_r[0] / 2, .27), xytext=(max(share_n[0], .4) / 2, .73), arrowprops=dict(arrowstyle="->", color=ACC, lw=1))
    ax.set_yticks([1, 0], [f"{unit} 수", "신고 수"])
    ax.set_xlim(0, 100); ax.set_ylim(-.45, 1.75)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(ncol=len(labs), frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.02), fontsize=7.2, handlelength=1, columnspacing=1.2)
    save(fig, q)
    return share_n, share_r


def lorenz(q, counts, unit, marks):
    a = np.sort(np.asarray(counts))[::-1]
    cx = np.arange(1, a.size + 1) / a.size * 100
    cy = np.cumsum(a) / a.sum() * 100
    fig, ax = new()
    ax.plot(cx, cy, color=INK, lw=1.4)
    ax.plot([cx[0], 100], [cx[0], 100], color=GREY, ls=(0, (3, 2)), lw=.9)
    ax.text(30, 22, "모두 똑같이 신고한다면", fontsize=7.2, color=MUTED, rotation=0)
    for m in marks:
        k = max(int(a.size * m / 100), 1)
        ax.scatter([m], [cy[k - 1]], color=ACC, s=16, zorder=4)
        ax.annotate(f"상위 {m:g}% {unit} → {cy[k - 1]:.0f}%", (m, cy[k - 1]), textcoords="offset points", xytext=(6, -10 if m < 1 else 4),
                    fontsize=7.6, color=ACC, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(cx[0], 100)
    ax.set_ylim(0, 102)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel(f"신고가 많은 순서로 센 {unit} 비율 (로그 눈금)")
    ax.set_ylabel("차지하는 신고 비율 (%)")
    ax.grid(alpha=.7)
    save(fig, q)


def street_only(addr):
    """개인 거주지일 수 있으므로 번지를 지우고 거리 이름만 남긴다."""
    t = addr.split()
    if len(t) >= 3 and re.search(r"\d", t[0]):
        t = t[1:]
    return " ".join(t).title()


ANON = {"655 EAST 230 STREET": "브롱크스 북부의 한 주소", "1869 83 STREET": "브루클린 남부의 한 주소",
        "5-47 50 AVENUE": "퀸스 서부의 한 주소", "1040B EAST 217 STREET": "브롱크스 북부의 한 건물"}


def addr_labels(idx, b):
    return [f"{i + 1}. {ANON[a]}" if a in ANON else f"{i + 1}. {KB.get(b.get(a, ''), '')} · {street_only(a)}" for i, a in enumerate(idx)]


def stack100(q, rows, parts, data, colors, labels_part, hi_row=None):
    """100% 누적 가로 막대. data[row][part] = %."""
    fig, ax = new()
    n = len(rows)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for p, c, lp in zip(parts, colors, labels_part):
        v = np.array([data[r].get(p, 0) for r in rows])
        ax.barh(y, v, left=left, color=c, height=.62, label=lp, edgecolor="white", lw=.6)
        for yi, l0, vv in zip(y, left, v):
            if vv >= 6:
                ax.text(l0 + vv / 2, yi, f"{vv:.0f}%", ha="center", va="center", fontsize=7.2,
                        color="white" if c in (ACC, INK, "#7d8696") else "#333")
        left += v
    ax.set_yticks(y, rows)
    if hi_row is not None:
        for t in ax.get_yticklabels():
            if t.get_text() == hi_row:
                t.set_color(ACC); t.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(ncol=len(parts), frameon=False, loc="lower center", bbox_to_anchor=(.45, 1.0), fontsize=7.2, handlelength=1, columnspacing=1)
    save(fig, q)


def r1(x):
    return round(float(x), 1)


# ═════════════════════════════════════════════
# 데이터
# ═════════════════════════════════════════════
cols = ["created_date", "agency", "problem", "problem_detail", "location_type", "status", "resolution_description",
        "community_board", "borough", "latitude", "longitude", "open_data_channel_type", "resolution_hours", "is_closed",
        "created_dow", "created_hour", "flag_system_duplicate", "flag_status_date_mismatch", "flag_midnight_close",
        "incident_group_id", "incident_group_size", "address_key"]
df = pd.read_parquet(ROOT / "data" / "processed" / "311_requests_clean.parquet", columns=cols)
df = df[~df["flag_system_duplicate"]].copy()  # [B]
for c in ["agency", "problem", "problem_detail", "location_type", "borough", "open_data_channel_type", "status", "community_board"]:
    df[c] = df[c].astype("category")
df["date"] = df["created_date"].dt.normalize()
df["m"] = df["created_date"].dt.month
N = len(df)
ex = df[df["address_key"] != SA]                      # SA 제외
full = df[(df["date"] > "2025-11-01") & (df["date"] < "2026-10-31")]   # FULL 363일
fex = full[full["address_key"] != SA]
DAYS = pd.date_range("2025-11-02", "2026-10-30")
d = full.groupby("date").size().reindex(DAYS, fill_value=0)
rd = df["resolution_description"].fillna("")
print("[B] rows", N, "| FULL days", len(DAYS))


def pc(mask, base=df):
    s = base[mask].groupby("borough", observed=True).size().reindex(BORO, fill_value=0)
    return (s / pop * 1e5).round(1).to_dict(), {k: int(v) for k, v in s.items()}


def nypd_cat(r):
    r = r.fillna("")
    return np.select(
        [r.str.contains("issued a summons"), r.str.contains("made an arrest"), r.str.contains("condition was corrected|took action to fix"),
         r.str.contains("no evidence|no criminal violation upon|were gone"), r.str.contains("police action was not necessary"),
         r.str.contains("unable to gain entry"), r.str.contains("observed an encampment|referred the complaint to the Department of Homeless"),
         r.str.contains("no encampment|no Encampment"), r.str.contains("jurisdiction"), r.str.contains("report was prepared"), r == ""],
        ["summons", "arrest", "corrected", "noevidence", "notnecessary", "noentry", "enc_found", "enc_none", "notjuris", "report", "blank"],
        "other")


# 기온
tj = json.load(open(ROOT / "data" / "external" / "nyc_daily_temp.json", encoding="utf-8"))
t = pd.DataFrame(tj["daily"]); t["date"] = pd.to_datetime(t["time"])
TALL = t.set_index("date")["temperature_2m_mean"]
T = TALL.reindex(DAYS)
bins = pd.cut(T, [-100, 0, 10, 20, 25, 100], right=False, labels=["<0", "0-10", "10-20", "20-25", "25+"])
BIN_DAYS = [int((bins == k).sum()) for k in ["<0", "0-10", "10-20", "20-25", "25+"]]
V["_temp_bin_days"] = dict(zip(["<0", "0-10", "10-20", "20-25", "25+"], BIN_DAYS))

# ═════════════════════════════════════════════
# 1장. 숫자로 만나는 311
# ═════════════════════════════════════════════
vc = df["problem"].value_counts(); vc = vc[vc > 0]
mon = df.groupby(df["created_date"].dt.to_period("M")).size()
V["Q001"] = {"total_rows": N, "first": str(df["created_date"].min()), "last": str(df["created_date"].max())}
vbar("Q001", [f"{p.month}월" for p in mon.index], [v / 1e4 for v in mon.values], hi=[],
     fmt=lambda v: f"{v:.0f}", ylabel="한 달 신고 수 (만 건)", xnotes=[(0, "2025년"), (2, "2026년")])

V["Q002"] = {"full_days": len(DAYS), "daily_avg": r1(d.mean()), "per_min": round(d.mean() / 1440, 2), "seconds_per_report": r1(86400 / d.mean())}
hpm = full.groupby("created_hour").size() / len(DAYS) / 60
fig, ax = new()
ax.bar(hpm.index, hpm.values, color=GREY, width=.72, zorder=2)
ax.axhline(d.mean() / 1440, color=ACC, lw=1.2, zorder=3)
ax.text(-.5, d.mean() / 1440 + .3, f"1년 평균 1분에 {d.mean() / 1440:.1f}건", ha="left", fontsize=7.6, color=ACC, fontweight="bold")
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
ax.set_ylabel("1분당 평균 신고 수"); ax.grid(axis="y", zorder=0); ax.tick_params(axis="x", length=0)
save(fig, "Q002")

V["Q003"] = {k: int(v) for k, v in vc.head(5).items()}
V["Q003"]["illegal_parking_share_pct"] = r1(vc.iloc[0] / N * 100)
top = vc.head(10)
hbar("Q003", [ko(k) for k in top.index], list(top.values / 1e4), hi=[0], fmt=lambda v: f"{v:.1f}만", xlabel="1년 신고 수 (만 건)")

ag = df["agency"].value_counts(); ag = ag[ag > 0]
V["Q004"] = {"NYPD": int(ag["NYPD"]), "NYPD_pct": r1(ag["NYPD"] / N * 100), "HPD_pct": r1(ag["HPD"] / N * 100), "DSNY_pct": r1(ag["DSNY"] / N * 100)}
a8 = (ag.head(8) / N * 100)
hbar("Q004", [ko(k) for k in a8.index], list(a8.values), hi=[0], fmt=pct, xlabel="전체 신고 중 비중 (%)")

V["Q005"] = {"n_types": len(vc), "top10_share_pct": r1(vc.head(10).sum() / N * 100), "bottom97_share_pct": round(vc.iloc[97:].sum() / N * 100, 2)}
cum = vc.cumsum() / N * 100
fig, ax = new()
ax.plot(np.arange(1, len(cum) + 1), cum.values, color=INK, lw=1.4)
for k, txt, off in [(10, f"상위 10종 = {cum.iloc[9]:.1f}%", (8, -12)), (97, f"97위까지 = {cum.iloc[96]:.1f}%\n(나머지 97종은 합쳐도 {100 - cum.iloc[96]:.2f}%)", (6, -26))]:
    ax.scatter([k], [cum.iloc[k - 1]], color=ACC, s=16, zorder=4)
    ax.annotate(txt, (k, cum.iloc[k - 1]), textcoords="offset points", xytext=off, fontsize=7.6, color=ACC, fontweight="bold")
ax.axvspan(98, len(cum), color=GREY, alpha=.25, lw=0)
ax.set_xlim(0, len(cum) + 2); ax.set_ylim(0, 103)
ax.set_xlabel("민원 유형 순위 (많은 순)"); ax.set_ylabel("누적 비중 (%)"); ax.grid(alpha=.7)
save(fig, "Q005")

V["Q006"] = {"date": str(d.idxmax().date()), "count": int(d.max()), "ratio_vs_avg": round(d.max() / d.mean(), 2),
             "snow_share_pct": r1((full[full["date"] == d.idxmax()]["problem"] == "Snow or Ice").mean() * 100)}
daily_line("Q006", d, d.idxmax(), f"2월 24일 폭설 {d.max():,}건", xytext=(-120, -4))

V["Q007"] = {"date": str(d.idxmin().date()), "count": int(d.min()), "pct_of_avg": r1(d.min() / d.mean() * 100), "2nd_lowest": str(d.nsmallest(2).index[1].date())}
low = d.nsmallest(8)
WD = "월화수목금토일"
hbar("Q007", [f"{x.year % 100}.{x.month}.{x.day} ({WD[x.day_of_week]})" for x in low.index], list(low.values), hi=[0],
     xlabel="하루 신고 수", ref=d.mean(), reflab=f"평소 {d.mean():,.0f}건", xmax=d.mean() * 1.25)

dw = full.groupby("created_dow").size() / full.groupby("created_dow")["date"].nunique()
V["Q008"] = {"mon": r1(dw[1]), "tue": r1(dw[2]), "sat": r1(dw[6]), "sun": r1(dw[7]), "tue_over_sat": round(dw[2] / dw[6], 2)}
vbar("Q008", [f"{c}요일" for c in WD], [dw[i] for i in range(1, 8)], hi=[1, 5], ylabel="하루 평균 신고 수")

# ═════════════════════════════════════════════
# 2장. 달력이 만든 민원
# ═════════════════════════════════════════════
fw = df[df["problem"] == "Illegal Fireworks"]
fwd = fw[(fw["date"] > "2025-11-01") & (fw["date"] < "2026-10-31")].groupby("date").size().reindex(DAYS, fill_value=0)
V["Q009"] = {"jul4": int(fwd["2026-07-04"]), "daily_avg": r1(fwd.mean()), "ratio": r1(fwd["2026-07-04"] / fwd.mean()), "jul5": int(fwd["2026-07-05"]), "median_day": float(fwd.median())}
fig, ax = new()
ax.fill_between(fwd.index, fwd.values, color=GREY, lw=0)
ax.plot(fwd.index, fwd.values, color=INK, lw=.6)
ax.scatter([pd.Timestamp("2026-07-04")], [fwd["2026-07-04"]], color=ACC, s=18, zorder=4)
ax.annotate(f"7월 4일 {fwd['2026-07-04']:,}건\n(하루 평균 {fwd.mean():.0f}건의 {fwd['2026-07-04'] / fwd.mean():.0f}배)", (pd.Timestamp("2026-07-04"), fwd["2026-07-04"]),
            textcoords="offset points", xytext=(-150, -14), fontsize=7.6, color=ACC, fontweight="bold")
ax.set_ylim(0, fwd.max() * 1.12); ax.yaxis.set_major_formatter(THOU); ax.set_ylabel("하루 불꽃놀이 신고"); month_axis(ax); ax.grid(axis="y")
save(fig, "Q009")

V["Q010"] = {"total": len(fw), "jun15_jul10_pct": r1(((fw["date"] >= "2026-06-15") & (fw["date"] <= "2026-07-10")).mean() * 100),
             "jul3_5_pct": r1(fw["date"].isin(pd.to_datetime(["2026-07-03", "2026-07-04", "2026-07-05"])).mean() * 100)}
wkly = fw.groupby(pd.Grouper(key="created_date", freq="W-SUN")).size()
fig, ax = new()
ax.axvspan(pd.Timestamp("2026-06-15"), pd.Timestamp("2026-07-11"), color=ACC, alpha=.13, lw=0)
ax.bar(wkly.index - pd.Timedelta(days=3), wkly.values, width=5.5, color=[ACC if pd.Timestamp("2026-06-15") <= w - pd.Timedelta(days=3) <= pd.Timestamp("2026-07-11") else GREY for w in wkly.index], zorder=2)
ax.text(pd.Timestamp("2026-06-12"), wkly.max() * .93, f"6/15~7/10\n1년 치의 {V['Q010']['jun15_jul10_pct']:.0f}%", ha="right", fontsize=7.6, color=ACC, fontweight="bold")
ax.yaxis.set_major_formatter(THOU); ax.set_ylabel("주별 불꽃놀이 신고"); month_axis(ax); ax.grid(axis="y", zorder=0)
save(fig, "Q010")

j4 = fw[fw["date"] == "2026-07-04"].groupby("created_hour").size().reindex(range(24), fill_value=0)
V["Q011"] = {"peak_hour": int(j4.idxmax()), "peak_count": int(j4.max()), "h20_23": {int(k): int(v) for k, v in j4.loc[20:23].items()},
             "h21_22_share_pct": r1(j4.loc[21:22].sum() / j4.sum() * 100)}
vbar("Q011", [f"{h}" for h in range(24)], list(j4.values), hi=[21, 22], ylabel="7월 4일 시간대별 신고")
plt.close("all")

th = fex[(fex["created_dow"] == 4) & (fex["date"] != "2025-12-25") & (fex["date"] != "2025-11-27") & (fex["m"].isin([11, 12, 1]))]
nth = th["date"].nunique()
xm = fex[fex["date"] == "2025-12-25"]


def xr(p):
    x = int((xm["problem"] == p).sum()); a = (th["problem"] == p).sum() / nth
    return {"xmas": x, "thu_avg": r1(a), "ratio": round(x / a, 2)}


V["Q012"] = {"basis": "Thursdays Nov2025-Jan2026 excl Thanksgiving & Christmas, n_days=%d" % nth, "noise_res": xr("Noise - Residential"),
             "heat": xr("HEAT/HOT WATER"), "illegal_parking": xr("Illegal Parking"),
             "xmas_noise_res_peak_hour": int(xm[xm["problem"] == "Noise - Residential"].groupby("created_hour").size().idxmax())}
V["Q013"] = {"missed_collection": xr("Missed Collection"), "street_condition": xr("Street Condition")}
tha = th.groupby("problem", observed=True).size() / nth
xmc = xm.groupby("problem", observed=True).size().reindex(tha.index, fill_value=0)
ratio = (xmc / tha)
tha = tha.drop("Snow or Ice")  # 눈 민원은 그날 날씨(눈 없음) 효과라 비교에서 뺀다
ratio = ratio.drop("Snow or Ice")
big = tha[tha >= 50].index
r12 = ratio[tha.sort_values(ascending=False).head(10).index].sort_values(ascending=False)
hbar("Q012", [ko(k) for k in r12.index], list(r12.values), hi=["이웃 소음"], fmt=lambda v: f"{v:.2f}배",
     ref=1, reflab="평소 목요일 = 1", xlabel="성탄절 신고 ÷ 평소 겨울 목요일 하루 평균")
r13 = ratio[big].sort_values().head(8)
hbar("Q013", [ko(k) for k in r13.index], list(r13.values), hi=["쓰레기 미수거"], fmt=lambda v: f"{v:.2f}배",
     ref=1, reflab="평소 = 1", xmax=1.15, xlabel="성탄절 신고 ÷ 평소 겨울 목요일 (가장 많이 줄어든 8종)")

fw0 = fw[fw["created_hour"] == 0]
fw0d = fw0[(fw0["date"] > "2025-11-01") & (fw0["date"] < "2026-10-31")].groupby("date").size().reindex(DAYS, fill_value=0)
ny = df[(df["created_date"] >= "2026-01-01") & (df["created_date"] < "2026-01-01 01:00")]
nyv = ny["problem"].value_counts(); nyv = nyv[nyv > 0]
V["Q014"] = {"fireworks_0h_jan1": int(fw0d["2026-01-01"]), "fireworks_0h_avg": round(fw0d.mean(), 2), "ratio": r1(fw0d["2026-01-01"] / fw0d.mean()),
             "all_reports_0h_jan1": len(ny), "top_problems": {k: int(v) for k, v in nyv.head(3).items()}}
hbar("Q014", [ko(k) for k in nyv.head(5).index] + ["평소 0시대 불꽃놀이"], list(nyv.head(5).values) + [fw0d.mean()], hi=["불꽃놀이"],
     fmt=lambda v: f"{v:,.0f}" if v >= 10 else f"{v:.1f}", xlabel="2026년 1월 1일 0시~1시 신고 수 (맨 아래: 평소 하루 0시대 평균)")

h2 = full.groupby(["date", "created_hour"]).size().unstack(fill_value=0).reindex(DAYS, fill_value=0)
m8 = df[df["date"] == "2026-03-08"]
V["Q015"] = {"mar8_hour2": int((m8["created_hour"] == 2).sum()), "avg_hour2": r1(h2[2].mean()),
             "mar8_hour1": int((m8["created_hour"] == 1).sum()), "mar8_hour3": int((m8["created_hour"] == 3).sum())}
m8h = m8.groupby("created_hour").size().reindex(range(7), fill_value=0)
avg_h = h2[list(range(7))].mean()
fig, ax = new()
x = np.arange(7)
ax.bar(x, m8h.values, color=[ACC if i == 2 else GREY for i in x], width=.62, zorder=2, label="2026년 3월 8일")
ax.plot(x, avg_h.values, color=INK, ls=(0, (3, 2)), marker="o", ms=3, lw=1, zorder=3, label="평소 같은 시간 평균")
for i, v in zip(x, m8h.values):
    ax.text(i, v + 8, f"{v:,}", ha="center", fontsize=7.2, color=ACC if i == 2 else "#555", fontweight="bold" if i == 2 else "normal")
ax.annotate("시계가 1시59분에서 3시로\n건너뛰어 '2시대'가 없다", (2, 12), xytext=(2.45, 440), textcoords="data", fontsize=7.6, color=ACC, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ACC, lw=.7))
ax.set_xticks(x, [f"{h}시" for h in x]); ax.set_ylabel("시간대별 신고 수"); ax.grid(axis="y", zorder=0); ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, loc="upper right")
save(fig, "Q015")

nrd = fex[fex["problem"] == "Noise - Residential"].groupby("date").size().reindex(DAYS, fill_value=0)
V["Q016"] = {"max_date": str(nrd.idxmax().date()), "max": int(nrd.max()), "avg": r1(nrd.mean()), "ratio": round(nrd.max() / nrd.mean(), 2),
             "top5": {str(k.date()): int(v) for k, v in nrd.nlargest(5).items()}}
calendar("Q016", nrd, nrd.idxmax(), "하루 이웃 소음 신고")

# ═════════════════════════════════════════════
# 3장. 날씨가 바꾸는 민원  (SA 제외, FULL 363일)
# ═════════════════════════════════════════════


def wb(mask):
    s = fex[mask].groupby("date").size().reindex(DAYS, fill_value=0)
    g = s.groupby(bins, observed=True).mean().round(1)
    return {str(k): float(v) for k, v in g.items()}, round(float(s.corr(T)), 2)


g, c = wb(fex["problem"] == "Noise - Street/Sidewalk"); V["Q017"] = {"by_bin": g, "corr": c, "ratio_25_vs_0": round(g["25+"] / g["<0"], 1)}
bins_bar("Q017", g, hi=[0, 4])
g18, c = wb(fex["problem"] == "Noise - Residential"); V["Q018"] = {"by_bin": g18, "corr": c, "ratio_25_vs_0": round(g18["25+"] / g18["<0"], 2)}
fig, ax = new()
ks = ["<0", "0-10", "10-20", "20-25", "25+"]
for gg, col, lab in [(V["Q017"]["by_bin"], GREY, "길거리 소음"), (g18, ACC, "이웃집 소음")]:
    idx = [gg[k] / gg["<0"] for k in ks]
    ax.plot(range(5), idx, marker="o", ms=4, color=col if col != GREY else "#8C939E", lw=1.8)
    ax.text(4.15, idx[-1] + (.35 if col == ACC else 0), f"{lab} {idx[-1]:.2f}배", va="center", fontsize=7.6, color=col if col != GREY else "#555", fontweight="bold")
ax.axhline(1, color=INK, lw=.6, ls=(0, (3, 2)))
ax.set_xticks(range(5), BIN_LAB); ax.set_xlim(-.3, 5.1); ax.set_ylim(0, 6.5)
ax.set_ylabel("영하인 날 = 1 로 본 배수"); ax.grid(axis="y")
save(fig, "Q018")
hyd = ["Hydrant Running (WC3)", "Hydrant Running Full (WA4)", "Illegal Use Of A Hydrant (CIN)"]
g, c = wb(fex["problem_detail"].isin(hyd)); V["Q019"] = {"by_bin": g, "corr": c, "ratio_25_vs_10_20": round(g["25+"] / g["10-20"], 1), "ratio_25_vs_0": round(g["25+"] / g["<0"], 1)}
bins_bar("Q019", g, hi=[0, 4])
g, c = wb(fex["problem_detail"] == "POWER OUTAGE")
V["Q020"] = {"by_bin": g, "corr": c, "ratio_cold_vs_20_25": round(g["<0"] / g["20-25"], 2), "ratio_hot_vs_20_25": round(g["25+"] / g["20-25"], 2)}
bins_bar("Q020", g, hi=[0, 4])
g, c = wb(fex["problem"] == "Drinking"); V["Q021"] = {"by_bin": g, "corr": c, "ratio_25_vs_0": round(g["25+"] / g["<0"], 1)}
pdp = fex.groupby(["date", "problem"], observed=True).size().unstack(fill_value=0).reindex(DAYS, fill_value=0)
bm_ = pdp.groupby(bins, observed=True).mean()
okp = bm_.columns[(bm_.loc["<0"] >= 4)]
rr = (bm_.loc["25+", okp] / bm_.loc["<0", okp]).sort_values(ascending=False)
rr = rr.drop(["Illegal Fireworks", "Non-Residential Heat", "School Maintenance"], errors="ignore")  # 영하 쪽 값이 거의 0 이거나 한 철 행사라 배수가 무의미
pick = [k for k in rr.index[:8]]
if "Drinking" not in pick:
    pick[-1] = "Drinking"
pick += [k for k in ["Noise - Residential", "HEAT/HOT WATER"] if k in rr.index]
rr = rr[pick].sort_values(ascending=False)
V["Q021"]["ratio_by_problem"] = {k: round(float(v), 2) for k, v in rr.items()}
fig, ax = new()
yy = np.arange(len(rr))[::-1]
on = [k == "Drinking" for k in rr.index]
ax.hlines(yy, 1, rr.values, color=GREY, lw=2)
ax.scatter(rr.values, yy, color=[ACC if o else INK for o in on], s=[80 if o else 36 for o in on], zorder=3)
for yi, v, o in zip(yy, rr.values, on):
    ax.text(v + rr.max() * .025, yi, f"{v:.1f}배", va="center", ha="left", fontsize=7.4,
            color=ACC if o else "#444", fontweight="bold" if o else "normal")
ax.axvline(1, color=INK, ls=(0, (3, 2)), lw=.9)
ax.set_yticks(yy, [ko(k) for k in rr.index])
for t, o in zip(ax.get_yticklabels(), on):
    if o:
        t.set_color(ACC); t.set_fontweight("bold")
ax.set_xlim(0, rr.max() * 1.15); ax.set_ylim(-.6, len(rr) - .4)
ax.set_xlabel("25도 이상인 날 ÷ 영하인 날 (하루 평균 신고, 1 = 변화 없음)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x")
save(fig, "Q021")
g, c = wb(fex["problem"] == "Encampment"); V["Q022"] = {"by_bin": g, "corr": c, "ratio_25_vs_0": round(g["25+"] / g["<0"], 1)}
enc = fex[fex["problem"] == "Encampment"].groupby("date").size().reindex(DAYS, fill_value=0)
fig, ax = new()
ax.scatter(T.values, enc.values, s=7, color=GREY, lw=0, label="하루 = 점 하나")
mids_ = [-4, 5, 15, 22.5, 27.5]
ax.plot(mids_, [g[k] for k in ["<0", "0-10", "10-20", "20-25", "25+"]], color=ACC, marker="o", ms=5, lw=1.8, label="기온 구간별 하루 평균")
ax.annotate(f"영하 {g['<0']:.0f}건", (-4, g["<0"]), textcoords="offset points", xytext=(-8, 12), fontsize=7.6, color=ACC, fontweight="bold")
ax.annotate(f"25도 이상 {g['25+']:.0f}건", (27.5, g["25+"]), textcoords="offset points", xytext=(-40, 14), fontsize=7.6, color=ACC, fontweight="bold")
ax.set_xlabel("하루 평균 기온 (°C)"); ax.set_ylabel("하루 노숙 텐트 신고"); ax.grid(alpha=.7)
ax.legend(frameon=False, loc="upper left")
save(fig, "Q022")

sn = df[df["problem"] == "Snow or Ice"]; snd = sn.groupby("date").size()
V["Q023"] = {"total": len(sn), "feb23": int(snd["2026-02-23"]), "feb24": int(snd["2026-02-24"]),
             "two_day_share_pct": r1((snd["2026-02-23"] + snd["2026-02-24"]) / len(sn) * 100), "days_with_any": int((snd > 0).sum())}
sw = snd.reindex(pd.date_range("2025-12-01", "2026-03-31"), fill_value=0)
fig, ax = new()
ax.bar(sw.index, sw.values / 1e3, width=.85, color=[ACC if str(x.date()) in ("2026-02-23", "2026-02-24") else GREY for x in sw.index], zorder=2)
ax.annotate(f"2월 23~24일 이틀 = 1년 치의 {V['Q023']['two_day_share_pct']:.0f}%", (pd.Timestamp("2026-02-24"), sw["2026-02-24"] / 1e3),
            textcoords="offset points", xytext=(-150, -6), fontsize=7.6, color=ACC, fontweight="bold")
ticks = pd.date_range("2025-12-01", "2026-03-01", freq="MS")
ax.set_xticks(ticks, [f"{x.month}월" for x in ticks]); ax.set_ylabel("하루 눈·얼음 신고 (천 건)"); ax.grid(axis="y", zorder=0)
save(fig, "Q023")

p, c_ = pc(df["problem"] == "Snow or Ice"); V["Q024"] = {"per100k": p, "counts": c_, "SI_over_MN": r1(p["STATEN ISLAND"] / p["MANHATTAN"])}
boro("Q024", p, ["STATEN ISLAND"], kind="tile", cnt=c_)

dt = fex[fex["problem"] == "Damaged Tree"].groupby("date").size().reindex(DAYS, fill_value=0)
V["Q025"] = {"max_date": str(dt.idxmax().date()), "max": int(dt.max()), "avg": r1(dt.mean()), "ratio": r1(dt.max() / dt.mean()),
             "detail_top_that_day": {k: int(v) for k, v in fex[(fex["problem"] == "Damaged Tree") & (fex["date"] == dt.idxmax())]["problem_detail"].value_counts().head(3).items()}}
daily_line("Q025", dt, dt.idxmax(), f"2월 23일 {dt.max():,}건 (평소 {dt.mean():.0f}건)", ylabel="하루 나무 피해 신고", xytext=(10, -3))

ph = df[(df["problem"] == "Street Condition") & (df["problem_detail"] == "Pothole")]; pm = ph.groupby("m").size()
V["Q026"] = {"by_month": {int(k): int(v) for k, v in pm.items()}, "mar": int(pm[3]), "nov": int(pm[11]), "mar_over_nov": r1(pm[3] / pm[11])}
V["Q026"]["mar_over_month_avg"] = r1(pm[3] / pm.mean())
vbar("Q026", MLAB, [V["Q026"]["by_month"].get(m, 0) for m in MONTHS], hi=[MONTHS.index(3)], ylabel="신고 수",
     ref=pm.mean(), reflab=f"월평균 {pm.mean():,.0f}건")

# ═════════════════════════════════════════════
# 4장. 소음의 다른 얼굴들 (SA 제외)
# ═════════════════════════════════════════════
nr = ex[ex["problem"] == "Noise - Residential"]
s = (nr["problem_detail"].value_counts(normalize=True) * 100).round(1); s = s[s > 0]
V["Q027"] = {"n_ex_SA": len(nr), "share_pct": {k: float(v) for k, v in s.head(4).items()}}
DK = {"Loud Music/Party": "음악·파티", "Banging/Pounding": "쿵쿵 소리", "Loud Talking": "말소리", "Loud Television": "TV 소리"}
hbar("Q027", [DK[k] for k in s.head(4).index], list(s.head(4).values), hi=[0], fmt=pct, xlabel="이웃 소음 신고 중 비중 (%)")

bd = ex[(ex["problem"] == "Noise") & ex["problem_detail"].isin(["Noise, Barking Dog (NR5)", "Dog"])]
bh = bd.groupby("created_hour").size().reindex(range(24), fill_value=0)
mp = nr[nr["problem_detail"] == "Loud Music/Party"].groupby("created_hour").size().reindex(range(24), fill_value=0)
NIGHT = [22, 23, 0, 1, 2, 3]
V["Q028"] = {"n": len(bd), "peak_hour": int(bh.idxmax()), "peak": int(bh.max()), "night22_03_share_pct": r1(bh.loc[NIGHT].sum() / bh.sum() * 100),
             "party_peak_hour": int(mp.idxmax()), "party_night_share_pct": r1(mp.loc[NIGHT].sum() / mp.sum() * 100)}
fig, ax = new()
for ser, col, lab in [(mp, "#8C939E", "음악·파티 소음"), (bh, ACC, "개 짖는 소리")]:
    sh = ser / ser.sum() * 100
    ax.plot(range(24), sh.values, color=col, marker="o", ms=2.5, lw=1.6, label=lab)
    k = int(sh.idxmax())
    ax.annotate(f"{lab} 정점 {k}시", (k, sh[k]), textcoords="offset points", xytext=(6 if k < 20 else -92, 2), fontsize=7.6, color=col, fontweight="bold")
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)]); ax.set_ylabel("하루 중 비중 (%)"); ax.grid(axis="y")
ax.set_ylim(0, 15.5)
save(fig, "Q028")

nv = ex[ex["problem"] == "Noise - Vehicle"]
s = (nv["problem_detail"].value_counts(normalize=True) * 100).round(1); s = s[s > 0]
V["Q029"] = {"n": len(nv), "share_pct": {k: float(v) for k, v in s.head(3).items()}}
hbar("Q029", ["차 안 음악", "경적", "공회전"], list(s.head(3).values), hi=[0], fmt=pct, xlabel="차량 소음 신고 중 비중 (%)")

ic = ex[ex["problem_detail"].isin(["Noise, Ice Cream Truck (NR4)", "Ice Cream Truck"])]
V["Q030"] = {"n": len(ic), "mar_oct_pct": r1(ic["m"].between(3, 10).mean() * 100), "by_month": {int(k): int(v) for k, v in ic.groupby("m").size().items()},
             "peak_hour": int(ic.groupby("created_hour").size().idxmax())}
month_bar("Q030", V["Q030"]["by_month"], hi=[3, 4, 5, 6, 7, 8, 9, 10])

hl = df[df["problem"] == "Noise - Helicopter"]; p, c_ = pc(df["problem"] == "Noise - Helicopter")
V["Q031"] = {"n": len(hl), "per100k": p, "counts": c_, "MN_over_BX": r1(p["MANHATTAN"] / p["BRONX"])}
boro("Q031", p, ["MANHATTAN"], cnt=c_)
cbv = hl["community_board"].value_counts(); cbv = cbv[cbv > 0]
V["Q032"] = {"cb07_manhattan": int((hl["community_board"] == "07 MANHATTAN").sum()), "share_pct": r1((hl["community_board"] == "07 MANHATTAN").mean() * 100),
             "top_cbs": {k: int(v) for k, v in cbv.head(4).items()}}
CBN = {"07 MANHATTAN": "맨해튼 7 (어퍼웨스트사이드)", "08 BROOKLYN": "브루클린 8 (크라운하이츠)", "06 MANHATTAN": "맨해튼 6 (미드타운 동쪽)"}


def cbname(k):
    if k in CBN:
        return CBN[k]
    n_, b_ = k.split(" ", 1)
    return f"{KB.get(b_, b_)} {int(n_)}" if n_.isdigit() else k


c8 = cbv.head(8) / len(hl) * 100
hbar("Q032", [cbname(k) for k in c8.index], list(c8.values), hi=[0], fmt=pct, xlabel="헬기 소음 신고 중 비중 (%), 커뮤니티보드별")

dn = ex[ex["problem"] == "Noise"]; ah = dn[dn["problem_detail"] == "Noise: Construction Before/After Hours (NM1)"]
dv = dn["problem_detail"].value_counts(); dv = dv[dv > 0]
V["Q033"] = {"dep_noise_n": len(dn), "afterhours_n": len(ah), "afterhours_share_pct": r1(len(ah) / len(dn) * 100),
             "afterhours_peak_hour": int(ah.groupby("created_hour").size().idxmax()), "top_details": {k: int(v) for k, v in dv.head(4).items()}}
DEPK = {"Noise: Construction Before/After Hours (NM1)": "허가 시간 밖 공사", "Construction": "공사(옛 분류)", "Noise, Barking Dog (NR5)": "개 짖음",
        "Noise: Alarms (NR3)": "경보기", "Noise: Construction Equipment (NC1)": "공사 장비", "Noise: air condition/ventilation equipment (NV1)": "에어컨·환풍기",
        "Dog": "개(옛 분류)", "Construction Equipment": "공사 장비(옛 분류)", "Alarm": "경보기(옛 분류)", "Noise: Jack Hammering (NC2)": "착암기"}
d8 = dv.head(8) / len(dn) * 100
hbar("Q033", [DEPK.get(k, k) for k in d8.index], list(d8.values), hi=[0], fmt=pct, xlabel="환경보호국 소음 신고 중 비중 (%)")

nz = ex[ex["problem"].astype(str).str.startswith("Noise")]
w = nz.assign(wk=nz["created_dow"] >= 6).groupby("problem", observed=True)["wk"].mean().mul(100).round(1)
V["Q034"] = {"weekend_share_pct": {k: float(v) for k, v in w.items()}, "expected_pct": 28.6}
ws_ = w.sort_values(ascending=False)
hbar("Q034", [ko(k) for k in ws_.index], list(ws_.values), hi=["공원 소음"], fmt=pct, ref=28.6, reflab="요일 고르면 28.6%",
     xlabel="토·일요일에 들어온 비중 (%)", xmax=60)

# ═════════════════════════════════════════════
# 5장. 동물의 도시
# ═════════════════════════════════════════════
AK = {"Cat": "고양이", "Raccoon": "너구리", "Rat or Mouse": "쥐", "Bird": "새", "Other": "기타", "Squirrel": "다람쥐", "Opossum": "주머니쥐",
      "Dog": "개", "Deer": "사슴", "Rooster": "수탉", "Farm Animal": "가축", "Snake": "뱀", "Ferret": "족제비(페럿)", "Monkey": "원숭이",
      "Iguana": "이구아나", "Turtle Under 4 inches Long": "10cm 미만 거북", "Neglected": "방치", "Other (complaint details)": "기타",
      "Tortured": "학대", "No Shelter": "쉼터 없음", "Chained": "묶어 둠", "In Car": "차 안 방치", "Dog Off Leash": "목줄 없는 개",
      "Injured Wildlife": "다친 야생동물", "Dead Animal": "동물 사체", "Wildlife Sighting": "야생동물 목격", "Animal Waste": "배설물",
      "Domestic Strays": "떠돌이 반려동물", "Removing Wildlife": "야생동물 퇴치"}


def counts_of(sub):
    s = sub["problem_detail"].value_counts(); return s[s > 0]


da = df[df["problem"] == "Dead Animal"]; s = counts_of(da)
V["Q035"] = {"n": len(da), "counts": {k: int(v) for k, v in s.items()}, "cat_share_pct": r1(s["Cat"] / len(da) * 100)}
hbar("Q035", [AK.get(k, k) for k in s.index], list(s.values), hi=[0], xlabel="동물 사체 신고 수")
p, c_ = pc((df["problem"] == "Dead Animal") & (df["problem_detail"] == "Raccoon"))
V["Q036"] = {"per100k": p, "counts": c_, "SI_over_MN": r1(p["STATEN ISLAND"] / p["MANHATTAN"])}
boro("Q036", p, ["STATEN ISLAND"], kind="tile", cnt=c_, fmt=f1)
deer_mask = (df["problem"] == "Dead Animal") & (df["problem_detail"] == "Deer")
p, c_ = pc(deer_mask)
V["Q037"] = {"counts": c_, "SI_share_pct": r1(c_["STATEN ISLAND"] / sum(c_.values()) * 100), "total": sum(c_.values())}
bg = df[["latitude", "longitude"]].dropna().sample(60000, random_state=42)
bg = bg[bg["latitude"].between(40.49, 40.92) & bg["longitude"].between(-74.27, -73.69)]
dr = df[deer_mask][["latitude", "longitude"]].dropna()
fig, ax = new()
ax.scatter(bg["longitude"], bg["latitude"], s=.15, color="#d6d9de", lw=0, rasterized=True)
drg = dr.assign(la=(dr["latitude"] / .015).round() * .015, lo=(dr["longitude"] / .015).round() * .015).groupby(["la", "lo"]).size().reset_index(name="n")
ax.scatter(drg["lo"], drg["la"], s=14 + drg["n"] * 16, color=ACC, alpha=.8, lw=.5, ec="white", zorder=3)
ax.scatter([-73.80], [40.585], s=14 + 3 * 16, color=ACC, alpha=.8, lw=.5, ec="white")
ax.text(-73.79, 40.585, " = 같은 동네 3건", fontsize=7.2, color="#555", va="center")
ax.set_aspect(1 / np.cos(np.radians(40.7)))
ax.text(-74.25, 40.66, f"스태튼아일랜드\n{c_['STATEN ISLAND']}건 ({V['Q037']['SI_share_pct']:.0f}%)", fontsize=7.2, color=ACC, fontweight="bold")
ax.text(-73.83, 40.52, f"나머지 4개 구\n합쳐서 {sum(c_.values()) - c_['STATEN ISLAND']}건", fontsize=7.6, color="#555")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
save(fig, "Q037")

ia = df[df["problem"] == "Illegal Animal Kept as Pet"]; s = counts_of(ia)
V["Q038"] = {"n": len(ia), "counts": {k: int(v) for k, v in s.items()}, "rooster_share_pct": r1(s["Rooster"] / len(ia) * 100),
             "rooster_per100k": pc((df["problem"] == "Illegal Animal Kept as Pet") & (df["problem_detail"] == "Rooster"))[0]}
hbar("Q038", [AK.get(k, k) for k in s.index], list(s.values), hi=[0], xlabel="불법 반려동물 신고 수")
isd = df[df["problem"] == "Illegal Animal Sold"]; s = counts_of(isd)
V["Q039"] = {"n": len(isd), "counts": {k: int(v) for k, v in s.items()}, "turtle_share_pct": r1(s["Turtle Under 4 inches Long"] / len(isd) * 100)}
hbar("Q039", [AK.get(k, k) for k in s.index], list(s.values), hi=[0], xlabel="불법 판매 동물 신고 수")
rs = (df["problem"] == "Rodent") & (df["problem_detail"] == "Rat Sighting"); p, c_ = pc(rs)
V["Q040"] = {"per100k": p, "counts": c_, "MN_over_SI": r1(p["MANHATTAN"] / p["STATEN ISLAND"])}
boro("Q040", p, ["MANHATTAN"], kind="dot", cnt=c_)
rm = df[rs].groupby("m").size()
s = fex[fex["problem"] == "Rodent"].groupby("date").size().reindex(DAYS, fill_value=0)
V["Q041"] = {"by_month": {int(k): int(v) for k, v in rm.items()}, "aug_over_dec": r1(rm[8] / rm[12]), "corr_daily_rodent_temp": round(float(s.corr(T)), 2)}
mt = TALL[(TALL.index >= "2025-11-01") & (TALL.index <= "2026-10-31")]
mtemp = mt.groupby(mt.index.month).mean()
fig, ax = new()
x = np.arange(12)
ax.bar(x, [rm[m] for m in MONTHS], color=[ACC if m in (8, 12) else GREY for m in MONTHS], width=.68, zorder=2)
for i, m_ in enumerate(MONTHS):
    if m_ in (8, 12):
        ax.text(i, rm[m_] + 40, f"{rm[m_]:,}", ha="center", fontsize=7.2, color=ACC, fontweight="bold")
ax.set_xticks(x, MLAB); ax.set_ylabel("쥐 목격 신고"); ax.yaxis.set_major_formatter(THOU); ax.grid(axis="y", zorder=0); ax.tick_params(axis="x", length=0)
ax2 = ax.twinx()
ax2.plot(x, [mtemp[m] for m in MONTHS], color=INK, marker="o", ms=2.5, lw=1.1)
ax2.set_ylabel("월평균 기온 (°C, 선)"); ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
ax2.set_ylim(-10, 32)
save(fig, "Q041")

mq = df[df["problem"] == "Mosquitoes"].groupby("m").size()
V["Q042"] = {"n": int(mq.sum()), "by_month": {int(k): int(v) for k, v in mq.items()}, "aug_share_pct": r1(mq[8] / mq.sum() * 100),
             "aug_sep_share_pct": r1((mq[8] + mq[9]) / mq.sum() * 100)}
month_bar("Q042", V["Q042"]["by_month"], hi=[8, 9])
bb = df[df["problem"] == "Harboring Bees/Wasps"].groupby("m").size()
V["Q043"] = {"n": int(bb.sum()), "by_month": {int(k): int(v) for k, v in bb.items()}, "sep_share_pct": r1(bb[9] / bb.sum() * 100)}
month_bar("Q043", V["Q043"]["by_month"], hi=[9])
p, c_ = pc(df["problem"] == "Unsanitary Pigeon Condition"); V["Q044"] = {"per100k": p, "counts": c_, "MN_over_BX": r1(p["MANHATTAN"] / p["BRONX"])}
boro("Q044", p, ["MANHATTAN"], cnt=c_)
ab = df[df["problem"] == "Animal-Abuse"]; s = counts_of(ab)
V["Q045"] = {"n": len(ab), "counts": {k: int(v) for k, v in s.items()}, "neglected_share_pct": r1(s["Neglected"] / len(ab) * 100)}
hbar("Q045", [AK.get(k, k) for k in s.index], list(s.values), hi=[0], xlabel="동물 학대 신고 수")
ap = df[df["problem"] == "Animal in a Park"]; s = counts_of(ap)
V["Q046"] = {"n": len(ap), "counts": {k: int(v) for k, v in s.items()}, "offleash_share_pct": r1(s["Dog Off Leash"] / len(ap) * 100)}
hbar("Q046", [AK.get(k, k) for k in s.index], list(s.values), hi=[0], xlabel="공원 동물 신고 수")

# ═════════════════════════════════════════════
# 6장. 도로 위의 신경전
# ═════════════════════════════════════════════
ip = df[df["problem"] == "Illegal Parking"]
s = (ip["problem_detail"].value_counts(normalize=True) * 100).round(1); s = s[s > 0]
V["Q047"] = {"n": len(ip), "share_pct": {k: float(v) for k, v in s.head(6).items()}}
PK = {"Blocked Hydrant": "소화전 앞", "Posted Parking Sign Violation": "주차 표지판 위반", "Blocked Sidewalk": "보도 막음",
      "Commercial Overnight Parking": "상용차 밤샘 주차", "Blocked Crosswalk": "횡단보도 막음", "Double Parked Blocking Traffic": "이중 주차(차로)",
      "Parking Permit Improper Use": "주차 허가증 악용", "Blocked Bike Lane": "자전거도로 막음"}
hbar("Q047", [PK[k] for k in s.head(8).index], list(s.head(8).values), hi=[0], fmt=pct, xlabel="불법주차 신고 중 비중 (%)")
p, c_ = pc(df["problem"] == "Blocked Driveway"); V["Q048"] = {"per100k": p, "counts": c_, "QN_over_MN": r1(p["QUEENS"] / p["MANHATTAN"])}
BORO_ROAD = {}
BORO_ROAD["차고 앞 막힘"] = (p, c_)
p, c_ = pc(df["problem"] == "Abandoned Vehicle"); V["Q049"] = {"per100k": p, "counts": c_, "QN_over_MN": r1(p["QUEENS"] / p["MANHATTAN"])}
boro("Q049", p, ["QUEENS"], kind="dot", cnt=c_)
BORO_ROAD["버려진 차"] = (p, c_)
p, c_ = pc((df["problem"] == "Illegal Parking") & (df["problem_detail"] == "Blocked Bike Lane")); V["Q050"] = {"per100k": p, "counts": c_, "MN_over_QN": r1(p["MANHATTAN"] / p["QUEENS"])}
V["Q050"]["MN_over_SI"] = r1(p["MANHATTAN"] / p["STATEN ISLAND"])
boro("Q050", p, ["MANHATTAN", "BROOKLYN"], cnt=c_)
BORO_ROAD["자전거도로 막음"] = (p, c_)
p, c_ = pc(df["problem"] == "Street Light Condition"); V["Q051"] = {"per100k": p, "counts": c_, "SI_over_BK": r1(p["STATEN ISLAND"] / p["BROOKLYN"])}
boro("Q051", p, ["STATEN ISLAND"], kind="tile", cnt=c_)
BORO_ROAD["가로등 고장"] = (p, c_)
p, c_ = pc(df["problem"] == "Abandoned Bike"); V["Q052"] = {"per100k": p, "counts": c_, "BK_over_BX": r1(p["BROOKLYN"] / p["BRONX"])}
boro("Q052", p, ["BROOKLYN"], kind="dot", cnt=c_, fmt=f1)
BORO_ROAD["버려진 자전거"] = (p, c_)
p, c_ = pc(df["problem"] == "E-Scooter")
V["Q053"] = {"counts": c_, "total": sum(c_.values()), "QN_share_pct": r1(c_["QUEENS"] / sum(c_.values()) * 100),
             "QN_BX_share_pct": r1((c_["QUEENS"] + c_["BRONX"]) / sum(c_.values()) * 100)}
boro("Q053", c_, ["QUEENS", "BRONX"], xlabel="전동킥보드 신고 수 (1년)", fmt=f0)
BORO_ROAD["전동킥보드"] = (p, c_)
p, c_ = pc(df["problem"] == "Broken Parking Meter")
V["Q054"] = {"per100k": p, "counts": c_, "MN_share_pct": r1(c_["MANHATTAN"] / sum(c_.values()) * 100), "MN_over_BX": r1(p["MANHATTAN"] / p["BRONX"])}
V["Q054"]["MN_over_SI"] = r1(p["MANHATTAN"] / p["STATEN ISLAND"])
BORO_ROAD["주차 미터 고장"] = (p, c_)
sh54 = pd.Series(c_) / sum(c_.values()) * 100
sh54 = sh54.sort_values(ascending=False)
hbar("Q054", [KB[b] for b in sh54.index], list(sh54.values), hi=["맨해튼"], fmt=pct, xlabel="고장 난 주차 미터 신고 중 자치구별 비중 (%)")
# Q048: 6장 도로 민원을 한 장에 — 자치구 x 민원, 뉴욕 평균 대비 배수
hm_rows = list(BORO_ROAD.keys())
hm = np.array([[BORO_ROAD[r][0][b] / (sum(BORO_ROAD[r][1].values()) / pop.sum() * 1e5) for b in BORO] for r in hm_rows])
fig, ax = new()
im = ax.imshow(hm, cmap=HEAT_CMAP, vmin=0, vmax=3, aspect="auto")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.6, color="white" if v > 1.9 else "#333",
                fontweight="bold" if (hm_rows[i] == "차고 앞 막힘" and BORO[j] == "QUEENS") else "normal")
ax.add_patch(plt.Rectangle((BORO.index("QUEENS") - .5, -.5), 1, 1, fill=False, ec=ACC, lw=2.2))
ax.set_xticks(range(5), [KB[b] for b in BORO]); ax.set_yticks(range(len(hm_rows)), hm_rows)
ax.xaxis.tick_top()
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.015, aspect=16)
cb.set_label("1인당 신고, 뉴욕 평균 = 1", fontsize=7.6, labelpad=8); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.2, length=0)
save(fig, "Q048")
ih = ip.groupby("created_hour").size().reindex(range(24), fill_value=0)
V["Q055"] = {"by_hour": {int(k): int(v) for k, v in ih.items()}, "peak_hour": int(ih.idxmax()), "min_hour": int(ih.idxmin()), "peak_over_min": r1(ih.max() / ih.min())}
vbar("Q055", [str(h) for h in range(24)], list(ih.values / 1e3), hi=[8, 3], fmt=lambda v: f"{v:.0f}", ylabel="1년 불법주차 신고 (천 건)")

# ═════════════════════════════════════════════
# 7장. 집이 아프다
# ═════════════════════════════════════════════
h = df[df["agency"] == "HPD"]; p, c_ = pc(df["agency"] == "HPD"); V["Q056"] = {"per100k": p, "counts": c_, "BX_over_SI": r1(p["BRONX"] / p["STATEN ISLAND"])}
boro("Q056", p, ["BRONX"], kind="tile", cnt=c_)
u = h[h["problem"] == "UNSANITARY CONDITION"]; s = counts_of(u)
V["Q057"] = {"n": len(u), "counts": {k: int(v) for k, v in s.items()}, "pests_share_pct": r1(s["PESTS"] / len(u) * 100), "mold_share_pct": r1(s["MOLD"] / len(u) * 100)}
UK = {"PESTS": "해충", "MOLD": "곰팡이", "GARBAGE/RECYCLING STORAGE": "쓰레기 보관", "SEWAGE": "오수"}
hbar("Q057", [UK.get(k, k) for k in s.index], list(s.values / len(u) * 100), hi=[0], fmt=pct, xlabel="집 안 위생 민원 중 비중 (%)")
pe = h[h["problem_detail"] == "PESTS"].groupby("m").size(); V["Q058"] = {"by_month": {int(k): int(v) for k, v in pe.items()}, "aug_over_feb": round(pe[8] / pe[2], 2)}
month_bar("Q058", V["Q058"]["by_month"], hi=[8, 2])
ht = h[h["problem"] == "HEAT/HOT WATER"]; a = ht["address_key"].value_counts()
V["Q059"] = {"n_heat": len(ht), "n_buildings": int(a.size), "top1pct_buildings": int(a.size // 100), "top1pct_share_pct": r1(a.head(a.size // 100).sum() / len(ht) * 100),
             "missing_address": int(ht["address_key"].isna().sum())}
concentration("Q059", a.values, "건물", [1, 10])
b1 = ht[ht["address_key"] == a.index[0]]
V["Q060"] = {"address": a.index[0], "borough": str(b1["borough"].mode()[0]), "count": int(a.iloc[0]), "days_with_report": int(b1["date"].nunique()),
             "second": {a.index[1]: int(a.iloc[1])}}
bmap = ht.groupby("address_key")["borough"].agg(lambda x: x.mode()[0]).astype(str).to_dict() if False else \
    ht[ht["address_key"].isin(a.head(10).index)].groupby("address_key", observed=True)["borough"].agg(lambda x: str(x.mode()[0])).to_dict()
hbar("Q060", addr_labels(a.head(10).index, bmap), list(a.head(10).values), hi=[0], xlabel="1년 난방·온수 민원 수 (번지는 가림)")
p, c_ = pc(df["problem"] == "Elevator")
V["Q061"] = {"per100k": p, "counts": c_, "BX_over_QN": r1(p["BRONX"] / p["QUEENS"]), "BX_share_pct": r1(c_["BRONX"] / sum(c_.values()) * 100)}
V["Q061"]["BX_over_SI"] = r1(p["BRONX"] / p["STATEN ISLAND"])
boro("Q061", p, ["BRONX"], cnt=c_)
apl = h[h["problem"] == "APPLIANCE"]; s = counts_of(apl)
V["Q062"] = {"n": len(apl), "counts": {k: int(v) for k, v in s.items()}, "range_share_pct": r1(s["ELECTRIC/GAS RANGE"] / len(apl) * 100)}
hbar("Q062", ["가스·전기레인지", "냉장고", "전자레인지"], list(s.values), hi=[0], xlabel="가전 고장 신고 수")
acc = rd.str.contains(r"gain access|gain entry|unable to complete the inspection|not able to gain", case=False)
hp = df["agency"] == "HPD"
accp = acc[hp].groupby(df["problem"][hp], observed=True).mean().mul(100)
V["Q063"] = {"HPD_noaccess_pct": r1(acc[hp].mean() * 100), "by_problem_pct": {k: r1(v) for k, v in accp.sort_values(ascending=False).head(6).items()}}
accn = df["problem"][hp].value_counts(); a63 = accp[accn[accn > 10000].index].sort_values(ascending=False)
a63 = a63[accn[a63.index].sort_values(ascending=False).index[:10]].sort_values(ascending=False)
hbar("Q063", [ko(k) for k in a63.index], list(a63.values), fmt=pct, ref=V["Q063"]["HPD_noaccess_pct"], reflab=f"전체 {V['Q063']['HPD_noaccess_pct']:.0f}%",
     xlabel="검사관이 집에 들어가지 못한 비율 (%)", hi=[], xmax=55)
vi = rd.str.contains("Violations were issued")
vg = vi[hp].groupby(df["problem"][hp], observed=True).agg(["mean", "size"]); vg = vg[vg["size"] > 10000]
V["Q064"] = {"HPD_violation_pct_all": r1(vi[hp].mean() * 100), "by_problem_pct": {k: r1(v * 100) for k, v in vg["mean"].sort_values().items()}}
v64 = (vg["mean"] * 100)[vg["size"].sort_values(ascending=False).index[:10]].sort_values()
hbar("Q064", [ko(k) for k in v64.index], list(v64.values), hi=["난방·온수"], fmt=pct, xlabel="위반 딱지가 발부된 비율 (%)",
     ref=V["Q064"]["HPD_violation_pct_all"], reflab=f"전체 {V['Q064']['HPD_violation_pct_all']:.0f}%")

# ═════════════════════════════════════════════
# 8장. 한 주소가 바꾼 통계 (SA 가 주인공)
# ═════════════════════════════════════════════
x = df[df["address_key"] == SA]
V["Q065"] = {"address": SA, "borough": "BRONX", "rows": len(x), "share_all_pct": round(len(x) / N * 100, 2),
             "noise_res_rows": int((x["problem"] == "Noise - Residential").sum()), "active_days": int(x["date"].nunique())}
aa = df["address_key"].value_counts()
a10 = aa.head(10)
bm = df[df["address_key"].isin(a10.index)].groupby("address_key")["borough"].agg(lambda z: str(z.mode()[0])).to_dict()
hbar("Q065", addr_labels(a10.index, bm), list(a10.values), hi=[0], xlabel="1년 신고 수, 주소별 (번지는 가림)")

nra = df[df["problem"] == "Noise - Residential"]; xs = x[x["problem"] == "Noise - Residential"]
nbx = int((nra["borough"] == "BRONX").sum())
V["Q066"] = {"share_NYC_noise_res_pct": r1(len(xs) / len(nra) * 100), "share_BX_noise_res_pct": r1(len(xs) / nbx * 100)}
fig, ax = new()
rows = [("뉴욕 전체 이웃 소음", len(nra)), ("브롱크스 이웃 소음", nbx)]
for i, (lab, tot) in enumerate(rows):
    yy = 1 - i
    sa_ = len(xs) / tot * 100
    ax.barh(yy, sa_, color=ACC, height=.55); ax.barh(yy, 100 - sa_, left=sa_, color=GREY, height=.55)
    ax.text(sa_ / 2, yy, f"한 주소\n{sa_:.1f}%", ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
    ax.text(sa_ + (100 - sa_) / 2, yy, f"나머지 모든 주소 {100 - sa_:.1f}%  ({tot - len(xs):,}건)", ha="center", va="center", fontsize=7.6, color="#333")
ax.set_yticks([1, 0], [r[0] for r in rows]); ax.set_xlim(0, 100); ax.set_ylim(-.6, 1.6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "Q066")

nrb = nra[nra["borough"] == "BRONX"]; dec = nrb[nrb["created_date"].dt.to_period("M") == "2025-12"]
V["Q067"] = {"bronx_dec2025_all": len(dec), "bronx_dec2025_exSA": int((dec["address_key"] != SA).sum()), "ratio": round(len(dec) / (dec["address_key"] != SA).sum(), 2),
             "bronx_jun2026_all": int((nrb["created_date"].dt.to_period("M") == "2026-06").sum())}
mall = nrb.groupby(nrb["created_date"].dt.to_period("M")).size()
mex = nrb[nrb["address_key"] != SA].groupby(nrb["created_date"].dt.to_period("M")).size().reindex(mall.index, fill_value=0)
fig, ax = new()
xx = np.arange(len(mall))
ax.fill_between(xx, mex.values / 1e3, mall.values / 1e3, color=ACC, alpha=.18, lw=0)
ax.plot(xx, mall.values / 1e3, color=ACC, marker="o", ms=3, lw=1.6, label="그 주소 포함")
ax.plot(xx, mex.values / 1e3, color=INK, marker="o", ms=3, lw=1.4, label="그 주소 제외")
ax.annotate(f"12월 {len(dec):,}건 → 빼면 {V['Q067']['bronx_dec2025_exSA']:,}건", (1, len(dec) / 1e3), textcoords="offset points", xytext=(10, -4),
            fontsize=7.6, color=ACC, fontweight="bold")
ax.set_xticks(xx, [f"{p_.month}월" for p_ in mall.index]); ax.set_ylabel("브롱크스 이웃 소음 (천 건)"); ax.grid(axis="y")
ax.legend(frameon=False, loc="upper right"); ax.set_ylim(0, None)
save(fig, "Q067")

xx_ = xs[xs["date"] == "2025-11-30"]["created_date"].sort_values()
V["Q068"] = {"date": "2025-11-30", "count": len(xx_), "median_gap_sec": float(xx_.diff().dt.total_seconds().median()), "hours_covered": int(xx_.dt.hour.nunique()),
             "nov7_dec16_share_pct": r1(x["date"].between("2025-11-07", "2025-12-16").mean() * 100)}
h68 = xx_.dt.hour.value_counts().reindex(range(24), fill_value=0)
vbar("Q068", [str(i) for i in range(24)], list(h68.values), hi=[], ylabel="2025년 11월 30일 시간대별 신고", show=False,
     ref=len(xx_) / 24, reflab=f"시간당 평균 {len(xx_) / 24:.0f}건 = 약 40초에 1건")

xc = pd.Series(nypd_cat(x[x["agency"] == "NYPD"]["resolution_description"])).value_counts(normalize=True).mul(100).round(1)
V["Q069"] = {"mobile_pct": r1((x["open_data_channel_type"] == "MOBILE").mean() * 100), "outcome_pct": {k: float(v) for k, v in xc.items()}}
o69 = xc.reindex(["noevidence", "corrected", "notnecessary", "noentry", "summons", "arrest"], fill_value=0)
hbar("Q069", [NYPD_KO[k] for k in o69.index], list(o69.values), hi=[0], fmt=pct, xlabel="그 주소 신고의 경찰 처리 결과 (%)")

y_ = df[df["address_key"] == "1869 83 STREET"]
V["Q070"] = {"address": "1869 83 STREET", "borough": "BROOKLYN", "rows": len(y_), "days_with_report": int(y_["date"].nunique()), "days_in_data": int(df["date"].nunique()),
             "noise_res_pct": r1((y_["problem"] == "Noise - Residential").mean() * 100), "per_day": r1(len(y_) / y_["date"].nunique())}
calendar("Q070", y_.groupby("date").size(), None, "하루 신고 수")

ipa = ip["address_key"].value_counts(); z = ip[ip["address_key"] == ipa.index[0]]
V["Q071"] = {"address": ipa.index[0], "borough": "QUEENS", "rows": int(ipa.iloc[0]), "days": int(z["date"].nunique()), "per_day": r1(ipa.iloc[0] / z["date"].nunique()),
             "blocked_sidewalk_pct": r1((z["problem_detail"] == "Blocked Sidewalk").mean() * 100)}
bm = ip[ip["address_key"].isin(ipa.head(10).index)].groupby("address_key")["borough"].agg(lambda z_: str(z_.mode()[0])).to_dict()
hbar("Q071", addr_labels(ipa.head(10).index, bm), list(ipa.head(10).values), hi=[0], xlabel="1년 불법주차 신고 수, 주소별 (번지는 가림)")

ws = df[df["problem"] == "Noise - House of Worship"]; wa = ws["address_key"].value_counts()
V["Q072"] = {"n": len(ws), "top_address": wa.index[0], "top_rows": int(wa.iloc[0]), "share_pct": r1(wa.iloc[0] / len(ws) * 100)}
bm = ws[ws["address_key"].isin(wa.head(5).index)].groupby("address_key")["borough"].agg(lambda z_: str(z_.mode()[0])).to_dict()
w5 = list(wa.head(5).values) + [len(ws) - wa.head(5).sum()]
hbar("Q072", addr_labels(wa.head(5).index, bm) + ["나머지 모든 주소"], w5, hi=[0], xlabel="종교시설 소음 신고 수 (번지는 가림)")

ha = hl["address_key"].value_counts()
V["Q073"] = {"n": len(hl), "n_addresses": int(ha.size), "top10_share_pct": r1(ha.head(10).sum() / len(hl) * 100), "top1": {ha.index[0]: int(ha.iloc[0])}}
fig, ax = new()
t20 = ha.head(20)
ax.bar(range(1, 21), t20.values, color=[ACC] * 10 + [GREY] * 10, width=.7, zorder=2)
ax.text(5.5, t20.max() * .85, f"상위 10곳 = 헬기 소음 신고의 {V['Q073']['top10_share_pct']:.0f}%\n(전체 {ha.size:,}개 주소 중)", ha="left", fontsize=7.2, color=ACC, fontweight="bold")
ax.set_xticks(range(1, 21)); ax.set_xlabel("신고 많은 주소 순위"); ax.set_ylabel("헬기 소음 신고 수"); ax.grid(axis="y", zorder=0); ax.tick_params(axis="x", length=0)
save(fig, "Q073")

V["Q074"] = {"n_addresses": int(aa.size), "rows_with_address": int(aa.sum()), "top1pct_share_pct": r1(aa.head(aa.size // 100).sum() / aa.sum() * 100),
             "top0.1pct_share_pct": r1(aa.head(aa.size // 1000).sum() / aa.sum() * 100), "single_report_addr_pct": r1((aa == 1).mean() * 100)}
concentration("Q074", aa.values, "주소", [0.1, 1, 10])

rep = df.assign(r=df["incident_group_size"] > 1).groupby("problem", observed=True).agg(r=("r", "mean"), n=("r", "size"))
rep = rep[rep["n"] > 5000].sort_values("r", ascending=False)
V["Q075"] = {"rows_in_repeat_groups_pct": r1((df["incident_group_size"] > 1).mean() * 100), "n_groups": int(df["incident_group_id"].nunique()),
             "top_problems_pct": {k: r1(v * 100) for k, v in rep["r"].head(5).items()}}
r10 = rep["r"].head(10) * 100
hbar("Q075", [ko(k) for k in r10.index], list(r10.values), hi=[], fmt=pct, ref=V["Q075"]["rows_in_repeat_groups_pct"],
     reflab=f"전체 {V['Q075']['rows_in_repeat_groups_pct']:.0f}%", xlabel="같은 날·같은 곳·같은 민원이 2건 이상인 묶음에 든 비율 (%)", xmax=68)

# ═════════════════════════════════════════════
# 9장. 경찰이 도착했을 때 (SA 제외)
# ═════════════════════════════════════════════
n = ex[ex["agency"] == "NYPD"].copy(); n["cat"] = nypd_cat(n["resolution_description"])
oc = n["cat"].value_counts(normalize=True).mul(100).round(1)
V["Q076"] = {"n_nypd_exSA": len(n), "outcome_pct": {k: float(v) for k, v in oc.items()}}
o8 = oc.head(8)
hbar("Q076", [NYPD_KO[k] for k in o8.index], list(o8.values), hi=[0], fmt=pct, xlabel="경찰 신고의 처리 결과 (%)")
tb = pd.crosstab(n["problem"].astype(str), n["cat"]); tb = tb[tb.sum(axis=1) > 3000]
tp = tb.div(tb.sum(axis=1), axis=0).mul(100).round(1)
V["Q077"] = {"noise_res_summons_pct": float(tp.loc["Noise - Residential", "summons"]), "noise_res_n": int(tb.loc["Noise - Residential"].sum()),
             "noise_res_corrected_pct": float(tp.loc["Noise - Residential", "corrected"])}
ns = tp.loc[[i for i in tp.index if i.startswith("Noise")], "summons"].sort_values(ascending=False)
hbar("Q077", [ko(k) for k in ns.index], list(ns.values), hi=["이웃 소음"], fmt=lambda v: f"{v:.1f}%", xmax=4, xlabel="소환장(딱지) 발부율 (%)")
V["Q078"] = {"summons_pct_top": {k: float(v) for k, v in tp["summons"].sort_values(ascending=False).head(5).items()}}
s78 = tp["summons"].sort_values(ascending=False).head(10)
hbar("Q078", [ko(k) for k in s78.index], list(s78.values), hi=[0], fmt=pct, xlabel="소환장(딱지) 발부율 (%)")
V["Q079"] = {"fireworks_noevidence_pct": float(tp.loc["Illegal Fireworks", "noevidence"]),
             "top_noevidence": {k: float(v) for k, v in tp["noevidence"].sort_values(ascending=False).head(5).items()}}
s79 = tp["noevidence"].sort_values(ascending=False).head(10)
hbar("Q079", [ko(k) for k in s79.index], list(s79.values), hi=[0], fmt=pct, xlabel="'가 보니 흔적 없음'으로 끝난 비율 (%)")
arr = n[n["cat"] == "arrest"]; av = arr["problem"].value_counts(); av = av[av > 0]
V["Q080"] = {"arrests": len(arr), "arrest_pct": round(len(arr) / len(n) * 100, 2), "by_problem": {k: int(v) for k, v in av.head(5).items()}}
hbar("Q080", [ko(k) for k in av.head(7).index], list(av.head(7).values), hi=[], xlabel=f"체포로 끝난 경찰 신고 {len(arr):,}건의 민원 유형")
V["Q081"] = {"noise_res_noentry_pct": float(tp.loc["Noise - Residential", "noentry"])}
o81 = tp.loc["Noise - Residential"].sort_values(ascending=False); o81 = o81[o81 >= 0.3]
hbar("Q081", [NYPD_KO[k] for k in o81.index], list(o81.values), hi=["들어가지 못함"], fmt=pct, xlabel="이웃 소음 신고의 경찰 처리 결과 (%)")
V["Q082"] = {"encampment_found_pct": float(tp.loc["Encampment", "enc_found"]), "encampment_none_pct": float(tp.loc["Encampment", "enc_none"]),
             "n": int(tb.loc["Encampment"].sum())}
o82 = tp.loc["Encampment"].sort_values(ascending=False); o82 = o82[o82 >= 0.5]
hbar("Q082", [NYPD_KO[k] for k in o82.index], list(o82.values), hi=["텐트 확인"], fmt=pct, xlabel="노숙 텐트 신고의 경찰 처리 결과 (%)")
nn = n[n["problem"].astype(str).str.startswith("Noise")]
cr = nn.groupby("created_hour")["cat"].apply(lambda s_: (s_ == "corrected").mean() * 100).round(1)
V["Q083"] = {"corrected_by_hour_pct": {int(k): float(v) for k, v in cr.items()}, "max_hour": int(cr.idxmax()), "min_hour": int(cr.idxmin())}
fig, ax = new()
ax.plot(cr.index, cr.values, color=INK, marker="o", ms=2.8, lw=1.4)
for k in [int(cr.idxmax()), int(cr.idxmin())]:
    ax.scatter([k], [cr[k]], color=ACC, s=22, zorder=4)
    ax.annotate(f"{k}시 {cr[k]:.0f}%", (k, cr[k]), textcoords="offset points", xytext=(-14, 7 if k == cr.idxmax() else -13), fontsize=7.2, color=ACC, fontweight="bold")
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)]); ax.set_ylim(0, 50); ax.set_ylabel("'현장 정리'로 끝난 비율 (%)"); ax.grid(axis="y")
save(fig, "Q083")
ipn = n[n["problem"] == "Illegal Parking"]
sb = ipn.groupby("borough", observed=True)["cat"].apply(lambda s_: (s_ == "summons").mean() * 100).round(1)
V["Q084"] = {"summons_pct_by_borough": {k: float(v) for k, v in sb.items()}}
boro("Q084", sb.to_dict(), ["BRONX", "STATEN ISLAND"], xlabel="불법주차 신고 중 소환장 발부율 (%)", fmt=pct)
cc_ = n[n["is_closed"] & (n["resolution_hours"] > 0) & ~n["flag_status_date_mismatch"] & ~n["flag_midnight_close"]]
md = cc_.groupby("problem", observed=True)["resolution_hours"].agg(["median", "size"]); md = md[md["size"] > 3000].sort_values("median")
V["Q085"] = {"median_hours": {k: round(float(v), 2) for k, v in md["median"].items()}, "nypd_overall_median_h": round(float(cc_["resolution_hours"].median()), 2)}
m85 = md["median"] * 60
sel = list(m85.index[:6]) + list(m85.index[-4:])
m85 = m85[sel]
hbar("Q085", [ko(k) for k in m85.index], list(m85.values), hi=[0], fmt=lambda v: f"{v:.0f}분",
     ref=V["Q085"]["nypd_overall_median_h"] * 60, reflab=f"경찰 전체 {V['Q085']['nypd_overall_median_h'] * 60:.0f}분",
     xlabel="접수에서 종료까지 걸린 시간의 중앙값 (분). 빠른 6종과 느린 4종")

# ═════════════════════════════════════════════
# 10장. 답변서에 숨은 이야기
# ═════════════════════════════════════════════
nf = rd.str.contains(r"could not find|found no condition|no evidence|no criminal violation upon|were gone|could not locate|unable to locate|did not find|no encampment|no Encampment|was not found", case=False)
gq = nf.groupby(df["agency"], observed=True).mean().mul(100).round(1); gn = df.groupby("agency", observed=True).size(); gq = gq[gn > 20000]
V["Q086"] = {"notfound_pct_by_agency": {k: float(v) for k, v in gq.sort_values(ascending=False).items()}}
g86 = gq.sort_values(ascending=False)
hbar("Q086", [ko(k) for k in g86.index], list(g86.values), hi=[0], fmt=pct, xlabel="'가 보니 없었다'는 답의 비율 (%)")

dh = df["agency"] == "DHS"; rdh = rd[dh]
dcat = pd.Series(np.select(
    [rdh.str.contains("could not find"), rdh.str.contains("did not accept|refused services"), rdh.str.contains("accepted assistance"),
     rdh.str.contains("attempted to engage"), rdh.str.contains("referred this request to the New York City Police|worked with the New York City Police"), rdh == ""],
    ["못 찾음", "도움 거절", "도움 받아들임", "접촉 시도·협의 중", "경찰로 넘김", "빈 값"], "기타 안내"), index=rdh.index)
dv_ = dcat.value_counts(normalize=True).mul(100).round(1)
V["Q087"] = {"DHS_n": int(dh.sum()), "declined_pct": r1(rdh.str.contains("did not accept").mean() * 100), "notfound_pct": r1(nf[dh].mean() * 100),
             "declined_incl_refused_pct": r1(rdh.str.contains("did not accept|refused services").mean() * 100),
             "accepted_pct": r1(rdh.str.contains("accepted assistance").mean() * 100), "categories_pct": dv_.to_dict()}
hbar("Q087", list(dv_.index), list(dv_.values), hi=["도움 거절"], fmt=pct, xlabel="노숙인서비스국 출동 결과 (%)")

dup = rd.str.contains(r"duplicate|earlier complaint about the same|already reported|addressed under another service request", case=False)
dp = dup.groupby(df["problem"], observed=True).mean(); dn_ = df.groupby("problem", observed=True).size(); dp = dp[dn_ > 5000].sort_values(ascending=False)
V["Q088"] = {"dup_pct_overall": r1(dup.mean() * 100), "top_problems_pct": {k: r1(v * 100) for k, v in dp.head(5).items()}}
d10 = dp.head(10) * 100
hbar("Q088", [ko(k) for k in d10.index], list(d10.values), hi=[0], fmt=pct, ref=V["Q088"]["dup_pct_overall"], reflab=f"전체 {V['Q088']['dup_pct_overall']:.0f}%",
     xlabel="'이미 접수된 건'으로 닫힌 비율 (%)")

yr = rd.str.contains("a year or more"); dpr = df["agency"] == "DPR"; rdp = rd[dpr]
V["Q089"] = {"count": int(yr.sum()), "pct_of_DPR": r1(yr[dpr].mean() * 100), "DPR_count": int(yr[dpr].sum())}
pcat = pd.Series(np.select(
    [rdp.str.contains("a year or more"), rdp.str.contains("within 120 days"),
     rdp.str.contains("completed the requested work|performed the work|corrected the problem"),
     rdp.str.contains("No work is necessary"), rdp.str.contains("could not find|not found"), rdp == ""],
    ["작업 지시 (1년 이상 걸릴 수 있음)", "작업 지시 (120일 안)", "작업 완료", "작업 필요 없음", "가 보니 없음", "빈 값"], "기타 안내"), index=rdp.index)
pv = pcat.value_counts()
V["Q089"]["DPR_categories"] = {k: int(v) for k, v in pv.items()}
hbar("Q089", list(pv.index), list(pv.values / 1e3), hi=["작업 지시 (1년 이상 걸릴 수 있음)"], fmt=lambda v: f"{v:.1f}천", xlabel="공원국 답변 유형별 건수 (천 건)")

hy = rd.str.contains("shut the running hydrant")
V["Q090"] = {"count": int(hy.sum()), "summer_jun_aug_pct": r1(df[hy]["m"].isin([6, 7, 8]).mean() * 100)}
month_bar("Q090", {int(k): int(v) for k, v in df[hy].groupby("m").size().items()}, hi=[6, 7, 8])

cl = df[df["is_closed"] & (df["resolution_hours"] > 0) & ~df["flag_status_date_mismatch"] & ~df["flag_midnight_close"]]
V["Q091"] = {"n_closed_valid": len(cl), "within_1h_pct": r1((cl["resolution_hours"] <= 1).mean() * 100), "over_30d_pct": r1((cl["resolution_hours"] > 720).mean() * 100),
             "median_h": round(float(cl["resolution_hours"].median()), 2)}
fig, ax = new()
hb = np.logspace(-2, 4, 61)
cnt, _ = np.histogram(cl["resolution_hours"].clip(upper=9999), bins=hb)
mids = np.sqrt(hb[:-1] * hb[1:])
ax.bar(hb[:-1], cnt / len(cl) * 100, width=np.diff(hb), align="edge", color=[ACC if m_ <= 1 else GREY for m_ in mids], lw=0, zorder=2)
for v_, lab in [(1, "1시간"), (24, "1일"), (720, "30일")]:
    ax.axvline(v_, color=INK, ls=(0, (3, 2)), lw=.8, zorder=3)
    ax.text(v_ * 1.08, ax.get_ylim()[1] * .92 if False else (cnt / len(cl) * 100).max() * 1.02, lab, fontsize=7.2, color=INK)
ax.text(.012, (cnt / len(cl) * 100).max() * .75, f"1시간 안에 끝남\n{V['Q091']['within_1h_pct']:.0f}%", fontsize=7.2, color=ACC, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(.01, 1e4)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: {0.01: "36초", 0.1: "6분", 1: "1시간", 10: "10시간", 100: "100시간", 1000: "1천시간", 10000: "1만시간"}.get(round(v, 2), "")))
ax.set_ylabel("전체 중 비중 (%)"); ax.set_xlabel("접수에서 종료까지 걸린 시간 (로그 눈금)"); ax.grid(axis="y", zorder=0)
save(fig, "Q091")

hst = hl["status"].value_counts(); hst = hst[hst > 0]
V["Q092"] = {"helicopter_n": len(hl), "closed": int(hl["is_closed"].sum()), "status": {k: int(v) for k, v in hst.items()}, "agency": str(hl["agency"].mode()[0])}
un = df.groupby("problem", observed=True).agg(n=("is_closed", "size"), op=("is_closed", lambda z_: (~z_).mean() * 100))
un = un[un["n"] > 1000].sort_values("op", ascending=False).head(8)
V["Q092"]["open_pct_top"] = {k: r1(v) for k, v in un["op"].items()}
hbar("Q092", [ko(k) for k in un.index], list(un["op"].values), hi=["헬기 소음", "공사 납 먼지"], fmt=pct, xlabel="아직 '종료'되지 않은 비율 (%, 1,000건 넘는 민원 중)", xmax=118)

db = df["agency"] == "DOB"; rdb = rd[db]
V["Q093"] = {"DOB_n": int(db.sum()), "twice_noaccess_pct": r1(rdb.str.contains("twice").mean() * 100), "twice_count": int(rdb.str.contains("twice").sum())}
bcat = pd.Series(np.select(
    [rdb.str.contains("twice"), rdb.str.contains("could not gain access|denied access"), rdb.str.contains("no further action was necessary"),
     rdb.str.contains("reviewed this complaint and closed"), rdb.str.contains("summons"), rdb.str.contains("another service request"),
     rdb.str.contains("check back later"), rdb == ""],
    ["두 번 가도 못 들어감", "한 번 못 들어감·거부", "조치 필요 없음", "검토 후 종결", "소환장 발부", "다른 신고로 처리", "처리 대기", "빈 값"], "기타"), index=rdb.index)
bv = bcat.value_counts(normalize=True).mul(100).round(1)
V["Q093"]["categories_pct"] = bv.to_dict()
hbar("Q093", list(bv.index), list(bv.values), hi=["두 번 가도 못 들어감"], fmt=pct, xlabel="건물국 처리 결과 (%)")

# ═════════════════════════════════════════════
# 11장. 전화·웹·앱 (SA 제외)
# ═════════════════════════════════════════════
ch = ex["open_data_channel_type"].astype(str)
V["Q094"] = {"share_pct_exSA": {k: r1(v * 100) for k, v in ch.value_counts(normalize=True).items()},
             "share_pct_all": {k: r1(v * 100) for k, v in df["open_data_channel_type"].astype(str).value_counts(normalize=True).items()}}
CHK = {"ONLINE": "웹(온라인)", "PHONE": "전화", "MOBILE": "앱", "OTHER": "기타"}
c94 = pd.Series(V["Q094"]["share_pct_exSA"])
hbar("Q094", [CHK[k] for k in c94.index], list(c94.values), hi=[0], fmt=pct, xlabel="신고 경로별 비중 (%)")
tpc = pd.crosstab(ex["problem"].astype(str), ch); tpc = tpc[tpc.sum(axis=1) > 5000]; tpp = tpc.div(tpc.sum(axis=1), axis=0).mul(100).round(1)
V["Q095"] = {"mobile_top": {k: float(v) for k, v in tpp["MOBILE"].sort_values(ascending=False).head(5).items()},
             "phone_top": {k: float(v) for k, v in tpp["PHONE"].sort_values(ascending=False).head(6).items()}}
fig, axs = plt.subplots(1, 2, figsize=(FW, FH))
mt5 = tpp["MOBILE"].sort_values(ascending=False).head(5)
pt5 = tpp["PHONE"].drop("Derelict Vehicles").sort_values(ascending=False).head(5)
for ax, ser, ttl, hi0 in [(axs[0], mt5, "앱 비율이 높은 민원", True), (axs[1], pt5, "전화 비율이 높은 민원", False)]:
    yy = np.arange(5)[::-1]
    ax.barh(yy, ser.values, color=[ACC if (hi0 and i == 0) else GREY for i in range(5)], height=.62)
    ax.set_yticks(yy, [ko(k) for k in ser.index])
    for yi, v_, i in zip(yy, ser.values, range(5)):
        ax.text(v_ + 2, yi, f"{v_:.0f}%", va="center", fontsize=7.2, color=ACC if (hi0 and i == 0) else "#444", fontweight="bold" if (hi0 and i == 0) else "normal")
    ax.set_xlim(0, 115); ax.set_title(ttl, loc="left", color="#333", fontsize=8)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
save(fig, "Q095")
bc = pd.crosstab(ex["borough"].astype(str), ch, normalize="index").mul(100).round(1)
V["Q096"] = {"by_borough_pct": {b: {k: float(v) for k, v in bc.loc[b].items()} for b in bc.index}}
order = bc["PHONE"].sort_values(ascending=False).index
stack100("Q096", [KB[b] for b in order], ["PHONE", "ONLINE", "MOBILE"], {KB[b]: bc.loc[b].to_dict() for b in order},
         [ACC, "#DADDE2", "#7d8696"], ["전화", "웹(온라인)", "앱"], hi_row="스태튼아일랜드")

# ═════════════════════════════════════════════
# 12장. 이런 것도 신고해?
# ═════════════════════════════════════════════
tc, tm = int((df["problem"] == "Taxi Complaint").sum()), int((df["problem"] == "Taxi Compliment").sum())
V["Q097"] = {"taxi_complaint": tc, "taxi_compliment": tm, "ratio": r1(tc / tm)}
fig, ax = new()
nsq = int(round(tc / tm)) + 1  # 불만 93칸 + 칭찬 1칸
cols_ = 19
for i in range(nsq):
    r_, c__ = divmod(i, cols_)
    col = ACC if i == nsq - 1 else GREY
    ax.add_patch(plt.Rectangle((c__ * 1.15, -r_ * 1.15), 1, 1, color=col))
ax.text(0, 1.6, f"택시 불만 {tc:,}건 (회색 한 칸 = 약 {tc / (nsq - 1):,.0f}건)", fontsize=8, color="#333")
r_, c__ = divmod(nsq - 1, cols_)
ax.annotate(f"칭찬 {tm}건", (c__ * 1.15 + 1, -r_ * 1.15 + .5), textcoords="offset points", xytext=(14, -2), fontsize=8, color=ACC, fontweight="bold",
            va="center", arrowprops=dict(arrowstyle="-", color=ACC, lw=.8))
ax.set_xlim(-.3, cols_ * 1.15 + 5.5); ax.set_ylim(-5 * 1.15 + .6, 2.4); ax.set_aspect("equal"); ax.axis("off")
save(fig, "Q097")

lp = df[df["problem"] == "Lost Property"]
V["Q098"] = {"n": len(lp), "detail_pct": {k: r1(v * 100) for k, v in lp["problem_detail"].value_counts(normalize=True).head(4).items()},
             "taxi_location_pct": r1((lp["location_type"] == "Taxi").mean() * 100),
             "airports_pct": r1(lp["address_key"].isin(["JOHN F KENNEDY AIRPORT", "LA GUARDIA AIRPORT"]).mean() * 100)}
LK = {"Bag/Wallet": "가방·지갑", "Electronics/Phones": "전자기기·휴대폰", "Other": "기타", "Clothing/Glasses": "옷·안경", "Jewelry": "보석",
      "Book/Stationery": "책·문구", "Sports Equipment": "운동용품", "Musical Instrument": "악기"}
l8 = lp["problem_detail"].value_counts(normalize=True).mul(100); l8 = l8[l8 > 0.1].head(8)
hbar("Q098", [LK.get(k, k) for k in l8.index], list(l8.values), hi=[0], fmt=pct, xlabel="분실물 신고 중 비중 (%)")
cc = df[df["problem"] == "Consumer Complaint"]
V["Q099"] = {"n": len(cc), "top_pct": {k: r1(v * 100) for k, v in cc["problem_detail"].value_counts(normalize=True).head(4).items()}}
CK = {"Tow Truck Company": "견인차 회사", "Other Store (Non-Food)": "기타 가게(비식품)", "Bodega, Deli, or Convenience Store": "보데가·편의점",
      "Restaurant": "식당", "Supermarket or Grocery Store": "슈퍼마켓", "Garage or Parking Lot": "주차장", "Stoop Line Stand": "가게 앞 진열대",
      "Street Fair Vendor": "거리 장터 노점", "Tobacco Sales": "담배 판매", "Hotel or Motel": "호텔·모텔"}
c10 = cc["problem_detail"].value_counts(normalize=True).mul(100).head(10)
hbar("Q099", [CK.get(k, k) for k in c10.index], list(c10.values), hi=[0], fmt=pct, xlabel="소비자 민원 중 비중 (%)")
V["Q100"] = {"types_le10": int((vc <= 10).sum()), "types_eq1": {k: int(v) for k, v in vc[vc == 1].items()},
             "examples": {k: int(vc[k]) for k in ["Squeegee", "Tanning", "Calorie Labeling", "Public Payphone Complaint", "Lifeguard", "Radioactive Material", "Public Toilet"]}}
FUN = ["Public Toilet", "Lifeguard", "Radioactive Material", "Tanning", "Calorie Labeling", "Squeegee", "Public Payphone Complaint",
       "Leaning Bar", "Trapping Pigeon"]
rare = vc[FUN].sort_values(ascending=False)
V["Q100"]["chart"] = {k: int(v) for k, v in rare.items()}
fig, ax = new()
yy = np.arange(len(rare))[::-1]
on = [v <= 10 for v in rare.values]
ax.axvspan(0, 10.5, color=ACC, alpha=.07, lw=0)
ax.text(10.8, len(rare) - .7, "10건 이하 = 17종 중 일부", fontsize=7.4, color=ACC, va="center")
ax.hlines(yy, 0, rare.values, color=GREY, lw=2)
ax.scatter(rare.values, yy, color=[ACC if k == "Trapping Pigeon" else (INK if o else "#8C939E") for k, o in zip(rare.index, on)],
           s=[80 if k == "Trapping Pigeon" else 36 for k in rare.index], zorder=3)
for yi, k, v_ in zip(yy, rare.index, rare.values):
    ax.text(v_ + 1.2, yi, f"{v_}건", va="center", fontsize=7.4, color=ACC if k == "Trapping Pigeon" else "#444",
            fontweight="bold" if k == "Trapping Pigeon" else "normal")
ax.set_yticks(yy, [ko(k) for k in rare.index])
for tl in ax.get_yticklabels():
    if tl.get_text() == "비둘기 포획":
        tl.set_color(ACC); tl.set_fontweight("bold")
ax.set_xlim(0, 75); ax.set_ylim(-.7, len(rare) - .3); ax.set_xlabel("1년 신고 수")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x")
save(fig, "Q100")

# ═════════════════════════════════════════════
# check_values 대조
# ═════════════════════════════════════════════
Q = json.load(open(ROOT / "docs" / "hundred" / "questions.json", encoding="utf-8"))
diffs = []


def cmp(a, b, path):
    """a = 기획 값(check_values), b = 이번 계산 값."""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            diffs.append((path, a, b)); return
        for k in a:
            kb = k if k in b else (int(k) if k.isdigit() and int(k) in b else None)
            if kb is None:
                diffs.append((path + "." + str(k), a[k], "(없음)"))
            else:
                cmp(a[k], b[kb], path + "." + str(k))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        try:
            bb_ = float(b)
        except (TypeError, ValueError):
            diffs.append((path, a, b)); return
        if abs(float(a) - bb_) > (0.051 if isinstance(a, float) else 0.5):
            diffs.append((path, a, b))
    elif str(a) != str(b):
        diffs.append((path, a, b))


for q in Q:
    cmp(q["check_values"], V.get(q["id"], {}), q["id"])
V["_check_diffs"] = [{"path": p_, "plan": a_, "data": b_} for p_, a_, b_ in diffs]
json.dump(V, open(OUT / "hundred_numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("check_values 와 다른 값:", len(diffs))
for p_, a_, b_ in diffs:
    print("  ", p_, "| 기획", a_, "| 계산", b_)
if MISSING:
    print("한글 이름 없는 라벨:", sorted(MISSING))
print("그림:", len(list(FIG.glob("h100_Q*.png"))), "개")
