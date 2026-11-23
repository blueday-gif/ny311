"""
[읽을거리] 뉴욕 311 다섯 가지 질문 — 책(PDF)용 분석

핵심 질문 : 밤 소음 / 난방 민원과 기온 / 처리 시간 병목 / 자치구 특화 민원 / 하루 리듬
입력      : data/processed/311_requests_clean.parquet
            data/external/nyc_daily_temp.json (Open-Meteo 과거 기상 API, 맨해튼 부근 일평균 기온)
출력      : outputs/figures/book_*.png, outputs/tables/book_*.csv, reports/book/book_numbers.json
실행 방법 : py -3.12 analysis/book_five_questions.py

필터 기준 (eda_clean_basic.py 와 같음)
- 신고량    : NOT flag_system_duplicate
- 처리 시간 : is_closed AND resolution_hours > 0 AND NOT flag_system_duplicate AND NOT flag_status_date_mismatch
"""

import json

from common import ROOT, np, pd, plt, save_fig, save_table, setup_plot

setup_plot()
plt.rcParams["figure.dpi"] = 160
# ── 책(PDF) 인쇄용 스타일: 그림 크기를 인쇄 폭(인치) 그대로 잡아 글자 크기가 책 전체에서 같게 나오게 한다 ──
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
plt.rcParams.update({
    "savefig.dpi": 300, "font.size": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
    "axes.labelcolor": "#333", "xtick.color": "#444", "ytick.color": "#444",
    "grid.color": "#ddd", "grid.linewidth": .6, "lines.linewidth": 1.4,
    "figure.facecolor": "white", "axes.titlepad": 4,
})
W_FULL, W_COL = 7.0, 4.55   # 본문 전체 폭(178mm), 본문 단 폭(116mm)
INK, MUTED = "#1d2433", "#6b7280"
C_WEEKDAY, C_WEEKEND = "#0072B2", "#D55E00"    # 평일 / 주말 (책 전체 공통)
C_HEAT, C_TEMP = "#D62F2F", "#0072B2"          # 난방 민원 / 기온
LOGFMT = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")  # 로그 축 눈금을 10ⁿ 대신 보통 숫자로
BOOK_CMAP = LinearSegmentedColormap.from_list("book", ["#FBF6EC", "#F2C48D", "#E07A3F", "#A3341A", "#3B1410"])
OUT = ROOT / "reports" / "book"
OUT.mkdir(parents=True, exist_ok=True)
NUM: dict = {}  # 본문에 쓸 숫자

KO_BORO = {"MANHATTAN": "맨해튼", "BROOKLYN": "브루클린", "QUEENS": "퀸스",
           "BRONX": "브롱크스", "STATEN ISLAND": "스태튼아일랜드"}
BORO_COLOR = {"맨해튼": "#0072B2", "브루클린": "#E69F00", "퀸스": "#009E73",
              "브롱크스": "#D55E00", "스태튼아일랜드": "#CC79A7"}
DOW = "월화수목금토일"

cols = ["created_date", "closed_date", "agency", "problem", "borough", "resolution_hours",
        "is_closed", "flag_system_duplicate", "flag_status_date_mismatch"]
df = pd.read_parquet(ROOT / "data" / "processed" / "311_requests_clean.parquet", columns=cols)
df = df[~df["flag_system_duplicate"]].copy()
df["boro"] = df["borough"].map(KO_BORO)
df["hour"] = df["created_date"].dt.hour
df["dow"] = df["created_date"].dt.day_of_week  # 0=월
df["date"] = df["created_date"].dt.normalize()
NUM["n_total"] = len(df)
NUM["period"] = [str(df["date"].min().date()), str(df["date"].max().date())]
n_days = df["date"].nunique()

# ═════════════════════════════════════════════
# 1장. 밤 소음
# ═════════════════════════════════════════════
noise = df[df["problem"].str.startswith("Noise", na=False)].copy()
NUM["noise_n"] = len(noise)
NUM["noise_types"] = noise["problem"].value_counts().to_dict()

# '밤의 날짜': 새벽 0~5시는 전날 밤에 붙인다 (금요일 밤 = 금 18시 ~ 토 05시)
noise["night_dow"] = (noise["created_date"] - pd.Timedelta(hours=6)).dt.day_of_week
noise["weekend_night"] = noise["night_dow"].isin([4, 5])  # 금·토 밤
# 요일 × 시간 히트맵 (하루 평균 건수)
dow_days = df.groupby("dow")["date"].nunique()
heat = noise.groupby(["dow", "hour"]).size().unstack(fill_value=0)
heat = heat.div(dow_days, axis=0)
fig, ax = plt.subplots(figsize=(W_FULL, 2.0))
im = ax.imshow(heat.values, aspect="auto", cmap=BOOK_CMAP)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.set_yticks(range(7), list(DOW))
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=.015, aspect=14)
cb.set_label("하루 평균 건수", fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=0)
save_fig(fig, "book_q1_heatmap")
save_table(heat.round(1).rename(index=dict(enumerate(DOW))), "book_q1_dow_hour", index=True)

# 주중 밤 vs 주말 밤 시간 곡선 (18시~05시 배치)
order = list(range(18, 24)) + list(range(0, 6))
grp = noise.assign(kind=np.where(noise["weekend_night"], "주말 밤(금·토)", "평일 밤(일~목)"))
n_nights = {"주말 밤(금·토)": noise.loc[noise.weekend_night, "night_dow"].count(),}
nights_per_kind = (df.assign(nd=(df["created_date"] - pd.Timedelta(hours=6)).dt.normalize())
                   .assign(we=lambda d: d["nd"].dt.day_of_week.isin([4, 5]))
                   .groupby("we")["nd"].nunique())
curve = grp.groupby(["kind", "hour"]).size().unstack(0).loc[order]
curve["주말 밤(금·토)"] /= nights_per_kind[True]
curve["평일 밤(일~목)"] /= nights_per_kind[False]
fig, ax = plt.subplots(figsize=(W_FULL, 2.15))
x = range(len(order))
for k, c in [("평일 밤(일~목)", C_WEEKDAY), ("주말 밤(금·토)", C_WEEKEND)]:
    ax.plot(x, curve[k], marker="o", ms=3, color=c, label=k)
    pk = curve[k].idxmax()
    ax.annotate(f"정점 {pk}시 · {curve[k].max():,.0f}건", (order.index(pk), curve[k].max()),
                textcoords="offset points", xytext=(7, 2), fontsize=7, color=c, fontweight="bold")
ax.set_xticks(list(x), [f"{h}시" for h in order])
ax.set_ylabel("밤 하루 평균 건수")
ax.set_ylim(0, curve.values.max() * 1.15)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y")
save_fig(fig, "book_q1_night_curve")
NUM["q1_peak"] = {k: int(curve[k].idxmax()) for k in curve}
NUM["q1_peak_val"] = {k: round(float(curve[k].max())) for k in curve}
NUM["q1_ratio_1am"] = round(float(curve.loc[1, "주말 밤(금·토)"] / curve.loc[1, "평일 밤(일~목)"]), 2)
NUM["q1_ratio_22"] = round(float(curve.loc[22, "주말 밤(금·토)"] / curve.loc[22, "평일 밤(일~목)"]), 2)
NUM["q1_ratio_3am"] = round(float(curve.loc[3, "주말 밤(금·토)"] / curve.loc[3, "평일 밤(일~목)"]), 2)

# 자치구 × 소음 유형: 밤 시간 프로필 (각 구·유형의 하루 총량 대비 시간별 비율)
sub = noise[noise["problem"].isin(["Noise - Residential", "Noise - Commercial", "Noise - Street/Sidewalk"])]
sub = sub[sub["boro"].notna()]
prof = sub.groupby(["problem", "boro", "hour"]).size()
prof = prof / prof.groupby(level=[0, 1]).transform("sum") * 100
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 1.85), sharey=True)
lab = {"Noise - Residential": "주거 소음", "Noise - Commercial": "상업(업소) 소음", "Noise - Street/Sidewalk": "거리·보도 소음"}
for ax, p in zip(axes, lab):
    for b, c in BORO_COLOR.items():
        s = prof.loc[p, b].reindex(order)
        ax.plot(range(len(order)), s.values, color=c, label=b, lw=1.3)
    ax.set_title(lab[p])
    ax.set_xticks(range(0, len(order), 2), [f"{h}시" for h in order[::2]])
    ax.grid(axis="y")
axes[0].set_ylabel("하루 중 비율(%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=(0, .07, 1, 1))
fig.tight_layout = lambda *a, **k: None  # save_fig 의 tight_layout 이 범례 자리를 덮지 않게
save_fig(fig, "book_q1_borough")
# 자치구별 정점 시각과 '자정 이후(0~5시) 비율'
peak_tab = []
for p in lab:
    for b in BORO_COLOR:
        s = prof.loc[p, b]
        peak_tab.append({"유형": lab[p], "자치구": b, "정점": int(s.idxmax()),
                         "자정이후비율": round(float(s.loc[0:5].sum()), 1),
                         "건수": int(sub[(sub.problem == p) & (sub.boro == b)].shape[0])})
peak_tab = pd.DataFrame(peak_tab)
save_table(peak_tab, "book_q1_borough_peak")
NUM["q1_borough_peak"] = peak_tab.to_dict("records")
# 자치구 × 유형: 주말 밤 한 밤 평균 / 평일 밤 한 밤 평균 (22~05시)
nt = sub[sub["hour"].isin([22, 23, 0, 1, 2, 3, 4])]
up = nt.groupby(["problem", "boro", "weekend_night"]).size().unstack()
up = (up[True] / nights_per_kind[True]) / (up[False] / nights_per_kind[False])
up = up.unstack().rename(index=lab).round(2)
save_table(up.reset_index(), "book_q1_weekend_uplift")
NUM["q1_weekend_uplift"] = up.to_dict("index")
# 1만 명당 밤(22~05시) 소음, 유형별
pop = pd.read_csv(ROOT / "docs" / "borough_context.csv")
print(pop.columns.tolist())

# ═════════════════════════════════════════════
# 2장. 난방/온수와 기온
# ═════════════════════════════════════════════
w = json.load(open(ROOT / "data" / "external" / "nyc_daily_temp.json", encoding="utf-8"))["daily"]
temp = pd.DataFrame({"date": pd.to_datetime(w["time"]), "tmean": w["temperature_2m_mean"],
                     "tmin": w["temperature_2m_min"]}).set_index("date")
heatd = df[df["problem"] == "HEAT/HOT WATER"]
daily_all = df.groupby("date").size()
daily_heat = heatd.groupby("date").size().reindex(daily_all.index, fill_value=0)
d = pd.DataFrame({"heat": daily_heat, "all": daily_all}).join(temp, how="left")
d = d.iloc[1:-1]  # 첫날·마지막 날은 하루가 다 안 찼다
d["share"] = d["heat"] / d["all"] * 100
save_table(d.reset_index(), "book_q2_daily")
fig, ax1 = plt.subplots(figsize=(W_FULL, 2.5))
ax1.bar(d.index, d["heat"], color=C_HEAT, width=1, alpha=.55, label="난방·온수 민원(일)")
ax1.set_ylabel("하루 민원 건수", color=C_HEAT)
ax1.set_ylim(0, d["heat"].max() * 1.18)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(d.index, d["tmean"], color=C_TEMP, lw=1.0, label="일평균 기온")
ax2.axhline((55 - 32) / 1.8, color=C_TEMP, ls=":", lw=.9)
ax2.text(pd.Timestamp("2026-07-20"), (55 - 32) / 1.8 - 1.0, "12.8℃(55℉): 법정 난방 기준", color=C_TEMP,
         fontsize=6.5, va="top", ha="center", bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
ax2.set_ylabel("일평균 기온(℃)", color=C_TEMP)
ax2.set_ylim(d["tmean"].min() - 4, d["tmean"].max() + 9)
for x_, t in [("2026-06-01", "6/1 난방 시즌 종료 "), ("2026-10-01", "10/1 시즌 시작 ")]:
    ax1.axvline(pd.Timestamp(x_), color="#777", ls="--", lw=.8)
    ax1.text(pd.Timestamp(x_), ax1.get_ylim()[1] * .97, t, fontsize=6.5, color="#555", ha="right", va="top")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
save_fig(fig, "book_q2_timeline")

# 기온 구간별 평균 민원 (난방 시즌 안/밖 나눠서)
d["season"] = ((d.index.month >= 10) | (d.index.month <= 5))
d["tbin"] = pd.cut(d["tmean"], bins=range(-14, 34, 3))
tb = d[d.season].groupby("tbin", observed=True)["heat"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(W_COL, 2.6))
ax.scatter(d.loc[d.season, "tmean"], d.loc[d.season, "heat"], s=7, alpha=.55, color=C_HEAT, lw=0, label="난방 시즌(10~5월)")
ax.scatter(d.loc[~d.season, "tmean"], d.loc[~d.season, "heat"], s=7, alpha=.55, color="#999", lw=0, label="비시즌(6~9월)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlabel("일평균 기온(℃)")
ax.set_ylabel("난방·온수 민원(일)")
ax.legend(frameon=False)
ax.grid()
save_fig(fig, "book_q2_scatter")
s_ = d[d.season]
NUM["q2_corr_pearson"] = round(float(s_["tmean"].corr(s_["heat"])), 2)
NUM["q2_corr_spearman"] = round(float(s_["tmean"].corr(s_["heat"], method="spearman")), 2)
# 기울기: 1℃ 내려갈 때 늘어나는 건수 (시즌 내, 단순회귀)
b1, b0 = np.polyfit(s_["tmean"], s_["heat"], 1)
NUM["q2_slope"] = round(float(-b1), 1)
NUM["q2_peak_day"] = [str(d["heat"].idxmax().date()), int(d["heat"].max()), round(float(d.loc[d["heat"].idxmax(), "tmean"]), 1)]
NUM["q2_summer_mean"] = round(float(d.loc[d.index.month.isin([7, 8]), "heat"].mean()), 1)
NUM["q2_jan_mean"] = round(float(d.loc[d.index.month == 1, "heat"].mean()), 1)
NUM["q2_share_jan"] = round(float(d.loc[d.index.month == 1, "share"].mean()), 1)
NUM["q2_share_jul"] = round(float(d.loc[d.index.month == 7, "share"].mean()), 1)
# 10월 시작: 9월 마지막 주 vs 10월 첫 주
NUM["q2_sep_last"] = round(float(d.loc["2026-09-24":"2026-09-30", "heat"].mean()), 1)
NUM["q2_oct_first"] = round(float(d.loc["2026-10-01":"2026-10-07", "heat"].mean()), 1)
NUM["q2_oct_last"] = round(float(d.loc["2026-10-25":"2026-10-30", "heat"].mean()), 1)
NUM["q2_temp_sep_last"] = round(float(d.loc["2026-09-24":"2026-09-30", "tmean"].mean()), 1)
NUM["q2_temp_oct_first"] = round(float(d.loc["2026-10-01":"2026-10-07", "tmean"].mean()), 1)
# 한파 전후 반응: 전날 대비 기온이 5℃ 이상 떨어진 날의 다음날 민원 증가율
d["dT"] = d["tmean"].diff()
d["heat_next_ratio"] = d["heat"].shift(-1) / d["heat"].rolling(7).mean().shift(1)
drop = d[d.season & (d["dT"] <= -5)]
NUM["q2_colddrop_days"] = int(len(drop))
NUM["q2_colddrop_ratio"] = round(float(drop["heat_next_ratio"].median()), 2)
# 래그 상관: 기온 t일 vs 민원 t+k일
lag = {k: round(float(s_["tmean"].corr(s_["heat"].shift(-k))), 3) for k in range(0, 4)}
NUM["q2_lag_corr"] = lag

# 자치구별: 1만 명당 1월 일평균 + 기온 민감도(기울기)
pop_col = [c for c in pop.columns if "인구" in c or "pop" in c.lower()]
print("pop columns:", pop_col)

hb = heatd[heatd.boro.notna()].groupby(["date", "boro"]).size().unstack(fill_value=0)
hb = hb.reindex(d.index, fill_value=0)
rows = []
for b in BORO_COLOR:
    y = hb.loc[d.season, b]
    sl = -np.polyfit(d.loc[d.season, "tmean"], y, 1)[0]
    # 시즌 시작 반응: 10월 첫주 / 10월 마지막주 비율 (얼마나 '빨리' 올라오나)
    early = hb.loc["2026-10-01":"2026-10-07", b].mean()
    late = hb.loc["2026-10-25":"2026-10-30", b].mean()
    jan = hb.loc[hb.index.month == 1, b].mean()
    rows.append({"자치구": b, "1월일평균": round(jan, 1), "기울기_1도당": round(sl, 1),
                 "10월첫주": round(early, 1), "10월말": round(late, 1),
                 "상관": round(float(d.loc[d.season, "tmean"].corr(y)), 2),
                 "피크대비_1월": round(float(hb[b].max() / jan), 2)})
q2b = pd.DataFrame(rows)
save_table(q2b, "book_q2_borough")
NUM["q2_borough"] = q2b.to_dict("records")
# 자치구별 정규화 곡선 (주간 합계를 1월 평균으로 나눔)
wk = hb.resample("W").sum()
wk_norm = wk / wk.loc[wk.index.month == 1].mean() * 100
fig, ax = plt.subplots(figsize=(W_FULL, 2.2))
for b, c in BORO_COLOR.items():
    ax.plot(wk_norm.index[1:-1], wk_norm[b].iloc[1:-1], color=c, label=b, lw=1.2)
ax.axhline(100, color="gray", lw=.6, ls=":")
ax.set_ylabel("1월 평균 = 100")
ax.legend(frameon=False, ncol=5, loc="upper center")
ax.set_ylim(0, wk_norm.iloc[1:-1].values.max() * 1.15)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.grid(axis="y")
save_fig(fig, "book_q2_borough")

# ═════════════════════════════════════════════
# 3장. 처리 시간 병목
# ═════════════════════════════════════════════
r = df[df["is_closed"] & (df["resolution_hours"] > 0) & ~df["flag_status_date_mismatch"]].copy()
NUM["q3_n"] = len(r)
NUM["q3_overall"] = {"median": round(float(r.resolution_hours.median()), 1),
                     "mean": round(float(r.resolution_hours.mean()), 1),
                     "p90": round(float(r.resolution_hours.quantile(.9)), 1)}

def q(s):
    return pd.Series({"건수": len(s), "중앙값": s.median(), "p25": s.quantile(.25),
                      "p75": s.quantile(.75), "p90": s.quantile(.9), "평균": s.mean()})

by_ag = r.groupby("agency")["resolution_hours"].apply(q).unstack()
by_ag = by_ag[by_ag["건수"] >= 2000].sort_values("중앙값")
save_table(by_ag.round(1).reset_index(), "book_q3_agency")
NUM["q3_agency"] = by_ag.round(1).reset_index().to_dict("records")
by_pb = r.groupby("problem")["resolution_hours"].apply(q).unstack()
by_pb = by_pb[by_pb["건수"] >= 1000]
save_table(by_pb.round(1).reset_index(), "book_q3_problem")
by_bo = r[r.boro.notna()].groupby("boro")["resolution_hours"].apply(q).unstack()
NUM["q3_borough"] = by_bo.round(1).reset_index().to_dict("records")

# 건수 vs 중앙값 (로그-로그) 산점도
from scipy.stats import spearmanr  # noqa: E402
rho, pval = spearmanr(by_pb["건수"], by_pb["중앙값"])
NUM["q3_spearman"] = [round(float(rho), 2), float(pval)]
fig, ax = plt.subplots(figsize=(W_COL, 3.1))
ax.scatter(by_pb["건수"], by_pb["중앙값"], s=10, color="#0072B2", alpha=.65, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_major_formatter(LOGFMT); ax.yaxis.set_major_formatter(LOGFMT)
lab_ = list(by_pb.nlargest(4, "건수").index) + list(by_pb.nlargest(6, "중앙값").index)
by_pb["ratio"] = by_pb["p90"] / by_pb["중앙값"]
ax.set_xlabel("1년 처리 완료 건수 (로그)")
ax.set_ylabel("처리 시간 중앙값, 시간 (로그)")
ax.text(.99, .02, f"스피어만 ρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
for yv, t in [(1, "1시간"), (24, "1일"), (24 * 7, "1주"), (24 * 30, "1달")]:
    ax.axhline(yv, color="gray", lw=.5, ls=":")
    ax.text(1.01, yv, t, fontsize=6, color="gray", va="center", transform=ax.get_yaxis_transform())
# 이름표가 서로 겹치지 않게: 축 배치를 먼저 확정한 뒤, 후보 위치를 차례로 시도해 먼저 놓인 이름표와 안 겹치는 곳에 둔다
fig.tight_layout()
fig.tight_layout = lambda *a, **k: None  # 이름표를 놓은 뒤 배치가 다시 바뀌면 겹침 검사가 무의미해진다
fig.canvas.draw()
placed = []
rend = fig.canvas.get_renderer()
for p in dict.fromkeys(lab_):
    right = by_pb.loc[p, "건수"] > 1e5
    side = [-1, 1] if right else [1, -1]
    cands = [(sd * 3, dy) for dy in (1.5, -7, 8, -13, 14, -19, 20, -25) for sd in side]
    best = None
    for k_, (dx, dy) in enumerate(cands):
        a_ = ax.annotate(p, (by_pb.loc[p, "건수"], by_pb.loc[p, "중앙값"]), fontsize=5.5, color="#333",
                         xytext=(dx, dy), ha="left" if dx > 0 else "right", textcoords="offset points")
        bb = a_.get_window_extent(rend).expanded(1.03, 1.15)  # 연결선 없이 글자 상자만 잰다
        inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
        hit = sum(1 for o in placed if bb.overlaps(o)) + (0 if inside else 5)
        a_.remove()
        if best is None or hit < best[0]:
            best = (hit, dx, dy, k_, bb)
        if hit == 0:
            break
    _, dx, dy, k_, bb = best
    ax.annotate(p, (by_pb.loc[p, "건수"], by_pb.loc[p, "중앙값"]), fontsize=5.5, color="#333",
                xytext=(dx, dy), ha="left" if dx > 0 else "right", textcoords="offset points",
                arrowprops=None if k_ < 2 else dict(arrowstyle="-", color="#aaa", lw=.4, shrinkA=0, shrinkB=2))
    placed.append(bb)
save_fig(fig, "book_q3_scatter")
# 숨은 병목: 건수 하위 절반인데 중앙값 상위 10
hidden = by_pb[by_pb["건수"] < by_pb["건수"].median()].nlargest(8, "중앙값")
NUM["q3_hidden"] = hidden.round(1).reset_index().to_dict("records")
NUM["q3_big"] = by_pb.nlargest(6, "건수").round(1).reset_index().to_dict("records")
# 꼬리가 긴 유형: p90/중앙값 비
tail = by_pb[by_pb["중앙값"] >= 1].nlargest(6, "ratio")
NUM["q3_tail"] = tail.round(1).reset_index().to_dict("records")

# 같은 민원, 다른 자치구: 주요 6개 유형의 자치구별 중앙값
key = ["HEAT/HOT WATER", "Illegal Parking", "Noise - Residential", "Street Condition", "UNSANITARY CONDITION", "Damaged Tree"]
mb = r[r.problem.isin(key) & r.boro.notna()].groupby(["problem", "boro"])["resolution_hours"].median().unstack()
mb = mb.loc[key]
save_table(mb.round(1).reset_index(), "book_q3_problem_borough")
NUM["q3_problem_borough"] = mb.round(1).reset_index().to_dict("records")
fig, ax = plt.subplots(figsize=(W_COL, 2.4))
show = np.log10(mb.values)
im = ax.imshow(show, cmap=BOOK_CMAP, aspect="auto")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.xaxis.tick_top()
ax.set_xticks(range(mb.shape[1]), [c.replace("스태튼아일랜드", "스태튼") for c in mb.columns])
ko_key = ["난방·온수", "불법 주차", "주거 소음", "도로 상태", "비위생", "가로수 피해"]
ax.set_yticks(range(len(key)), ko_key)
for i in range(mb.shape[0]):
    for j in range(mb.shape[1]):
        v = mb.values[i, j]
        t = f"{v:.1f}시간" if v < 48 else f"{v / 24:.1f}일"
        ax.text(j, i, t, ha="center", va="center", fontsize=6.5,
                color=INK if (show[i, j] - np.nanmin(show)) / (np.nanmax(show) - np.nanmin(show)) < .55 else "white")
save_fig(fig, "book_q3_heat")

# 처리 시간 분포 예시: 로그 스케일 히스토그램 (평균이 왜 오해를 부르는지)
fig, ax = plt.subplots(figsize=(W_FULL, 2.4))
bins = np.logspace(-2, 4, 70)
ax.hist(r["resolution_hours"].clip(upper=9999), bins=bins, color="#7A8CA8", alpha=.9)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(LOGFMT)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
for v, t, c in [(r.resolution_hours.median(), "중앙값", INK), (r.resolution_hours.mean(), "평균", "#D55E00")]:
    ax.axvline(v, color=c, ls="--", lw=1); ax.text(v * 1.12, ax.get_ylim()[1] * .97, f"{t} {v:.1f}시간", color=c, fontsize=7, va="top", fontweight="bold")
ax.set_xlabel("처리 시간(시간, 로그 축)")
ax.set_ylabel("건수")
save_fig(fig, "book_q3_hist")

# ═════════════════════════════════════════════
# 4장. 자치구 '유독' 민원 (입지계수 LQ)
# ═════════════════════════════════════════════
bp = df[df.boro.notna()].groupby(["boro", "problem"]).size().unstack(fill_value=0)
city_share = bp.sum() / bp.values.sum()
share = bp.div(bp.sum(axis=1), axis=0)
lq = share / city_share
rows = []
for b in BORO_COLOR:
    ok = bp.loc[b] >= 300  # 너무 적은 건수는 비율이 튄다
    top = lq.loc[b][ok & (bp.sum() >= 3000)].nlargest(6)
    for p, v in top.items():
        rows.append({"자치구": b, "민원": p, "LQ": round(float(v), 2), "구내비중%": round(float(share.loc[b, p] * 100), 2),
                     "시전체비중%": round(float(city_share[p] * 100), 2), "건수": int(bp.loc[b, p]),
                     "건수순위": int(bp.loc[b].rank(ascending=False)[p])})
q4 = pd.DataFrame(rows)
save_table(q4, "book_q4_lq_top")
NUM["q4_lq"] = q4.to_dict("records")
NUM["q4_top1_count"] = {b: [bp.loc[b].idxmax(), int(bp.loc[b].max()), round(float(share.loc[b].max() * 100), 1)] for b in BORO_COLOR}
NUM["q4_boro_n"] = {b: int(bp.loc[b].sum()) for b in BORO_COLOR}
fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 3.8))
axes = axes.ravel()
for ax, b in zip(axes, BORO_COLOR):
    t = q4[q4.자치구 == b].iloc[::-1]
    ax.barh(range(len(t)), t["LQ"], color=BORO_COLOR[b], height=.7)
    ax.set_yticks(range(len(t)), [__import__("textwrap").fill(m, 20) for m in t["민원"]])
    ax.axvline(1, color="#666", ls=":", lw=.9)
    ax.set_title(b, color=INK, loc="left")
    ax.tick_params(axis="y", labelsize=6.2, length=0)
    ax.tick_params(axis="x", labelsize=6)
    for tl in ax.get_yticklabels():
        tl.set_linespacing(.95)
    for i, v in enumerate(t["LQ"]):
        ax.text(v, i, f" {v:.1f}배", va="center", fontsize=6.3, color="#333")
    ax.set_xlim(0, t["LQ"].max() * 1.4)
axes[5].axis("off")
save_fig(fig, "book_q4_lq")
# 건수 1위 vs LQ 1위 비교용 '두 순위' 표
cmp_ = pd.DataFrame({"건수1위": {b: bp.loc[b].idxmax() for b in BORO_COLOR},
                     "LQ1위": {b: q4[q4.자치구 == b].iloc[0]["민원"] for b in BORO_COLOR}})
save_table(cmp_.reset_index(names="자치구"), "book_q4_compare")
# 주제 묶음 비중 (실내설비 vs 거리/차량) — 자치구 색깔 한눈에
theme = {
    "실내 설비(HPD)": ["HEAT/HOT WATER", "UNSANITARY CONDITION", "PLUMBING", "PAINT/PLASTER", "DOOR/WINDOW", "WATER LEAK", "GENERAL", "ELECTRIC", "FLOORING/STAIRS", "APPLIANCE"],
    "차량·주차": ["Illegal Parking", "Blocked Driveway", "Abandoned Vehicle", "Derelict Vehicles"],
    "나무·도로·하수": ["Street Condition", "Damaged Tree", "Overgrown Tree/Branches", "Sewer", "Sidewalk Condition", "Street Light Condition", "Water System"],
    "소음": [p for p in bp.columns if p.startswith("Noise")],
}
th = pd.DataFrame({k: bp[v].sum(axis=1) for k, v in theme.items()})
th["기타"] = bp.sum(axis=1) - th.sum(axis=1)
th_share = th.div(th.sum(axis=1), axis=0) * 100
save_table(th_share.round(1).reset_index(), "book_q4_theme")
NUM["q4_theme"] = th_share.round(1).to_dict("index")
fig, ax = plt.subplots(figsize=(W_COL, 2.1))
left = np.zeros(len(th_share))
tc = ["#B8452A", "#E69F00", "#4E9A6B", "#3E6FA8", "#D3D0C8"]
for k, c in zip(th_share.columns, tc):
    ax.barh(th_share.index, th_share[k], left=left, color=c, label=k, height=.7, edgecolor="white", lw=.5)
    for i, v in enumerate(th_share[k]):
        if v > 6:
            ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.3, color="white" if k not in ("기타", "차량·주차") else "#222")
    left += th_share[k].values
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(ncol=5, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(.42, -.14), handlelength=.9, columnspacing=.8, handletextpad=.4)
save_fig(fig, "book_q4_theme")

# ═════════════════════════════════════════════
# 5장. 하루의 리듬
# ═════════════════════════════════════════════
hr = df.groupby("hour").size() / n_days
NUM["q5_hour_min"] = [int(hr.idxmin()), round(float(hr.min()))]
NUM["q5_hour_max"] = [int(hr.idxmax()), round(float(hr.max()))]
NUM["q5_ratio"] = round(float(hr.max() / hr.min()), 1)
NUM["q5_hour"] = {int(k): round(float(v)) for k, v in hr.items()}
# 시간대 × 주제 (누적 면적)
inv = {p: k for k, v in theme.items() for p in v}
df["theme"] = df["problem"].map(inv).fillna("기타")
ht = df.groupby(["hour", "theme"]).size().unstack(fill_value=0) / n_days
ht = ht[list(theme) + ["기타"]]
fig, ax = plt.subplots(figsize=(W_FULL, 2.6))
ax.stackplot(ht.index, ht.T.values, labels=ht.columns, colors=tc, alpha=.95, edgecolor="white", lw=.4)
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
ax.set_ylabel("시간당 평균 건수")
ax.legend(frameon=False, loc="upper left", ncol=5)
ax.set_ylim(0, ht.sum(axis=1).max() * 1.18)
ax.margins(x=0)
ax.grid(axis="y", alpha=.6)
save_fig(fig, "book_q5_stack")
NUM["q5_theme_share"] = (ht.div(ht.sum(axis=1), axis=0) * 100).round(1).loc[[4, 9, 14, 23]].to_dict("index")
# 기상 순서: 상위 유형별 '새벽 바닥(4시) 대비 몇 시에 하루 평균을 처음 넘는가'
top = df["problem"].value_counts().head(25).index
pt = df[df.problem.isin(top)].groupby(["problem", "hour"]).size().unstack(fill_value=0)
norm = pt.div(pt.max(axis=1), axis=0)
wake = {}
for p in pt.index:
    s = norm.loc[p]
    # 새벽 4시부터 보면서 '하루 최대치의 50%'를 처음 넘는 시각을 선형보간으로 구한다 (예: 7.4 = 7시 24분)
    wake[p] = None
    for h in range(4, 23):
        if s[h] < .5 <= s[h + 1]:
            wake[p] = round(h + (.5 - s[h]) / (s[h + 1] - s[h]), 2)
            break
        if s[h] >= .5 and h == 4:
            wake[p] = 4.0
            break
wake = pd.Series(wake).sort_values()
order_p = wake.index
NUM["q5_wake"] = {k: float(v) for k, v in wake.items()}
fig, ax = plt.subplots(figsize=(W_FULL, 4.3))
im = ax.imshow(norm.loc[order_p].values, aspect="auto", cmap=BOOK_CMAP)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.set_yticks(range(len(order_p)), [f"{p}  ({int(wake[p])}:{int(round(wake[p] % 1 * 60)):02d})" for p in order_p], fontsize=6.5)
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
cb = fig.colorbar(im, ax=ax, shrink=.5, pad=.015, aspect=16)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=0); cb.set_label("최대 시간 = 1", fontsize=6.5)
save_fig(fig, "book_q5_wake")
# 주중 vs 주말 하루 곡선
we = df["dow"] >= 5
cw = pd.DataFrame({"평일": df[~we].groupby("hour").size() / df[~we]["date"].nunique(),
                   "주말": df[we].groupby("hour").size() / df[we]["date"].nunique()})
NUM["q5_weekday_peak"] = int(cw["평일"].idxmax()); NUM["q5_weekend_peak"] = int(cw["주말"].idxmax())
NUM["q5_weekday_min"] = int(cw["평일"].idxmin()); NUM["q5_weekend_min"] = int(cw["주말"].idxmin())
fig, ax = plt.subplots(figsize=(W_COL, 2.3))
ax.plot(cw.index, cw["평일"], color=C_WEEKDAY, marker="o", ms=2.2, label="평일 하루")
ax.plot(cw.index, cw["주말"], color=C_WEEKEND, marker="o", ms=2.2, label="주말 하루")
ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
ax.set_ylabel("시간당 평균 건수")
ax.set_ylim(0, cw.values.max() * 1.12)
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y")
save_fig(fig, "book_q5_weekday")
NUM["q5_cw"] = cw.round(0).astype(int).to_dict("index")

json.dump(NUM, open(OUT / "book_numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(NUM, ensure_ascii=False, indent=1, default=str))
