"""
[전처리 후 EDA] 민원 유형으로 읽는 자치구 성격

핵심 질문 : 민원 유형 구성만 보고 각 자치구가 어떤 곳인지(도심인지, 임대 아파트 지역인지, 교외 주택지인지)
            짐작할 수 있는가? 그 "특징 유형"은 자치구 전체의 모습인가, 한두 동네의 모습인가?
입력      : data/processed/311_requests_clean.parquet, docs/borough_context.csv (인구·배경)
출력      : outputs/tables/edac10_*.csv, outputs/figures/edac10_*.png
실행 방법 : py -3.12 analysis/eda_borough_character.py

핵심 지표: 배율(특화 지수)
- 배율 = (이 자치구 안에서 그 유형의 비중) / (나머지 4개 자치구에서 그 유형의 비중)
- 1보다 크면 그 자치구에 "유난히 많은" 유형, 1보다 작으면 "유난히 적은" 유형
- 건수가 아니라 비중끼리 나누므로, 자치구 크기(신고량) 차이는 걷어 낸다
- 지역경제학의 입지계수(LQ)와 같은 생각. 여기서는 비교 대상을 "시 전체"가 아니라 "나머지 자치구"로 둬서
  큰 자치구(브루클린)가 시 평균을 끌어당기는 효과를 뺀다

필터 : NOT flag_system_duplicate (신고량 기준과 같음), 자치구 5개만(결측 제외)
배율은 비중이 작은 유형에서 크게 흔들리므로, "자치구 안 비중 0.3% 이상 또는 나머지 비중 0.5% 이상"인 유형만 순위에 올린다.
"""

import numpy as np

from common import ROOT, pd, plt, print_header, save_fig, save_table, setup_plot

CLEAN = ROOT / "data" / "processed" / "311_requests_clean.parquet"
POP = ROOT / "docs" / "borough_context.csv"
PER = 10_000
BOROUGHS = ["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"]
MIN_ME, MIN_REST = 0.3, 0.5  # 배율 순위에 올릴 최소 비중(%)
setup_plot()

# ─────────────────────────────────────────────
# 주제(테마) 묶음: 비슷한 성격의 유형을 묶어 자치구 성격을 큰 그림으로 본다
# 묶음에 없는 유형은 "기타"
# ─────────────────────────────────────────────
THEMES = {
    "주거 설비(실내)": ["HEAT/HOT WATER", "UNSANITARY CONDITION", "PLUMBING", "PAINT/PLASTER", "DOOR/WINDOW",
                   "WATER LEAK", "GENERAL", "ELECTRIC", "FLOORING/STAIRS", "APPLIANCE", "SAFETY", "Elevator",
                   "Boilers", "Indoor Air Quality", "Lead", "OUTSIDE BUILDING", "Mold"],
    "주거 소음": ["Noise - Residential"],
    "거리·상업 소음": ["Noise - Street/Sidewalk", "Noise - Commercial", "Noise", "Noise - Vehicle",
                  "Noise - Helicopter", "Noise - Park", "Illegal Fireworks"],
    "주차·차량": ["Illegal Parking", "Blocked Driveway", "Abandoned Vehicle", "Derelict Vehicles"],
    "도로·상하수도": ["Street Condition", "Traffic Signal Condition", "Street Light Condition", "Sidewalk Condition",
                "Curb Condition", "Highway Condition", "Street Sign - Damaged", "Street Sign - Dangling",
                "Street Sign - Missing", "Sewer", "Water System", "Water Maintenance", "Sewer Maintenance",
                "Root/Sewer/Sidewalk Condition", "Water Conservation"],
    "청소·쓰레기": ["Dirty Condition", "Illegal Dumping", "Missed Collection", "Residential Disposal Complaint",
                "Graffiti", "Illegal Posting", "Litter Basket Request", "Street Sweeping Complaint", "Dead Animal",
                "Sanitation Worker or Vehicle Complaint", "Obstruction", "Rodent"],
    "나무·공원·눈": ["Damaged Tree", "Overgrown Tree/Branches", "Dead/Dying Tree", "Illegal Tree Damage",
                "New Tree Request", "Snow or Ice", "Maintenance or Facility", "Animal in a Park",
                "Violation of Park Rules"],
    "노숙·거리 질서": ["Encampment", "Homeless Person Assistance", "Panhandling", "Drug Activity", "Drinking",
                  "Non-Emergency Police Matter", "Vendor Enforcement", "Smoking or Vaping", "Cannabis Retailer"],
    "교통수단·방문객": ["Taxi Complaint", "For Hire Vehicle Complaint", "Lost Property", "Traffic"],
    "상업·소비자": ["Consumer Complaint", "Food Establishment", "Food Poisoning", "Outdoor Dining"],
    "건물·공사": ["General Construction/Plumbing", "Building/Use", "Real Time Enforcement",
              "Special Projects Inspection Team (SPIT)", "Emergency Response Team (ERT)"],
}
THEME_OF = {p: t for t, ps in THEMES.items() for p in ps}
THEME_ORDER = list(THEMES) + ["기타"]

# ─────────────────────────────────────────────
# 0. 데이터 불러오기
# ─────────────────────────────────────────────
df = pd.read_parquet(CLEAN, columns=["borough", "problem", "community_board", "incident_group_id",
                                     "created_date", "flag_system_duplicate"])
n0 = len(df)
df = df[~df["flag_system_duplicate"] & df["borough"].isin(BOROUGHS)].copy()
df["borough"] = df["borough"].astype(str)
df["problem"] = df["problem"].astype(str)
df["theme"] = df["problem"].map(THEME_OF).fillna("기타")
print(f"분석 대상 {len(df):,}행 (중복 의심·자치구 결측 {n0 - len(df):,}행 제외)")

pop = pd.read_csv(POP, encoding="utf-8").set_index("borough")
KO = pop["borough_ko"].to_dict()

# ─────────────────────────────────────────────
# 1. 자치구 × 유형 배율 (신고 기준 + 사건 기준)
# ─────────────────────────────────────────────
print_header("1. 자치구 × 유형 배율")
cnt = df.groupby(["borough", "problem"]).size().unstack(fill_value=0)          # 신고 수
grp = df.groupby(["borough", "problem"])["incident_group_id"].nunique().unstack(fill_value=0)  # 사건 수


def ratio_table(c: pd.DataFrame) -> pd.DataFrame:
    """자치구별로 '자치구 비중'과 '나머지 자치구 비중', 배율을 긴 표로 만든다."""
    rows = []
    for b in c.index:
        me = c.loc[b] / c.loc[b].sum() * 100
        rest_c = c.drop(index=b).sum()
        rest = rest_c / rest_c.sum() * 100
        rows.append(pd.DataFrame({"borough": b, "problem": c.columns, "건수": c.loc[b].values,
                                  "자치구 비중(%)": me.values, "나머지 비중(%)": rest.values}))
    out = pd.concat(rows, ignore_index=True)
    out["배율"] = out["자치구 비중(%)"] / out["나머지 비중(%)"].replace(0, np.nan)
    return out


lq = ratio_table(cnt)
lq_g = ratio_table(grp).rename(columns={"배율": "사건 기준 배율"})[["borough", "problem", "사건 기준 배율"]]
lq = lq.merge(lq_g, on=["borough", "problem"])
lq["theme"] = lq["problem"].map(THEME_OF).fillna("기타")
lq["1만명당"] = lq["건수"] / lq["borough"].map(pop["population_census2020"]) * PER
lq["순위 대상"] = (lq["자치구 비중(%)"] >= MIN_ME) | (lq["나머지 비중(%)"] >= MIN_REST)

# ─────────────────────────────────────────────
# 2. 특징 유형이 자치구 전체의 모습인가? 커뮤니티 보드(CB) 일관성
#    CB 배율 = CB 안 비중 / 나머지 자치구 비중. 자치구 CB 중 몇 %가 배율 1을 넘는지 센다
# ─────────────────────────────────────────────
print_header("2. 커뮤니티 보드 일관성")
cbn = pd.to_numeric(df["community_board"].astype(str).str[:2], errors="coerce")
# 공원·공항 등 특수 코드(26번 이상)를 빼고, 자치구 이름이 다른 CB(예: borough=MANHATTAN 인데 "01 QUEENS", 몇 건)도 뺀다
cb_boro = df["community_board"].astype(str).str[3:]
cbdf = df[(cbn >= 1) & (cbn <= 18) & (cb_boro == df["borough"])]
cbc = cbdf.groupby(["borough", "community_board"], observed=True)["problem"].value_counts().rename("n").reset_index()
cbc["cb_share"] = cbc["n"] / cbc.groupby(["borough", "community_board"])["n"].transform("sum") * 100
cbc = cbc.merge(lq[["borough", "problem", "나머지 비중(%)"]], on=["borough", "problem"], how="left")
cbc["cb_ratio"] = cbc["cb_share"] / cbc["나머지 비중(%)"]
n_cb = cbdf.groupby("borough")["community_board"].nunique()
cons = (cbc[cbc["cb_ratio"] > 1].groupby(["borough", "problem"]).size().rename("배율>1 CB 수").reset_index())
lq = lq.merge(cons, on=["borough", "problem"], how="left")
lq["배율>1 CB 수"] = lq["배율>1 CB 수"].fillna(0).astype(int)
lq["CB 수"] = lq["borough"].map(n_cb)
lq["배율>1 CB 비율(%)"] = lq["배율>1 CB 수"] / lq["CB 수"] * 100
top_cb = (cbc.sort_values("n", ascending=False).drop_duplicates(["borough", "problem"])
          .set_index(["borough", "problem"]))
tot = cbc.groupby(["borough", "problem"])["n"].sum()
lq = lq.set_index(["borough", "problem"])
lq["최다 CB"] = top_cb["community_board"]
lq["최다 CB 점유(%)"] = top_cb["n"] / tot * 100
lq = lq.reset_index()

lq["borough_ko"] = lq["borough"].map(KO)
lq_out = lq[["borough", "borough_ko", "problem", "theme", "건수", "1만명당", "자치구 비중(%)", "나머지 비중(%)",
             "배율", "사건 기준 배율", "배율>1 CB 수", "CB 수", "배율>1 CB 비율(%)", "최다 CB", "최다 CB 점유(%)",
             "순위 대상"]]
save_table(lq_out.round(3), "edac10_problem_ratio")

# 자치구별 특징 유형(많음 상위 10, 적음 하위 8)
sig = []
for b in BOROUGHS:
    s = lq_out[(lq_out["borough"] == b) & lq_out["순위 대상"]]
    hi = s.sort_values("배율", ascending=False).head(10).assign(방향="많음")
    lo = s[s["나머지 비중(%)"] >= MIN_REST].sort_values("배율").head(8).assign(방향="적음")
    sig.append(pd.concat([hi, lo]))
sig = pd.concat(sig, ignore_index=True)
print(sig[["borough_ko", "방향", "problem", "자치구 비중(%)", "나머지 비중(%)", "배율", "사건 기준 배율",
           "배율>1 CB 비율(%)"]].round(2).to_string(index=False))
save_table(sig.round(3), "edac10_signature")

# ─────────────────────────────────────────────
# 3. 주제별 비중과 배율
# ─────────────────────────────────────────────
print_header("3. 주제별 비중")
th = df.groupby(["borough", "theme"]).size().unstack(fill_value=0).reindex(columns=THEME_ORDER, fill_value=0)
th_share = th.div(th.sum(axis=1), axis=0) * 100
th_ratio = pd.DataFrame({b: th_share.loc[b] / (th.drop(index=b).sum() / th.drop(index=b).sum().sum() * 100)
                         for b in th.index}).T
th_pc = th.div(pop["population_census2020"], axis=0).dropna() * PER
city_pc = th.sum() / pop.loc[th.index, "population_census2020"].sum() * PER
th_pc_ratio = th_pc / city_pc
th_share, th_ratio, th_pc, th_pc_ratio = (x.loc[BOROUGHS] for x in (th_share, th_ratio, th_pc, th_pc_ratio))
print(th_share.round(1).to_string())
print(th_ratio.round(2).to_string())


def long(x: pd.DataFrame, val: str) -> pd.DataFrame:
    return x.rename_axis(index="borough", columns="theme").stack().rename(val).reset_index()


theme_long = (long(th_share, "비중(%)").merge(long(th_ratio, "배율"))
              .merge(long(th_pc, "1만명당")).merge(long(th_pc_ratio, "1만명당 시 평균 대비")))
theme_long["borough_ko"] = theme_long["borough"].map(KO)
save_table(theme_long.round(3), "edac10_theme")

# 주제별 대표 유형 목록 (대시보드 설명용)
save_table(pd.DataFrame([(t, p) for t, ps in THEMES.items() for p in ps], columns=["theme", "problem"]),
           "edac10_theme_members")

# ─────────────────────────────────────────────
# 4. 월별 주제 비중 (계절에 따라 성격이 바뀌는가: 겨울 난방·눈, 여름 소음)
# ─────────────────────────────────────────────
print_header("4. 월별 주제 비중")
df["ym"] = df["created_date"].dt.strftime("%Y-%m")
m = df.groupby(["borough", "ym", "theme"]).size().rename("n").reset_index()
m["비중(%)"] = m["n"] / m.groupby(["borough", "ym"])["n"].transform("sum") * 100
m["borough_ko"] = m["borough"].map(KO)
save_table(m.round(3), "edac10_theme_monthly")

# ─────────────────────────────────────────────
# 5. 자치구끼리 얼마나 닮았나: 유형 비중 벡터의 코사인 유사도 + 시 전체와의 거리
#    코사인 유사도 1 = 구성이 똑같음. JS 거리 0 = 비교 대상과 구성이 똑같음
# ─────────────────────────────────────────────
print_header("5. 자치구 유사도")
P = cnt.loc[BOROUGHS].div(cnt.loc[BOROUGHS].sum(axis=1), axis=0)
U = P.values / np.linalg.norm(P.values, axis=1, keepdims=True)
sim = pd.DataFrame(U @ U.T, index=[KO[b] for b in BOROUGHS], columns=[KO[b] for b in BOROUGHS])
print(sim.round(3).to_string())
save_table(sim.round(4).reset_index(names="자치구"), "edac10_similarity")

city_p = cnt.sum() / cnt.values.sum()


def js(p, q):
    """젠슨-섀넌 거리: 두 비중 분포가 얼마나 다른가. 0 = 똑같음, 1 = 전혀 겹치지 않음."""
    mm = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda a, b: np.nansum(np.where(a > 0, a * np.log2(a / b), 0))  # noqa: E731
        return np.sqrt((kl(p, mm) + kl(q, mm)) / 2)


def rest_p(b):
    """b를 뺀 나머지 자치구의 유형 비중. 큰 자치구가 시 전체를 끌어당기는 효과를 뺀다."""
    r = cnt.loc[BOROUGHS].drop(index=b).sum()
    return (r / r.sum()).values


dist = pd.DataFrame({"borough": BOROUGHS, "borough_ko": [KO[b] for b in BOROUGHS],
                     "시 전체와의 JS 거리": [js(P.loc[b].values, city_p.values) for b in BOROUGHS],
                     "나머지 자치구와의 JS 거리": [js(P.loc[b].values, rest_p(b)) for b in BOROUGHS],
                     "50% 도달 유형 수": [int((P.loc[b].sort_values(ascending=False).cumsum() < 0.5).sum() + 1)
                                     for b in BOROUGHS],
                     "유효 유형 수": [1 / (P.loc[b] ** 2).sum() for b in BOROUGHS],
                     "신고수": cnt.loc[BOROUGHS].sum(axis=1).values})
dist["시 전체 신고 중 비중(%)"] = dist["신고수"] / dist["신고수"].sum() * 100
print(dist.round(3).to_string(index=False))
save_table(dist.round(4), "edac10_distance")

# ─────────────────────────────────────────────
# 6. 그림: 주제 배율 히트맵 / 자치구별 특징 유형 / 유사도
# ─────────────────────────────────────────────
ko_rows = [KO[b] for b in BOROUGHS]
cols = THEME_ORDER[:-1]
fig, ax = plt.subplots(figsize=(13, 4.6))
v = np.log2(th_ratio[cols].values)
im = ax.imshow(v, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right")
ax.set_yticks(range(len(ko_rows)), ko_rows)
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        r = th_ratio[cols].iat[i, j]
        ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v[i, j]) > 1.2 else "black")
ax.set_title("주제별 배율 (자치구 비중 ÷ 나머지 자치구 비중, 빨강 = 많음 · 파랑 = 적음)")
cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(["0.25배", "0.5배", "1배", "2배", "4배"])
save_fig(fig, "edac10_theme_ratio")

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
axes = axes.ravel()
axes[-1].axis("off")
for ax, b in zip(axes, BOROUGHS):
    s = sig[sig["borough"] == b]
    s = pd.concat([s[s["방향"] == "많음"].head(7), s[s["방향"] == "적음"].head(5).iloc[::-1]]).iloc[::-1]
    y = np.log2(s["배율"].clip(lower=1 / 16, upper=16))
    ax.barh(s["problem"], y, color=["#D6604D" if x > 0 else "#4393C3" for x in y])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title(KO[b], fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4], ["⅛", "¼", "½", "1", "2", "4", "8", "16"])
    ax.tick_params(axis="x", labelbottom=True)
fig.suptitle("자치구별 특징 유형: 나머지 자치구 대비 배율 (로그 눈금)")
save_fig(fig, "edac10_signature")
print("완료")
