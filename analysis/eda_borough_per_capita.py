"""
[전처리 후 EDA] 자치구 인구 보정: 주민 1만 명당 민원

핵심 질문 : 건수로는 브루클린이 1위, 스태튼아일랜드가 꼴찌다. 인구 차이(최대 5배)를 걷어 내면
            "누가 311을 더 많이 쓰는가"의 순위가 바뀌는가? 어떤 기관(민원 성격)에서 차이가 나는가?
입력      : data/processed/311_requests_clean.parquet, docs/borough_context.csv (자치구 인구)
출력      : outputs/tables/edac09_*.csv, outputs/figures/edac09_*.png
실행 방법 : py -3.12 analysis/eda_borough_per_capita.py

인구는 두 가지를 쓴다.
- population_census2020 : 2020년 인구총조사(Decennial Census) 공식값. 기본값으로 쓴다
- population_approx     : 최근 ACS 기준 대략값(docs/borough_context.md). 순위가 인구 기준에 따라 흔들리는지 보는 민감도 확인용
필터 : NOT flag_system_duplicate (신고량 기준과 같음), borough 결측 제외
"""

from common import ROOT, pd, plt, print_header, save_fig, save_table, setup_plot

CLEAN = ROOT / "data" / "processed" / "311_requests_clean.parquet"
POP = ROOT / "docs" / "borough_context.csv"
PER = 10_000  # 1만 명당
setup_plot()

# ─────────────────────────────────────────────
# 0. 데이터 불러오기
# ─────────────────────────────────────────────
df = pd.read_parquet(CLEAN, columns=["borough", "agency", "problem", "incident_group_id", "flag_system_duplicate"])
n0 = len(df)
df = df[~df["flag_system_duplicate"] & df["borough"].notna()].copy()
df["borough"] = df["borough"].astype(str)
print(f"분석 대상 {len(df):,}행 (중복 의심·자치구 결측 {n0 - len(df):,}행 제외)")

pop = pd.read_csv(POP, encoding="utf-8")[["borough", "borough_ko", "population_census2020", "population_approx"]]

# ─────────────────────────────────────────────
# 1. 자치구별 1만 명당 신고 수 / 사건(그룹) 수
# ─────────────────────────────────────────────
print_header("1. 자치구별 1만 명당 신고 수")
g = df.groupby("borough").agg(신고수=("borough", "size"), 사건수=("incident_group_id", "nunique"))
t = pop.set_index("borough").join(g, how="inner")
t["1만명당 신고"] = t["신고수"] / t["population_census2020"] * PER
t["1만명당 사건"] = t["사건수"] / t["population_census2020"] * PER
t["1만명당 신고(대략 인구)"] = t["신고수"] / t["population_approx"] * PER
t["사건당 신고"] = t["신고수"] / t["사건수"]
t["건수 순위"] = t["신고수"].rank(ascending=False).astype(int)
t["1만명당 순위"] = t["1만명당 신고"].rank(ascending=False).astype(int)
t["1만명당 순위(대략 인구)"] = t["1만명당 신고(대략 인구)"].rank(ascending=False).astype(int)
t["사건 기준 순위"] = t["1만명당 사건"].rank(ascending=False).astype(int)
city = t["신고수"].sum() / t["population_census2020"].sum() * PER
t["시 평균 대비(배)"] = t["1만명당 신고"] / city
t = t.sort_values("1만명당 신고", ascending=False).reset_index()
print(f"시 전체 1만 명당 {city:,.0f}건")
print(t.drop(columns=["population_approx"]).round(2).to_string(index=False))
save_table(t.round(3), "edac09_borough_per_capita")

# ─────────────────────────────────────────────
# 2. 기관별 1만 명당 신고 (민원 성격별로 어디서 차이가 나나)
#    NYPD=소음·주차, HPD=주거 설비(난방·누수·위생), DSNY=쓰레기·청소, DOT=도로·신호, DEP=수도·소음, DPR=공원·가로수
# ─────────────────────────────────────────────
print_header("2. 기관별 1만 명당 신고")
AGENCIES = ["NYPD", "HPD", "DSNY", "DOT", "DEP", "DPR", "DOB"]
a = (df[df["agency"].isin(AGENCIES)].groupby(["borough", "agency"], observed=True).size()
     .unstack(fill_value=0).reindex(columns=AGENCIES, fill_value=0))
a = a.div(pop.set_index("borough")["population_census2020"], axis=0).dropna() * PER
a = a.loc[t["borough"]]
print(a.round(0).to_string())
save_table(a.round(1).reset_index(), "edac09_agency_per_capita")

# 기관별로 시 평균 대비 배율 (1보다 크면 그 자치구에서 그 성격의 민원이 인구에 비해 많다)
city_a = a.mul(pop.set_index("borough")["population_census2020"], axis=0).sum() / pop["population_census2020"].sum()
ratio = a / city_a
print(ratio.round(2).to_string())
save_table(ratio.round(3).reset_index(), "edac09_agency_ratio")

# ─────────────────────────────────────────────
# 3. 가설 확인용 대표 유형 (docs/borough_context.md 가설 A)
#    실내 설비(임대 주거) vs 외부 인프라(자가·차량 중심)
# ─────────────────────────────────────────────
print_header("3. 대표 유형별 1만 명당 신고")
GROUPS = {
    "난방·온수": ["HEAT/HOT WATER"],
    "실내 설비(누수·위생·도장 등)": ["UNSANITARY CONDITION", "PLUMBING", "PAINT/PLASTER", "WATER LEAK", "DOOR/WINDOW"],
    "주거 소음": ["Noise - Residential"],
    "거리·상업 소음": ["Noise - Street/Sidewalk", "Noise - Commercial", "Noise - Vehicle"],
    "불법 주차·진입로": ["Illegal Parking", "Blocked Driveway"],
    "도로 파손": ["Street Condition"],
    "가로수": ["Damaged Tree", "Overgrown Tree/Branches", "Dead/Dying Tree", "New Tree Request"],
}
rows = []
for name, probs in GROUPS.items():
    s = df[df["problem"].isin(probs)].groupby("borough").size()
    rows.append((s / pop.set_index("borough")["population_census2020"] * PER).rename(name))
gr = pd.concat(rows, axis=1).loc[t["borough"]]
print(gr.round(0).to_string())
save_table(gr.round(1).reset_index(), "edac09_group_per_capita")

# ─────────────────────────────────────────────
# 4. 그림: 건수 vs 1만 명당 신고 / 대표 유형 배율
# ─────────────────────────────────────────────
ko = dict(zip(t["borough"], t["borough_ko"]))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
c = t.sort_values("신고수")
axes[0].barh(c["borough_ko"], c["신고수"] / 1e4, color="#BBBBBB")
axes[0].set_title("신고 수 (만 건)")
c = t.sort_values("1만명당 신고")
axes[1].barh(c["borough_ko"], c["1만명당 신고"], color="#4C78A8")
axes[1].axvline(city, color="gray", ls="--", lw=1, label=f"시 평균 {city:,.0f}")
axes[1].legend(loc="lower right")
axes[1].set_title("주민 1만 명당 신고 수 (2020 인구총조사)")
save_fig(fig, "edac09_borough_per_capita")

gratio = gr / (gr.mul(pop.set_index("borough")["population_census2020"], axis=0).sum()
               / pop["population_census2020"].sum())
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(gratio.values, cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
ax.set_xticks(range(len(gratio.columns)), gratio.columns, rotation=20, ha="right")
ax.set_yticks(range(len(gratio.index)), [ko[b] for b in gratio.index])
for i in range(gratio.shape[0]):
    for j in range(gratio.shape[1]):
        v = gratio.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v - 1) > 0.6 else "black")
ax.set_title("대표 유형별 1만 명당 신고: 시 평균 대비 배율 (1 = 시 평균)")
fig.colorbar(im, ax=ax, shrink=0.8)
save_fig(fig, "edac09_group_ratio")
