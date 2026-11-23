"""
[전처리 후 EDA] 기본 탐색 분석

핵심 질문 : 전처리한 메인 테이블은 어떤 모습이고, 분석할 때 무엇을 조심해야 하는가?
입력      : data/processed/311_requests_clean.parquet, agency_lookup.parquet (scripts/preprocess.py 결과)
출력      : outputs/tables/edac_*.csv, outputs/figures/edac_*.png
실행 방법 : py -3.12 analysis/eda_clean_basic.py

필터 기준 (preprocessing 안내 문서의 예시와 같음)
- 신고량      : NOT flag_system_duplicate
- 처리 시간   : is_closed AND resolution_hours > 0 AND NOT flag_system_duplicate AND NOT flag_status_date_mismatch
- 지역 분석   : NOT flag_no_location AND NOT flag_system_duplicate
"""

import pyarrow.parquet as pq

from common import ROOT, np, pd, plt, print_header, save_fig, save_table, setup_plot

CLEAN = ROOT / "data" / "processed" / "311_requests_clean.parquet"
AGENCY = ROOT / "data" / "processed" / "agency_lookup.parquet"

setup_plot()

# ─────────────────────────────────────────────
# 0. 불러오기: 분석에 쓸 열만 읽는다
# ─────────────────────────────────────────────
pf = pq.ParquetFile(CLEAN)
schema = pf.schema_arrow
n_rows = pf.metadata.num_rows

USE = [
    "unique_key", "created_date", "closed_date", "agency", "problem", "borough", "status",
    "open_data_channel_type", "location_type", "latitude", "longitude",
    "resolution_hours", "is_closed", "created_month", "created_dow", "created_hour",
    "incident_group_id", "incident_group_size", "borough_imputed",
] + [c for c in schema.names if c.startswith("flag_")]
df = pd.read_parquet(CLEAN, columns=USE)
agency_name = pd.read_parquet(AGENCY).set_index("agency")["agency_name"]
for c in ["agency", "problem", "borough", "status", "open_data_channel_type", "location_type"]:
    df[c] = df[c].astype("category")

base = df[~df["flag_system_duplicate"]]  # 중복 의심만 뺀 기본 집합

# ─────────────────────────────────────────────
# 1. 개요: 행·열, 자료형, 결측률
# ─────────────────────────────────────────────
print_header("1. 개요")
print(f"행 수 {n_rows:,} / 열 수 {len(schema.names)} / 기간 {df['created_date'].min()} ~ {df['created_date'].max()}")
dtypes = pd.Series([str(t) for t in schema.types], index=schema.names)
print(dtypes.value_counts().to_string())

# 결측률은 전체 열에 대해 메타데이터(통계)로 계산한다. 읽지 않고도 null 개수를 알 수 있다
nulls = {name: 0 for name in schema.names}
for rg in range(pf.metadata.num_row_groups):
    meta = pf.metadata.row_group(rg)
    for i in range(meta.num_columns):
        st = meta.column(i).statistics
        nulls[meta.column(i).path_in_schema] += st.null_count if st is not None and st.has_null_count else 0
missing = (
    pd.DataFrame({"열": list(nulls), "자료형": dtypes.values, "결측 수": list(nulls.values())})
    .assign(**{"결측률(%)": lambda d: (d["결측 수"] / n_rows * 100).round(2)})
    .sort_values("결측률(%)", ascending=False)
)
save_table(missing, "edac01_missing")
print(missing[missing["결측 수"] > 0].to_string(index=False))

show = missing[missing["결측 수"] > 0]
fig, ax = plt.subplots(figsize=(8, max(4, len(show) * 0.28)))
ax.barh(show["열"][::-1], show["결측률(%)"][::-1], color="#4C78A8")
ax.set_xlabel("결측률 (%)")
ax.set_title("전처리 후 열별 결측률 (결측이 있는 열만)")
save_fig(fig, "edac01_missing_bar")

# ─────────────────────────────────────────────
# 2. 품질 플래그
# ─────────────────────────────────────────────
print_header("2. 품질 플래그")
flag_cols = [c for c in df.columns if c.startswith("flag_")] + ["borough_imputed"]
flags = pd.DataFrame({
    "플래그": flag_cols,
    "건수": [int(df[c].sum()) for c in flag_cols],
})
flags["비율(%)"] = (flags["건수"] / n_rows * 100).round(3)
save_table(flags, "edac02_flags")
print(flags.to_string(index=False))

# 플래그끼리 얼마나 겹치는지 (처리 시간 관련 플래그만)
time_flags = ["flag_negative_duration", "flag_zero_duration", "flag_midnight_close",
              "flag_status_date_mismatch", "flag_updated_before_created"]
overlap = pd.DataFrame(
    [[int((df[a] & df[b]).sum()) for b in time_flags] for a in time_flags],
    index=time_flags, columns=time_flags,
)
save_table(overlap, "edac02_flag_overlap", index=True)

# ─────────────────────────────────────────────
# 3. 신고량: 월·요일·시간
# ─────────────────────────────────────────────
print_header("3. 신고량 (중복 의심 제외)")
monthly = base.groupby("created_month").agg(
    신고수=("unique_key", "size"), 동일조건그룹수=("incident_group_id", "nunique"),
).reset_index()
monthly["그룹당 신고수"] = (monthly["신고수"] / monthly["동일조건그룹수"]).round(3)
save_table(monthly, "edac03_monthly")
print(monthly.to_string(index=False))

dow = base["created_dow"].value_counts().sort_index().rename_axis("요일").reset_index(name="신고수")
dow["요일"] = dow["요일"].map(dict(zip(range(1, 8), "월화수목금토일")))
hour = base["created_hour"].value_counts().sort_index().rename_axis("시").reset_index(name="신고수")
save_table(dow, "edac03_by_dow")
save_table(hour, "edac03_by_hour")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(monthly["created_month"], monthly["신고수"], marker="o", label="신고 수")
axes[0].plot(monthly["created_month"], monthly["동일조건그룹수"], marker="o", label="동일 조건 그룹 수")
axes[0].set_title("월별 신고량")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(dow["요일"], dow["신고수"], color="#4C78A8")
axes[1].set_title("요일별 신고량")
axes[2].bar(hour["시"], hour["신고수"], color="#4C78A8")
axes[2].set_title("시간대별 신고량 (접수 시각)")
axes[2].set_xlabel("시")
save_fig(fig, "edac03_volume")

# ─────────────────────────────────────────────
# 4. 범주 분포: 기관, 민원 유형, 자치구, 접수 경로, 상태
# ─────────────────────────────────────────────
print_header("4. 범주 분포 (중복 의심 제외)")


def share_table(col: str, top: int | None = None) -> pd.DataFrame:
    vc = base[col].value_counts(dropna=False)
    t = vc.rename_axis(col).reset_index(name="신고수")
    t[col] = t[col].astype(object).where(t[col].notna(), "(결측)")
    t["비율(%)"] = (t["신고수"] / len(base) * 100).round(2)
    return t.head(top) if top else t


agency_t = share_table("agency")
agency_t.insert(1, "기관 이름", agency_t["agency"].map(agency_name))
problem_t = share_table("problem", 20)
borough_t = share_table("borough")
channel_t = share_table("open_data_channel_type")
status_t = share_table("status")
for name, t in [("agency", agency_t), ("problem_top20", problem_t), ("borough", borough_t),
                ("channel", channel_t), ("status", status_t)]:
    save_table(t, f"edac04_{name}")
    print(t.to_string(index=False), "\n")

fig, ax = plt.subplots(figsize=(8, 6))
p = problem_t.iloc[::-1]
ax.barh(p["problem"].astype(str), p["신고수"], color="#4C78A8")
ax.set_title("민원 유형 상위 20개")
ax.set_xlabel("신고 수")
save_fig(fig, "edac04_problem_top20")

# ─────────────────────────────────────────────
# 5. 처리 시간
# ─────────────────────────────────────────────
print_header("5. 처리 시간")
ok = (df["is_closed"] & (df["resolution_hours"] > 0)
      & ~df["flag_system_duplicate"] & ~df["flag_status_date_mismatch"])
rt = df.loc[ok, ["agency", "problem", "borough", "resolution_hours", "flag_midnight_close"]]
print(f"분석 대상 {len(rt):,}건 (전체의 {len(rt) / n_rows:.1%})")

q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
desc = pd.DataFrame({
    "집합": ["전체 정상 건", "자정 종료 제외"],
    "건수": [len(rt), int((~rt["flag_midnight_close"]).sum())],
    "평균": [rt["resolution_hours"].mean(), rt.loc[~rt["flag_midnight_close"], "resolution_hours"].mean()],
    **{f"p{int(x * 100)}": [rt["resolution_hours"].quantile(x),
                            rt.loc[~rt["flag_midnight_close"], "resolution_hours"].quantile(x)] for x in q},
}).round(2)
save_table(desc, "edac05_hours_summary")
print(desc.to_string(index=False))

by_agency = rt.groupby("agency", observed=True)["resolution_hours"].agg(
    건수="size", 중앙값="median", 평균="mean", p90=lambda s: s.quantile(0.9))
by_agency["중앙값(자정 종료 제외)"] = (
    rt[~rt["flag_midnight_close"]].groupby("agency", observed=True)["resolution_hours"].median())
by_agency["자정 종료 비율(%)"] = rt.groupby("agency", observed=True)["flag_midnight_close"].mean() * 100
by_agency.insert(0, "기관 이름", by_agency.index.map(agency_name))
by_agency = by_agency.sort_values("중앙값").round(2).reset_index()
save_table(by_agency, "edac05_hours_by_agency")
print(by_agency.to_string(index=False))

by_borough = rt.groupby("borough", observed=True, dropna=False)["resolution_hours"].agg(
    건수="size", 중앙값="median", p90=lambda s: s.quantile(0.9)).round(2).reset_index()
save_table(by_borough, "edac05_hours_by_borough")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(np.log10(rt["resolution_hours"]), bins=80, color="#4C78A8")
for h, lab in [(1, "1시간"), (24, "1일"), (24 * 7, "1주"), (24 * 30, "30일")]:
    axes[0].axvline(np.log10(h), color="gray", ls="--", lw=0.8)
    axes[0].text(np.log10(h), axes[0].get_ylim()[1] * 0.95, lab, ha="center", fontsize=8)
axes[0].set_xlabel("log10(처리 시간, 시간)")
axes[0].set_title("처리 시간 분포 (로그 척도)")
a = by_agency.sort_values("중앙값")
axes[1].barh(a["agency"].astype(str), a["중앙값"], color="#4C78A8", label="전체")
axes[1].scatter(a["중앙값(자정 종료 제외)"], a["agency"].astype(str), color="#E45756", zorder=3,
                label="자정 종료 제외")
axes[1].set_xscale("log")
axes[1].set_xlabel("처리 시간 중앙값 (시간, 로그 척도)")
axes[1].set_title("기관별 처리 시간 중앙값")
axes[1].legend()
save_fig(fig, "edac05_hours")

# ─────────────────────────────────────────────
# 6. 위치
# ─────────────────────────────────────────────
print_header("6. 위치 (좌표 있는 행, 중복 의심 제외)")
loc = base[~base["flag_no_location"]]
no_loc = df.groupby("agency", observed=True)["flag_no_location"].mean().mul(100).round(2)
no_loc = no_loc.sort_values(ascending=False).rename("좌표 없음 비율(%)").reset_index()
save_table(no_loc, "edac06_no_location_by_agency")
print(no_loc.head(8).to_string(index=False))

sample = loc.sample(n=min(300_000, len(loc)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 7))
hb = ax.hexbin(sample["longitude"], sample["latitude"], gridsize=120, bins="log", cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax, label="신고 수 (log)")
ax.set_title("신고 위치 밀도 (30만 건 표본)")
ax.set_xlabel("경도")
ax.set_ylabel("위도")
save_fig(fig, "edac06_location_hexbin")

# ─────────────────────────────────────────────
# 7. 중복: 동일 조건 신고 그룹, 기관 판정 중복
# ─────────────────────────────────────────────
print_header("7. 중복")
gsize = df.drop_duplicates("incident_group_id")["incident_group_size"]
bins = [0, 1, 2, 3, 5, 10, 50, np.inf]
labels = ["1", "2", "3", "4-5", "6-10", "11-50", "51+"]
gdist = pd.cut(gsize, bins=bins, labels=labels).value_counts().reindex(labels).rename_axis("그룹 크기")
gdist = gdist.reset_index(name="그룹 수")
gdist["행 수"] = pd.cut(df["incident_group_size"], bins=bins, labels=labels).value_counts().reindex(labels).values
save_table(gdist, "edac07_group_size")
print(gdist.to_string(index=False))

big = (df[df["incident_group_size"] >= 2].groupby("problem", observed=True)
       .size().sort_values(ascending=False).head(10).rename("반복 그룹 소속 행 수").reset_index())
big["해당 유형 전체 대비(%)"] = (big["반복 그룹 소속 행 수"]
                           / big["problem"].map(df["problem"].value_counts()) * 100).round(1)
save_table(big, "edac07_repeat_by_problem")
print(big.to_string(index=False))

dup_ag = base.groupby("agency", observed=True)["flag_agency_duplicate"].agg(["sum", "mean"])
dup_ag.columns = ["기관 판정 중복 수", "비율(%)"]
dup_ag["비율(%)"] = (dup_ag["비율(%)"] * 100).round(2)
dup_ag = dup_ag.sort_values("기관 판정 중복 수", ascending=False).reset_index()
save_table(dup_ag, "edac07_agency_duplicate")
print(dup_ag.head(8).to_string(index=False))

print("\n완료")
