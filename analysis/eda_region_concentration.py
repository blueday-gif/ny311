"""
[전처리 후 EDA] 지역별 민원 유형 쏠림

핵심 질문 : 시 전체로는 소수 유형(불법 주차·주거 소음·난방)에 신고가 몰리는데,
            모든 지역이 그런가? 쏠린 지역과 고르게 퍼진 지역은 무엇이 다른가?
입력      : data/processed/311_requests_clean.parquet
출력      : outputs/tables/edac08_*.csv, outputs/figures/edac08_*.png
실행 방법 : py -3.12 analysis/eda_region_concentration.py

단위 : 커뮤니티 보드(CB) 59개. 공원·공항 등 특수 코드(26번 이상)와 'Unspecified'는 뺀다.
필터 : NOT flag_system_duplicate (신고량 기준과 같음)

쏠림 지표 (CB 안에서 유형별 비중 p_i 로 계산)
- 상위3 점유율 : 그 CB에서 가장 많은 3개 유형의 비중 합
- HHI          : sum(p_i^2). 한 유형만 있으면 1, 고르게 퍼질수록 0에 가깝다
- 유효 유형 수 : 1 / HHI. "사실상 몇 개 유형이 나눠 갖는가"
- 50% 도달 유형 수 : 비중이 큰 순서로 더해 50%를 넘길 때까지 필요한 유형 수
그룹 기준(incident_group_id 고유 수)으로도 같은 지표를 계산해, 반복 신고가 쏠림을 부풀리는지 본다.
"""

from common import ROOT, np, pd, plt, print_header, save_fig, save_table, setup_plot

CLEAN = ROOT / "data" / "processed" / "311_requests_clean.parquet"
setup_plot()

df = pd.read_parquet(
    CLEAN,
    columns=["community_board", "borough", "problem", "incident_group_id", "flag_system_duplicate"],
)
df = df[~df["flag_system_duplicate"]].copy()
cb_num = pd.to_numeric(df["community_board"].str[:2], errors="coerce")
df = df[cb_num.between(1, 18)].copy()
df["cb"] = df["community_board"].astype(str)
df["cb_borough"] = df["cb"].str[3:].str.title()
print(f"분석 대상 {len(df):,}행, CB {df['cb'].nunique()}개")

CITY_TOP3 = ["Illegal Parking", "Noise - Residential", "HEAT/HOT WATER"]


def concentration(counts: pd.DataFrame) -> pd.DataFrame:
    """counts: index=cb, columns=problem, 값=건수. CB별 쏠림 지표를 돌려준다."""
    share = counts.div(counts.sum(axis=1), axis=0)
    srt = np.sort(share.values, axis=1)[:, ::-1]
    cum = srt.cumsum(axis=1)
    hhi = (share**2).sum(axis=1)
    out = pd.DataFrame(
        {
            "건수": counts.sum(axis=1),
            "상위3 점유율(%)": cum[:, 2] * 100,
            "상위20 점유율(%)": cum[:, 19] * 100,
            "HHI": hhi,
            "유효 유형 수": 1 / hhi,
            "50% 도달 유형 수": (cum < 0.5).sum(axis=1) + 1,
            "시 상위3 유형 비중(%)": share.reindex(columns=CITY_TOP3, fill_value=0).sum(axis=1) * 100,
        },
        index=counts.index,
    )
    top = share.apply(lambda r: r.nlargest(3), axis=1)
    out["1위 유형"] = share.idxmax(axis=1)
    out["1위 비중(%)"] = share.max(axis=1) * 100
    out["상위3 유형"] = share.apply(lambda r: " / ".join(r.nlargest(3).index), axis=1)
    for p in CITY_TOP3:
        out[f"{p}(%)"] = share.get(p, 0) * 100
    return out


# ─────────────────────────────────────────────
# 1. CB별 쏠림 지표 (건수 기준, 그룹 기준)
# ─────────────────────────────────────────────
print_header("1. CB별 쏠림 지표")
cnt = df.groupby(["cb", "problem"], observed=True).size().unstack(fill_value=0)
grp = df.groupby(["cb", "problem"], observed=True)["incident_group_id"].nunique().unstack(fill_value=0)
conc = concentration(cnt)
conc_g = concentration(grp)
conc["자치구"] = df.groupby("cb")["cb_borough"].first()
conc["상위3 점유율_그룹(%)"] = conc_g["상위3 점유율(%)"]
conc["유효 유형 수_그룹"] = conc_g["유효 유형 수"]
conc["반복 부풀림(%p)"] = conc["상위3 점유율(%)"] - conc["상위3 점유율_그룹(%)"]

# 쏠림 3단계: 상위3 점유율의 1/3, 2/3 분위로 나눈다
q1, q2 = conc["상위3 점유율(%)"].quantile([1 / 3, 2 / 3])
conc["쏠림 단계"] = pd.cut(
    conc["상위3 점유율(%)"], [-np.inf, q1, q2, np.inf], labels=["분산형", "중간형", "집중형"]
)


# 집중형이면 무엇 때문에 쏠렸는지(1위 유형의 성격)로 다시 나눈다
def driver(r):
    top = r["1위 유형"]
    if r["쏠림 단계"] != "집중형":
        return "-"
    if top in ("Illegal Parking", "Blocked Driveway"):
        return "주차 주도"
    if top.startswith("Noise"):
        return "소음 주도"
    if top in ("HEAT/HOT WATER", "UNSANITARY CONDITION", "PLUMBING", "PAINT/PLASTER"):
        return "주거 설비 주도"
    return "기타"


conc["집중 원인"] = conc.apply(driver, axis=1)
conc = conc.sort_values("상위3 점유율(%)", ascending=False)
print(f"상위3 점유율 분위 경계: {q1:.1f}% / {q2:.1f}%")
print(conc[["자치구", "건수", "상위3 점유율(%)", "유효 유형 수", "쏠림 단계", "집중 원인", "상위3 유형"]].round(1).to_string())
save_table(conc.round(3).reset_index().rename(columns={"cb": "community_board"}), "edac08_cb_concentration")

# ─────────────────────────────────────────────
# 2. 쏠림 단계별 프로필
# ─────────────────────────────────────────────
print_header("2. 쏠림 단계별 프로필")
cols = ["건수", "상위3 점유율(%)", "유효 유형 수", "50% 도달 유형 수", "시 상위3 유형 비중(%)",
        "Illegal Parking(%)", "Noise - Residential(%)", "HEAT/HOT WATER(%)", "반복 부풀림(%p)"]
prof = conc.groupby("쏠림 단계", observed=True)[cols].median()
prof.insert(0, "CB 수", conc.groupby("쏠림 단계", observed=True).size())
print(prof.round(1).to_string())
save_table(prof.round(2).reset_index(), "edac08_tier_profile")

# 자치구 × 쏠림 단계
bt = pd.crosstab(conc["자치구"], conc["쏠림 단계"])
print(bt.to_string())
save_table(bt.reset_index(), "edac08_borough_tier")

drv = conc["집중 원인"].value_counts()
print(drv.to_string())

# 분산형 CB에서 시 평균보다 비중이 특히 높은 유형 (분산형의 '대신 무엇이 많은가')
city_share = cnt.sum() / cnt.values.sum()
share = cnt.div(cnt.sum(axis=1), axis=0)
lift_rows = []
for tier in ["분산형", "집중형"]:
    idx = conc.index[conc["쏠림 단계"] == tier]
    tshare = cnt.loc[idx].sum() / cnt.loc[idx].values.sum()
    lift = (tshare / city_share).where(city_share > 0.003)  # 시 비중 0.3% 이상 유형만
    for p in lift.dropna().nlargest(10).index:
        lift_rows.append({"쏠림 단계": tier, "유형": p, "단계 내 비중(%)": tshare[p] * 100,
                          "시 전체 비중(%)": city_share[p] * 100, "배율": lift[p]})
lift_df = pd.DataFrame(lift_rows)
print(lift_df.round(2).to_string(index=False))
save_table(lift_df.round(3), "edac08_tier_overrep")

# ─────────────────────────────────────────────
# 3. 그림: CB별 상위3 점유율 vs 유효 유형 수
# ─────────────────────────────────────────────
colors = {"분산형": "#4C78A8", "중간형": "#BBBBBB", "집중형": "#E45756"}
fig, ax = plt.subplots(figsize=(9, 6))
for t, c in colors.items():
    s = conc[conc["쏠림 단계"] == t]
    ax.scatter(s["상위3 점유율(%)"], s["유효 유형 수"], s=s["건수"] / 1500, c=c, alpha=0.75, label=t)
for cb, r in conc.iloc[list(range(5)) + list(range(-5, 0))].iterrows():
    ax.annotate(cb, (r["상위3 점유율(%)"], r["유효 유형 수"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("상위 3개 유형 점유율 (%)")
ax.set_ylabel("유효 유형 수 (1/HHI)")
ax.set_title("커뮤니티 보드별 민원 유형 쏠림 (점 크기 = 신고 수)")
ax.legend()
save_fig(fig, "edac08_cb_concentration")

fig, ax = plt.subplots(figsize=(10, 12))
mix = conc[["Illegal Parking(%)", "Noise - Residential(%)", "HEAT/HOT WATER(%)"]].copy()
mix["그 외(%)"] = 100 - mix.sum(axis=1)
mix = mix.iloc[::-1]
mix.plot.barh(stacked=True, ax=ax, color=["#F58518", "#54A24B", "#E45756", "#DDDDDD"], width=0.8)
ax.set_xlabel("CB 안 비중 (%)")
ax.set_ylabel("")
ax.set_title("CB별 시 상위 3개 유형 구성 (위에서부터 쏠림 큰 순)")
ax.legend(["불법 주차", "주거 소음", "난방·온수", "그 외"], loc="lower right")
ax.tick_params(axis="y", labelsize=7)
save_fig(fig, "edac08_cb_top3_mix")
