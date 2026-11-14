"""유형 · 지역 · 시간대별 기술통계 (표 + 그래프).

입력:  data/processed/nyc311_noise_clean.csv         (유형/시간: 전체 102,859행)
       data/processed/nyc311_noise_zip_analysis.csv  (지역: zip 유효 102,585행)
       -> 먼저 analysis/01_preprocess.py 를 실행해야 한다.
출력:  outputs/tables/desc_*.csv
       outputs/figures/desc_*.png

실행: 프로젝트 루트에서  python analysis/02_descriptive_stats.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

# ---- 색상 / 스타일 (dataviz 기본 팔레트, 라이트 모드) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_GRAY = "#b5b3ab"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE = CATEGORICAL[0]

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

TYPE_KO = {
    "Noise - Street/Sidewalk": "거리/보도",
    "Noise - Residential": "주거",
    "Noise - Commercial": "상업시설",
    "Noise": "일반(공사·설비 등)",
    "Noise - Helicopter": "헬리콥터",
    "Noise - Vehicle": "차량",
    "Noise - Park": "공원",
    "Noise - House of Worship": "종교시설",
}
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
BAND_ORDER = ["night", "morning", "afternoon", "evening", "late_night"]
BAND_KO = {
    "night": "심야 00–05시",
    "morning": "오전 06–11시",
    "afternoon": "오후 12–17시",
    "evening": "저녁 18–21시",
    "late_night": "밤 22–23시",
}


def pct(s: pd.Series) -> pd.Series:
    return (s / s.sum() * 100).round(2)


def save_fig(fig, name: str) -> None:
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(TABLES / name, index=index, encoding="utf-8-sig")


def hbar(ax, labels, values, fmt="{:,.0f}", color=SINGLE):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=color, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    vmax = max(values)
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, vmax * 1.12)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROC / "nyc311_noise_clean.csv", dtype={"unique_key": str, "incident_zip": str})
    dz = pd.read_csv(PROC / "nyc311_noise_zip_analysis.csv", dtype={"unique_key": str, "incident_zip": str})
    n_all, n_zip = len(df), len(dz)
    df["type_ko"] = df["complaint_std"].map(TYPE_KO)
    dz["type_ko"] = dz["complaint_std"].map(TYPE_KO)
    type_order = df["type_ko"].value_counts().index.tolist()

    # =====================================================================
    # 1. 유형별
    # =====================================================================
    t_type = df["type_ko"].value_counts().rename_axis("유형").reset_index(name="건수")
    t_type["비율(%)"] = pct(t_type["건수"])
    t_type["누적비율(%)"] = t_type["비율(%)"].cumsum().round(2)
    save_table(t_type, "desc_type_counts.csv")

    t_desc = (
        df.groupby(["type_ko", "descriptor"]).size().rename("건수").reset_index()
        .rename(columns={"type_ko": "유형", "descriptor": "세부유형"})
    )
    t_desc["유형내비율(%)"] = t_desc.groupby("유형")["건수"].transform(lambda s: s / s.sum() * 100).round(2)
    t_desc["전체비율(%)"] = (t_desc["건수"] / n_all * 100).round(2)
    t_desc["유형"] = pd.Categorical(t_desc["유형"], type_order, ordered=True)
    t_desc = t_desc.sort_values(["유형", "건수"], ascending=[True, False])
    save_table(t_desc, "desc_type_descriptor_counts.csv")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    hbar(ax, t_type["유형"], t_type["건수"])
    ax.set_title(f"유형별 소음 민원 건수 (맨해튼, 2024.4–9, n={n_all:,})")
    ax.set_xlabel("건수")
    save_fig(fig, "desc_01_type_counts.png")

    top_desc = df.assign(label=df["type_ko"] + " · " + df["descriptor"].str.replace(r"\s*\([A-Z0-9]+\)$", "", regex=True))
    top_desc = top_desc["label"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    hbar(ax, top_desc.index, top_desc.values)
    ax.set_title("세부유형 상위 12개")
    ax.set_xlabel("건수")
    save_fig(fig, "desc_02_descriptor_top12.png")

    # =====================================================================
    # 2. 지역별 (zip)
    # =====================================================================
    zip_n = dz["incident_zip"].value_counts()
    zip_type = pd.crosstab(dz["incident_zip"], dz["type_ko"])[type_order]
    zip_band = pd.crosstab(dz["incident_zip"], dz["time_band"])
    t_zip = pd.DataFrame({
        "zip": zip_n.index,
        "건수": zip_n.values,
    })
    t_zip["비율(%)"] = pct(t_zip["건수"])
    t_zip["순위"] = np.arange(1, len(t_zip) + 1)
    t_zip["최다유형"] = zip_type.idxmax(axis=1).reindex(t_zip["zip"]).values
    t_zip["최다유형비율(%)"] = (zip_type.max(axis=1) / zip_type.sum(axis=1) * 100).round(1).reindex(t_zip["zip"]).values
    night = dz.groupby("incident_zip")["is_night"].mean().mul(100).round(1)
    t_zip["야간비율(%)"] = night.reindex(t_zip["zip"]).values
    t_zip["30건이상"] = t_zip["건수"] >= 30
    save_table(t_zip, "desc_zip_counts.csv")

    zip_type_pct = zip_type.div(zip_type.sum(axis=1), axis=0).mul(100).round(2)
    out = zip_type.add_suffix("_건수").join(zip_type_pct.add_suffix("_비율(%)"))
    out.insert(0, "합계", zip_type.sum(axis=1))
    save_table(out.sort_values("합계", ascending=False), "desc_zip_type_crosstab.csv", index=True)

    # 분포 요약 (zip 단위 건수의 기술통계)
    zs = zip_n.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1)
    zs.index = ["zip수", "평균", "표준편차", "최소", "25%", "중앙값", "75%", "90%", "최대"]
    top10_share = zip_n.head(10).sum() / n_zip * 100
    zs["상위10개zip_점유율(%)"] = round(top10_share, 1)
    save_table(zs.rename("값").rename_axis("통계량").reset_index(), "desc_zip_summary.csv")

    top15 = zip_n.head(15)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    hbar(ax, top15.index, top15.values)
    ax.set_title(f"민원 건수 상위 15개 zip (상위 10개가 전체의 {top10_share:.1f}%)")
    ax.set_xlabel("건수")
    save_fig(fig, "desc_03_zip_top15.png")

    # 상위 15 zip × 유형 구성비 (100% 누적 막대, 상위 5유형 + 기타)
    keep_types = type_order[:5]
    comp = zip_type.loc[top15.index].copy()
    comp["기타"] = comp.drop(columns=keep_types).sum(axis=1)
    comp = comp[keep_types + ["기타"]]
    comp = comp.div(comp.sum(axis=1), axis=0) * 100
    colors = CATEGORICAL[:5] + [OTHER_GRAY]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    left = np.zeros(len(comp))
    y = np.arange(len(comp))
    for col, c in zip(comp.columns, colors):
        ax.barh(y, comp[col], left=left, height=0.66, color=c, edgecolor=SURFACE, linewidth=1.5, label=col)
        for yi, (l, v) in enumerate(zip(left, comp[col])):
            if v >= 12:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color=INK if c == CATEGORICAL[3] else "#ffffff")
        left += comp[col].values
    ax.set_yticks(y, comp.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("유형 구성비 (%)")
    ax.tick_params(axis="y", length=0)
    ax.set_title("상위 15개 zip의 유형 구성비", pad=30)
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1, columnspacing=1)
    save_fig(fig, "desc_04_zip_type_composition.png")

    # =====================================================================
    # 3. 시간대별
    # =====================================================================
    t_hour = df["hour"].value_counts().sort_index().rename_axis("시").reset_index(name="건수")
    t_hour["비율(%)"] = pct(t_hour["건수"])
    t_hour["일평균건수"] = (t_hour["건수"] / df["created_dt"].str[:10].nunique()).round(2)
    save_table(t_hour, "desc_hour_counts.csv")

    band = df["time_band"].value_counts().reindex(BAND_ORDER)
    t_band = pd.DataFrame({"시간대": [BAND_KO[b] for b in BAND_ORDER], "시간수": [6, 6, 6, 4, 2], "건수": band.values})
    t_band["비율(%)"] = pct(t_band["건수"])
    t_band["시간당건수"] = (t_band["건수"] / t_band["시간수"]).round(1)
    save_table(t_band, "desc_timeband_counts.csv")

    wd = df["weekday"].value_counts().sort_index()
    n_days_wd = pd.to_datetime(df["created_dt"].str[:10]).drop_duplicates().dt.day_of_week.value_counts().sort_index()
    t_wd = pd.DataFrame({"요일": WEEKDAY_KO, "건수": wd.values, "해당요일수": n_days_wd.values})
    t_wd["비율(%)"] = pct(t_wd["건수"])
    t_wd["일평균건수"] = (t_wd["건수"] / t_wd["해당요일수"]).round(1)
    save_table(t_wd, "desc_weekday_counts.csv")

    mo = df["month"].value_counts().sort_index()
    t_mo = pd.DataFrame({"월": [f"{m}월" for m in mo.index], "건수": mo.values})
    days_in_month = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}
    t_mo["비율(%)"] = pct(t_mo["건수"])
    t_mo["일평균건수"] = (mo.values / np.array([days_in_month[m] for m in mo.index])).round(1)
    save_table(t_mo, "desc_month_counts.csv")

    wh = pd.crosstab(df["weekday"], df["hour"])
    wh.index = WEEKDAY_KO
    save_table(wh, "desc_weekday_hour_crosstab.csv", index=True)

    # 시간별 건수 (라인)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_hour["시"], t_hour["건수"], color=SINGLE, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    h_max = int(t_hour.loc[t_hour["건수"].idxmax(), "시"])
    h_min = int(t_hour.loc[t_hour["건수"].idxmin(), "시"])
    for h in (h_max, h_min):
        v = t_hour.loc[t_hour["시"] == h, "건수"].iloc[0]
        ax.annotate(f"{h}시  {v:,}", (h, v), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=INK2)
    ax.axvspan(-0.5, 5.5, color=GRID, alpha=0.35, linewidth=0)
    ax.axvspan(21.5, 23.5, color=GRID, alpha=0.35, linewidth=0)
    ax.text(2.5, ax.get_ylim()[1] * 0.97, "야간(22–05시)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, t_hour["건수"].max() * 1.15)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("시 (뉴욕 현지 시각)")
    ax.set_ylabel("건수")
    ax.set_title(f"시간별 민원 건수 — {h_max}시 정점, {h_min:02d}시 최저")
    save_fig(fig, "desc_05_hour_line.png")

    # 요일 × 시간 히트맵
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(wh.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(24), [str(h) for h in range(24)])
    ax.set_yticks(range(7), WEEKDAY_KO)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.set_label("건수", color=INK2, fontsize=9)
    ax.set_xlabel("시")
    ax.set_title("요일 × 시간 민원 건수 — 금·토 밤에서 토·일 새벽으로 이어지는 집중")
    save_fig(fig, "desc_06_weekday_hour_heatmap.png")

    # 요일 / 월 (일평균) — 두 개의 작은 그림
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, labels, vals, title in [
        (axes[0], t_wd["요일"], t_wd["일평균건수"], "요일별 일평균 건수"),
        (axes[1], t_mo["월"], t_mo["일평균건수"], "월별 일평균 건수"),
    ]:
        x = np.arange(len(labels))
        ax.bar(x, vals, width=0.62, color=SINGLE, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", length=0, colors=INK2)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.15)
        imax = int(np.argmax(vals))
        ax.text(imax, vals.iloc[imax] + max(vals) * 0.02, f"{vals.iloc[imax]:.0f}", ha="center",
                fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=11)
    save_fig(fig, "desc_07_weekday_month.png")

    # =====================================================================
    # 4. 유형 × 시간대 (교차)
    # =====================================================================
    tb = pd.crosstab(df["type_ko"], df["time_band"])[BAND_ORDER].loc[type_order]
    tb.columns = [BAND_KO[b] for b in BAND_ORDER]
    tb_pct = tb.div(tb.sum(axis=1), axis=0).mul(100).round(2)
    out = tb.add_suffix("_건수").join(tb_pct.add_suffix("_비율(%)"))
    out.insert(0, "합계", tb.sum(axis=1))
    save_table(out, "desc_type_timeband_crosstab.csv", index=True)

    th = pd.crosstab(df["type_ko"], df["hour"]).loc[type_order]
    save_table(th, "desc_type_hour_crosstab.csv", index=True)
    th_pct = th.div(th.sum(axis=1), axis=0) * 100

    # 주요 4유형의 시간 분포 (유형 내 비율 → 하나의 축에서 모양 비교)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    main4 = type_order[:4]
    for i, t in enumerate(main4):
        ax.plot(th_pct.columns, th_pct.loc[t], color=CATEGORICAL[i], linewidth=2, label=t)
    for i, t in enumerate(main4):
        y_end = th_pct.loc[t].iloc[-1]
        ax.annotate(t, (23, y_end), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("시")
    ax.set_ylabel("유형 내 비율 (%)")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1.5)
    ax.set_title("주요 4개 유형의 시간 분포 — 일반(공사·설비)만 오전 9시에 별도 정점", pad=30)
    save_fig(fig, "desc_08_type_hour_profile.png")

    # 유형 × 시간대 구성비 히트맵 (행 = 유형, 100%)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    im = ax.imshow(tb_pct.values, aspect="auto", cmap=cmap, vmin=0, vmax=tb_pct.values.max())
    ax.set_xticks(range(tb_pct.shape[1]), tb_pct.columns)
    ax.set_yticks(range(tb_pct.shape[0]), tb_pct.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    vmax = tb_pct.values.max()
    for i in range(tb_pct.shape[0]):
        for j in range(tb_pct.shape[1]):
            v = tb_pct.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > vmax * 0.55 else INK)
    ax.set_xticks(np.arange(-0.5, tb_pct.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, tb_pct.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("유형별 시간대 구성비 (%, 행 합계 100)")
    save_fig(fig, "desc_09_type_timeband_heatmap.png")

    print(f"전체 {n_all:,}행 / zip 분석 {n_zip:,}행")
    print("표:", sorted(p.name for p in TABLES.glob("desc_*.csv")))
    print("그림:", sorted(p.name for p in FIGS.glob("desc_*.png")))


if __name__ == "__main__":
    main()
