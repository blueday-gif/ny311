"""zip 군집분석 재실행 (이상치 분리 + 최소 건수 상향 + 부트스트랩 안정성).

03_clustering.py 결과에서 단일 zip 군집과 헬리콥터 지배 zip이 군집 구조를 흔들었기 때문에
다음 기준으로 다시 수행한다.
  1. 최소 건수 기준: 30건 -> 100건 이상
  2. 헬리콥터 비율 30% 이상 zip은 '특수 유형'으로 분리 보고 (군집 입력에서 제외)
  3. 헬리콥터 비율 변수는 제외 (남은 zip에서 거의 0)
  4. k 선택: 최소 군집 크기 3 이상인 k 중 부트스트랩 안정성(ARI 평균)이 가장 높은 k

입력:  data/processed/nyc311_zip_features.csv
출력:  outputs/tables/clust2_*.csv, outputs/figures/clust2_*.png
실행:  프로젝트 루트에서  python analysis/04_clustering_refined.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

SEED = 42
MIN_COUNT = 100
HELI_THRESHOLD = 0.30
K_RANGE = range(2, 8)
MIN_CLUSTER_SIZE = 3
N_BOOT = 300

FEATURES = [
    "share_street", "share_residential", "share_commercial", "share_generic",
    "share_vehicle", "share_park", "night_share", "weekend_share", "log_n_complaints",
]
FEAT_KO = {
    "share_street": "거리/보도 비율", "share_residential": "주거 비율",
    "share_commercial": "상업시설 비율", "share_generic": "일반(공사·설비) 비율",
    "share_helicopter": "헬리콥터 비율", "share_vehicle": "차량 비율", "share_park": "공원 비율",
    "night_share": "야간(22–05시) 비율", "weekend_share": "주말 비율",
    "log_n_complaints": "민원 규모 log(건수)",
}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32a2a"]
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 9,
})


def save_fig(fig, name):
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(df, name, index=False):
    df.to_csv(TABLES / name, index=index, encoding="utf-8-sig")


def bootstrap_ari(X, k, n_boot, rng):
    """전체 표본 군집과 부트스트랩 표본 군집의 일치도(ARI). 높을수록 안정."""
    full = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
    out = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        uniq = np.unique(idx)
        lb = KMeans(k, n_init=10, random_state=b).fit(X[idx]).predict(X[uniq])
        out.append(adjusted_rand_score(full[uniq], lb))
    return np.array(out)


def relabel_by_size(lab):
    order = pd.Series(lab).value_counts().index.tolist()
    m = {old: new + 1 for new, old in enumerate(order)}
    return pd.Series(lab).map(m).values


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(PROC / "nyc311_zip_features.csv", dtype={"zip": str})
    small = feat["n_complaints"] < MIN_COUNT
    heli = ~small & (feat["share_helicopter"] >= HELI_THRESHOLD)
    data = feat[~small & ~heli].reset_index(drop=True)

    log = pd.DataFrame([
        ("zip 전체", len(feat), int(feat["n_complaints"].sum()), ""),
        (f"제외: 건수 {MIN_COUNT}건 미만", int(small.sum()), int(feat.loc[small, "n_complaints"].sum()),
         "비율 변수가 불안정"),
        (f"분리: 헬리콥터 비율 {HELI_THRESHOLD:.0%} 이상", int(heli.sum()), int(feat.loc[heli, "n_complaints"].sum()),
         ", ".join(feat.loc[heli, "zip"])),
        ("군집분석 대상", len(data), int(data["n_complaints"].sum()), ""),
    ], columns=["항목", "zip수", "민원건수", "비고"])
    save_table(log, "clust2_input_log.csv")

    special = feat[heli].copy()
    sp = special[["zip", "n_complaints", "share_helicopter", "share_street", "share_residential",
                  "share_commercial", "share_generic", "night_share", "weekend_share"]].copy()
    for c in sp.columns[2:]:
        sp[c] = (sp[c] * 100).round(1)
    sp = sp.rename(columns={"n_complaints": "건수", **{c: FEAT_KO[c] + "(%)" for c in sp.columns[2:]}})
    save_table(sp.sort_values("건수", ascending=False), "clust2_special_helicopter_zips.csv")

    X = StandardScaler().fit_transform(data[FEATURES])

    # ---- k 선택 ----
    rng = np.random.default_rng(0)
    rows, labels = [], {}
    for k in K_RANGE:
        lab = KMeans(k, n_init=50, random_state=SEED).fit_predict(X)
        labels[k] = lab
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        boot = bootstrap_ari(X, k, N_BOOT, rng)
        sizes = np.bincount(lab)
        rows.append({
            "k": k,
            "silhouette": round(silhouette_score(X, lab), 4),
            "음수실루엣_zip수": int((silhouette_samples(X, lab) < 0).sum()),
            "ARI_vs_ward": round(adjusted_rand_score(lab, ward), 4),
            "bootstrap_ARI_mean": round(boot.mean(), 4),
            "bootstrap_ARI_p10": round(np.percentile(boot, 10), 4),
            "최소군집크기": int(sizes.min()),
            "군집크기": "-".join(map(str, sorted(sizes, reverse=True))),
        })
    ksel = pd.DataFrame(rows)
    eligible = ksel[ksel["최소군집크기"] >= MIN_CLUSTER_SIZE]
    best_k = int(eligible.loc[eligible["bootstrap_ARI_mean"].idxmax(), "k"])
    ksel["선택"] = ksel["k"].eq(best_k)
    save_table(ksel, "clust2_k_selection.csv")

    data["cluster"] = relabel_by_size(labels[best_k])
    data["silhouette"] = silhouette_samples(X, labels[best_k]).round(3)
    k_ids = sorted(data["cluster"].unique())

    # 민감도: 규모 변수(log 건수)를 뺐을 때도 같은 군집이 나오는지
    X_nosize = StandardScaler().fit_transform(data[FEATURES[:-1]])
    lab_nosize = KMeans(best_k, n_init=50, random_state=SEED).fit_predict(X_nosize)
    ari_nosize = adjusted_rand_score(labels[best_k], lab_nosize)

    # ---- 프로파일 ----
    Z = pd.DataFrame(X, columns=FEATURES)
    Z["cluster"] = data["cluster"].values
    zmean = Z.groupby("cluster")[FEATURES].mean()
    prof = data.groupby("cluster").agg(
        zip수=("zip", "size"), 민원건수합계=("n_complaints", "sum"),
        zip당평균건수=("n_complaints", "mean"), 평균실루엣=("silhouette", "mean"),
        **{FEAT_KO[f]: (f, "mean") for f in FEATURES if f != "log_n_complaints"},
    )
    prof["zip당평균건수"] = prof["zip당평균건수"].round(0)
    prof["평균실루엣"] = prof["평균실루엣"].round(3)
    for f in FEATURES[:-1]:
        prof[FEAT_KO[f]] = (prof[FEAT_KO[f]] * 100).round(1)
    prof.insert(0, "소속 zip", [", ".join(data.loc[data["cluster"] == c].sort_values(
        "n_complaints", ascending=False)["zip"]) for c in prof.index])
    save_table(prof, "clust2_profiles.csv", index=True)
    save_table(zmean.round(3).rename(columns=FEAT_KO), "clust2_profiles_zscore.csv", index=True)

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    P = pca.transform(X)
    data["pc1"], data["pc2"] = P[:, 0].round(4), P[:, 1].round(4)
    # 참고용 대안: k=3 소속 (선택 k와 다를 때만)
    alt = []
    if best_k != 3:
        data["cluster_k3"] = relabel_by_size(labels[3])
        alt = ["cluster_k3"]
    out = data[["zip", "cluster"] + alt + ["silhouette", "n_complaints"] + FEATURES[:-1] + ["pc1", "pc2"]].copy()
    for f in FEATURES[:-1]:
        out[f] = (out[f] * 100).round(1)
    out = out.rename(columns={f: FEAT_KO[f] + "(%)" for f in FEATURES[:-1]})
    save_table(out.sort_values(["cluster", "n_complaints"], ascending=[True, False]), "clust2_assignments.csv")

    # ---- 그림 1: k 선택 (실루엣 / 부트스트랩 안정성, 각자 축) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "silhouette", "실루엣 계수 (분리도)"),
                           (axes[1], "bootstrap_ARI_mean", f"부트스트랩 안정성 (ARI 평균, {N_BOOT}회)")]:
        ax.plot(ksel["k"], ksel[col], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        v = ksel.loc[ksel["k"] == best_k, col].iloc[0]
        ax.plot(best_k, v, marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=1.5)
        ax.annotate(f"선택 k={best_k}", (best_k, v), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=INK2)
        ax.set_xticks(list(K_RANGE))
        ax.set_xlabel("군집 수 k")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11)
    save_fig(fig, "clust2_01_k_selection.png")

    # ---- 그림 2: PCA 산점도 ----
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, c in enumerate(k_ids):
        m = data["cluster"] == c
        ax.scatter(data.loc[m, "pc1"], data.loc[m, "pc2"], s=70, marker=MARKERS[i], color=CATEGORICAL[i],
                   edgecolor=SURFACE, linewidth=1.5, label=f"군집 {c} (n={m.sum()})", zorder=3)
    for _, r in data.iterrows():
        ax.annotate(r["zip"], (r["pc1"], r["pc2"]), textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.legend(ncol=len(k_ids), loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3)
    ax.set_title(f"zip 군집 재분석 (K-means, k={best_k}, 헬리콥터형 3개 zip 분리)", pad=32)
    save_fig(fig, "clust2_02_pca_scatter.png")

    # ---- 그림 3: 프로파일 히트맵 ----
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    zm = zmean.loc[k_ids, FEATURES]
    lim = max(2.0, float(np.ceil(np.abs(zm.values).max())))
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(k_ids) + 2.2))
    im = ax.imshow(zm.values, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(FEATURES)), [FEAT_KO[f] for f in FEATURES], rotation=35, ha="right")
    ax.set_yticks(range(len(k_ids)), [f"군집 {c} (n={int((data['cluster'] == c).sum())})" for c in k_ids])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(zm.shape[0]):
        for j in range(zm.shape[1]):
            v = zm.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > lim * 0.55 else INK)
    ax.set_xticks(np.arange(-0.5, zm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, zm.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.set_label("표준화 평균 (0 = 대상 zip 평균)", color=INK2, fontsize=9)
    ax.set_title("군집별 특징 — 파랑은 평균보다 낮음, 빨강은 높음")
    save_fig(fig, "clust2_03_profile_heatmap.png")

    # ---- 그림 4: 군집별 유형 구성 (건수 가중) + 헬리콥터형 ----
    share_cols = ["share_street", "share_residential", "share_commercial", "share_generic",
                  "share_helicopter", "share_vehicle", "share_park", "share_worship"]
    both = pd.concat([data.assign(group=[f"군집 {c}" for c in data["cluster"]]),
                      special.assign(group="특수: 헬리콥터형")])
    w = both[share_cols].mul(both["n_complaints"], axis=0)
    w["group"] = both["group"].values
    comp = w.groupby("group")[share_cols].sum()
    comp = comp.div(comp.sum(axis=1), axis=0) * 100
    main5 = share_cols[:5]
    comp["기타"] = comp.drop(columns=main5).sum(axis=1)
    comp = comp[main5 + ["기타"]].loc[[f"군집 {c}" for c in k_ids] + ["특수: 헬리콥터형"]]
    names = [FEAT_KO[f].replace(" 비율", "") for f in main5] + ["기타(차량·공원 등)"]
    colors = CATEGORICAL[:5] + ["#b5b3ab"]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(comp) + 2))
    left = np.zeros(len(comp))
    y = np.arange(len(comp))
    for col, nm, c in zip(comp.columns, names, colors):
        vals = comp[col].values
        ax.barh(y, vals, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=1.5, label=nm)
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v >= 9:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=INK if c == CATEGORICAL[3] else "#ffffff")
        left += vals
    ax.set_yticks(y, comp.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("유형 구성비 (%, 민원 건수 가중)")
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1, columnspacing=1)
    ax.set_title("군집별 민원 유형 구성", pad=30)
    save_fig(fig, "clust2_04_cluster_composition.png")

    print(log.to_string(index=False))
    print()
    print(ksel.to_string(index=False))
    print(f"\n선택 k = {best_k}  |  규모 변수 제외 시 ARI = {ari_nosize:.3f}")
    print(prof.T.to_string())


if __name__ == "__main__":
    main()
