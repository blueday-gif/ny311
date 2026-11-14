"""zip 단위 소음 민원 군집분석 (K-means + Ward 계층군집 교차검증).

입력:  data/processed/nyc311_zip_features.csv  (analysis/01_preprocess.py 출력)
출력:  outputs/tables/clust_*.csv
       outputs/figures/clust_*.png

분석 단위: zip (건수 30건 이상인 zip만; 미만은 비율이 불안정해 제외)
변수: 유형 구성비 7개 + 야간비율 + 주말비율 + log(건수)  -> 표준화
      (zip 코드 자체는 식별자이므로 변수로 쓰지 않음, 종교시설 비율은 거의 0이라 제외)

실행: 프로젝트 루트에서  python analysis/03_clustering.py
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
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

SEED = 42
K_RANGE = range(2, 9)
MIN_CLUSTER_SIZE = 2  # 단일 zip 군집만 생기는 k는 선택 대상에서 제외하지 않고 표시만 함

FEATURES = [
    "share_street", "share_residential", "share_commercial", "share_generic",
    "share_helicopter", "share_vehicle", "share_park",
    "night_share", "weekend_share", "log_n_complaints",
]
FEAT_KO = {
    "share_street": "거리/보도 비율",
    "share_residential": "주거 비율",
    "share_commercial": "상업시설 비율",
    "share_generic": "일반(공사·설비) 비율",
    "share_helicopter": "헬리콥터 비율",
    "share_vehicle": "차량 비율",
    "share_park": "공원 비율",
    "night_share": "야간(22–05시) 비율",
    "weekend_share": "주말 비율",
    "log_n_complaints": "민원 규모 log(건수)",
}

# ---- 스타일 (dataviz 기본 팔레트, 라이트) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
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


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(PROC / "nyc311_zip_features.csv", dtype={"zip": str})
    stable = feat[feat["is_stable"]].reset_index(drop=True)
    excluded = feat[~feat["is_stable"]]
    log = pd.DataFrame([
        ("zip 전체", len(feat), int(feat["n_complaints"].sum())),
        ("제외: 건수 30건 미만 zip", len(excluded), int(excluded["n_complaints"].sum())),
        ("군집분석 대상 zip", len(stable), int(stable["n_complaints"].sum())),
    ], columns=["항목", "zip수", "민원건수"])
    save_table(log, "clust_input_log.csv")

    X = StandardScaler().fit_transform(stable[FEATURES])

    # ---- k 선택 ----
    rows, km_labels = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
        lab = km.labels_
        km_labels[k] = lab
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sizes = np.bincount(lab)
        rows.append({
            "k": k,
            "silhouette": round(silhouette_score(X, lab), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X, lab), 2),
            "davies_bouldin": round(davies_bouldin_score(X, lab), 4),
            "inertia": round(km.inertia_, 2),
            "ARI_vs_ward": round(adjusted_rand_score(lab, ward), 4),
            "최소군집크기": int(sizes.min()),
            "군집크기": "-".join(map(str, sorted(sizes, reverse=True))),
        })
    ksel = pd.DataFrame(rows)
    # 선택 규칙: 최소 군집 크기 >= 2 인 k 중 실루엣 최대
    eligible = ksel[ksel["최소군집크기"] >= MIN_CLUSTER_SIZE]
    best_k = int(eligible.loc[eligible["silhouette"].idxmax(), "k"])
    ksel["선택"] = ksel["k"].eq(best_k)
    save_table(ksel, "clust_k_selection.csv")

    # ---- 최종 군집 (크기 순으로 번호 재부여) ----
    lab = km_labels[best_k]
    order = pd.Series(lab).value_counts().index.tolist()
    remap = {old: new + 1 for new, old in enumerate(order)}
    stable["cluster"] = pd.Series(lab).map(remap).values
    k_ids = sorted(stable["cluster"].unique())

    # 군집 특징: 표준화 평균에서 절댓값이 큰 상위 2개 변수
    Z = pd.DataFrame(X, columns=FEATURES)
    Z["cluster"] = stable["cluster"].values
    zmean = Z.groupby("cluster")[FEATURES].mean()

    def describe(cid):
        top = zmean.loc[cid].abs().sort_values(ascending=False).index[:2]
        return ", ".join(f"{FEAT_KO[f]} {'↑' if zmean.loc[cid, f] > 0 else '↓'}" for f in top)

    prof = stable.groupby("cluster").agg(
        zip수=("zip", "size"),
        민원건수합계=("n_complaints", "sum"),
        zip당평균건수=("n_complaints", "mean"),
        **{FEAT_KO[f]: (f, "mean") for f in FEATURES if f != "log_n_complaints"},
    )
    prof["zip당평균건수"] = prof["zip당평균건수"].round(0)
    for f in FEATURES:
        if f.endswith("share") or f.startswith("share_"):
            prof[FEAT_KO[f]] = (prof[FEAT_KO[f]] * 100).round(1)
    prof.insert(0, "주요특징(표준화 평균 기준)", [describe(c) for c in prof.index])
    prof.insert(1, "소속 zip", [", ".join(stable.loc[stable["cluster"] == c, "zip"]) for c in prof.index])
    save_table(prof, "clust_profiles.csv", index=True)
    save_table(zmean.round(3).rename(columns=FEAT_KO), "clust_profiles_zscore.csv", index=True)

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    P = pca.transform(X)
    stable["pc1"], stable["pc2"] = P[:, 0].round(4), P[:, 1].round(4)
    # 참고용: 대안 k의 소속 (k=3, 4) — 소수 군집이 어떤 zip인지 확인
    alt_cols = []
    for k_alt in (3, 4):
        if k_alt != best_k and k_alt in km_labels:
            a = km_labels[k_alt]
            a_order = pd.Series(a).value_counts().index.tolist()
            a_map = {old: new + 1 for new, old in enumerate(a_order)}
            stable[f"cluster_k{k_alt}"] = pd.Series(a).map(a_map).values
            alt_cols.append(f"cluster_k{k_alt}")
    out = stable[["zip", "cluster"] + alt_cols + ["n_complaints"] + FEATURES[:-1] + ["pc1", "pc2"]].copy()
    for f in FEATURES[:-1]:
        out[f] = (out[f] * 100).round(1)
    out = out.rename(columns={f: FEAT_KO[f] + "(%)" for f in FEATURES[:-1]})
    save_table(out.sort_values(["cluster", "n_complaints"], ascending=[True, False]), "clust_assignments.csv")
    load = pd.DataFrame(pca.components_.T, index=[FEAT_KO[f] for f in FEATURES], columns=["PC1", "PC2"]).round(3)
    load.loc["설명분산비율"] = pca.explained_variance_ratio_.round(3)
    save_table(load, "clust_pca_loadings.csv", index=True)

    # =================================================================
    # 그림 1: k 선택 (두 지표를 각자의 축에)
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "silhouette", "실루엣 계수 (높을수록 좋음)"),
                           (axes[1], "inertia", "군집 내 제곱합 (엘보우)")]:
        ax.plot(ksel["k"], ksel[col], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        row = ksel[ksel["k"] == best_k].iloc[0]
        ax.plot(best_k, row[col], marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.5)
        ax.annotate(f"선택 k={best_k}", (best_k, row[col]), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=INK2)
        ax.set_xticks(list(K_RANGE))
        ax.set_xlabel("군집 수 k")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11)
    save_fig(fig, "clust_01_k_selection.png")

    # =================================================================
    # 그림 2: PCA 산점도 (색 + 모양 이중 부호화, 군집 중심에 직접 라벨)
    # =================================================================
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, c in enumerate(k_ids):
        m = stable["cluster"] == c
        ax.scatter(stable.loc[m, "pc1"], stable.loc[m, "pc2"], s=70, marker=MARKERS[i],
                   color=CATEGORICAL[i], edgecolor=SURFACE, linewidth=1.5, label=f"군집 {c} (n={m.sum()})",
                   zorder=3)
    # 건수 상위 zip과 극단값 zip만 라벨
    lab_zip = set(stable.nlargest(6, "n_complaints")["zip"])
    d = np.hypot(stable["pc1"], stable["pc2"])
    lab_zip |= set(stable.loc[d.nlargest(4).index, "zip"])
    for _, r in stable[stable["zip"].isin(lab_zip)].iterrows():
        ax.annotate(r["zip"], (r["pc1"], r["pc2"]), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=INK2)
    ev = pca.explained_variance_ratio_ * 100
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.legend(ncol=min(len(k_ids), 4), loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3)
    ax.set_title(f"zip 군집 (K-means, k={best_k}) — 주성분 2차원 투영", pad=32)
    save_fig(fig, "clust_02_pca_scatter.png")

    # =================================================================
    # 그림 3: 군집 프로파일 히트맵 (표준화 평균, 발산형)
    # =================================================================
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    zm = zmean.loc[k_ids, FEATURES]
    lim = max(2.0, float(np.ceil(np.abs(zm.values).max())))
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(k_ids) + 2.2))
    im = ax.imshow(zm.values, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(FEATURES)), [FEAT_KO[f] for f in FEATURES], rotation=35, ha="right")
    ax.set_yticks(range(len(k_ids)), [f"군집 {c} (n={int((stable['cluster'] == c).sum())})" for c in k_ids])
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
    cb.set_label("표준화 평균 (0 = 전체 평균)", color=INK2, fontsize=9)
    ax.set_title("군집별 특징 — 파랑은 평균보다 낮음, 빨강은 높음")
    save_fig(fig, "clust_03_profile_heatmap.png")

    # =================================================================
    # 그림 4: 군집별 유형 구성비 (민원 건수 가중, 100% 누적 막대)
    # =================================================================
    share_cols = [f for f in FEATURES if f.startswith("share_")]
    w = stable[share_cols].mul(stable["n_complaints"], axis=0)
    w["cluster"] = stable["cluster"]
    comp = w.groupby("cluster")[share_cols].sum()
    comp = comp.div(comp.sum(axis=1), axis=0) * 100
    main5 = ["share_street", "share_residential", "share_commercial", "share_generic", "share_helicopter"]
    comp["기타"] = comp.drop(columns=main5).sum(axis=1)
    comp = comp[main5 + ["기타"]]
    names = [FEAT_KO[f].replace(" 비율", "") for f in main5] + ["기타(차량·공원)"]
    colors = CATEGORICAL[:5] + ["#b5b3ab"]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(k_ids) + 2))
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
    ax.set_yticks(y, [f"군집 {c}" for c in comp.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("유형 구성비 (%, 민원 건수 가중)")
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, handlelength=1, columnspacing=1)
    ax.set_title("군집별 민원 유형 구성", pad=30)
    save_fig(fig, "clust_04_cluster_composition.png")

    print(log.to_string(index=False))
    print()
    print(ksel.to_string(index=False))
    print(f"\n선택 k = {best_k}")
    print(prof.iloc[:, :4].to_string())
    print(f"\nPCA 설명분산: {ev.round(1)}")


if __name__ == "__main__":
    main()
