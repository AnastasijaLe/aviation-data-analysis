"""Nepārraudzītā ML segmentācija — ES aviācijas dati.

Piemēri:
    python segment.py --analysis co2_intensity
    python segment.py --analysis passenger_dynamics --clusters 4
    python segment.py --analysis combined --clusters 3
    python segment.py --analysis all
    python segment.py --analysis all --no-plots
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

import ml_core

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).parent
DB_PATH  = ml_core.get_db_path(BASE_DIR)
OUT_DIR  = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#7b3294", "#f4a582",
    "#1a9850", "#e08214",
]

BAR = "═" * 72

FEATURE_LABELS = {
    "co2_per_pax":    "CO2/pasažieris",
    "co2_per_capita": "CO2/iedzīvotājs",
    "co2_cagr_13_24": "CO2 CAGR 2013-24",
    "co2_recovery":   "CO2 atveseļošanās",
    "co2_total_log":  "log(CO2 kopā)",
    "pax_per_cap_19": "Pax/iedzīvotājs 2019",
    "pax_per_cap_24": "Pax/iedzīvotājs 2024",
    "pax_cagr_13_19": "Pax CAGR 2013-19",
    "pax_recovery":   "Pax atveseļošanās",
    "pax_cagr_19_24": "Pax CAGR 2019-24",
    "seasonality_cv": "Sezonalitāte (CV)",
    "peak_share":     "Virsotnes mēnesis",
    "gdp_per_capita": "IKP/iedzīvotājs",
}

CLUSTER_DESCRIPTIONS = {
    "co2_intensity": {
        "high":   "Augsta intensitāte  — prioritāra intervence",
        "medium": "Vidēja intensitāte  — uzraudzīt dinamiku",
        "low":    "Zema intensitāte    — labas prakse",
    },
    "passenger_dynamics": {
        "high":   "Augsta dinamika     — nobriedis tirgus",
        "medium": "Vidēja dinamika     — izaugsmes potenciāls",
        "low":    "Zema dinamika       — mazattīstīts tirgus",
    },
    "combined": {
        "high":   "Augsts CO2 + zema pieejamība  — pārskatīt reisus",
        "medium": "Balanss             — uzturēt pašreizējo līmeni",
        "low":    "Efektīvs + pieejams — labās prakses modelis",
    },
}


def interpret_clusters(
    df:       pd.DataFrame,
    features: list[str],
    analysis: str,
) -> dict[int, str]:
    """Piešķir katram klasterim cilvēklasāmu apzīmējumu pēc kompozīta ranga."""
    ranking = ml_core.composite_cluster_ranking(df, features, "km_cluster")

    labels_map = {}
    levels     = ["high", "medium", "low", "low", "low"]
    for rank, (cid, _) in enumerate(ranking.items()):
        key  = levels[min(rank, len(levels) - 1)]
        desc = CLUSTER_DESCRIPTIONS.get(analysis, {}).get(key, f"Klasteris {rank + 1}")
        labels_map[int(cid)] = desc
    return labels_map


def print_silhouette_table(scores: dict, optimal_k: int) -> None:
    print("\n  Siluetes koeficients pa k vērtībām:")
    print(f"  {'k':>3}  {'Siluets':>8}")
    for k, s in scores.items():
        bar  = "█" * int(s * 40)
        star = " ← optimāls" if k == optimal_k else ""
        print(f"  {k:>3}  {s:>8.4f}  {bar}{star}")


def print_cluster_assignments(
    df:        pd.DataFrame,
    k:         int,
    label_map: dict,
    algo:      str,
) -> None:
    col = f"{algo}_cluster"
    print(f"\n  Klastera piešķīrumi — {algo.upper()} (k={k}):")
    print(f"  {'Klasteris':<4}  {'Apraksts':<46}  Valstis")
    print(f"  {'-'*4}  {'-'*46}  {'-'*40}")

    for cid in sorted(df[col].unique()):
        desc    = label_map.get(int(cid), f"Klasteris {cid}")
        members = sorted(df[df[col] == cid]["country"].tolist())
        first   = True
        for m in members:
            if first:
                print(f"  {cid:<4}  {desc:<46}  {m}")
                first = False
            else:
                print(f"  {'':4}  {'':46}  {m}")


def print_cluster_stats(
    df:       pd.DataFrame,
    features: list[str],
    col:      str = "km_cluster",
) -> None:
    print(f"\n  Klastera centroīdi (vidējās vērtības, pirms normalizēšanas):")
    stats = df.groupby(col)[features].mean()
    stats = stats.rename(columns=FEATURE_LABELS)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(stats.T.to_string())
    pd.reset_option("display.float_format")


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_dendrogram(Z, labels, k, analysis, title):
    plt = _import_matplotlib()
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(
        f"Hierarhiskā klasterizācija — {title}\nWard saite  |  k = {k}",
        fontsize=13, fontweight="bold",
    )
    dendrogram(
        Z, labels=labels,
        leaf_rotation=45, leaf_font_size=9,
        color_threshold=Z[-(k - 1), 2],
        above_threshold_color="lightgray",
        ax=ax,
    )
    ax.set_ylabel("Ward attālums", fontsize=10)
    ax.axhline(
        y=Z[-(k - 1), 2], color="red",
        linestyle="--", linewidth=1.2, label=f"k={k} griezums",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = OUT_DIR / f"dendrogram_{analysis}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_silhouette(X_scaled, labels, analysis, algo):
    plt = _import_matplotlib()

    k        = len(np.unique(labels))
    sil      = silhouette_score(X_scaled, labels)
    sil_vals = silhouette_samples(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for cid in range(k):
        c_vals  = np.sort(sil_vals[labels == cid])
        size    = c_vals.shape[0]
        y_upper = y_lower + size
        color   = PALETTE[cid % len(PALETTE)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, c_vals,
            facecolor=color, edgecolor=color, alpha=0.75,
        )
        ax.text(
            -0.05, y_lower + 0.5 * size, str(cid), fontsize=9,
            color=color, fontweight="bold",
        )
        y_lower = y_upper + 10

    ax.axvline(
        x=sil, color="red", linestyle="--",
        label=f"Vid. siluets = {sil:.4f}",
    )
    ax.set_title(
        f"Siluetes diagramma — {analysis.replace('_', ' ').title()}\n"
        f"{algo.upper()}  |  k={k}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Siluetes koeficients")
    ax.set_ylabel("Klasteris")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = OUT_DIR / f"silhouette_{analysis}_{algo}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scatter_2d(df, feat_x, feat_y, label_col, analysis, algo, label_map):
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 7))

    for cid in sorted(df[label_col].unique()):
        subset     = df[df[label_col] == cid]
        color      = PALETTE[int(cid) % len(PALETTE)]
        desc       = label_map.get(int(cid), f"K{cid}")
        short_desc = desc.split("—")[-1].strip() if "—" in desc else desc
        ax.scatter(
            subset[feat_x], subset[feat_y],
            color=color, s=120, zorder=3,
            label=f"K{cid}: {short_desc}",
            edgecolors="white", linewidths=0.8,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["country_code"] if row["country_code"] else row["country"][:3],
                (row[feat_x], row[feat_y]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
            )

    ax.set_xlabel(FEATURE_LABELS.get(feat_x, feat_x), fontsize=11)
    ax.set_ylabel(FEATURE_LABELS.get(feat_y, feat_y), fontsize=11)
    ax.set_title(
        f"ES dalībvalstu segmentācija — {analysis.replace('_', ' ').title()}\n"
        f"{algo.upper()}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = OUT_DIR / f"scatter_{analysis}_{algo}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_radar(df, features, label_col, analysis):
    plt = _import_matplotlib()

    centroids   = df.groupby(label_col)[features].mean()
    centroids_n = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-9)

    angles  = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in features], fontsize=8)

    for cid, row in centroids_n.iterrows():
        vals  = row.tolist() + row.tolist()[:1]
        color = PALETTE[int(cid) % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2, label=f"K{int(cid)}")
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_title(
        f"Klasteru profils — {analysis.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()

    out = OUT_DIR / f"radar_{analysis}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_analysis(
    base_df:    pd.DataFrame,
    analysis:   str,
    k:          int | None,
    make_plots: bool,
) -> pd.DataFrame:
    if base_df is None or len(base_df) == 0:
        conn    = sqlite3.connect(DB_PATH)
        base_df = ml_core.build_seg_features(conn)
        conn.close()

    features = ml_core.FEATURE_SETS[analysis]

    if k is None:
        df_temp = base_df.dropna(subset=features).copy()
        X_scan  = df_temp[features].copy()
        for f in features:
            if f in ml_core.SKEWED_FEATURES:
                X_scan[f] = np.log1p(X_scan[f].clip(lower=0))
        X_scaled_scan = RobustScaler().fit_transform(X_scan.values)
        sil_scores    = ml_core.find_optimal_k(X_scaled_scan)
        k_used        = ml_core.pick_optimal_k(X_scaled_scan, min_cluster_size=3)
        print_silhouette_table(sil_scores, k_used)
    else:
        k_used = k

    result = ml_core.cluster_data(base_df, analysis, k_used)

    df        = result["df"]
    X_scaled  = result["X_scaled"]
    Z         = result["Z"]
    features  = result["features"]
    km_sil    = result["km_sil"]
    hi_sil    = result["hi_sil"]
    agreement = result["agreement"]

    print(f"\n{BAR}")
    print(f"  SEGMENTĀCIJA: {analysis.replace('_', ' ').upper()}")
    print(BAR)
    print(f"  Pazīmes ({len(features)}): "
          f"{', '.join(FEATURE_LABELS.get(f, f) for f in features)}")
    print(f"  Valstu skaits: {len(df)}")
    print(f"  Izmantotais k = {k_used}")

    print(f"\n  Siluetes koeficients:")
    print(f"    K-Means        : {km_sil:.4f}")
    print(f"    Hierarhiskais  : {hi_sil:.4f}")

    print(f"\n  Metožu vienprātība:")
    print(f"    ARI            : {agreement['ari']:.4f}   "
          f"(1.0 = identiski, 0.0 = nejauši)")
    print(f"    NMI            : {agreement['nmi']:.4f}   "
          f"(0..1, informācijas-teorētisks)")
    print(f"    Pa pāriem      : {agreement['pairwise_pct']:.1f}%")

    km_label_map = interpret_clusters(df, features, analysis)
    hi_label_map = {
        cid: km_label_map.get(
            df[df["hi_cluster"] == cid]["km_cluster"].mode().iloc[0]
            if len(df[df["hi_cluster"] == cid]["km_cluster"].mode()) > 0
            else int(cid),
            f"K{cid}",
        )
        for cid in df["hi_cluster"].unique()
    }

    print_cluster_assignments(df, k_used, km_label_map, "km")
    print_cluster_assignments(df, k_used, hi_label_map, "hi")
    print_cluster_stats(df, features, col="km_cluster")

    if make_plots:
        plots_saved = []
        plots_saved.append(plot_dendrogram(
            Z, df["country"].tolist(), k_used, analysis,
            analysis.replace("_", " ").title(),
        ))
        plots_saved.append(plot_silhouette(
            X_scaled, df["km_cluster"].values, analysis, "kmeans",
        ))
        plots_saved.append(plot_silhouette(
            X_scaled, df["hi_cluster"].values, analysis, "hierarchical",
        ))

        feat_x = features[0]
        feat_y = features[1] if len(features) > 1 else features[0]

        plots_saved.append(plot_scatter_2d(
            df, feat_x, feat_y, "km_cluster", analysis, "kmeans", km_label_map,
        ))
        plots_saved.append(plot_scatter_2d(
            df, feat_x, feat_y, "hi_cluster", analysis, "hierarchical", hi_label_map,
        ))
        plots_saved.append(plot_radar(df, features, "km_cluster", analysis))
        print(f"\n  Saglabāti {len(plots_saved)} grafiki → results/")

    out_cols = ["country", "country_code"] + features + ["km_cluster", "hi_cluster"]
    csv_path = OUT_DIR / f"segment_{analysis}.csv"
    df[out_cols].to_csv(csv_path, index=False)
    print(f"  Klasteru CSV: results/segment_{analysis}.csv")
    print(BAR)

    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Nepārraudzītā ML segmentācija — ES aviācijas dati",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--analysis",
        choices=["co2_intensity", "passenger_dynamics", "combined", "all"],
        required=True,
        help="Segmentācijas veids",
    )
    p.add_argument(
        "--clusters", "-k",
        type=int, default=None,
        help="Klasteru skaits k (noklusējums: auto pēc siluetes koeficienta)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Neģenerēt grafikus (tikai terminālā un CSV; ātrāka palaišana)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        sys.exit(f"[KĻŪDA] Datubāze nav atrasta: {DB_PATH}")

    if args.clusters is not None and (args.clusters < 2 or args.clusters > 10):
        sys.exit("[KĻŪDA] --clusters jābūt diapazonā 2..10")

    print(f"\n{BAR}")
    print(f"  ES Aviācijas Segmentācija — Nepārraudzītā ML")
    print(BAR)

    conn    = sqlite3.connect(DB_PATH)
    base_df = ml_core.build_seg_features(conn)
    conn.close()

    print(f"  Dati ielādēti: {len(base_df)} valstis, "
          f"{sum(len(v) for v in ml_core.FEATURE_SETS.values())} pazīmes kopā")

    analyses = (
        ["co2_intensity", "passenger_dynamics", "combined"]
        if args.analysis == "all"
        else [args.analysis]
    )

    results = {}
    for analysis in analyses:
        print(f"\n{'─'*72}")
        print(f"  Analīze: {analysis.upper()}")
        print(f"{'─'*72}")
        df_out = run_analysis(
            base_df, analysis,
            k=args.clusters,
            make_plots=not args.no_plots,
        )
        results[analysis] = df_out

    if len(results) > 1:
        print(f"\n{BAR}")
        print(f"  KOPSAVILKUMS — visu analīžu klasteru piešķīrumi")
        print(BAR)
        summary = base_df[["country", "country_code"]].copy()
        for analysis, df_out in results.items():
            col    = f"km_{analysis[:6]}"
            merged = df_out[["country", "km_cluster"]].rename(
                columns={"km_cluster": col},
            )
            summary = summary.merge(merged, on="country", how="left")

        print(summary.to_string(index=False))

        summary_path = OUT_DIR / "segment_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Kopsavilkums saglabāts: results/segment_summary.csv")
        print(BAR)

    print(f"\n  Pabeigts.\n")


if __name__ == "__main__":
    main()
