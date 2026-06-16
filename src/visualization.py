"""
visualization.py
----------------
Publication-quality figures for the PAM50 classification pipeline:

  • Confusion matrices (XGBoost + RF)
  • Multiclass ROC curves
  • PCA coloured by subtype
  • Gene expression heatmap (PAM50 genes × samples)
  • Model comparison bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

SUBTYPE_ORDER = ["LumA", "LumB", "HER2", "Basal", "Normal"]
SUBTYPE_COLORS = {
    "LumA":   "#2196F3",
    "LumB":   "#03A9F4",
    "HER2":   "#FF9800",
    "Basal":  "#F44336",
    "Normal": "#4CAF50",
}

STYLE = {
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
}


def plot_confusion_matrix(cm: np.ndarray, le, model_name: str, out_path: str):
    plt.rcParams.update(STYLE)
    class_names = le.classes_
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title_sfx in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["(Counts)", "(Row-normalised)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar=True,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{model_name} — Confusion Matrix {title_sfx}",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


def plot_roc_curves(y_test: np.ndarray, y_prob: np.ndarray,
                    le, model_name: str, out_path: str):
    """One-vs-rest multiclass ROC curves."""
    plt.rcParams.update(STYLE)
    class_names = le.classes_
    n_classes   = len(class_names)
    y_bin       = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, subtype in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=SUBTYPE_COLORS[subtype], lw=2,
                label=f"{subtype} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved → {out_path}")


def plot_pca(X_train: np.ndarray, y_train: np.ndarray,
             le, out_path: str):
    """PCA scatter coloured by PAM50 subtype."""
    plt.rcParams.update(STYLE)
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_train)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, subtype in enumerate(le.classes_):
        mask = y_train == i
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=SUBTYPE_COLORS[subtype], label=subtype,
                   alpha=0.60, s=22, edgecolors="none")

    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", fontsize=11)
    ax.set_title("PCA of PAM50 Gene Expression (Training Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(markerscale=1.5, fontsize=9, framealpha=0.85)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA plot saved → {out_path}")
    return pca


def plot_expression_heatmap(df: pd.DataFrame,
                             feature_names: list,
                             out_path: str,
                             n_per_subtype: int = 30,
                             top_genes: int = 30):
    """
    Heatmap of top variable PAM50 genes across balanced subtype samples.
    Rows = genes, Columns = samples (sorted by subtype).
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    gene_cols = [g for g in feature_names if g in df.columns]

    # Sample n_per_subtype from each subtype
    frames = []
    for sub in SUBTYPE_ORDER:
        sub_df = df[df["PAM50_subtype"] == sub][gene_cols + ["PAM50_subtype"]]
        frames.append(sub_df.sample(min(n_per_subtype, len(sub_df)),
                                     random_state=42))
    balanced = pd.concat(frames)

    # Select top variable genes
    variances   = balanced[gene_cols].var()
    top_g       = variances.nlargest(top_genes).index.tolist()
    mat         = balanced[top_g].T      # (genes × samples)

    # Build column colour bar
    col_colors = balanced["PAM50_subtype"].map(SUBTYPE_COLORS)

    g = sns.clustermap(
        mat,
        cmap="RdYlBu_r",
        col_colors=col_colors.values,
        col_cluster=False,       # keep subtype order
        row_cluster=True,
        yticklabels=True,
        xticklabels=False,
        figsize=(14, 9),
        cbar_kws={"label": "log2 Expression"},
        dendrogram_ratio=(0.12, 0.04),
    )
    g.ax_heatmap.set_xlabel("Samples (sorted by subtype)", fontsize=10)
    g.ax_heatmap.set_ylabel("Genes", fontsize=10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.figure.suptitle(f"Top {top_genes} Variable PAM50 Genes — Expression Heatmap",
                       fontsize=13, fontweight="bold", y=1.01)

    # Legend for subtype colours
    patches = [mpatches.Patch(color=SUBTYPE_COLORS[s], label=s) for s in SUBTYPE_ORDER]
    g.ax_heatmap.legend(handles=patches, loc="upper right",
                         bbox_to_anchor=(1.18, 1.05), fontsize=8,
                         title="Subtype", title_fontsize=8, framealpha=0.8)

    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Heatmap saved → {out_path}")


def plot_model_comparison(xgb_met: dict, rf_met: dict, out_path: str):
    """Grouped bar chart comparing XGBoost vs RF across 3 metrics."""
    plt.rcParams.update(STYLE)
    metrics = ["accuracy", "balanced_accuracy", "roc_auc_macro"]
    labels  = ["Accuracy", "Balanced\nAccuracy", "Macro\nROC-AUC"]
    xgb_vals = [xgb_met[m] for m in metrics]
    rf_vals  = [rf_met[m]  for m in metrics]

    x   = np.arange(len(metrics))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, xgb_vals, w, label="XGBoost",
                color="#1565C0", edgecolor="white")
    b2 = ax.bar(x + w/2, rf_vals,  w, label="Random Forest",
                color="#2E7D32", edgecolor="white")

    for bar, val in zip(list(b1) + list(b2), xgb_vals + rf_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — XGBoost vs Random Forest",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Model comparison saved → {out_path}")


def run_all_visualizations(df, X_train, X_test, y_train, y_test,
                            le, feature_names,
                            xgb_model, rf_model,
                            xgb_met, rf_met,
                            figures_dir, tables_dir):
    """Master call for all visualizations."""
    print("\n── Generating Figures ────────────────────────────────────")

    # 1. PCA
    plot_pca(X_train, y_train, le, f"{figures_dir}/pca_subtypes.png")

    # 2. Heatmap
    plot_expression_heatmap(df, feature_names,
                             f"{figures_dir}/expression_heatmap.png")

    # 3. Confusion matrices
    for met, model_name in [(xgb_met, "XGBoost"), (rf_met, "RandomForest")]:
        plot_confusion_matrix(
            met["confusion_matrix"], le, model_name,
            f"{figures_dir}/confusion_matrix_{model_name}.png",
        )

    # 4. ROC curves
    xgb_prob = xgb_model.predict_proba(X_test)
    rf_prob  = rf_model.predict_proba(X_test)
    plot_roc_curves(y_test, xgb_prob, le, "XGBoost",
                    f"{figures_dir}/roc_curves_XGBoost.png")
    plot_roc_curves(y_test, rf_prob,  le, "Random Forest",
                    f"{figures_dir}/roc_curves_RandomForest.png")

    # 5. Model comparison
    plot_model_comparison(xgb_met, rf_met,
                           f"{figures_dir}/model_comparison.png")
