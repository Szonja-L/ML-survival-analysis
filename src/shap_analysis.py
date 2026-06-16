"""
shap_analysis.py
----------------
SHAP (SHapley Additive exPlanations) interpretability for both XGBoost
and Random Forest PAM50 classifiers.

Produces:
  • Summary beeswarm plots per class (top-20 features)
  • Mean absolute SHAP feature importance bar chart
  • SHAP-ranked gene tables
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

SUBTYPE_ORDER  = ["LumA", "LumB", "HER2", "Basal", "Normal"]
SUBTYPE_COLORS = {
    "LumA":   "#2196F3",
    "LumB":   "#03A9F4",
    "HER2":   "#FF9800",
    "Basal":  "#F44336",
    "Normal": "#4CAF50",
}

STYLE = {
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
}


def compute_shap_xgb(xgb_model, X_test: np.ndarray,
                     feature_names: list) -> shap.Explanation:
    """Compute TreeExplainer SHAP values for XGBoost (fast, exact)."""
    print("  Computing XGBoost SHAP values …")
    explainer = shap.TreeExplainer(xgb_model)
    sv        = explainer.shap_values(X_test)     # list of arrays, one per class
    # Build shap.Explanation for unified API
    sv_arr = np.stack(sv, axis=-1)                # (n_samples, n_features, n_classes)
    return sv_arr, explainer


def compute_shap_rf(rf_model, X_test: np.ndarray,
                    feature_names: list, n_samples: int = 200):
    """Compute TreeExplainer SHAP values for Random Forest."""
    print("  Computing Random Forest SHAP values …")
    explainer = shap.TreeExplainer(rf_model)
    # Use a subset to keep it tractable for RF
    idx = np.random.default_rng(42).choice(len(X_test),
                                            min(n_samples, len(X_test)),
                                            replace=False)
    sv  = explainer.shap_values(X_test[idx])      # list, one per class
    sv_arr = np.stack(sv, axis=-1)
    return sv_arr, idx, explainer


def plot_shap_summary(sv_arr: np.ndarray, X_test: np.ndarray,
                      feature_names: list, le,
                      model_name: str, out_dir: str, top_n: int = 20):
    """
    One beeswarm summary plot per subtype class.
    sv_arr shape: (n_samples, n_features, n_classes)
    """
    plt.rcParams.update(STYLE)
    class_names = le.classes_

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    fig.suptitle(f"{model_name} — SHAP Summary (top {top_n} features per subtype)",
                 fontsize=14, fontweight="bold", y=1.02)

    for cls_idx, (subtype, ax) in enumerate(zip(class_names, axes)):
        sv_cls  = sv_arr[:, :, cls_idx]          # (n, n_features)
        mean_abs = np.abs(sv_cls).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-top_n:][::-1]

        # Beeswarm-style scatter (approximate)
        n_samples = sv_cls.shape[0]
        for rank, fi in enumerate(top_idx[::-1]):
            vals      = sv_cls[:, fi]
            feat_vals = X_test[:n_samples, fi]
            # Normalise feature values for colour
            fv_range  = feat_vals.max() - feat_vals.min()
            fv_norm   = (feat_vals - feat_vals.min()) / (fv_range + 1e-8)
            y_jitter  = rank + np.random.default_rng(fi).uniform(-0.25, 0.25, n_samples)
            sc = ax.scatter(vals, y_jitter,
                            c=plt.cm.coolwarm(fv_norm.astype(float)),
                            alpha=0.4, s=6, linewidths=0)

        color = SUBTYPE_COLORS.get(subtype, "steelblue")
        ax.set_title(subtype, fontsize=12, color=color, fontweight="bold")
        actual_n = len(top_idx)
        ax.set_yticks(range(actual_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]][:actual_n], fontsize=7)
        ax.axvline(0, lw=0.8, color="grey", ls="--")
        ax.set_xlabel("SHAP value", fontsize=9)

    plt.tight_layout()
    path = f"{out_dir}/{model_name.replace(' ', '_')}_shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved SHAP summary → {path}")
    return path


def plot_shap_importance_bar(sv_arr: np.ndarray, feature_names: list,
                              le, model_name: str, out_dir: str,
                              top_n: int = 20):
    """
    Global mean |SHAP| bar chart (averaged across classes).
    """
    plt.rcParams.update(STYLE)
    # Mean across classes then samples
    mean_abs_global = np.abs(sv_arr).mean(axis=(0, 2))  # (n_features,)
    top_idx   = np.argsort(mean_abs_global)[-top_n:]
    top_vals  = mean_abs_global[top_idx]
    top_names = [feature_names[i] for i in top_idx]
    actual_n  = len(top_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(actual_n), top_vals,
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, actual_n)),
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(actual_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (global importance)", fontsize=10)
    ax.set_title(f"{model_name} — Global SHAP Feature Importance",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/{model_name.replace(' ', '_')}_shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved SHAP importance → {path}")
    return path


def shap_gene_tables(sv_arr: np.ndarray, feature_names: list,
                     le, out_dir: str) -> pd.DataFrame:
    """Return DataFrame of top SHAP genes per subtype."""
    rows = []
    for cls_idx, subtype in enumerate(le.classes_):
        sv_cls   = sv_arr[:, :, cls_idx]
        mean_abs = np.abs(sv_cls).mean(axis=0)
        mean_dir = sv_cls.mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]
        for rank, fi in enumerate(sorted_idx[:20]):
            rows.append({
                "Subtype": subtype,
                "Rank":    rank + 1,
                "Gene":    feature_names[fi],
                "Mean_SHAP": round(mean_dir[fi], 4),
                "Mean_abs_SHAP": round(mean_abs[fi], 4),
                "Direction": "Up" if mean_dir[fi] > 0 else "Down",
            })
    df = pd.DataFrame(rows)
    path = f"{out_dir}/shap_gene_rankings.csv"
    df.to_csv(path, index=False)
    print(f"  Saved SHAP gene table → {path}")
    return df


def run_shap_analysis(xgb_model, rf_model,
                      X_train, X_test,
                      feature_names, le,
                      figures_dir: str, tables_dir: str):
    """
    Full SHAP pipeline for both models.
    Returns shap value arrays for downstream use.
    """
    print("\n── SHAP Interpretability ─────────────────────────────────")

    # XGBoost SHAP
    xgb_sv, _ = compute_shap_xgb(xgb_model, X_test, feature_names)
    plot_shap_summary(xgb_sv, X_test, feature_names, le,
                      "XGBoost", figures_dir)
    plot_shap_importance_bar(xgb_sv, feature_names, le,
                              "XGBoost", figures_dir)
    xgb_table = shap_gene_tables(xgb_sv, feature_names, le, tables_dir)

    # Random Forest SHAP (subsample)
    rf_sv, rf_idx, _ = compute_shap_rf(rf_model, X_test, feature_names)
    plot_shap_summary(rf_sv, X_test[rf_idx], feature_names, le,
                      "Random_Forest", figures_dir)
    plot_shap_importance_bar(rf_sv, feature_names, le,
                              "Random_Forest", figures_dir)

    print("\n  Top XGBoost SHAP genes per subtype:")
    for sub in le.classes_:
        top5 = xgb_table[xgb_table.Subtype == sub].head(5)["Gene"].tolist()
        print(f"    {sub:8s}: {', '.join(top5)}")

    return xgb_sv, rf_sv
