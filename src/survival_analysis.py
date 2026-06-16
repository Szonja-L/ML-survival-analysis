"""
survival_analysis.py
--------------------
Survival analysis pipeline for PAM50-stratified outcomes:

  1. Kaplan-Meier survival curves per PAM50 subtype
  2. Log-rank pairwise tests
  3. Cox Proportional Hazards model with clinical + expression covariates
  4. Cox hazard ratio forest plot
  5. Predicted vs. true subtype KM comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

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
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
}


# ── 1. Kaplan-Meier ──────────────────────────────────────────────────────────
def plot_km_curves(df: pd.DataFrame,
                   subtype_col: str,
                   title: str,
                   out_path: str,
                   ci: bool = True):
    """
    Draw KM curves for each PAM50 subtype with 95% CI ribbon.
    """
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    median_os = {}
    for subtype in SUBTYPE_ORDER:
        mask = df[subtype_col] == subtype
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "OS_months"],
                df.loc[mask, "OS_event"],
                label=subtype)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=ci,
            color=SUBTYPE_COLORS[subtype],
            linewidth=2.2,
            show_censors=True,
            censor_styles={"ms": 4, "marker": "|"},
        )
        median_os[subtype] = kmf.median_survival_time_

    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Overall Survival Probability", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)

    # Annotate medians in legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        f"{l}  (median={median_os.get(l, 'NR'):.0f} mo)"
        if isinstance(median_os.get(l), float) else l
        for l in labels
    ]
    ax.legend(handles, new_labels, frameon=True,
              fontsize=9, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  KM curves saved → {out_path}")
    return median_os


def run_logrank(df: pd.DataFrame, subtype_col: str) -> pd.DataFrame:
    """Pairwise log-rank tests between all subtype pairs."""
    rows = []
    subtypes = [s for s in SUBTYPE_ORDER if s in df[subtype_col].values]
    for i, s1 in enumerate(subtypes):
        for s2 in subtypes[i+1:]:
            g1 = df[df[subtype_col] == s1]
            g2 = df[df[subtype_col] == s2]
            result = logrank_test(
                g1["OS_months"], g2["OS_months"],
                g1["OS_event"],  g2["OS_event"],
            )
            rows.append({
                "Subtype_A": s1, "Subtype_B": s2,
                "Test_stat": round(result.test_statistic, 3),
                "p_value":   round(result.p_value, 4),
                "Significant": "***" if result.p_value < 0.001
                               else "**"  if result.p_value < 0.01
                               else "*"   if result.p_value < 0.05
                               else "ns",
            })
    df_lr = pd.DataFrame(rows)
    print("\n  Pairwise log-rank results:")
    print(df_lr.to_string(index=False))
    return df_lr


# ── 2. Cox PH Model ───────────────────────────────────────────────────────────
def build_cox_dataframe(clin_df: pd.DataFrame,
                        X_test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble covariates for Cox model:
      - Clinical: Age, Stage_num, Node_positive, Tumour_size
      - Subtype dummies (LumB/HER2/Basal/Normal vs LumA)
      - Key expression scores: proliferation index, ER-pathway score
    """
    df = clin_df[["OS_months", "OS_event",
                  "Age", "Stage_num", "Node_positive", "Tumour_size",
                  "PAM50_true"]].copy()

    # Subtype dummies (LumA = reference)
    dummies = pd.get_dummies(df["PAM50_true"], prefix="Sub", drop_first=False)
    for sub in ["Sub_LumB", "Sub_HER2", "Sub_Basal", "Sub_Normal"]:
        if sub not in dummies.columns:
            dummies[sub] = 0
    df = pd.concat([df.drop(columns=["PAM50_true"]), dummies], axis=1)

    # Proliferation index: mean of MKI67, CCNB1, BIRC5, MELK (z-scored)
    prolif_genes = [g for g in ["MKI67","CCNB1","BIRC5","MELK","UBE2C","CDC20"]
                    if g in X_test_df.columns]
    if prolif_genes:
        scores = X_test_df[prolif_genes].mean(axis=1)
        df["Proliferation_score"] = ((scores - scores.mean()) / scores.std()).values

    # ER pathway score: mean of ESR1, PGR, FOXA1, MLPH
    er_genes = [g for g in ["ESR1","PGR","FOXA1","MLPH","NAT1","GATA3"]
                if g in X_test_df.columns]
    if er_genes:
        scores = X_test_df[er_genes].mean(axis=1)
        df["ER_pathway_score"] = ((scores - scores.mean()) / scores.std()).values

    # Standardise continuous clinical vars
    for col in ["Age", "Stage_num", "Tumour_size"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Drop reference subtype column
    df = df.drop(columns=["Sub_LumA"], errors="ignore")
    return df


def run_cox(cox_df: pd.DataFrame, out_path_csv: str) -> CoxPHFitter:
    """Fit CoxPH model and save coefficient table."""
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col="OS_months", event_col="OS_event")
    print("\n── Cox Proportional Hazards Model ────────────────────────")
    cph.print_summary(decimals=4)

    summary = cph.summary.copy()
    summary.to_csv(out_path_csv)
    print(f"  Cox summary saved → {out_path_csv}")
    return cph


def plot_cox_forest(cph: CoxPHFitter, out_path: str):
    """
    Horizontal forest plot of Cox hazard ratios with 95% CI.
    """
    plt.rcParams.update(STYLE)
    summary = cph.summary.copy()
    summary = summary.sort_values("coef")

    # Human-readable covariate labels
    label_map = {
        "Age":                  "Age (SD)",
        "Stage_num":            "Tumour Stage (SD)",
        "Node_positive":        "Node Positive",
        "Tumour_size":          "Tumour Size (SD)",
        "Proliferation_score":  "Proliferation Score",
        "ER_pathway_score":     "ER Pathway Score",
        "Sub_LumB":             "LumB vs LumA",
        "Sub_HER2":             "HER2 vs LumA",
        "Sub_Basal":            "Basal vs LumA",
        "Sub_Normal":           "Normal vs LumA",
    }
    summary.index = [label_map.get(i, i) for i in summary.index]

    hr   = np.exp(summary["coef"])
    ci_lo = np.exp(summary["coef lower 95%"])
    ci_hi = np.exp(summary["coef upper 95%"])
    pvals = summary["p"]

    colors = ["#D32F2F" if p < 0.05 else "#9E9E9E" for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(summary))
    ax.barh(y_pos, hr, xerr=np.vstack([hr - ci_lo, ci_hi - hr]),
            height=0.45, color=colors, ecolor="black", capsize=4,
            error_kw={"linewidth": 1.2})

    ax.axvline(1.0, lw=1.2, color="black", ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index, fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
    ax.set_title("Cox PH Model — Hazard Ratios\n(red = p < 0.05)",
                 fontsize=12, fontweight="bold")

    # p-value annotations
    for y, (p, h) in enumerate(zip(pvals, hr)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            ax.text(ci_hi.iloc[y] + 0.02, y, sig, va="center", fontsize=10,
                    color="#D32F2F")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cox forest plot saved → {out_path}")


# ── 3. Predicted vs true KM comparison ───────────────────────────────────────
def plot_km_predicted_vs_true(clin_df: pd.DataFrame,
                               y_pred_labels: np.ndarray,
                               out_path: str):
    """
    Side-by-side KM: true PAM50 labels vs XGBoost-predicted labels.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, col, title in zip(
        axes,
        ["PAM50_true", "PAM50_predicted"],
        ["True PAM50 Labels", "XGBoost Predicted Labels"],
    ):
        for subtype in SUBTYPE_ORDER:
            mask = clin_df[col] == subtype
            if mask.sum() < 3:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(clin_df.loc[mask, "OS_months"],
                    clin_df.loc[mask, "OS_event"],
                    label=subtype)
            kmf.plot_survival_function(
                ax=ax, ci_show=False,
                color=SUBTYPE_COLORS[subtype], linewidth=2.2,
                show_censors=True,
                censor_styles={"ms": 4, "marker": "|"},
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (months)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0)

    axes[0].set_ylabel("Overall Survival Probability", fontsize=11)
    fig.suptitle("Kaplan-Meier: True vs. Predicted PAM50 Subtypes",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Predicted vs True KM saved → {out_path}")


# ── Master runner ─────────────────────────────────────────────────────────────
def run_survival_analysis(clin_df: pd.DataFrame,
                           X_test_df: pd.DataFrame,
                           xgb_pred: np.ndarray,
                           le,
                           figures_dir: str,
                           tables_dir: str):
    """Run the full survival analysis pipeline."""
    print("\n── Survival Analysis ─────────────────────────────────────")

    # 1. KM on true labels
    plot_km_curves(
        clin_df, "PAM50_true",
        "Kaplan-Meier by True PAM50 Subtype (Test Set)",
        f"{figures_dir}/km_true_subtypes.png",
    )

    # 2. Log-rank tests
    lr_df = run_logrank(clin_df, "PAM50_true")
    lr_df.to_csv(f"{tables_dir}/logrank_pairwise.csv", index=False)

    # 3. Cox regression
    clin_df["PAM50_predicted"] = le.inverse_transform(xgb_pred)
    cox_df = build_cox_dataframe(clin_df, X_test_df)
    cph    = run_cox(cox_df, f"{tables_dir}/cox_summary.csv")
    plot_cox_forest(cph, f"{figures_dir}/cox_forest_plot.png")

    # 4. Predicted vs True KM
    plot_km_predicted_vs_true(
        clin_df, le.inverse_transform(xgb_pred),
        f"{figures_dir}/km_predicted_vs_true.png",
    )

    return cph, lr_df
