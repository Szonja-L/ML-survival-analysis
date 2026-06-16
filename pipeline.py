#!/usr/bin/env python3
"""
pipeline.py
-----------
End-to-end runner for the TCGA-BRCA PAM50 Classification & Survival Analysis project.

Usage:
    python pipeline.py [--seed SEED] [--skip-shap]

Steps
-----
1.  Generate synthetic TCGA-BRCA dataset
2.  Preprocess: scale features, stratified train/test split
3.  Train XGBoost + Random Forest classifiers
4.  Generate publication-quality figures (PCA, heatmap, ROC, confusion)
5.  SHAP interpretability analysis
6.  Kaplan-Meier survival curves + log-rank tests
7.  Cox Proportional Hazards regression
8.  Save all results to results/figures/ and results/tables/
"""

import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data    import generate_dataset
from preprocessing    import load_and_split, get_clinical_df
from models           import train_and_evaluate, save_metrics
from visualization    import run_all_visualizations
from shap_analysis    import run_shap_analysis
from survival_analysis import run_survival_analysis

FIGURES_DIR = "results/figures"
TABLES_DIR  = "results/tables"
DATA_DIR    = "data"


def banner(msg: str):
    print(f"\n{'━'*62}")
    print(f"  {msg}")
    print(f"{'━'*62}")


def main(seed: int = 42, skip_shap: bool = False):
    t0 = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR,    exist_ok=True)

    # ── Step 1 ─ Data Generation ─────────────────────────────────────────────
    banner("STEP 1 — Synthetic TCGA-BRCA Data Generation")
    df = generate_dataset(seed=seed)
    df.to_csv(f"{DATA_DIR}/synthetic_tcga_brca.csv", index=False)
    print(f"  Dataset shape    : {df.shape}")
    print(f"  Subtype counts   :\n{df['PAM50_subtype'].value_counts().to_string()}")
    print(f"  Event rate       : {df['OS_event'].mean():.2%}")
    print(f"  Median OS        : {df['OS_months'].median():.1f} months")

    # ── Step 2 ─ Preprocessing ───────────────────────────────────────────────
    banner("STEP 2 — Preprocessing")
    (X_train, X_test, y_train, y_test,
     le, scaler, feature_names,
     X_train_df, X_test_df) = load_and_split(df, test_size=0.20,
                                               random_state=seed)
    print(f"  Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")

    # ── Step 3 ─ Model Training ───────────────────────────────────────────────
    banner("STEP 3 — Model Training & Evaluation")
    (xgb_model, rf_model,
     xgb_met, rf_met,
     xgb_pred, rf_pred) = train_and_evaluate(
        X_train, X_test, y_train, y_test, le
    )
    save_metrics(xgb_met, rf_met, f"{TABLES_DIR}/model_metrics.csv")

    # Per-class report tables
    import pandas as pd
    for met, name in [(xgb_met, "XGBoost"), (rf_met, "RandomForest")]:
        rpt = pd.DataFrame(met["report"]).T
        rpt.to_csv(f"{TABLES_DIR}/classification_report_{name}.csv")

    # ── Step 4 ─ Visualizations ───────────────────────────────────────────────
    banner("STEP 4 — Figures")
    run_all_visualizations(
        df, X_train, X_test, y_train, y_test,
        le, feature_names,
        xgb_model, rf_model,
        xgb_met, rf_met,
        FIGURES_DIR, TABLES_DIR,
    )

    # ── Step 5 ─ SHAP ────────────────────────────────────────────────────────
    if not skip_shap:
        banner("STEP 5 — SHAP Interpretability")
        run_shap_analysis(
            xgb_model, rf_model,
            X_train, X_test,
            feature_names, le,
            FIGURES_DIR, TABLES_DIR,
        )
    else:
        print("\n  SHAP skipped (--skip-shap flag set)")

    # ── Step 6 & 7 ─ Survival Analysis ───────────────────────────────────────
    banner("STEP 6 + 7 — Survival Analysis (KM + Cox)")
    clin_df = get_clinical_df(df, X_test_df, y_test, le)
    cph, lr_df = run_survival_analysis(
        clin_df, X_test_df, xgb_pred, le,
        FIGURES_DIR, TABLES_DIR,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"\n  XGBoost  accuracy : {xgb_met['accuracy']:.4f}")
    print(f"  RF       accuracy : {rf_met['accuracy']:.4f}")
    print(f"  XGBoost  ROC-AUC  : {xgb_met['roc_auc_macro']:.4f}")
    print(f"  RF       ROC-AUC  : {rf_met['roc_auc_macro']:.4f}")
    print(f"\n  Figures  → {FIGURES_DIR}/")
    print(f"  Tables   → {TABLES_DIR}/")
    print(f"  Data     → {DATA_DIR}/")

    files_f = sorted(os.listdir(FIGURES_DIR))
    files_t = sorted(os.listdir(TABLES_DIR))
    print(f"\n  Generated figures  ({len(files_f)}):")
    for f in files_f:
        print(f"    • {f}")
    print(f"\n  Generated tables   ({len(files_t)}):")
    for f in files_t:
        print(f"    • {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TCGA-BRCA PAM50 Classification + Survival Analysis Pipeline"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis (faster run)")
    args = parser.parse_args()
    main(seed=args.seed, skip_shap=args.skip_shap)
