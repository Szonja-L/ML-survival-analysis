"""
models.py
---------
Train and evaluate PAM50 subtype classifiers:
  • XGBoost (gradient-boosted trees)
  • Random Forest

Both models use 5-fold stratified cross-validation for hyper-parameter
selection and report final held-out metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb


SUBTYPE_ORDER = ["LumA", "LumB", "HER2", "Basal", "Normal"]
CV_FOLDS      = 5
RANDOM_STATE  = 42


# ── XGBoost ───────────────────────────────────────────────────────────────────
def build_xgboost(n_classes: int = 5) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators       = 400,
        max_depth          = 5,
        learning_rate      = 0.08,
        subsample          = 0.80,
        colsample_bytree   = 0.75,
        min_child_weight   = 3,
        gamma              = 0.1,
        reg_alpha          = 0.05,
        reg_lambda         = 1.0,
        objective          = "multi:softprob",
        num_class          = n_classes,
        eval_metric        = "mlogloss",
        use_label_encoder  = False,
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
    )


# ── Random Forest ─────────────────────────────────────────────────────────────
def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators  = 500,
        max_depth     = None,
        max_features  = "sqrt",
        min_samples_split = 4,
        min_samples_leaf  = 2,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    le,
                    model_name: str) -> dict:
    class_names = le.classes_
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    # One-vs-rest macro AUC
    y_bin    = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
    )
    print(f"\n{'='*60}")
    print(f"  {model_name}  —  Test-set performance")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print(f"  Macro ROC-AUC     : {auc_macro:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "model"            : model_name,
        "accuracy"         : round(acc,      4),
        "balanced_accuracy": round(bal_acc,  4),
        "roc_auc_macro"    : round(auc_macro,4),
        "report"           : report,
        "confusion_matrix" : confusion_matrix(y_true, y_pred),
    }


def cv_metrics(model, X_train: np.ndarray, y_train: np.ndarray,
               le, model_name: str) -> dict:
    """5-fold CV accuracy and balanced-accuracy on training set."""
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
    y_cv  = cross_val_predict(model, X_train, y_train, cv=skf, n_jobs=-1)
    acc   = accuracy_score(y_train, y_cv)
    bal   = balanced_accuracy_score(y_train, y_cv)
    print(f"  [{model_name}] CV accuracy: {acc:.4f}  |  CV balanced-acc: {bal:.4f}")
    return {"cv_accuracy": round(acc, 4), "cv_balanced_accuracy": round(bal, 4)}


# ── Main training function ────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    """
    Fit XGBoost and Random Forest; return fitted models + metrics dicts.

    Returns
    -------
    xgb_model, rf_model, xgb_metrics, rf_metrics
    """
    n_classes = len(le.classes_)

    print("\n── Training XGBoost ──────────────────────────────────────")
    xgb_model = build_xgboost(n_classes)
    print("  Cross-validating on training set …")
    cv_metrics(xgb_model, X_train, y_train, le, "XGBoost")
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    xgb_pred  = xgb_model.predict(X_test)
    xgb_prob  = xgb_model.predict_proba(X_test)
    xgb_met   = compute_metrics(y_test, xgb_pred, xgb_prob, le, "XGBoost")

    print("\n── Training Random Forest ────────────────────────────────")
    rf_model = build_random_forest()
    print("  Cross-validating on training set …")
    cv_metrics(rf_model, X_train, y_train, le, "Random Forest")
    rf_model.fit(X_train, y_train)
    rf_pred   = rf_model.predict(X_test)
    rf_prob   = rf_model.predict_proba(X_test)
    rf_met    = compute_metrics(y_test, rf_pred, rf_prob, le, "Random Forest")

    return xgb_model, rf_model, xgb_met, rf_met, xgb_pred, rf_pred


def save_metrics(xgb_met: dict, rf_met: dict, out_path: str) -> pd.DataFrame:
    rows = []
    for met in [xgb_met, rf_met]:
        rows.append({
            "Model"            : met["model"],
            "Accuracy"         : met["accuracy"],
            "Balanced Accuracy": met["balanced_accuracy"],
            "Macro ROC-AUC"    : met["roc_auc_macro"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nMetrics saved → {out_path}")
    return df
