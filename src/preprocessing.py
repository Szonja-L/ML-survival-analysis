"""
preprocessing.py
----------------
Preprocessing pipeline for PAM50 classification:
  - Feature / label extraction
  - Stratified train/test split
  - StandardScaler fitting on train, applied to test
  - Label encoding for multi-class targets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

PAM50_GENES = [
    "ACTR3B", "ANLN",   "BAG1",   "BCL2",   "BIRC5",
    "BLVRA",  "CCNB1",  "CCNE1",  "CDC20",  "CDC6",
    "CDH3",   "CENPF",  "CEP55",  "CXXC5",  "EGFR",
    "ERBB2",  "ESR1",   "EXO1",   "FGFR4",  "FOXA1",
    "FOXC1",  "GRB7",   "GUSB",   "KNTC2",  "KRT14",
    "KRT17",  "KRT5",   "MAPT",   "MDM2",   "MELK",
    "MIA",    "MKI67",  "MLPH",   "MMP11",  "MYBL2",
    "MYC",    "NAT1",   "NDC80",  "NUF2",   "ORC6L",
    "PGR",    "PHGDH",  "PTTG1",  "RRM2",   "SFRP1",
    "SLC39A6","TMEM45B","TYMS",   "UBE2T",  "UBE2C",
]

EXTRA_GENES = [
    "TP53", "PIK3CA", "PTEN", "CDH1", "GATA3",
    "MAP3K1", "CDKN1B", "TBX3", "RUNX1", "AKT1",
]

ALL_GENES = PAM50_GENES + EXTRA_GENES

SUBTYPE_ORDER = ["LumA", "LumB", "HER2", "Basal", "Normal"]
SUBTYPE_COLORS = {
    "LumA":   "#2196F3",   # blue
    "LumB":   "#03A9F4",   # light blue
    "HER2":   "#FF9800",   # orange
    "Basal":  "#F44336",   # red
    "Normal": "#4CAF50",   # green
}


def load_and_split(df: pd.DataFrame,
                   test_size: float = 0.20,
                   random_state: int = 42):
    """
    Extract feature matrix and labels; return stratified train/test splits.

    Returns
    -------
    X_train, X_test : np.ndarray   (scaled)
    y_train, y_test : np.ndarray   (integer-encoded)
    le              : LabelEncoder
    scaler          : StandardScaler
    feature_names   : list[str]
    X_train_raw, X_test_raw : pd.DataFrame  (unscaled, for SHAP display)
    """
    gene_cols = [g for g in ALL_GENES if g in df.columns]
    X = df[gene_cols].values
    y_str = df["PAM50_subtype"].values

    le = LabelEncoder()
    le.fit(SUBTYPE_ORDER)           # deterministic class ordering
    y = le.transform(y_str)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # Wrap raw splits as DataFrames for SHAP / inspection
    X_train_df = pd.DataFrame(X_train_raw, columns=gene_cols)
    X_test_df  = pd.DataFrame(X_test_raw,  columns=gene_cols)

    return X_train, X_test, y_train, y_test, le, scaler, gene_cols, X_train_df, X_test_df


def get_clinical_df(df: pd.DataFrame,
                    X_test_raw: pd.DataFrame,
                    y_test: np.ndarray,
                    le: LabelEncoder) -> pd.DataFrame:
    """
    Return a clinical dataframe aligned with the test set for survival analysis.
    Matches rows by gene expression fingerprint (gene sum).
    """
    gene_cols = X_test_raw.columns.tolist()
    # Match test rows back to original df via nearest gene sum
    test_sums  = X_test_raw[gene_cols].sum(axis=1).values
    orig_sums  = df[gene_cols].sum(axis=1).values
    indices    = []
    for ts in test_sums:
        idx = int(np.argmin(np.abs(orig_sums - ts)))
        indices.append(idx)
    clin = df.iloc[indices][
        ["Patient_ID", "OS_months", "OS_event",
         "Age", "Stage", "Stage_num", "Node_positive", "Tumour_size"]
    ].copy().reset_index(drop=True)
    clin["PAM50_true"]      = le.inverse_transform(y_test)
    return clin
