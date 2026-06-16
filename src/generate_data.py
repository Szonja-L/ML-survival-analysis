"""
generate_data.py
----------------
Generates biologically realistic synthetic TCGA-BRCA data for PAM50 subtype
classification and survival analysis.

Expression means derived from published TCGA-BRCA studies:
  - Parker et al. (2009) J Clin Oncol - PAM50 gene signatures
  - Cancer Genome Atlas Network (2012) Nature - TCGA BRCA
  - Koboldt et al. for subtype-specific expression patterns

Output: 600 samples × (50 PAM50 genes + 10 extra genes + clinical variables)
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

# ── PAM50 canonical gene list ─────────────────────────────────────────────────
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

# Extra clinically relevant genes added for richer features
EXTRA_GENES = [
    "TP53", "PIK3CA", "PTEN", "CDH1", "GATA3",
    "MAP3K1", "CDKN1B", "TBX3", "RUNX1", "AKT1",
]

ALL_GENES = PAM50_GENES + EXTRA_GENES
SUBTYPES   = ["LumA", "LumB", "HER2", "Basal", "Normal"]

# ── Subtype-specific expression means (log2 TPM-like units, 0–15 scale) ───────
# Derived from Parker 2009, TCGA 2012, and Prat & Perou 2011 review
EXPRESSION_PROFILES = {
    #           LumA   LumB   HER2  Basal  Normal
    "ACTR3B": [ 5.0,   5.5,   5.5,  5.0,   5.5],
    "ANLN":   [ 4.0,   7.0,   8.0,  9.0,   3.5],
    "BAG1":   [ 8.5,   7.0,   5.0,  4.0,   7.5],
    "BCL2":   [ 9.0,   7.0,   4.0,  3.5,   7.5],
    "BIRC5":  [ 4.0,   7.5,   8.0,  9.5,   3.5],
    "BLVRA":  [ 7.0,   6.5,   5.5,  5.0,   6.5],
    "CCNB1":  [ 4.5,   7.0,   8.5,  9.0,   4.0],
    "CCNE1":  [ 4.5,   7.0,   7.0,  8.0,   4.0],
    "CDC20":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "CDC6":   [ 4.5,   7.0,   7.0,  8.5,   4.0],
    "CDH3":   [ 3.5,   4.5,   6.5,  8.0,   4.0],
    "CENPF":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "CEP55":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "CXXC5":  [ 8.0,   6.5,   4.5,  3.0,   7.0],
    "EGFR":   [ 3.0,   3.5,   4.0,  9.0,   4.0],
    "ERBB2":  [ 4.0,   5.0,  11.5,  3.5,   5.0],
    "ESR1":   [10.5,   9.0,   4.0,  2.5,   7.0],
    "EXO1":   [ 4.0,   7.0,   7.0,  8.5,   3.5],
    "FGFR4":  [ 7.5,   6.5,   7.5,  3.5,   6.0],
    "FOXA1":  [10.0,   8.5,   4.5,  2.0,   7.0],
    "FOXC1":  [ 2.5,   3.0,   3.5,  9.0,   3.0],
    "GRB7":   [ 3.5,   4.5,  11.0,  3.0,   4.5],
    "GUSB":   [ 7.0,   7.0,   7.0,  7.0,   7.0],  # housekeeping
    "KNTC2":  [ 4.0,   7.0,   7.5,  8.5,   3.5],
    "KRT14":  [ 2.0,   2.5,   2.5, 10.5,   3.0],
    "KRT17":  [ 2.0,   2.5,   2.5,  9.5,   3.0],
    "KRT5":   [ 2.0,   2.5,   2.5, 10.0,   3.5],
    "MAPT":   [ 8.5,   7.0,   4.0,  2.5,   6.5],
    "MDM2":   [ 5.0,   5.5,   6.0,  6.0,   5.5],
    "MELK":   [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "MIA":    [ 4.0,   4.0,   4.0,  3.5,   4.0],
    "MKI67":  [ 4.0,   7.5,   8.0,  9.5,   3.5],
    "MLPH":   [ 9.5,   8.0,   4.5,  2.5,   7.0],
    "MMP11":  [ 4.5,   6.0,   7.0,  6.5,   5.0],
    "MYBL2":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "MYC":    [ 5.0,   6.5,   7.5,  8.0,   5.0],
    "NAT1":   [ 9.0,   7.5,   4.0,  2.5,   7.0],
    "NDC80":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "NUF2":   [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "ORC6L":  [ 4.0,   7.0,   7.5,  8.5,   3.5],
    "PGR":    [ 9.5,   7.5,   3.0,  2.0,   6.5],
    "PHGDH":  [ 4.0,   5.0,   5.0,  8.5,   4.5],
    "PTTG1":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "RRM2":   [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "SFRP1":  [ 8.0,   6.5,   4.0,  3.0,   7.0],
    "SLC39A6":[ 9.0,   7.5,   4.5,  3.0,   7.0],
    "TMEM45B":[ 7.0,   6.0,   5.0,  5.5,   6.5],
    "TYMS":   [ 4.5,   7.0,   7.5,  8.5,   4.0],
    "UBE2T":  [ 4.0,   7.0,   7.5,  9.0,   3.5],
    "UBE2C":  [ 4.0,   7.5,   8.0,  9.5,   3.5],
    # Extra genes
    "TP53":   [ 5.0,   6.0,   6.5,  9.0,   5.0],
    "PIK3CA": [ 6.5,   6.5,   7.0,  5.5,   6.5],
    "PTEN":   [ 7.0,   6.0,   5.5,  5.0,   7.0],
    "CDH1":   [ 8.5,   7.5,   5.0,  3.0,   8.0],
    "GATA3":  [ 9.5,   8.0,   4.0,  2.5,   7.5],
    "MAP3K1": [ 6.5,   5.5,   5.0,  4.5,   6.0],
    "CDKN1B": [ 7.5,   6.0,   5.5,  4.5,   7.0],
    "TBX3":   [ 7.0,   6.0,   5.5,  4.0,   6.5],
    "RUNX1":  [ 7.0,   6.0,   5.5,  4.5,   6.5],
    "AKT1":   [ 6.0,   6.0,   6.5,  5.5,   6.0],
}

# Survival parameters (Weibull distribution) per subtype
# scale → location of survival time in months; shape → hazard shape
SURVIVAL_PARAMS = {
    #        shape   scale   event_prob
    "LumA":  (2.0,   140.0,  0.18),
    "LumB":  (2.0,    90.0,  0.30),
    "HER2":  (1.8,    70.0,  0.42),
    "Basal": (1.6,    55.0,  0.48),
    "Normal":(2.0,   120.0,  0.22),
}

# Subtype sample counts (approximate TCGA-BRCA proportions; total = 600)
SUBTYPE_N = {"LumA": 228, "LumB": 120, "HER2": 66, "Basal": 132, "Normal": 54}


def generate_expression(subtype: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate log2-TPM expression matrix for n samples of a given subtype."""
    idx = SUBTYPES.index(subtype)
    rows = {}
    for gene in ALL_GENES:
        mu  = EXPRESSION_PROFILES[gene][idx]
        # Add mild within-subtype correlation noise
        sd  = 0.9 if gene != "GUSB" else 0.3   # GUSB = housekeeping, tight
        rows[gene] = rng.normal(mu, sd, n).clip(0, 15)
    return pd.DataFrame(rows)


def generate_survival(subtype: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate overall survival time (months) and event indicator."""
    shape, scale, event_prob = SURVIVAL_PARAMS[subtype]
    # Raw survival times from Weibull
    raw_times   = weibull_min.rvs(shape, scale=scale, size=n, random_state=int(rng.integers(1e6)))
    raw_times   = np.clip(raw_times, 1, 240)
    # Censor a fraction of samples (observed = 0 if censored)
    event_flags = rng.binomial(1, event_prob, n)
    # Censored patients have a censoring time ≤ their raw time
    # Ensure lower bound is always less than upper bound
    low_bound    = np.minimum(12, raw_times * 0.5)
    censor_times = np.array([rng.uniform(lo, hi) for lo, hi in zip(low_bound, raw_times)])
    observed_time = np.where(event_flags, raw_times, censor_times)
    return pd.DataFrame({
        "OS_months": observed_time.round(1),
        "OS_event":  event_flags,
    })


def add_clinical_covariates(df_expr: pd.DataFrame, subtype: str,
                             n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Append realistic clinical variables derived from subtype biology."""
    # Age at diagnosis
    age_mu = {"LumA": 61, "LumB": 58, "HER2": 54, "Basal": 50, "Normal": 60}
    age = rng.normal(age_mu[subtype], 10, n).clip(25, 90).astype(int)

    # Tumour stage (I–IV), subtype-specific distribution
    stage_probs = {
        "LumA":   [0.30, 0.45, 0.20, 0.05],
        "LumB":   [0.20, 0.40, 0.30, 0.10],
        "HER2":   [0.15, 0.35, 0.35, 0.15],
        "Basal":  [0.15, 0.35, 0.35, 0.15],
        "Normal": [0.30, 0.40, 0.25, 0.05],
    }
    stage_num = rng.choice([1, 2, 3, 4], size=n, p=stage_probs[subtype])
    stage_str = [f"Stage_{s}" for s in stage_num]

    # Node status
    node_pos_prob = {"LumA": 0.30, "LumB": 0.42, "HER2": 0.55,
                     "Basal": 0.40, "Normal": 0.28}
    node_positive = rng.binomial(1, node_pos_prob[subtype], n)

    # Tumour size (cm)
    size_mu = {"LumA": 1.8, "LumB": 2.4, "HER2": 2.8, "Basal": 2.6, "Normal": 1.9}
    tumour_size = rng.normal(size_mu[subtype], 0.8, n).clip(0.1, 10).round(1)

    df_expr = df_expr.copy()
    df_expr["Age"]          = age
    df_expr["Stage"]        = stage_str
    df_expr["Stage_num"]    = stage_num
    df_expr["Node_positive"]= node_positive
    df_expr["Tumour_size"]  = tumour_size
    return df_expr


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate the full synthetic TCGA-BRCA dataset.

    Returns
    -------
    pd.DataFrame  shape (600, 60 + clinical + survival columns)
    """
    rng = np.random.default_rng(seed)
    frames = []
    for subtype, n in SUBTYPE_N.items():
        expr   = generate_expression(subtype, n, rng)
        surv   = generate_survival(subtype, n, rng)
        clin   = add_clinical_covariates(expr, subtype, n, rng)
        chunk  = pd.concat([clin, surv], axis=1)
        chunk["PAM50_subtype"] = subtype
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows so subtypes are interleaved
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Patient IDs in TCGA style
    df.insert(0, "Patient_ID",
              [f"TCGA-BX-{str(i).zfill(4)}" for i in range(len(df))])

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"\nSubtype distribution:\n{df['PAM50_subtype'].value_counts()}")
    print(f"\nEvent rate:  {df['OS_event'].mean():.2%}")
    print(f"Median OS:   {df['OS_months'].median():.1f} months")
    df.to_csv("data/synthetic_tcga_brca.csv", index=False)
    print("\nSaved → data/synthetic_tcga_brca.csv")
