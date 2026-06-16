# Data Directory

## Synthetic Data (default)

`synthetic_tcga_brca.csv` is generated automatically when you run `python pipeline.py`.

It contains 600 biologically realistic samples with:
- 50 PAM50 canonical genes
- 10 extra clinically relevant genes (TP53, PIK3CA, PTEN, CDH1, GATA3, ...)
- Clinical covariates: Age, Stage, Node_positive, Tumour_size
- Survival data: OS_months, OS_event
- Ground-truth PAM50 subtype label

Expression profiles are derived from published TCGA-BRCA PAM50 studies.

---

## Using Real TCGA-BRCA Data

### Option 1: GDC Data Portal (recommended)
1. Go to https://portal.gdc.cancer.gov/projects/TCGA-BRCA
2. Select "RNA-Seq" → "Gene Expression Quantification" → "HTSeq - Counts"
3. Download the manifest and use the GDC client:
   ```bash
   gdc-client download -m manifest.txt
   ```
4. Use the `TCGAbiolinks` R package to aggregate and normalise:
   ```R
   library(TCGAbiolinks)
   query <- GDCquery(
     project = "TCGA-BRCA",
     data.category = "Transcriptome Profiling",
     data.type = "Gene Expression Quantification",
     workflow.type = "STAR - Counts"
   )
   GDCdownload(query)
   data <- GDCprepare(query)
   ```

### Option 2: Pre-processed matrix from recount3
```R
library(recount3)
proj <- create_rse_manual(
  project = "BRCA",
  project_home = "data_sources/tcga",
  organism = "human",
  annotation = "gencode_v29",
  type = "gene"
)
```

### Option 3: UCSC Xena Browser
Download the TCGA-BRCA gene expression RNAseq (IlluminaHiSeq) matrix from:
https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)

---

## Expected Data Format

To use real data with this pipeline, create a CSV with:

| Column | Type | Description |
|--------|------|-------------|
| Patient_ID | str | TCGA barcode (e.g. TCGA-BH-A0AK) |
| ACTR3B, ANLN, ... | float | log2 gene expression (50 PAM50 genes required) |
| PAM50_subtype | str | LumA / LumB / HER2 / Basal / Normal |
| OS_months | float | Overall survival time in months |
| OS_event | int | 1 = deceased, 0 = censored |
| Age | int | Age at diagnosis |
| Stage | str | Stage_1 / Stage_2 / Stage_3 / Stage_4 |
| Node_positive | int | 1 = lymph node positive |
| Tumour_size | float | Tumour size in cm |

PAM50 subtype assignments can be obtained using the `genefu` R package:
```R
library(genefu)
subtypes <- molecular.subtypes(data = expr_matrix,
                                y = NULL,
                                mol.biol = "PAM50")
```

---

*Data: Synthetic data generated for portfolio demonstration. Real analysis should use TCGA portal data.*
