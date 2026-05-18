# The Impact of OECD Pillar Two Global Minimum Tax on the Valuation of Multinational Technology Enterprises

**Lead Researcher & Developer:** Boyan Zhang (bzhang@student.eae.es)  
**Institution:** EAE Business School Barcelona  
**Academic Supervisor:** Aleix Ruiz de Villa Robert (aleix.ruizdevillarobert@campus.eae.es)

---

## 1. Intellectual Property & Contribution Notice
**All rights reserved.** This repository contains the original code, empirical architecture, and data processing logic developed exclusively by **Boyan Zhang** for the Master in Big Data & Analytics (TFM). 

* **Code Sovereignty:** The core Python implementation for data normalization, Difference-in-Differences (DiD) modeling, Synthetic Control Method (SCM) optimization, and Causal ML heterogeneity estimation is the sole intellectual property of the lead researcher.
* **Collaboration Policy:** Access to this private repository is granted for academic review purposes. Any unauthorized duplication or inclusion of this logic in external submissions without explicit written consent is strictly prohibited.

---

## 2. Abstract
When the OECD/G20 Inclusive Framework reached its historic agreement on Pillar Two in October 2021, it established a 15% global minimum tax floor that, by design, targeted the world's largest technology companies. This study examines the real-world consequences of that shift for their market valuations. 

Utilizing a quasi-experimental **Big Data framework**, this research integrates heterogeneous datasets ranging from structured corporate tax disclosures (XBRL) to financial market indicators. The primary analytical strategy employs a **Difference-in-Differences (DiD)** panel regression to isolate the valuation impact of the policy shock. However, recognizing that mega-capitalization firms often follow unique, idiosyncratic trajectories that challenge the "parallel trends" assumption, the study incorporates the **Synthetic Control Method (SCM)** as a robust complementary check. 

Furthermore, to unpack the high-dimensional cross-sectional heterogeneity within the treatment group, the pipeline integrates **Unsupervised Time-Series Clustering (K-Means)** and **Non-parametric Causal Machine Learning (Random Forest Feature Importance)**. This dual econometric and data science architecture allows for the construction of mathematically weighted counterfactuals and the non-parametric unmasking of policy elasticity determinants, significantly strengthening the internal and external validity of the causal inference.

---

## 3. Repository Topology Architecture

```text
OECD-Pillar-Two-Tech-Valuation/
├── data/                                 # Storage layer for raw and derived datasets
│   ├── analytical_panel_dataset.csv      # Baseline integrated macro-micro panel (508 Obs)
│   └── derived_results/                  # Convex-optimized matrix outputs and aggregations
│       ├── full_cohort_scm_trajectories.csv  # High-dimensional individual firm counterfactuals
│       ├── global_scm_att_results.csv        # Consolidated cohort mean ATT gaps
│       └── derived_data_clusters.csv         # Unsupervised K-Means behavioral labels
├── src/                                  # Pure execution and algorithm layer
│   ├── models/
│   │   ├── did_regression.py             # Linear TWFE OLS model with clustered standard errors
│   │   └── synthetic_control.py          # Full-cohort convex-optimized non-parametric SCM solver
│   └── analysis/
│       ├── plot_scm_results.py           # Canvas alignment and rendering engine for Figure 1
│       ├── plot_supplementary_figures.py # Mathematical graphics engines for Figure 2 & 3
│       └── advanced_data_science_models.py # Heterogeneity forest, secondary audits & K-Means pipelines
└── docs/                                 # Hardcopy output assets and textual deliverables
    └── analysis_reports/                 # Publication-ready econometric summaries 
        ├── Table3_DiD_Regression_Results.txt  # Linear baseline summary (Pooled bias)
        ├── Table4_Secondary_OLS_Audit.txt     # Matrix singularity and collinearity diagnostic report
        ├── Table5_ML_Feature_Importances.csv  # Informational variance shares from forest splits
        └── figures/                      # High-resolution vector PDF/PNG charts (Type-42 Embedded)
            ├── Figure1_SCM_Global_ATT.pdf
            ├── Figure2_DiD_Event_Study.pdf
            ├── Figure3_SCM_Distribution_Gaps.pdf
            ├── Figure4_KMeans_SCM_Clusters.pdf
            └── Figure5_CausalML_Feature_Importance.pdf