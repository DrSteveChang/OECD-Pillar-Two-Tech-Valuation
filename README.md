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
├── .vscode/                              # IDE environment configurations
├── data/                                 # Storage layer for raw, analytic, and derived datasets
│   ├── derived_results/                  # Convex-optimized matrix outputs and aggregations
│   │   ├── AAPL_synthetic_trajectory_indexed.csv
│   │   ├── AAPL_synthetic_trajectory.csv
│   │   ├── full_cohort_scm_trajectories.csv
│   │   └── global_scm_att_results.csv
│   ├── .gitkeep
│   ├── analytical_panel_dataset.csv      # Baseline integrated macro-micro panel (508 Obs)
│   ├── raw_balance_sheets.csv            # Unprocessed structural balance sheet pulls
│   ├── raw_corporate_financials.csv      # Ingested XBRL fundamental disclosures
│   ├── raw_market_dimensions.csv         # High-frequency financial market metrics
│   └── verified_100_constituents.csv     # Harmonized stratified sampling master list
├── doc/                                  # Theoretical and data dictionary documentation
│   ├── DATA_SCHEMA.md                    # Econometric codebook and variable schemas
│   └── SAMPLE_SELECTION_METHODOLOGY.md   # Statutory screening logic and definitions
├── docs/
│   └── analysis_reports/                 # Hardcopy output assets and textual deliverables
│       ├── figures/                      # High-resolution vector PDF/PNG charts
│       │   ├── Figure1_SCM_Global_ATT.pdf
│       │   ├── Figure1_SCM_Global_ATT.png
│       │   ├── Figure2_DiD_Event_Study.pdf
│       │   ├── Figure2_DiD_Event_Study.png
│       │   ├── Figure3_SCM_Distribution_Gaps.pdf
│       │   ├── Figure3_SCM_Distribution_Gaps.png
│       │   ├── Figure4_KMeans_SCM_Clusters.pdf
│       │   ├── Figure4_KMeans_SCM_Clusters.png
│       │   ├── Figure5_CausalML_Feature_Importance.pdf
│       │   └── Figure5_CausalML_Feature_Importance.png
│       ├── derived_data_clusters.csv     # Unsupervised K-Means behavioral cohort labels
│       ├── Table1_Summary_Statistics.csv # Descriptive moments tracking matrix
│       ├── Table2_Correlation_Matrix.csv # Multicollinearity linear association grid
│       ├── Table3_DiD_Regression_Results.txt # TWFE baseline panel regression summary
│       ├── Table4_Secondary_OLS_Audit.txt    # Rank deficiency and matrix singularity diagnostic
│       └── Table5_ML_Feature_Importances.csv # Permutation information share statistics
├── src/                                  # Pure execution and algorithm layer
│   ├── analysis/                         # Analytical plotting and machine learning engines
│   │   ├── advanced_data_science_models.py
│   │   ├── descriptive_stats.py          # Generates Table 1 and Table 2 descriptive matrices
│   │   ├── plot_scm_results.py           # Core visualization engine for Figure 1
│   │   └── plot_supplementary_figures.py # Graphical vector compilers for Figures 2 & 3
│   ├── cleaning/                         # Pre-processing scripts
│   │   └── clean_data.py                 # XBRL parser and data harmonization script
│   ├── model/                            # Structural econometric estimators
│   │   ├── did_regression.py             # Computes multi-period panel TWFE equations
│   │   └── synthetic_control.py          # Solves SLSQP non-parametric donor weights
│   ├── .gitkeep
│   ├── fetch_data.py                     # Automated async API macro-micro data retriever
│   └── validate_sample.py                # Ingestion pipeline currency normalizer script
├── .gitignore
├── LICENSE
└── README.md