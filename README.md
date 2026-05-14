# The Impact of OECD Pillar Two Global Minimum Tax on the Valuation of Multinational Technology Enterprises

**Lead Researcher & Developer:** Boyan Zhang (bzhang@student.eae.es)  
**Institution:** EAE Business School Barcelona  
**Academic Supervisor:** Aleix Ruiz de Villa Robert (aleix.ruizdevillarobert@campus.eae.es)

---

## 🔒 Intellectual Property & Contribution Notice
**All rights reserved.** This repository contains the original code, empirical architecture, and data processing logic developed exclusively by **Boyan Zhang** for the Master in Big Data & Analytics (TFM). 

* **Code Sovereignty:** The core Python implementation for data normalization, Difference-in-Differences (DiD) modeling, and Synthetic Control Method (SCM) optimization is the sole intellectual property of the lead researcher.
* **Collaboration Policy:** Access to this private repository is granted for academic review purposes. Any unauthorized duplication or inclusion of this logic in external submissions without explicit written consent is strictly prohibited.

---

## 📝 Abstract
When the OECD/G20 Inclusive Framework reached its historic agreement on Pillar Two in October 2021, it established a 15% global minimum tax floor that, by design, targeted the world’s largest technology companies. This study examines the real-world consequences of that shift for their market valuations. For decades, Big Tech has navigated complex tax positions across low-tax jurisdictions; Pillar Two represents a direct, structural response to these long-standing practices. 

Utilizing a quasi-experimental **Big Data framework**, this research integrates heterogeneous datasets—ranging from structured corporate tax disclosures (XBRL) to high-frequency financial market data. The primary analytical strategy employs a **Difference-in-Differences (DiD)** panel regression to isolate the valuation impact of the 2021 policy shock. However, recognizing that mega-capitalization firms often follow unique, idiosyncratic trajectories that challenge the "parallel trends" assumption, the study incorporates the **Synthetic Control Method (SCM)** as a robust complementary check. This dual-model architecture allows for the construction of mathematically weighted counterfactuals, significantly strengthening the internal validity of the causal inference.

**Keywords:** OECD Pillar Two, Global Minimum Tax, Technology Sector Valuation, Difference-in-Differences, Synthetic Control Method, Big Data Analytics, Causal Inference.

---

## 🛠 Technical Implementation

### 1. Dual-Model Causal Inference
* **Baseline (DiD):** Captures sector-wide average treatment effects (ATE) following the October 8, 2021 threshold.
* **Robustness (SCM):** Constructs "synthetic" counterparts for idiosyncratic mega-caps (e.g., Apple, Microsoft) to mitigate pre-trend deviations.

### 2. Automated Data Pipeline
* **Normalization:** Automated mapping of heterogeneous tax tags into a unified schema for 50 technology MNEs.
* **Cleaning:** Implementation of Winsorization (1%/99%) and historical exchange rate alignment.

---

## 📂 Repository Structure
- `/data`: Normalized financial datasets and macro-governance indices.
- `/src/cleaning`: Python scripts for data standardization and XBRL mapping.
- `/src/models`: Core implementation of DiD and SCM optimization loops.
- `/notebooks`: Exploratory Data Analysis (EDA) and visualization.
- `/docs`: Literature review and methodological foundations.

---

## 📊 Development Status & Audit
Current phase: **Phase 2 - Data Normalization & Structural Mapping.** All technical milestones and code contributions are tracked via the GitHub Commit History to ensure absolute transparency of individual contributions.
