# Statutory and Econometric Framework for Sample Selection

This document outlines the methodological, statutory, and econometric rationale behind the selection and classification of the 100 technology multinational enterprises (MNEs) analyzed in this study. To ensure causal inference validity, the sample is strictly divided into a 50/50 Treatment and Control structure based on objective legislative thresholds.

---

## 1. Statutory Rationale (OECD Pillar Two Scope)
In accordance with the **OECD (2021) Pillar Two Model Rules (GloBE Rules), Chapter 1, Article 1.1 (Scope)**, the Global Minimum Tax framework applies exclusively to Constituent Entities that are members of an MNE Group that has annual revenues of **€750 million or more** in the Consolidated Financial Statements of the Ultimate Parent Entity (UPE) in at least two of the four Fiscal Years immediately preceding the tested Fiscal Year.

By leveraging this statutory bright-line threshold, this study eliminates subjective selection bias. Firms crossing the €750M consolidated revenue line are legally bound to the 15% tax floor, while those falling below it remain structurally exempt.

---

## 2. Econometric Rationale (DiD & SCM Requirements)

### A. Difference-in-Differences (DiD) Parallel Trends
The foundational identifier for a DiD model is the **Parallel Trends Assumption**, which dictates that in the absence of treatment, the average outcomes of the Treatment and Control groups would have followed parallel trajectories over time. 
* To satisfy this, the Control Group is selected from the **same GICS Technology Sectors/Sub-sectors** as the Treatment Group. 
* This ensures that both groups are exposed to identical macroeconomic shocks, sector-specific cycles, and technological shifts, isolating the regulatory enforcement as the sole source of divergence post-2021.

### B. Synthetic Control Method (SCM) Donor Pool Integrity
For mega-cap firms (e.g., Apple, Microsoft), finding a single identical control firm is statistically impossible due to idiosyncratic market power. SCM resolves this by constructing a "synthetic counterfactual" from a **Donor Pool**.
* The 50 firms in the Control Group constitute this clean Donor Pool.
* Because their revenues are strictly below €750M, they are completely unpolluted by the Pillar Two policy shock, ensuring that the optimization loop minimizes pre-trend mean squared prediction error (MSPE) without endogenous bias.

---

## 3. Sample Architecture (50/50 Split Matrix)

| Cohort | Sample Size | Statutory Selection Criteria (OECD Rule 1.1) | Econometric Function in the Research |
| :--- | :--- | :--- | :--- |
| **Treatment Group** | 50 Companies | • GICS Technology Sector / Digital Platforms<br>• Consolidated Pre-policy Revenue (2018-2020) **$\ge$ €750M** | Captures the structural valuation impact of direct policy exposure and top-up tax liabilities. |
| **Control Group** | 50 Companies | • GICS Technology Sector / Digital Platforms<br>• Consolidated Pre-policy Revenue (2018-2020) **< €750M** | Serves as the baseline control for DiD and forms the unpolluted Donor Pool for SCM counterfactual optimization. |

---

## 4. Algorithmic Screening & Currency Normalization Pipeline
Because multinational enterprises disclose financial statements in their native reporting currencies (USD, TWD, EUR, INR, etc.), a direct nominal comparison introduces severe measurement error. 

The ingestion pipeline (`src/validate_sample.py`) executes the following multi-step normalization:
1. **Retrieval:** Extracts the FY2020 consolidated `TotalRevenue` and the native `financialCurrency` metric via the `yahooquery` async engine.
2. **FX Harmonization:** Applies the historical annualized average exchange rate relative to the Euro (€) for the benchmark year 2020 (e.g., USD/EUR = 0.877).
3. **Threshold Audit:** Evaluates the mathematical condition:
   $$	ext{Revenue}_{	ext{EUR}} \ge 750,000,000$$
4. **Stratified Sorting:** Sorts compliant candidates by revenue size to capture the top 50 mega/large-caps for the Treatment Group, and the top 50 mid/small-caps within the exempt boundary for the Control Group, preventing severe asset-size asymmetry from skewing the econometric weights.
