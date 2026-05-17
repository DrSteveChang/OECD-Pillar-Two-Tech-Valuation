## 📊 Data Schema & Resource Mapping

To ensure the strict algebraic convergence of the Difference-in-Differences (DiD) and Synthetic Control Method (SCM) models, this project constructs a balanced panel dataset encompassing 50 global technology multinational enterprises (MNEs) from 2014 to 2025. All data variables are retrieved via standardized financial API interfaces, entirely bypassing the manual parsing of non-standardized raw filings.

| Variable Category | Variable Name | Academic Definition / Proxy | Official Data Source | Python Retrieval Path / Core Library |
| :--- | :--- | :--- | :--- | :--- |
| **Dependent Variable (Y)**<br>*Core Valuation Metrics* | `Market_Cap` | **Total Market Capitalization**: The primary target of causal inference, used to evaluate the market's real-time response to policy shocks. | Yahoo Finance | `yahooquery.Ticker.valuation_measures` |
| | `PE_Ratio` | **Trailing P/E Ratio**: A secondary dependent variable reflecting the adjustment of market expectations and valuation premiums. | Yahoo Finance | `yahooquery.Ticker.summary_detail` |
| **Core Treatment & Threshold**<br>*Policy Exposure & Selection* | `ETR` | **Effective Tax Rate (ETR)**: Calculated as $\text{Income Tax Expense} / \text{Pre-tax Income}$. Used to identify the baseline level of exposure to the global minimum tax shock. | Yahoo Finance / FMP | `yahooquery.Ticker.income_statement` (TaxProvision / PretaxIncome) |
| | `Foreign_Income_Ratio` | **Foreign Pre-tax Income Ratio**: Foreign Pre-tax Income / Total Pre-tax Income. Measures the risk exposure to international profit shifting. | Compustat Global / FMP Segment API | Corporate segment disclosure interface (Segment Income Data) |
| | `Total_Revenue` | **Consolidated Annual Revenue**: Used for the €750M statutory threshold verification to mathematically separate the treatment group from the control group. | Yahoo Finance | `yahooquery.Ticker.income_statement` (TotalRevenue) |
| **Micro Control Variable (X)**<br>*Firm Financial & Structural* | `Firm_Size` | **Firm Size**: Controlled using the natural logarithm of total assets, $\ln(\text{Total Assets})$, to eliminate systemic scale interference. | Yahoo Finance | `yahooquery.Ticker.balance_sheet` (TotalAssets) |
| | `Net_Income` | **Net Income**: Controls for endogenous fluctuations in firm-level internal profitability. | Yahoo Finance | `yahooquery.Ticker.income_statement` (NetIncome) |
| | `Rev_Growth` | **Revenue Growth Rate**: Controls for the industry life cycle and high-growth premiums of individual firms. | Yahoo Finance | Computed longitudinally based on `TotalRevenue` |
| | `Leverage` | **Leverage Ratio**: Total Liabilities / Total Assets. Controls for the impact of capital structure and leverage-related financial risks. | Yahoo Finance | `yahooquery.Ticker.balance_sheet` (TotalLiabilitiesNetMinorityInterest) |
| | `RD_Intensity` | **R&D Intensity**: R&D Expense / Total Revenue. Acts as a critical structural predictor to optimize pre-trend alignment within the Synthetic Control framework. | Yahoo Finance | `yahooquery.Ticker.income_statement` (ResearchAndDevelopment) |
| | `Intangible_Ratio` | **Intangible Assets Ratio**: Net Intangible Assets / Total Assets. Controls for intellectual property (IP) concentration and shifting capacity. | Yahoo Finance | `yahooquery.Ticker.balance_sheet` (GrossIntangibleAssets / TotalAssets) |
| **Macro & Temporal Controls (X)**<br>*External & Policy Context* | `Gov_Quality` | **Regulatory Quality Index**: Controls for the institutional environment, governance quality, and compliance costs within core operating markets. | World Bank | `wbgapi` (Index: PV.EST / SG.REG.LLWS.XQ) |
| | `GDP_Growth` | **Real GDP Growth Rate**: Controls for macroeconomic economic cycles and systematic market risks. | World Bank | `wbgapi` (Index: NY.GDP.MKTP.KD.ZG) |
| | `Anticipation_Dim` | **Anticipation Time Dummy**: A temporal indicator (0 or 1) capturing the 2021-2023 transitional pricing-in effects prior to the formal 2024 implementation. | Policy Timeline | Manual Coding / Temporal hardcoding mapping |

---

### 🗄️ Data Storage Specifications & Naming Conventions

Upon executing the data ingestion pipeline, variables are cleaned, standardized, and formatted into a unified panel structure, then solidified in `CSV` format within the `/data` directory under the following naming conventions:

1. `/data/raw_market_dimensions.csv`: Stores high-frequency market capitalization and P/E ratio panel series.
2. `/data/raw_corporate_financials.csv`: Stores normalized and standardized annual financial metrics for multinational technology enterprises.
3. `/data/macro_institutional_controls.csv`: Stores macroeconomic governance and economic indicators retrieved from the World Bank.