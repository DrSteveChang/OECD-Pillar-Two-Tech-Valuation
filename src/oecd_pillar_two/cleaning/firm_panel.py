from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ANALYTICAL_GOLD, BRONZE, REFERENCE, SILVER


def build_firm_year_panel() -> pd.DataFrame:
    income_paths = [BRONZE / "yahoo" / "corporate_financials.csv", BRONZE / "yahoo" / "candidate_corporate_financials.csv"]
    balance_paths = [BRONZE / "yahoo" / "balance_sheets.csv", BRONZE / "yahoo" / "candidate_balance_sheets.csv"]
    income = pd.concat([pd.read_csv(path) for path in income_paths if path.exists()], ignore_index=True).drop_duplicates()
    balance = pd.concat([pd.read_csv(path) for path in balance_paths if path.exists()], ignore_index=True).drop_duplicates()
    registry = REFERENCE / "analysis_registry.csv"
    sample = pd.read_csv(registry if registry.exists() else REFERENCE / "sample_registry.csv").rename(
        columns={"Treatment_Group": "pillar_two_in_scope_proxy"}
    )

    income = income[income["periodType"].eq("12M")].copy()
    balance = balance[balance["periodType"].eq("12M")].copy()
    income["FiscalYear"] = pd.to_datetime(income["asOfDate"]).dt.year
    balance["FiscalYear"] = pd.to_datetime(balance["asOfDate"]).dt.year
    income = income[income["FiscalYear"] < 2026]
    balance = balance[balance["FiscalYear"] < 2026]
    income = income.sort_values("asOfDate").drop_duplicates(["symbol", "FiscalYear"], keep="last")
    balance = balance.sort_values("asOfDate").drop_duplicates(["symbol", "FiscalYear"], keep="last")

    panel = income.merge(balance, on=["symbol", "FiscalYear"], suffixes=("_inc", "_bal"))
    panel = panel.merge(
        sample[["Ticker", "pillar_two_in_scope_proxy", "Revenue_EUR", "Currency"]],
        left_on="symbol",
        right_on="Ticker",
        how="inner",
    )
    revenue = pd.to_numeric(panel["TotalRevenue"], errors="coerce")
    assets = pd.to_numeric(panel["TotalAssets"], errors="coerce")
    liabilities = pd.to_numeric(panel["TotalLiabilitiesNetMinorityInterest"], errors="coerce")
    rd = pd.to_numeric(panel.get("ResearchAndDevelopment"), errors="coerce").fillna(0)
    tax = pd.to_numeric(panel.get("TaxProvision"), errors="coerce")
    pretax = pd.to_numeric(panel.get("PretaxIncome"), errors="coerce")
    intangible = pd.to_numeric(panel.get("GoodwillAndOtherIntangibleAssets"), errors="coerce")

    panel["Log_Revenue"] = np.where(revenue > 0, np.log(revenue), np.nan)
    panel["Firm_Size"] = np.where(assets > 0, np.log(assets), np.nan)
    panel["Leverage"] = liabilities / assets
    panel["RD_Intensity"] = rd / revenue
    panel["ETR"] = (tax / pretax).where(pretax > 0).clip(0, 1)
    panel["Intangible_Ratio"] = intangible / assets
    exposure = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")[
        ["Ticker", "pre_policy_tax_exposure_score", "pillar_two_exposure_intensity"]
    ]
    panel = panel.merge(exposure, on="Ticker", how="left")
    columns = [
        "Ticker", "FiscalYear", "pillar_two_in_scope_proxy", "Currency", "Revenue_EUR",
        "TotalRevenue", "PretaxIncome", "NetIncome", "Log_Revenue", "Firm_Size",
        "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio",
        "pre_policy_tax_exposure_score", "pillar_two_exposure_intensity",
    ]
    result = panel[columns].sort_values(["Ticker", "FiscalYear"]).reset_index(drop=True)
    result.insert(0, "firm_id", "FIRM_" + result["Ticker"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True))
    if result.duplicated(["Ticker", "FiscalYear"]).any():
        raise ValueError("Canonical firm-year panel is not unique")
    result.to_csv(SILVER / "fact_firm_financial_year.csv", index=False)
    return result
