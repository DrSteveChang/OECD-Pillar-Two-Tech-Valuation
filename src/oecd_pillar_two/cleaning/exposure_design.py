from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ANALYTICAL_GOLD, BRONZE, REFERENCE, SILVER
from ..utils import write_json


SCOPE_YEARS = [2020, 2021, 2022, 2023]
THRESHOLD_EUR = 750_000_000
FX_TO_EUR = {"USD": 0.92, "EUR": 1.0, "CHF": 1.04, "TWD": 0.029, "INR": 0.011}
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]


def _sec_revenue_panel() -> pd.DataFrame:
    rows = []
    for path in sorted((BRONZE / "sec" / "companyfacts").glob("*.json")):
        facts = json.loads(path.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {})
        for tag in REVENUE_TAGS:
            if tag not in facts:
                continue
            units = facts[tag].get("units", {})
            values = units.get("USD", [])
            for item in values:
                if item.get("form") not in {"10-K", "20-F"} or item.get("fp") != "FY":
                    continue
                end = pd.to_datetime(item.get("end"), errors="coerce")
                start = pd.to_datetime(item.get("start"), errors="coerce")
                if pd.isna(end) or pd.isna(start) or not 300 <= (end - start).days <= 400:
                    continue
                rows.append(
                    {
                        "Ticker": path.stem,
                        "FiscalYear": int(end.year),
                        "Revenue_EUR": float(item["val"]) * FX_TO_EUR["USD"],
                        "revenue_source": f"SEC Company Facts:{tag}",
                        "filed": item.get("filed", ""),
                    }
                )
            if values:
                break
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    return (
        panel.sort_values("filed")
        .drop_duplicates(["Ticker", "FiscalYear"], keep="last")
        .drop(columns="filed")
    )


def _yahoo_revenue_panel() -> pd.DataFrame:
    paths = [BRONZE / "yahoo" / "candidate_corporate_financials.csv", BRONZE / "yahoo" / "corporate_financials.csv"]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    data = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    if data.empty:
        return data
    data = data[data["periodType"].eq("12M")].copy()
    data["FiscalYear"] = pd.to_datetime(data["asOfDate"]).dt.year
    data["Revenue_EUR"] = pd.to_numeric(data["TotalRevenue"], errors="coerce") * data["currencyCode"].map(FX_TO_EUR)
    data["Ticker"] = data["symbol"]
    data["revenue_source"] = "Yahoo annual income statement"
    return data[["Ticker", "FiscalYear", "Revenue_EUR", "revenue_source"]].dropna(subset=["Revenue_EUR"])


def _accounting_exposure() -> pd.DataFrame:
    income_paths = [BRONZE / "yahoo" / "candidate_corporate_financials.csv", BRONZE / "yahoo" / "corporate_financials.csv"]
    balance_paths = [BRONZE / "yahoo" / "candidate_balance_sheets.csv", BRONZE / "yahoo" / "balance_sheets.csv"]
    income = pd.concat([pd.read_csv(path) for path in income_paths if path.exists()], ignore_index=True).drop_duplicates()
    balance = pd.concat([pd.read_csv(path) for path in balance_paths if path.exists()], ignore_index=True).drop_duplicates()
    for frame in (income, balance):
        frame["FiscalYear"] = pd.to_datetime(frame["asOfDate"]).dt.year
    income = income[income["periodType"].eq("12M") & income["FiscalYear"].le(2022)]
    balance = balance[balance["periodType"].eq("12M") & balance["FiscalYear"].le(2022)]
    income = income.sort_values("FiscalYear").drop_duplicates(["symbol", "FiscalYear"], keep="last")
    balance = balance.sort_values("FiscalYear").drop_duplicates(["symbol", "FiscalYear"], keep="last")
    panel = income.merge(balance, on=["symbol", "FiscalYear"], suffixes=("_inc", "_bal"))
    revenue = pd.to_numeric(panel["TotalRevenue"], errors="coerce")
    pretax = pd.to_numeric(panel.get("PretaxIncome"), errors="coerce")
    tax = pd.to_numeric(panel.get("TaxProvision"), errors="coerce")
    rd = pd.to_numeric(panel.get("ResearchAndDevelopment"), errors="coerce")
    assets = pd.to_numeric(panel.get("TotalAssets"), errors="coerce")
    intangible = pd.to_numeric(panel.get("GoodwillAndOtherIntangibleAssets"), errors="coerce")
    liabilities = pd.to_numeric(panel.get("TotalLiabilitiesNetMinorityInterest"), errors="coerce")
    panel["ETR"] = (tax / pretax).where(pretax > 0).clip(0, 1)
    panel["low_tax_gap"] = (0.15 - panel["ETR"]).clip(lower=0)
    panel["RD_Intensity"] = (rd / revenue).clip(0, 1)
    panel["Intangible_Ratio"] = (intangible / assets).clip(0, 1)
    panel["Firm_Size"] = np.log(assets.where(assets > 0))
    panel["Leverage"] = (liabilities / assets).clip(0, 3)
    latest = panel.sort_values("FiscalYear").groupby("symbol", as_index=False).tail(1).copy()
    components = ["low_tax_gap", "RD_Intensity", "Intangible_Ratio"]
    for component in components:
        latest[f"{component}_rank"] = latest[component].rank(pct=True)
    latest["exposure_component_count"] = latest[components].notna().sum(axis=1)
    latest["pre_policy_tax_exposure_score"] = latest[[f"{x}_rank" for x in components]].mean(axis=1)
    latest.loc[latest["exposure_component_count"] < 2, "pre_policy_tax_exposure_score"] = np.nan
    return latest.rename(columns={"symbol": "Ticker"})[
        ["Ticker", "FiscalYear", "ETR", "low_tax_gap", "RD_Intensity", "Intangible_Ratio",
         "Firm_Size", "Leverage", "exposure_component_count", "pre_policy_tax_exposure_score"]
    ]


def build_exposure_design() -> pd.DataFrame:
    sec = _sec_revenue_panel()
    yahoo = _yahoo_revenue_panel()
    revenue = pd.concat([sec, yahoo], ignore_index=True)
    revenue = revenue[revenue["FiscalYear"].isin(SCOPE_YEARS)].copy()
    revenue = revenue.sort_values("revenue_source").drop_duplicates(["Ticker", "FiscalYear"], keep="first")
    revenue.insert(0, "firm_id", "FIRM_" + revenue["Ticker"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True))
    revenue.to_csv(SILVER / "fact_firm_revenue_year.csv", index=False)
    wide = revenue.pivot(index="Ticker", columns="FiscalYear", values="Revenue_EUR").reindex(columns=SCOPE_YEARS)
    design = wide.reset_index()
    design["four_years_observed"] = design[SCOPE_YEARS].notna().sum(axis=1)
    design["years_above_threshold"] = design[SCOPE_YEARS].ge(THRESHOLD_EUR).sum(axis=1)
    design["pillar_two_four_year_scope_proxy"] = (
        design["four_years_observed"].eq(4) & design["years_above_threshold"].ge(2)
    ).astype(int)
    design["median_pre_policy_revenue_eur"] = design[SCOPE_YEARS].median(axis=1)
    design["threshold_distance_log"] = np.log(design["median_pre_policy_revenue_eur"] / THRESHOLD_EUR)
    exposure = _accounting_exposure()
    design = design.merge(exposure, on="Ticker", how="left")
    design["pillar_two_exposure_intensity"] = (
        design["pillar_two_four_year_scope_proxy"] * design["pre_policy_tax_exposure_score"]
    )
    design["eligible_for_main_design"] = (
        design["four_years_observed"].eq(4) & design["pre_policy_tax_exposure_score"].notna()
    )
    design.to_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv", index=False)
    registry = design.loc[design["eligible_for_main_design"], [
        "Ticker", "median_pre_policy_revenue_eur", "pillar_two_four_year_scope_proxy",
        "pre_policy_tax_exposure_score", "pillar_two_exposure_intensity", "four_years_observed",
    ]].rename(columns={
        "median_pre_policy_revenue_eur": "Revenue_EUR",
        "pillar_two_four_year_scope_proxy": "Treatment_Group",
    })
    registry["Currency"] = "EUR"
    registry.to_csv(REFERENCE / "analysis_registry.csv", index=False)
    write_json(
        ANALYTICAL_GOLD / "exposure_design_metadata.json",
        {
            "threshold_eur": THRESHOLD_EUR,
            "scope_years": SCOPE_YEARS,
            "scope_rule": "At least EUR 750m in at least two of four years; all four years must be observed.",
            "exposure_score": "Mean percentile rank of low-tax gap, R&D intensity, and intangible-asset ratio; minimum two components.",
            "eligible_firms": int(len(registry)),
            "in_scope_firms": int(registry["Treatment_Group"].sum()),
        },
    )
    return design
