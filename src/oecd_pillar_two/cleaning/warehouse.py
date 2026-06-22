from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ANALYTICAL_GOLD, REFERENCE, SILVER, load_config


def _firm_id(ticker: pd.Series) -> pd.Series:
    return "FIRM_" + ticker.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def build_silver_star_schema() -> dict[str, pd.DataFrame]:
    registry = pd.read_csv(REFERENCE / "candidate_universe.csv")
    benchmarks = pd.DataFrame({"Ticker": load_config()["project"]["benchmarks"], "universe_source": "market_benchmark"})
    registry = pd.concat([registry, benchmarks], ignore_index=True).drop_duplicates("Ticker")
    mapping = pd.read_csv(REFERENCE / "ticker_cik_mapping.csv")
    analysis = pd.read_csv(REFERENCE / "analysis_registry.csv")
    firms = registry.merge(mapping, on="Ticker", how="left").merge(
        analysis[["Ticker"]].assign(analysis_registry_flag=True), on="Ticker", how="left"
    )
    firms["firm_id"] = _firm_id(firms["Ticker"])
    firms["analysis_registry_flag"] = firms["analysis_registry_flag"].fillna(False)
    firms = firms.rename(columns={"Ticker": "ticker", "CIK": "cik"})
    firms[["firm_id", "ticker", "cik", "universe_source", "analysis_registry_flag"]].to_csv(
        SILVER / "dim_firm.csv", index=False
    )

    daily = pd.read_csv(SILVER / "fact_market_daily.csv")
    dates = pd.to_datetime(daily["Date"]).drop_duplicates().sort_values()
    dim_date = pd.DataFrame({"date": dates})
    dim_date["date_id"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date["month"] = dim_date["date"].dt.to_period("M").astype(str)
    dim_date["trading_day_flag"] = True
    dim_date["post_policy_flag"] = dim_date["month"].ge(load_config()["project"]["policy_post_month"])
    dim_date.to_csv(SILVER / "dim_date.csv", index=False)

    events = pd.DataFrame(load_config()["events"]).rename(columns={"id": "event_id", "date": "event_date", "label": "event_name"})
    events["event_type"] = events["event_id"].str.replace("_", " ")
    events.to_csv(SILVER / "dim_policy_event.csv", index=False)

    cbcr = pd.read_csv(SILVER / "fact_cbcr_jurisdiction_year.csv")
    jurisdictions = pd.concat([cbcr["REF_AREA"], cbcr["COUNTERPART_AREA"]]).dropna().drop_duplicates().sort_values()
    pd.DataFrame({"jurisdiction_id": jurisdictions, "jurisdiction_code": jurisdictions}).to_csv(
        SILVER / "dim_jurisdiction.csv", index=False
    )

    firm_lookup = pd.read_csv(SILVER / "dim_firm.csv")[["firm_id", "ticker"]]
    financial = pd.read_csv(SILVER / "fact_firm_financial_year.csv").drop(columns=["firm_id", "ticker"], errors="ignore").merge(firm_lookup, left_on="Ticker", right_on="ticker")
    financial.to_csv(SILVER / "fact_firm_financial_year.csv", index=False)
    daily = daily.drop(columns=["firm_id", "ticker"], errors="ignore").merge(firm_lookup, left_on="Ticker", right_on="ticker", how="left")
    daily["date_id"] = pd.to_datetime(daily["Date"]).dt.strftime("%Y%m%d").astype(int)
    daily.to_csv(SILVER / "fact_market_daily.csv", index=False)
    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv").drop(columns=["firm_id", "ticker"], errors="ignore").merge(firm_lookup, left_on="Ticker", right_on="ticker", how="left")
    monthly.to_csv(SILVER / "fact_market_monthly.csv", index=False)
    cbcr.to_csv(SILVER / "fact_cbcr_jurisdiction_year.csv", index=False)
    return {"dim_firm": firms, "dim_date": dim_date, "fact_firm_financial_year": financial, "fact_market_daily": daily, "fact_market_monthly": monthly}


def build_gold_analytical_schema() -> dict[str, pd.DataFrame]:
    firms = pd.read_csv(SILVER / "dim_firm.csv")[["firm_id", "ticker"]]
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv").drop(columns=["firm_id", "ticker"], errors="ignore").merge(firms, left_on="Ticker", right_on="ticker", how="left")
    design.to_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv", index=False)
    daily = pd.read_csv(SILVER / "fact_market_daily.csv", parse_dates=["Date"])
    cars = []
    for event in load_config()["events"]:
        available = daily.loc[daily["Date"] >= pd.Timestamp(event["date"]), "Date"]
        if available.empty:
            continue
        event_date = available.min()
        for ticker, group in daily.dropna(subset=["pillar_two_in_scope_proxy"]).groupby("Ticker"):
            group = group.sort_values("Date").reset_index(drop=True)
            matches = group.index[group["Date"].eq(event_date)]
            if matches.empty:
                continue
            center = int(matches[0])
            for width in (1, 3, 5):
                subset = group.iloc[max(0, center - width): center + width + 1]
                if len(subset) < width + 1:
                    continue
                cars.append({
                    "event_window_id": f"{event['id']}_m{width}_p{width}",
                    "event_id": event["id"],
                    "event_date": event_date.date().isoformat(),
                    "firm_id": group["firm_id"].iloc[0],
                    "ticker": ticker,
                    "pillar_two_in_scope_proxy": int(group["pillar_two_in_scope_proxy"].iloc[0]),
                    "window": f"[-{width},+{width}]",
                    "car": subset["abnormal_return"].sum(),
                })
    pd.DataFrame(cars).to_csv(ANALYTICAL_GOLD / "fact_event_firm_car.csv", index=False)
    design["exclusion_reason"] = design["eligible_for_main_design"].map({True: "", False: "incomplete_four_year_revenue_or_exposure_components"})
    dim_analysis = design.rename(columns={
        "pillar_two_four_year_scope_proxy": "pillar_two_scope_proxy",
        "pre_policy_tax_exposure_score": "pre_policy_exposure_score",
    })[["firm_id", "ticker", "eligible_for_main_design", "four_years_observed", "years_above_threshold",
        "pillar_two_scope_proxy", "pre_policy_exposure_score", "exposure_component_count", "exclusion_reason"]]
    dim_analysis.to_csv(ANALYTICAL_GOLD / "dim_analysis_firm.csv", index=False)

    models = [
        ("exposure_event_study", "event_study", "Continuous exposure event study", "car", "pillar_two_exposure_intensity", "HC3", "main_model", False),
        ("weighted_did", "did", "Overlap-weighted monthly return DiD", "abnormal_return", "DiD", "clustered_firm", "supplementary_validation", False),
        ("binary_did", "did", "Binary monthly return DiD", "abnormal_return", "DiD", "clustered_firm", "legacy_sensitivity", False),
        ("scm", "scm", "Synthetic control", "abnormal_return", "pillar_two_scope_proxy", "placebo", "complementary", False),
        ("revenue_did", "did", "Revenue mechanism DiD", "Log_Revenue", "DiD", "clustered_firm", "exploratory", False),
        ("cs_did", "staggered_did", "Callaway & Sant'Anna staggered DiD", "abnormal_return", "pillar_two_scope_proxy", "clustered_firm", "main_model", False),
        ("sa_did", "staggered_did", "Sun & Abraham event study", "abnormal_return", "pillar_two_scope_proxy", "clustered_firm", "robustness", False),
        ("gsynth", "gscm", "Generalized synthetic control", "abnormal_return", "pillar_two_scope_proxy", "bootstrap", "main_model", False),
        ("grf_heterogeneity", "causal_ml", "Causal forest heterogeneity", "Log_Revenue_change", "pillar_two_scope_proxy", "forest", "heterogeneity", False),
        ("bacon_decomposition", "diagnostic", "Bacon decomposition of TWFE", "abnormal_return", "pillar_two_scope_proxy", "bacon", "assumption_diagnostic", False),
        ("honest_did", "sensitivity", "HonestDiD parallel trends sensitivity", "abnormal_return", "pillar_two_scope_proxy", "sensitivity_bounds", "assumption_diagnostic", False),
    ]
    dim_model = pd.DataFrame(models, columns=["model_id", "model_family", "model_name", "outcome_variable", "treatment_variable", "standard_error_method", "evidence_role", "causal_claim_allowed"])
    dim_model.to_csv(ANALYTICAL_GOLD / "dim_model.csv", index=False)

    assumptions = [
        ("parallel_trends", "Parallel trends", "Pre-policy outcome trends are comparable"),
        ("common_support", "Common support", "Treatment groups overlap on observed covariates"),
        ("event_exogeneity", "Event exogeneity", "No concurrent exposure-correlated news"),
        ("no_unobserved_confounding", "No unobserved confounding", "No omitted time-varying causes"),
        ("staggered_timing", "Staggered treatment timing", "Treatment timing heterogeneity is correctly modeled"),
        ("limited_anticipation", "Limited anticipation", "Firms do not change behavior before treatment"),
        ("correct_specification", "Correct specification", "Interactive fixed effects model is correctly specified"),
        ("parallel_trends_sensitivity", "Parallel trends sensitivity", "Conclusions are robust to bounded violations of parallel trends"),
    ]
    pd.DataFrame(assumptions, columns=["assumption_id", "assumption_name", "interpretation_boundary"]).to_csv(
        ANALYTICAL_GOLD / "dim_model_assumption.csv", index=False
    )
    windows = []
    for event in load_config()["events"]:
        for width in (1, 3, 5):
            windows.append({"event_window_id": f"{event['id']}_m{width}_p{width}", "event_id": event["id"], "window_start": -width, "window_end": width, "benchmark": "QQQ"})
    pd.DataFrame(windows).to_csv(ANALYTICAL_GOLD / "dim_event_window.csv", index=False)

    scope = pd.read_csv(SILVER / "fact_firm_revenue_year.csv").merge(firms, left_on="Ticker", right_on="ticker", how="left")
    scope = scope.merge(design[["Ticker", "pillar_two_four_year_scope_proxy"]], on="Ticker", how="left")
    scope["above_threshold_flag"] = scope["Revenue_EUR"].ge(750_000_000)
    scope.rename(columns={"FiscalYear": "scope_year", "Revenue_EUR": "revenue_eur", "pillar_two_four_year_scope_proxy": "four_year_scope_proxy"}).to_csv(
        ANALYTICAL_GOLD / "fact_scope_classification.csv", index=False
    )
    return {"dim_analysis_firm": dim_analysis, "dim_model": dim_model}
