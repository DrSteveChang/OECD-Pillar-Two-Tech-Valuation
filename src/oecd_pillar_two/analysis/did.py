from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..config import SILVER, PYTHON_RESULTS
from ..utils import write_json


BASELINE_COVARIATES = ["Firm_Size", "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio"]


def _result_payload(result, term: str, label: str) -> dict:
    return {
        "model": label,
        "term": term,
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
        "ci_low": float(result.conf_int().loc[term, 0]),
        "ci_high": float(result.conf_int().loc[term, 1]),
        "nobs": int(result.nobs),
        "statistically_significant_5pct": bool(result.pvalues[term] < 0.05),
    }


def _fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["Ticker"]})


def _baseline_covariates() -> pd.DataFrame:
    firm = pd.read_csv(SILVER / "fact_firm_financial_year.csv")
    baseline = firm[firm["FiscalYear"].eq(2022)][["Ticker", "pillar_two_in_scope_proxy", *BASELINE_COVARIATES]].copy()
    return baseline.rename(columns={column: f"baseline_{column}" for column in BASELINE_COVARIATES})


def _balance_diagnostics(baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    for variable in BASELINE_COVARIATES:
        column = f"baseline_{variable}"
        treated = baseline.loc[baseline["pillar_two_in_scope_proxy"].eq(1), column].dropna()
        control = baseline.loc[baseline["pillar_two_in_scope_proxy"].eq(0), column].dropna()
        pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = (treated.mean() - control.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        records.append(
            {
                "variable": variable,
                "treated_mean": treated.mean(),
                "control_mean": control.mean(),
                "standardized_mean_difference": smd,
                "treated_n": len(treated),
                "control_n": len(control),
            }
        )
    balance = pd.DataFrame(records)
    balance.to_csv(PYTHON_RESULTS / "did_baseline_balance.csv", index=False)

    size = baseline.dropna(subset=["baseline_Firm_Size"])
    treated_size = size.loc[size["pillar_two_in_scope_proxy"].eq(1), "baseline_Firm_Size"]
    control_size = size.loc[size["pillar_two_in_scope_proxy"].eq(0), "baseline_Firm_Size"]
    lower = max(treated_size.min(), control_size.min())
    upper = min(treated_size.max(), control_size.max())
    common_support = size["baseline_Firm_Size"].between(lower, upper) if lower <= upper else pd.Series(False, index=size.index)
    diagnostic = {
        "largest_absolute_standardized_difference": float(balance["standardized_mean_difference"].abs().max()),
        "covariates_with_abs_smd_above_0_25": balance.loc[
            balance["standardized_mean_difference"].abs() > 0.25, "variable"
        ].tolist(),
        "firm_size_common_support_share": float(common_support.mean()),
        "overlap_status": "material_concern" if common_support.mean() < 0.8 else "acceptable",
    }
    return balance, diagnostic


def _dynamic_did(data: pd.DataFrame) -> dict:
    dynamic = data.copy()
    dynamic["CalendarYear"] = dynamic["Month"].str[:4].astype(int)
    formula = (
        "abnormal_return ~ C(Ticker) + C(Month) + "
        "pillar_two_in_scope_proxy * C(CalendarYear, Treatment(reference=2023))"
    )
    result = _fit(formula, dynamic)
    rows = []
    for term in result.params.index:
        if "pillar_two_in_scope_proxy:C(CalendarYear" not in term:
            continue
        year = int(term.split("[")[1].split("]")[0].replace("T.", ""))
        rows.append(
            {
                "calendar_year": year,
                "estimate": result.params[term],
                "std_error": result.bse[term],
                "ci_low": result.conf_int().loc[term, 0],
                "ci_high": result.conf_int().loc[term, 1],
                "p_value": result.pvalues[term],
                "reference_year": 2023,
            }
        )
    coefficients = pd.DataFrame(rows).sort_values("calendar_year")
    reference = pd.DataFrame(
        [{"calendar_year": 2023, "estimate": 0.0, "std_error": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": np.nan, "reference_year": 2023}]
    )
    coefficients = pd.concat([coefficients, reference], ignore_index=True).sort_values("calendar_year")
    coefficients.to_csv(PYTHON_RESULTS / "did_dynamic_coefficients.csv", index=False)

    pre_terms = [
        term for term in result.params.index
        if "pillar_two_in_scope_proxy:C(CalendarYear" in term
        and any(f"[T.{year}]" in term or f"[{year}]" in term for year in (2020, 2021, 2022))
    ]
    restrictions = np.zeros((len(pre_terms), len(result.params)))
    for row, term in enumerate(pre_terms):
        restrictions[row, result.params.index.get_loc(term)] = 1
    test = result.wald_test(restrictions, scalar=True) if pre_terms else None
    return {
        "reference_year": 2023,
        "pre_period_years": [2020, 2021, 2022],
        "joint_pretrend_p_value": float(test.pvalue) if test is not None else None,
        "joint_pretrend_not_rejected_5pct": bool(test.pvalue >= 0.05) if test is not None else False,
        "interpretation": "Failure to reject does not prove parallel trends; it only reports whether detectable annual pre-period differences are present.",
    }


def run_market_did() -> dict:
    data = pd.read_csv(SILVER / "fact_market_monthly.csv")
    data = data[data["trading_days"] >= 10].copy()
    data["time_index"] = data["Month"].rank(method="dense").astype(int) - 1
    baseline = _baseline_covariates()
    _, overlap = _balance_diagnostics(baseline)
    data = data.merge(baseline.drop(columns="pillar_two_in_scope_proxy"), on="Ticker", how="left")

    specifications = []
    formulas = [
        ("baseline_qqq", "abnormal_return ~ DiD + C(Ticker) + C(Month)", data),
        ("alternative_benchmark_spy", "abnormal_return_spy ~ DiD + C(Ticker) + C(Month)", data),
        ("alternative_benchmark_xlk", "abnormal_return_xlk ~ DiD + C(Ticker) + C(Month)", data),
        ("group_linear_trend", "abnormal_return ~ DiD + pillar_two_in_scope_proxy:time_index + C(Ticker) + C(Month)", data),
    ]
    complete_covariates = data.dropna(subset=[f"baseline_{variable}" for variable in BASELINE_COVARIATES]).copy()
    covariate_terms = " + ".join(f"Post:baseline_{variable}" for variable in BASELINE_COVARIATES)
    formulas.append(("baseline_covariate_post_interactions", f"abnormal_return ~ DiD + {covariate_terms} + C(Ticker) + C(Month)", complete_covariates))
    for name, formula, frame in formulas:
        result = _fit(formula, frame)
        payload = _result_payload(result, "DiD", name)
        specifications.append({"specification": name, **payload})
    pd.DataFrame(specifications).to_csv(PYTHON_RESULTS / "did_robustness_specifications.csv", index=False)

    main = next(item for item in specifications if item["specification"] == "baseline_qqq")
    dynamic = _dynamic_did(data)
    assumptions = {
        "parallel_trends": {
            "status": "not_rejected_but_not_proven" if dynamic["joint_pretrend_not_rejected_5pct"] else "concern",
            **dynamic,
        },
        "covariate_balance_and_overlap": overlap,
        "no_anticipation": {
            "status": "at_risk",
            "reason": "Pillar Two announcements occurred in 2021-2022 before the January 2024 implementation indicator.",
        },
        "no_time_varying_unobserved_confounding": {
            "status": "untestable_assumption",
            "reason": "Firm and month fixed effects do not absorb firm-specific time-varying shocks.",
        },
        "robustness_specifications": len(specifications),
    }
    write_json(PYTHON_RESULTS / "did_assumption_diagnostics.json", assumptions)
    payload = {key: value for key, value in main.items() if key not in {"specification"}}
    payload["model"] = "Monthly abnormal return DiD with firm and month FE"
    payload["assumption_diagnostics"] = assumptions
    write_json(PYTHON_RESULTS / "market_did.json", payload)
    return payload


def run_revenue_mechanism() -> dict:
    data = pd.read_csv(SILVER / "fact_firm_financial_year.csv")
    data = data.dropna(subset=["Log_Revenue"]).copy()
    data["Post"] = (data["FiscalYear"] >= 2024).astype(int)
    data["DiD"] = data["pillar_two_in_scope_proxy"] * data["Post"]
    result = _fit("Log_Revenue ~ DiD + C(Ticker) + C(FiscalYear)", data)
    payload = _result_payload(result, "DiD", "Exploratory revenue mechanism DiD")
    payload["formal_causal_claim"] = False
    payload["limitation"] = "Only one well-populated pre-implementation fiscal year is available; parallel trends cannot be credibly assessed."
    write_json(PYTHON_RESULTS / "revenue_mechanism.json", payload)
    return payload
