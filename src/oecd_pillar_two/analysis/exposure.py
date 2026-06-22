from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ..config import ANALYTICAL_GOLD, PYTHON_RESULTS, SILVER
from ..utils import write_json


MATCHING_COVARIATES = ["threshold_distance_log", "Firm_Size", "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio"]


def _payload(result, term: str, model: str) -> dict:
    ci = result.conf_int().loc[term]
    return {
        "model": model,
        "term": term,
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "nobs": int(result.nobs),
        "statistically_significant_5pct": bool(result.pvalues[term] < 0.05),
    }


def run_exposure_event_study() -> dict:
    cars = pd.read_csv(ANALYTICAL_GOLD / "fact_event_firm_car.csv")
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")
    covariates = ["Ticker", "pillar_two_exposure_intensity", "threshold_distance_log", "Firm_Size", "Leverage"]
    data = cars.merge(design[covariates], left_on="ticker", right_on="Ticker", how="inner")
    rows = []
    for (event_id, window), group in data.groupby(["event_id", "window"]):
        frame = group.dropna(subset=["car", "pillar_two_exposure_intensity", "threshold_distance_log", "Firm_Size", "Leverage"])
        if len(frame) < 20 or frame["pillar_two_exposure_intensity"].nunique() < 3:
            continue
        result = smf.ols(
            "car ~ pillar_two_exposure_intensity + threshold_distance_log + Firm_Size + Leverage",
            data=frame,
        ).fit(cov_type="HC3")
        rows.append({"event_id": event_id, "window": window, **_payload(
            result, "pillar_two_exposure_intensity", "Continuous pre-policy tax-exposure event study"
        )})
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["p_value_holm"] = multipletests(summary["p_value"], method="holm")[1]
        summary["statistically_significant_holm_5pct"] = summary["p_value_holm"] < 0.05
    summary.to_csv(PYTHON_RESULTS / "exposure_event_study.csv", index=False)
    payload = {
        "method": "Continuous pre-policy tax-exposure event study with HC3 inference",
        "events": summary.to_dict("records"),
        "assumptions_and_limitations": {
            "exposure_pre_determined": "Exposure components use accounting observations no later than fiscal 2022.",
            "conditional_event_exogeneity": "After included pre-policy covariates, exposure must be unrelated to concurrent firm-specific news; this is not fully testable.",
            "functional_form": "The coefficient assumes a linear relation between the constructed exposure score and CAR.",
            "measurement": "The score is a proxy based on ETR, R&D intensity, and intangible assets, not observed top-up tax.",
            "multiple_testing": "Holm-adjusted p-values are reported across event-window specifications.",
        },
    }
    write_json(PYTHON_RESULTS / "exposure_event_study.json", payload)
    return payload


def _smd(frame: pd.DataFrame, variable: str, weights: str | None = None) -> float:
    treated = frame[frame["pillar_two_four_year_scope_proxy"].eq(1)]
    control = frame[frame["pillar_two_four_year_scope_proxy"].eq(0)]
    if weights:
        mt = np.average(treated[variable], weights=treated[weights])
        mc = np.average(control[variable], weights=control[weights])
        vt = np.average((treated[variable] - mt) ** 2, weights=treated[weights])
        vc = np.average((control[variable] - mc) ** 2, weights=control[weights])
    else:
        mt, mc = treated[variable].mean(), control[variable].mean()
        vt, vc = treated[variable].var(ddof=1), control[variable].var(ddof=1)
    pooled = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / pooled) if pooled > 0 else np.nan


def _weighted_dynamic(data: pd.DataFrame) -> dict:
    dynamic = data.copy()
    dynamic["CalendarYear"] = dynamic["Month"].str[:4].astype(int)
    result = smf.wls(
        "abnormal_return ~ pillar_two_in_scope_proxy * C(CalendarYear, Treatment(reference=2023)) + C(Ticker) + C(Month)",
        data=dynamic,
        weights=dynamic["overlap_weight"],
    ).fit(cov_type="cluster", cov_kwds={"groups": dynamic["Ticker"]})
    rows, pre_terms = [], []
    for term in result.params.index:
        if "pillar_two_in_scope_proxy:C(CalendarYear" not in term:
            continue
        year = int(term.split("[")[1].split("]")[0].replace("T.", ""))
        if year in (2020, 2021, 2022):
            pre_terms.append(term)
        rows.append({
            "calendar_year": year, "estimate": result.params[term], "std_error": result.bse[term],
            "ci_low": result.conf_int().loc[term, 0], "ci_high": result.conf_int().loc[term, 1],
            "p_value": result.pvalues[term], "reference_year": 2023,
        })
    rows.append({"calendar_year": 2023, "estimate": 0, "std_error": 0, "ci_low": 0, "ci_high": 0, "p_value": np.nan, "reference_year": 2023})
    pd.DataFrame(rows).sort_values("calendar_year").to_csv(PYTHON_RESULTS / "weighted_did_dynamic.csv", index=False)
    restrictions = np.zeros((len(pre_terms), len(result.params)))
    for row, term in enumerate(pre_terms):
        restrictions[row, result.params.index.get_loc(term)] = 1
    test = result.wald_test(restrictions, scalar=True) if pre_terms else None
    return {"joint_pretrend_p_value": float(test.pvalue) if test is not None else None}


def run_weighted_did() -> dict:
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")
    baseline = design.loc[design["eligible_for_main_design"].eq(True)].dropna(subset=MATCHING_COVARIATES).copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(baseline[MATCHING_COVARIATES])
    treatment = baseline["pillar_two_four_year_scope_proxy"].astype(int)
    propensity = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(x, treatment).predict_proba(x)[:, 1]
    baseline["propensity_score"] = propensity
    treated_range = baseline.loc[treatment.eq(1), "propensity_score"].agg(["min", "max"])
    control_range = baseline.loc[treatment.eq(0), "propensity_score"].agg(["min", "max"])
    lower, upper = max(treated_range["min"], control_range["min"]), min(treated_range["max"], control_range["max"])
    baseline["in_common_support"] = baseline["propensity_score"].between(lower, upper)
    baseline["overlap_weight"] = np.where(treatment.eq(1), 1 - propensity, propensity)
    support_has_both_groups = baseline.loc[baseline["in_common_support"], "pillar_two_four_year_scope_proxy"].nunique() == 2
    baseline["support_trim_applied"] = support_has_both_groups
    if support_has_both_groups:
        baseline.loc[~baseline["in_common_support"], "overlap_weight"] = 0.0
    balance = pd.DataFrame({
        "variable": MATCHING_COVARIATES,
        "smd_unweighted": [_smd(baseline, variable) for variable in MATCHING_COVARIATES],
        "smd_overlap_weighted": [_smd(baseline[baseline["overlap_weight"] > 0], variable, "overlap_weight") for variable in MATCHING_COVARIATES],
    })
    balance.to_csv(PYTHON_RESULTS / "weighted_did_balance.csv", index=False)
    baseline[["Ticker", "pillar_two_four_year_scope_proxy", "propensity_score", "in_common_support", "overlap_weight"]].to_csv(
        PYTHON_RESULTS / "propensity_scores.csv", index=False
    )
    market = pd.read_csv(SILVER / "fact_market_monthly.csv")
    data = market.merge(baseline[["Ticker", "overlap_weight"]], on="Ticker", how="inner")
    data = data[(data["trading_days"] >= 10) & (data["overlap_weight"] > 0)].copy()
    result = smf.wls(
        "abnormal_return ~ DiD + C(Ticker) + C(Month)", data=data, weights=data["overlap_weight"]
    ).fit(cov_type="cluster", cov_kwds={"groups": data["Ticker"]})
    payload = _payload(result, "DiD", "Overlap-weighted monthly abnormal-return DiD")
    dynamic = _weighted_dynamic(data)
    weights = baseline.loc[baseline["overlap_weight"] > 0, "overlap_weight"]
    payload["assumption_diagnostics"] = {
        "overlap_status": "failed" if not support_has_both_groups else "available",
        "common_support_share": float(baseline["in_common_support"].mean()),
        "common_support_interval": [float(lower), float(upper)],
        "support_trim_applied": bool(support_has_both_groups),
        "effective_sample_size": float(weights.sum() ** 2 / (weights ** 2).sum()),
        "max_absolute_smd_unweighted": float(balance["smd_unweighted"].abs().max()),
        "max_absolute_smd_weighted": float(balance["smd_overlap_weighted"].abs().max()),
        "joint_pretrend_p_value": dynamic["joint_pretrend_p_value"],
        "limitations": "Weighting addresses observed baseline differences only; unobserved time-varying confounding, anticipation, and treatment misclassification remain. If strict common-support trimming removes one treatment arm, untrimmed overlap weights are retained and weak overlap is explicitly flagged.",
    }
    write_json(PYTHON_RESULTS / "weighted_did.json", payload)
    return payload
