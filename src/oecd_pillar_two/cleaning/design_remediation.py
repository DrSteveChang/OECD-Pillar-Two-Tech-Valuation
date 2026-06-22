from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import ANALYTICAL_GOLD, PYTHON_RESULTS, REFERENCE, SILVER
from ..utils import write_json


CORE_COVARIATES = ["Firm_Size", "ETR", "RD_Intensity"]
FIRM_JURISDICTION_EXPOSURE_COLUMNS = {
    "Ticker",
    "jurisdiction_code",
    "FiscalYear",
    "exposure_weight",
    "first_treat_year",
    "treated",
}
EU_COUNCIL_SOURCE = (
    "https://www.consilium.europa.eu/en/press/press-releases/2022/12/12/"
    "international-taxation-council-reaches-agreement-on-a-minimum-level-of-taxation-for-largest-corporations/"
)
ADOPTION_COLUMNS = [
    "jurisdiction_code",
    "jurisdiction_name",
    "rule_type",
    "effective_date",
    "status",
    "source_url",
    "source_type",
    "notes",
]
EU_MEMBER_CODES = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czechia",
    "DNK": "Denmark",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "DEU": "Germany",
    "GRC": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SVK": "Slovak Republic",
    "SVN": "Slovenia",
    "ESP": "Spain",
    "SWE": "Sweden",
}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate_firm_jurisdiction_exposure(path: Path | None = None) -> dict[str, object]:
    path = path or (ANALYTICAL_GOLD / "fact_firm_jurisdiction_pillar_two_exposure.csv")
    if not path.exists():
        return {
            "valid": False,
            "failure_reason": "Missing firm-jurisdiction revenue/profit exposure needed for staggered cohorts.",
            "required_columns": sorted(FIRM_JURISDICTION_EXPOSURE_COLUMNS),
        }
    try:
        exposure = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {
            "valid": False,
            "failure_reason": "Firm-jurisdiction exposure file is empty.",
            "required_columns": sorted(FIRM_JURISDICTION_EXPOSURE_COLUMNS),
        }
    missing = sorted(FIRM_JURISDICTION_EXPOSURE_COLUMNS - set(exposure.columns))
    if missing:
        return {
            "valid": False,
            "failure_reason": "Missing required columns: " + ", ".join(missing),
            "required_columns": sorted(FIRM_JURISDICTION_EXPOSURE_COLUMNS),
        }
    if exposure.empty:
        return {"valid": False, "failure_reason": "Firm-jurisdiction exposure file has no rows."}
    exposure = exposure.copy()
    exposure["treated"] = pd.to_numeric(exposure["treated"], errors="coerce")
    exposure["FiscalYear"] = pd.to_numeric(exposure["FiscalYear"], errors="coerce")
    exposure["first_treat_year"] = pd.to_numeric(exposure["first_treat_year"], errors="coerce")
    exposure["exposure_weight"] = pd.to_numeric(exposure["exposure_weight"], errors="coerce")
    if exposure["exposure_weight"].isna().any() or not exposure["exposure_weight"].between(0, 1).all():
        return {"valid": False, "failure_reason": "Exposure weights must be numeric and between 0 and 1."}
    treated = exposure[exposure["treated"].eq(1)]
    cohorts = sorted(treated["first_treat_year"].dropna().unique())
    if len(cohorts) < 2:
        return {"valid": False, "failure_reason": "At least two treatment cohorts are required for staggered DiD."}
    has_never_treated = exposure["treated"].eq(0).any()
    has_not_yet_treated = (exposure["first_treat_year"] > exposure["FiscalYear"]).any()
    if not (has_never_treated or has_not_yet_treated):
        return {"valid": False, "failure_reason": "Never-treated or not-yet-treated support is required."}
    return {
        "valid": True,
        "failure_reason": "",
        "cohort_count": int(len(cohorts)),
        "has_never_treated": bool(has_never_treated),
        "has_not_yet_treated": bool(has_not_yet_treated),
        "rows": int(len(exposure)),
    }


def build_overlap_restricted_sample() -> pd.DataFrame:
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")
    design = design.copy()
    design["pillar_two_in_scope_proxy"] = design["pillar_two_four_year_scope_proxy"].astype(int)
    design["eligible_for_main_design"] = design["eligible_for_main_design"].map(_bool)
    design["core_covariate_count"] = design[CORE_COVARIATES].notna().sum(axis=1)

    eligible = design[design["eligible_for_main_design"] & design["core_covariate_count"].ge(2)].copy()
    design["propensity_score"] = np.nan
    if eligible["pillar_two_in_scope_proxy"].nunique() == 2:
        features = eligible[CORE_COVARIATES].copy()
        features = features.fillna(features.median(numeric_only=True))
        scaled = StandardScaler().fit_transform(features)
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(scaled, eligible["pillar_two_in_scope_proxy"])
        design.loc[eligible.index, "propensity_score"] = model.predict_proba(scaled)[:, 1]

    scored = design.dropna(subset=["propensity_score"])
    treated = scored.loc[scored["pillar_two_in_scope_proxy"].eq(1), "propensity_score"]
    control = scored.loc[scored["pillar_two_in_scope_proxy"].eq(0), "propensity_score"]
    if not treated.empty and not control.empty:
        lower = max(treated.min(), control.min())
        upper = min(treated.max(), control.max())
        design["within_common_support"] = design["propensity_score"].between(lower, upper) if lower <= upper else False
    else:
        lower = np.nan
        upper = np.nan
        design["within_common_support"] = False

    design["restricted_sample_flag"] = (
        design["eligible_for_main_design"]
        & design["core_covariate_count"].ge(2)
        & design["within_common_support"].fillna(False)
    )
    design["exclusion_reason"] = np.select(
        [
            ~design["eligible_for_main_design"],
            design["core_covariate_count"].lt(2),
            ~design["within_common_support"].fillna(False),
        ],
        ["not_eligible_for_main_design", "insufficient_core_covariates", "outside_common_support"],
        default="retained",
    )
    columns = [
        "Ticker",
        "pillar_two_in_scope_proxy",
        "eligible_for_main_design",
        "core_covariate_count",
        "Firm_Size",
        "ETR",
        "RD_Intensity",
        "propensity_score",
        "within_common_support",
        "restricted_sample_flag",
        "exclusion_reason",
    ]
    output = design[columns].sort_values("Ticker")
    output.attrs["common_support_bounds"] = {"lower": lower, "upper": upper}
    output.to_csv(ANALYTICAL_GOLD / "fact_overlap_restricted_sample.csv", index=False)
    return output


def build_restricted_sample_did(sample: pd.DataFrame | None = None) -> dict:
    if sample is None:
        sample = pd.read_csv(ANALYTICAL_GOLD / "fact_overlap_restricted_sample.csv")
    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv")
    retained = sample.loc[sample["restricted_sample_flag"].eq(True), ["Ticker"]]
    data = monthly.merge(retained, on="Ticker", how="inner")
    data = data[data["trading_days"] >= 10].copy()
    attrition = {
        "initial_firms": int(sample["Ticker"].nunique()),
        "eligible_main_design": int(sample.loc[sample["eligible_for_main_design"].eq(True), "Ticker"].nunique()),
        "core_covariates": int(sample.loc[sample["core_covariate_count"].ge(2), "Ticker"].nunique()),
        "common_support": int(sample.loc[sample["restricted_sample_flag"].eq(True), "Ticker"].nunique()),
    }
    payload: dict[str, object] = {
        "model": "Restricted overlap-sample monthly return DiD",
        "evidence_role": "diagnostic_only",
        "causal_upgrade": False,
        "attrition_counts": attrition,
        "sample_rule": "eligible_for_main_design, at least two core covariates, and propensity common support",
    }
    if data["pillar_two_in_scope_proxy"].nunique() < 2 or data["Post"].nunique() < 2:
        payload.update(
            {
                "status": "not_estimated",
                "failure_reason": "Restricted overlap sample does not contain treated and control observations in pre/post periods.",
            }
        )
    else:
        try:
            result = smf.ols("abnormal_return ~ DiD + C(Ticker) + C(Month)", data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data["Ticker"]}
            )
            payload.update(
                {
                    "status": "estimated",
                    "estimate": float(result.params["DiD"]),
                    "std_error": float(result.bse["DiD"]),
                    "p_value": float(result.pvalues["DiD"]),
                    "nobs": int(result.nobs),
                    "interpretation": "Diagnostic restricted-sample sensitivity only; not promoted to the main causal estimate.",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive against singular designs
            payload.update({"status": "not_estimated", "failure_reason": str(exc)})
    write_json(PYTHON_RESULTS / "restricted_sample_did.json", payload)
    return payload


def build_event_confound_screen() -> pd.DataFrame:
    events = pd.read_csv(REFERENCE / "policy_events.csv")
    daily = pd.read_csv(SILVER / "fact_market_daily.csv", parse_dates=["Date"])
    daily = daily.dropna(subset=["abnormal_return"]).sort_values(["Ticker", "Date"])
    records = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["date"])
        parts = []
        for _, firm in daily.groupby("Ticker", sort=False):
            dates = firm["Date"].to_numpy()
            insert_at = int(np.searchsorted(dates, np.datetime64(event_date), side="left"))
            if insert_at >= len(firm):
                continue
            start = max(0, insert_at - 3)
            stop = min(len(firm), insert_at + 4)
            window = firm.iloc[start:stop].copy()
            window["trading_day_offset"] = range(start - insert_at, stop - insert_at)
            parts.append(window)
        window_data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        records.append(
            {
                "event_id": event["event_id"],
                "event_date": event["date"],
                "window_start_trading_day": -3,
                "window_end_trading_day": 3,
                "firm_observations": int(len(window_data)),
                "firms": int(window_data["Ticker"].nunique()) if not window_data.empty else 0,
                "abnormal_return_mean": float(window_data["abnormal_return"].mean()) if not window_data.empty else np.nan,
                "abnormal_return_dispersion": float(window_data["abnormal_return"].std(ddof=1)) if len(window_data) > 1 else np.nan,
                "market_benchmark_volatility": float(window_data["benchmark_return"].std(ddof=1)) if len(window_data) > 1 else np.nan,
                "diagnostic_only": True,
                "event_retained_for_analysis": True,
            }
        )
    screen = pd.DataFrame(records)
    dispersion_cutoff = screen["abnormal_return_dispersion"].quantile(0.75)
    market_cutoff = screen["market_benchmark_volatility"].quantile(0.75)
    screen["market_wide_volatility_flag"] = (
        screen["abnormal_return_dispersion"].ge(dispersion_cutoff)
        | screen["market_benchmark_volatility"].ge(market_cutoff)
    )
    screen["paper_interpretation"] = np.where(
        screen["market_wide_volatility_flag"],
        "High-volatility diagnostic window; retain event but discuss concurrent-news risk.",
        "No high-volatility flag under the project diagnostic threshold.",
    )
    screen.to_csv(ANALYTICAL_GOLD / "fact_event_confound_screen.csv", index=False)
    return screen


def build_modern_method_applicability() -> pd.DataFrame:
    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv")
    sample = pd.read_csv(ANALYTICAL_GOLD / "fact_overlap_restricted_sample.csv")
    exposure_validation = validate_firm_jurisdiction_exposure()
    firm_jurisdiction_exposure_available = bool(exposure_validation["valid"])
    exposure_failure = str(exposure_validation.get("failure_reason", ""))
    treated_share = sample["pillar_two_in_scope_proxy"].mean()
    balanced_firms = int(monthly.groupby("Ticker")["Month"].nunique().eq(monthly["Month"].nunique()).sum())
    pre_years = sorted(monthly.loc[monthly["Post"].eq(0), "Month"].str[:4].astype(int).unique())
    common_support_firms = int(sample["restricted_sample_flag"].sum())

    rows = [
        {
            "method_id": "bacon_decomposition",
            "method_label": "Bacon decomposition",
            "applicable": False,
            "failure_reason": "No staggered firm-jurisdiction treatment timing; revenue-threshold proxy has uniform timing.",
            "required_data_condition": "Multiple treatment cohorts with early, late, and never/not-yet treated comparisons.",
            "observed_data_condition": "Firm-level jurisdiction exposure file absent; treatment proxy is time-invariant with implementation-year post indicator.",
            "paper_interpretation": "Use as a diagnostic showing why TWFE bad-comparison decomposition cannot identify the design.",
        },
        {
            "method_id": "cs_did",
            "method_label": "Callaway and Sant'Anna DiD",
            "applicable": bool(firm_jurisdiction_exposure_available),
            "failure_reason": "" if firm_jurisdiction_exposure_available else exposure_failure,
            "required_data_condition": "Firm-level cohort timing plus never-treated or not-yet-treated support.",
            "observed_data_condition": json.dumps(exposure_validation, ensure_ascii=False),
            "paper_interpretation": "Report as not applicable until jurisdiction adoption can be linked to firm geographic exposure.",
        },
        {
            "method_id": "sa_did",
            "method_label": "Sun and Abraham event study",
            "applicable": bool(firm_jurisdiction_exposure_available),
            "failure_reason": "" if firm_jurisdiction_exposure_available else exposure_failure,
            "required_data_condition": "Heterogeneous cohort timing with usable pre-period event coefficients.",
            "observed_data_condition": f"Pre-years available: {pre_years}; exposure_validation={json.dumps(exposure_validation, ensure_ascii=False)}",
            "paper_interpretation": "Do not interpret SA coefficients as causal ATT without geographic exposure timing.",
        },
        {
            "method_id": "gsynth",
            "method_label": "Generalized synthetic control",
            "applicable": False,
            "failure_reason": "Interactive fixed-effect estimator did not produce stable ATT under current proxy design.",
            "required_data_condition": "Balanced panel with credible treated/control support and enough pre-periods for factor fit.",
            "observed_data_condition": f"Balanced market-panel firms: {balanced_firms}; treated proxy share: {treated_share:.3f}.",
            "paper_interpretation": "Keep as a diagnostic for sample support and avoid upgrading failed gsynth output to evidence.",
        },
        {
            "method_id": "honestdid",
            "method_label": "HonestDiD sensitivity",
            "applicable": False,
            "failure_reason": "Required valid pre-period event-study coefficients are not credible under failed timing design.",
            "required_data_condition": "Reliable pre-period and post-period dynamic DiD coefficients from a supported design.",
            "observed_data_condition": f"Available pre-period years: {pre_years}; staggered firm-level exposure absent.",
            "paper_interpretation": "Use only to explain sensitivity-data requirements; do not present bounds as causal.",
        },
        {
            "method_id": "grf_blp_rate",
            "method_label": "GRF BLP/RATE",
            "applicable": False,
            "failure_reason": "Treatment proxy and overlap limitations make causal heterogeneity interpretation unsupported.",
            "required_data_condition": "Observed-treatment ignorability, overlap, sufficient sample, and complete covariates.",
            "observed_data_condition": f"Restricted common-support firms: {common_support_firms}; core covariates are incomplete for some firms.",
            "paper_interpretation": "Frame as heterogeneity diagnostic, not as proof of firm-level causal effect variation.",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(ANALYTICAL_GOLD / "fact_modern_method_applicability.csv", index=False)
    return table


def _default_adoption_table() -> pd.DataFrame:
    rows = [
        {
            "jurisdiction_code": code,
            "jurisdiction_name": name,
            "rule_type": "directive",
            "effective_date": "2024-01-01",
            "status": "announced",
            "source_url": EU_COUNCIL_SOURCE,
            "source_type": "eu",
            "notes": "EU-level minimum-tax directive context; not firm-level treatment exposure.",
        }
        for code, name in sorted(EU_MEMBER_CODES.items())
    ]
    return pd.DataFrame(rows, columns=ADOPTION_COLUMNS)


def ensure_jurisdiction_adoption_reference() -> pd.DataFrame:
    path = REFERENCE / "pillar_two_jurisdiction_adoption.csv"
    if path.exists():
        adoption = pd.read_csv(path).fillna("")
    else:
        adoption = _default_adoption_table()
        adoption.to_csv(path, index=False)
    if list(adoption.columns) != ADOPTION_COLUMNS:
        raise ValueError(f"Invalid adoption table schema: {list(adoption.columns)}")
    known = adoption[adoption["status"].ne("unknown")]
    missing_sources = known[known["source_url"].eq("")]
    if not missing_sources.empty:
        raise ValueError("Non-unknown adoption rows must include source_url")
    return adoption


def build_jurisdiction_policy_timing() -> pd.DataFrame:
    adoption = ensure_jurisdiction_adoption_reference()
    cbcr = pd.read_csv(SILVER / "fact_cbcr_jurisdiction_year.csv")
    jurisdiction = cbcr[["REF_AREA", "Reference area", "TIME_PERIOD", "cash_etr", "accrued_etr"]].drop_duplicates()
    jurisdiction = jurisdiction.rename(
        columns={
            "REF_AREA": "jurisdiction_code",
            "Reference area": "cbcr_jurisdiction_name",
        }
    )
    adoption = adoption.rename(columns={"jurisdiction_name": "adoption_jurisdiction_name"})
    timing = jurisdiction.merge(adoption, on="jurisdiction_code", how="left")
    timing["jurisdiction_name"] = timing["cbcr_jurisdiction_name"]
    timing["rule_type"] = timing["rule_type"].fillna("directive")
    timing["effective_date"] = timing["effective_date"].fillna("")
    timing["status"] = timing["status"].fillna("unknown")
    timing["source_url"] = timing["source_url"].fillna("")
    timing["source_type"] = timing["source_type"].fillna("")
    timing["notes"] = timing["notes"].fillna("No project-validated adoption source; retained as CbCR context only.")
    timing["formal_analysis_eligible"] = timing["status"].ne("unknown") & timing["source_type"].isin(
        ["official", "oecd", "eu", "tax_authority"]
    )
    timing["firm_level_treatment_allowed"] = False
    timing["paper_interpretation"] = np.where(
        timing["formal_analysis_eligible"],
        "Jurisdiction-level policy context only; cannot become firm treatment without firm geographic revenue/profit exposure.",
        "CbCR context row with unknown policy adoption status in the project reference table.",
    )
    timing = timing.sort_values(["jurisdiction_code", "TIME_PERIOD"])
    timing.to_csv(ANALYTICAL_GOLD / "fact_jurisdiction_policy_timing.csv", index=False)
    return timing


def _write_not_applicable_r_placeholders() -> None:
    cs_att = ANALYTICAL_GOLD.parent / "statistical" / "r_validation" / "r_cs_did_att.csv"
    cs_event = ANALYTICAL_GOLD.parent / "statistical" / "r_validation" / "r_cs_did_event_study.csv"
    cs_json = ANALYTICAL_GOLD.parent / "statistical" / "r_validation" / "r_cs_did_aggregate.json"
    sa_event = ANALYTICAL_GOLD.parent / "statistical" / "r_validation" / "r_sa_did_event_study.csv"
    message = "not_applicable: missing firm-jurisdiction revenue/profit exposure for staggered cohorts"
    pd.DataFrame([{"status": "not_applicable", "note": message}]).to_csv(cs_att, index=False)
    pd.DataFrame([{"status": "not_applicable", "note": message}]).to_csv(cs_event, index=False)
    write_json(cs_json, {"method": "Callaway and Sant'Anna (2021)", "status": "not_applicable", "note": message})
    pd.DataFrame([{"status": "not_applicable", "note": message}]).to_csv(sa_event, index=False)


def build_design_remediation() -> dict[str, object]:
    ANALYTICAL_GOLD.mkdir(parents=True, exist_ok=True)
    PYTHON_RESULTS.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    sample = build_overlap_restricted_sample()
    restricted = build_restricted_sample_did(sample)
    event_screen = build_event_confound_screen()
    applicability = build_modern_method_applicability()
    adoption = ensure_jurisdiction_adoption_reference()
    timing = build_jurisdiction_policy_timing()
    exposure_validation = validate_firm_jurisdiction_exposure()
    if not exposure_validation["valid"]:
        _write_not_applicable_r_placeholders()
    summary = {
        "restricted_sample_firms": int(sample["restricted_sample_flag"].sum()),
        "restricted_sample_result_status": restricted.get("status"),
        "high_volatility_event_windows": int(event_screen["market_wide_volatility_flag"].sum()),
        "applicable_modern_methods": applicability.loc[applicability["applicable"].eq(True), "method_id"].tolist(),
        "adoption_reference_rows": int(len(adoption)),
        "jurisdiction_policy_timing_rows": int(len(timing)),
        "firm_level_staggered_exposure_generated": bool(exposure_validation["valid"]),
        "firm_jurisdiction_exposure_validation": exposure_validation,
    }
    write_json(ANALYTICAL_GOLD / "design_remediation_summary.json", summary)
    return summary
