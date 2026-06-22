from __future__ import annotations

import json

import pandas as pd

from ..config import ANALYTICAL_GOLD, PYTHON_RESULTS, R_RESULTS, SCOREBOARDS, VERIFIED_RESULTS


def _read_json(name: str) -> dict:
    path = PYTHON_RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_scoreboard() -> pd.DataFrame:
    verified_path = VERIFIED_RESULTS / "verified_model_results.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    python = verified.get("python") or _read_json("python_model_results.json")
    r_path = R_RESULTS / "r_validation_results.json"
    r_results = verified.get("r") or (json.loads(r_path.read_text()) if r_path.exists() else {})
    did = python.get("market_did", {})
    assumptions = did.get("assumption_diagnostics", {})
    scm = python.get("scm", {})
    revenue = python.get("revenue_mechanism", {})
    exposure_events = python.get("exposure_event_study", {}).get("events", [])
    weighted = python.get("weighted_did", {})
    restricted = _read_json("restricted_sample_did.json")
    applicability_path = ANALYTICAL_GOLD / "fact_modern_method_applicability.csv"
    applicability = pd.read_csv(applicability_path) if applicability_path.exists() else pd.DataFrame()
    events = python.get("event_study", {}).get("events", [])
    adjusted_negative = sum(
        event.get("difference", 0) < 0 and event.get("statistically_significant_holm_5pct", False)
        for event in events
    )
    adjusted_positive = sum(
        event.get("difference", 0) > 0 and event.get("statistically_significant_holm_5pct", False)
        for event in events
    )
    robustness = pd.read_csv(PYTHON_RESULTS / "did_robustness_specifications.csv")
    modern_rows = []
    if not applicability.empty:
        for row in applicability.to_dict("records"):
            applicable = str(row.get("applicable", "")).lower() in {"true", "1", "yes"}
            modern_rows.append(
                {
                    "analysis": row.get("method_label", row.get("method_id")),
                    "evidence_role": "modern_method" if applicable else "not_applicable_diagnostic",
                    "estimate": None,
                    "std_error": None,
                    "p_value": None,
                    "assumption_status": "applicable" if applicable else row.get("failure_reason", "not_applicable"),
                    "interpretation": row.get("paper_interpretation", ""),
                }
            )

    rows = [
        {
            "analysis": "Continuous exposure-intensity event study",
            "evidence_role": "main_model",
            "estimate": None,
            "std_error": None,
            "p_value": min((event.get("p_value_holm", 1) for event in exposure_events), default=None),
            "assumption_status": "conditional event exogeneity remains untestable",
            "interpretation": f"{sum(event.get('statistically_significant_holm_5pct', False) for event in exposure_events)} Holm-significant event-window estimates; exposure is a pre-policy proxy.",
        },
        {
            "analysis": "Overlap-weighted monthly-return DiD",
            "evidence_role": "supplementary_validation",
            "estimate": weighted.get("estimate"),
            "std_error": weighted.get("std_error"),
            "p_value": weighted.get("p_value"),
            "assumption_status": weighted.get("assumption_diagnostics", {}).get("overlap_status", "not_run"),
            "interpretation": "Supplementary estimate after overlap weighting; common-support failure prevents a strong causal interpretation.",
        },
        {
            "analysis": "Monthly-return DiD",
            "evidence_role": "legacy_binary_sensitivity",
            "estimate": did.get("estimate"),
            "std_error": did.get("std_error"),
            "p_value": did.get("p_value"),
            "assumption_status": "material_concerns",
            "interpretation": "Stable non-significant association; not a strong causal estimate because pretrends and overlap fail diagnostics.",
        },
        {
            "analysis": "DiD robustness specifications",
            "evidence_role": "robustness",
            "estimate": robustness["estimate"].min(),
            "std_error": None,
            "p_value": robustness["p_value"].min(),
            "assumption_status": f"{len(robustness)} specifications; none significant at 5%",
            "interpretation": "Direction and magnitude vary when group trends and baseline-covariate interactions are added.",
        },
        {
            "analysis": "Event study",
            "evidence_role": "short_window_association",
            "estimate": None,
            "std_error": None,
            "p_value": None,
            "assumption_status": "concurrent-news and cross-firm-dependence risks",
            "interpretation": f"After Holm adjustment: {adjusted_negative} significant negative and {adjusted_positive} significant positive comparisons.",
        },
        {
            "analysis": "SCM",
            "evidence_role": "complementary_proxy_evidence",
            "estimate": scm.get("post_mean_gap"),
            "std_error": None,
            "p_value": scm.get("placebo_p_value"),
            "assumption_status": "no-spillover and donor-validity assumptions untestable",
            "interpretation": "Complete-case SCM; placebo inference is not significant at 5%.",
        },
        {
            "analysis": "Revenue DiD",
            "evidence_role": "exploratory_only",
            "estimate": revenue.get("estimate"),
            "std_error": revenue.get("std_error"),
            "p_value": revenue.get("p_value"),
            "assumption_status": "insufficient pre-periods",
            "interpretation": "Not formal causal evidence.",
        },
        {
            "analysis": "R independent replication",
            "evidence_role": "computational_replication",
            "estimate": r_results.get("market_did", {}).get("estimate"),
            "std_error": r_results.get("market_did", {}).get("std_error"),
            "p_value": r_results.get("market_did", {}).get("p_value"),
            "assumption_status": verified.get("causal_assumption_status", "unknown"),
            "interpretation": verified.get("status", "not_run"),
        },
        *modern_rows,
        {
            "analysis": "Restricted overlap-sample DiD",
            "evidence_role": "diagnostic_sensitivity",
            "estimate": restricted.get("estimate"),
            "std_error": restricted.get("std_error"),
            "p_value": restricted.get("p_value"),
            "assumption_status": restricted.get("status", "not_run"),
            "interpretation": "Common-support restricted result is diagnostic only and is not promoted to the main causal estimate.",
        },
        {
            "analysis": "Modern method applicability matrix",
            "evidence_role": "design_diagnostic",
            "estimate": None,
            "std_error": None,
            "p_value": None,
            "assumption_status": f"{int(applicability['applicable'].sum()) if not applicability.empty else 0} methods currently applicable",
            "interpretation": "Bacon, CS DiD, SA DiD, gsynth, HonestDiD, and GRF are reported with explicit data-condition boundaries.",
        },
        {
            "analysis": "Jurisdiction policy timing context",
            "evidence_role": "context_only",
            "estimate": None,
            "std_error": None,
            "p_value": None,
            "assumption_status": "not firm-level treatment",
            "interpretation": "Jurisdiction adoption context is not converted into firm-level staggered exposure without geographic revenue/profit splits.",
        },
    ]
    scoreboard = pd.DataFrame(rows)
    scoreboard.to_csv(SCOREBOARDS / "scoreboard.csv", index=False)

    parallel = assumptions.get("parallel_trends", {})
    overlap = assumptions.get("covariate_balance_and_overlap", {})
    assumption_rows = [
        {
            "method": "Exposure event study",
            "assumption": "Pre-determined exposure and isolated policy news",
            "diagnostic": "Exposure components are restricted to fiscal years no later than 2022; Holm adjustment is reported.",
            "status": "partially_addressed",
            "limitation": "Concurrent firm-specific news and proxy measurement error remain possible.",
        },
        {
            "method": "Overlap-weighted DiD",
            "assumption": "Observed covariate balance and common support",
            "diagnostic": f"Common-support share={weighted.get('assumption_diagnostics', {}).get('common_support_share', float('nan')):.3f}; weighted max |SMD|={weighted.get('assumption_diagnostics', {}).get('max_absolute_smd_weighted', float('nan')):.3f}",
            "status": "acceptable" if weighted.get("assumption_diagnostics", {}).get("max_absolute_smd_weighted", 1) <= 0.1 else "concern",
            "limitation": "Balance on observed covariates does not establish balance on unobserved determinants.",
        },
        {
            "method": "Overlap-weighted DiD",
            "assumption": "Parallel trends",
            "diagnostic": f"Weighted joint annual pretrend p={weighted.get('assumption_diagnostics', {}).get('joint_pretrend_p_value', float('nan')):.6g}",
            "status": "not_rejected_but_not_proven" if weighted.get("assumption_diagnostics", {}).get("joint_pretrend_p_value", 0) >= 0.05 else "concern",
            "limitation": "A pretrend test has limited power and cannot prove the identifying assumption.",
        },
        {
            "method": "DiD",
            "assumption": "Parallel trends",
            "diagnostic": f"Joint annual pretrend p={parallel.get('joint_pretrend_p_value', float('nan')):.6g}",
            "status": parallel.get("status", "not_run"),
            "limitation": "A diagnostic rejection indicates the baseline DiD cannot support a strong causal interpretation.",
        },
        {
            "method": "DiD",
            "assumption": "Covariate overlap/common support",
            "diagnostic": f"Firm-size common-support share={overlap.get('firm_size_common_support_share', float('nan')):.3f}; max |SMD|={overlap.get('largest_absolute_standardized_difference', float('nan')):.3f}",
            "status": overlap.get("overlap_status", "not_run"),
            "limitation": "Treatment status is strongly associated with firm size and other baseline attributes.",
        },
        {
            "method": "DiD",
            "assumption": "No anticipation",
            "diagnostic": assumptions.get("no_anticipation", {}).get("reason", ""),
            "status": assumptions.get("no_anticipation", {}).get("status", "not_run"),
            "limitation": "Announcements precede the implementation indicator.",
        },
        {
            "method": "Event study",
            "assumption": "Isolated event and valid inference",
            "diagnostic": "Holm and BH adjusted p-values reported across 12 comparisons.",
            "status": "partially_addressed",
            "limitation": "Concurrent news and cross-firm dependence remain possible.",
        },
        {
            "method": "SCM",
            "assumption": "Good pre-fit, valid donor pool, no spillovers",
            "diagnostic": f"Pre-RMSPE={scm.get('pre_rmspe', float('nan')):.4f}; placebo p={scm.get('placebo_p_value', float('nan')):.4f}; max donor weight={scm.get('max_donor_weight', float('nan')):.3f}",
            "status": "partially_addressed",
            "limitation": "Donor validity and spillovers cannot be fully tested.",
        },
        {
            "method": "Revenue DiD",
            "assumption": "Sufficient pre-periods",
            "diagnostic": "Only one well-populated pre-implementation fiscal year.",
            "status": "failed",
            "limitation": "Exploratory descriptive mechanism only.",
        },
        {
            "method": "Modern staggered estimators",
            "assumption": "Observed firm-jurisdiction exposure timing",
            "diagnostic": "fact_modern_method_applicability.csv records missing firm-jurisdiction revenue/profit exposure for CS DiD and SA DiD.",
            "status": "not_applicable",
            "limitation": "Jurisdiction adoption dates alone cannot identify firm-level Pillar Two treatment.",
        },
        {
            "method": "Event study",
            "assumption": "No concurrent market-wide shocks",
            "diagnostic": "fact_event_confound_screen.csv flags high-volatility [-3,+3] trading-day windows without deleting events.",
            "status": "diagnostic_only",
            "limitation": "High-volatility flags support discussion of confounding risk, not event exclusion.",
        },
        {
            "method": "R replication",
            "assumption": "Independent computational reproduction",
            "diagnostic": "Estimate, standard error, p-value, sample size, and pretrend p-value compared.",
            "status": verified.get("status", "not_run"),
            "limitation": "Computational replication does not validate causal assumptions.",
        },
    ]
    assumption_scoreboard = pd.DataFrame(assumption_rows)
    assumption_scoreboard.to_csv(SCOREBOARDS / "assumption_scoreboard.csv", index=False)

    markdown = [
        "# Formal Analysis Scoreboard",
        "",
        "This scoreboard separates numerical replication from causal-assumption credibility.",
        "",
        scoreboard.fillna("").to_markdown(index=False),
        "",
        "## Assumption Diagnostics",
        "",
        assumption_scoreboard.fillna("").to_markdown(index=False),
        "",
        f"Python/R computational replication status: **{verified.get('status', 'not_run')}**.",
        f"Causal-assumption status: **{verified.get('causal_assumption_status', 'unknown')}**.",
        "",
    ]
    (SCOREBOARDS / "scoreboard.md").write_text("\n".join(markdown), encoding="utf-8")
    return scoreboard
