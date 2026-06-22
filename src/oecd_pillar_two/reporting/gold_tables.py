from __future__ import annotations

import json

import pandas as pd

from ..config import ANALYTICAL_GOLD, PYTHON_RESULTS, R_RESULTS, VERIFIED_RESULTS


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def build_python_gold_facts() -> None:
    estimates = []
    for name in ("market_did", "weighted_did", "revenue_mechanism", "scm"):
        value = _json(PYTHON_RESULTS / f"{name}.json")
        if "estimate" in value:
            estimates.append({"model_id": name, "term": value.get("term", "effect"), **{
                key: value.get(key) for key in ("estimate", "std_error", "ci_low", "ci_high", "p_value", "nobs")
            }})
        elif name == "scm":
            estimates.append({"model_id": name, "term": "post_mean_gap", "estimate": value.get("post_mean_gap"), "p_value": value.get("placebo_p_value")})
    events = pd.read_csv(PYTHON_RESULTS / "exposure_event_study.csv")
    if not events.empty:
        event_facts = events.assign(model_id="exposure_event_study", term="pillar_two_exposure_intensity")
        event_facts.to_csv(PYTHON_RESULTS / "fact_event_study_result.csv", index=False)
        estimates.extend(event_facts[["model_id", "term", "estimate", "std_error", "ci_low", "ci_high", "p_value", "nobs"]].to_dict("records"))
    pd.DataFrame(estimates).to_csv(PYTHON_RESULTS / "fact_model_estimate.csv", index=False)
    pd.DataFrame([row for row in estimates if row["model_id"] in {"market_did", "weighted_did", "revenue_mechanism"}]).to_csv(
        PYTHON_RESULTS / "fact_did_result.csv", index=False
    )

    diagnostics = []
    did = _json(PYTHON_RESULTS / "did_assumption_diagnostics.json")
    diagnostics.extend([
        {"model_id": "binary_did", "assumption_id": "parallel_trends", "diagnostic_name": "joint_pretrend_p_value", "diagnostic_value": did.get("parallel_trends", {}).get("joint_pretrend_p_value"), "status": did.get("parallel_trends", {}).get("status")},
        {"model_id": "binary_did", "assumption_id": "common_support", "diagnostic_name": "firm_size_common_support_share", "diagnostic_value": did.get("covariate_balance_and_overlap", {}).get("firm_size_common_support_share"), "status": did.get("covariate_balance_and_overlap", {}).get("overlap_status")},
    ])
    weighted = _json(PYTHON_RESULTS / "weighted_did.json").get("assumption_diagnostics", {})
    for name in ("common_support_share", "effective_sample_size", "max_absolute_smd_weighted", "joint_pretrend_p_value"):
        diagnostics.append({"model_id": "weighted_did", "assumption_id": "common_support" if "support" in name or "smd" in name else "parallel_trends", "diagnostic_name": name, "diagnostic_value": weighted.get(name), "status": weighted.get("overlap_status")})
    pd.DataFrame(diagnostics).to_csv(PYTHON_RESULTS / "fact_assumption_diagnostic.csv", index=False)

    balance = pd.read_csv(PYTHON_RESULTS / "weighted_did_balance.csv")
    balance.insert(0, "model_id", "weighted_did")
    balance["balance_threshold"] = 0.1
    balance["balance_passed"] = balance["smd_overlap_weighted"].abs().le(0.1)
    balance.to_csv(PYTHON_RESULTS / "fact_covariate_balance.csv", index=False)


def build_verified_gold_facts() -> None:
    verified = _json(VERIFIED_RESULTS / "verified_model_results.json")
    python = pd.read_csv(PYTHON_RESULTS / "fact_model_estimate.csv")
    python["python_r_match"] = verified.get("status") == "computationally_replicated"
    python["assumption_status"] = verified.get("causal_assumption_status")
    python["causal_claim_allowed"] = False
    python.to_csv(VERIFIED_RESULTS / "fact_verified_estimate.csv", index=False)
    pd.read_csv(PYTHON_RESULTS / "fact_assumption_diagnostic.csv").to_csv(
        VERIFIED_RESULTS / "fact_verified_assumption_diagnostic.csv", index=False
    )
    models = pd.read_csv(ANALYTICAL_GOLD / "dim_model.csv")
    models["verification_status"] = verified.get("status", "not_run")
    models.to_csv(VERIFIED_RESULTS / "dim_verified_model.csv", index=False)

    r = _json(R_RESULTS / "r_validation_results.json")
    comparisons = []
    for key, value in r.get("python_comparison", {}).items():
        comparisons.append({"comparison_metric": key, "difference": value, "validation_status": verified.get("status")})
    pd.DataFrame(comparisons).to_csv(VERIFIED_RESULTS / "fact_python_r_comparison.csv", index=False)
    summary = {
        "verification_status": verified.get("status", "not_run"),
        "causal_assumption_status": verified.get("causal_assumption_status", "unknown"),
        "verified_estimate_rows": int(len(python)),
        "python_r_comparison_rows": int(len(comparisons)),
    }
    from ..utils import write_json
    write_json(VERIFIED_RESULTS / "validation_summary.json", summary)
