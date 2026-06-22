from __future__ import annotations

import json

from ..config import PYTHON_RESULTS, R_RESULTS, VERIFIED_RESULTS
from ..utils import write_json


def verify_results() -> dict:
    python_results = json.loads((PYTHON_RESULTS / "python_model_results.json").read_text())
    r_path = R_RESULTS / "r_validation_results.json"
    r_results = json.loads(r_path.read_text()) if r_path.exists() else {"status": "not_run"}
    comparisons = []
    if r_results.get("market_did"):
        py = python_results["market_did"]
        r = r_results["market_did"]
        comparisons.append(
            {
                "model": "market_did",
                "estimate_difference": abs(py["estimate"] - r["estimate"]),
                "nobs_match": py["nobs"] == r["nobs"],
            }
        )
    comparison = comparisons[0] if comparisons else {}
    r_comparison = r_results.get("python_comparison", {})
    replicated = bool(
        comparison
        and comparison["estimate_difference"] < 1e-6
        and comparison["nobs_match"]
        and r_comparison.get("market_did_standard_error_difference", 1) < 1e-6
        and r_comparison.get("market_did_p_value_difference", 1) < 1e-6
        and r_comparison.get("pretrend_p_value_difference", 1) < 1e-6
        and r_comparison.get("robustness_max_estimate_difference", 1) < 1e-6
        and r_comparison.get("robustness_max_standard_error_difference", 1) < 1e-6
        and r_comparison.get("event_max_difference_difference", 1) < 1e-6
        and r_comparison.get("event_max_bh_p_value_difference", 1) < 1e-6
        and r_comparison.get("event_max_holm_p_value_difference", 1) < 1e-6
    )
    payload = {
        "status": "computationally_replicated" if replicated else "partial",
        "causal_assumption_status": "material_concerns",
        "python": python_results,
        "r": r_results,
        "comparisons": comparisons,
    }
    write_json(VERIFIED_RESULTS / "verified_model_results.json", payload)
    return payload
