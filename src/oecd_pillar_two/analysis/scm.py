from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import SILVER, PYTHON_RESULTS
from ..utils import write_json


def _weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    count = donors.shape[1]
    result = minimize(
        lambda weights: np.sum((target - donors @ weights) ** 2),
        np.repeat(1 / count, count),
        bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"SCM optimization failed: {result.message}")
    return result.x


def run_scm() -> dict:
    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv")
    monthly = monthly[monthly["trading_days"] >= 10].copy()
    pivot = monthly.pivot(index="Month", columns="Ticker", values="abnormal_return").sort_index()
    membership = monthly.groupby("Ticker")["pillar_two_in_scope_proxy"].first()
    treated = membership[membership.eq(1)].index.intersection(pivot.columns)
    controls = membership[membership.eq(0)].index.intersection(pivot.columns)
    complete_controls = pivot[controls].columns[pivot[controls].notna().all()]
    complete_treated = pivot[treated].columns[pivot[treated].notna().all()]
    treated_series = pivot[complete_treated].mean(axis=1)
    donor_frame = pivot[complete_controls]
    pre = pivot.index < "2024-01"
    weights = _weights(treated_series[pre].to_numpy(), donor_frame.loc[pre].to_numpy())
    synthetic = donor_frame.to_numpy() @ weights
    result = pd.DataFrame(
        {
            "Month": pivot.index,
            "treated_abnormal_return": treated_series.to_numpy(),
            "synthetic_abnormal_return": synthetic,
        }
    )
    result["gap"] = result["treated_abnormal_return"] - result["synthetic_abnormal_return"]
    result.to_csv(PYTHON_RESULTS / "scm_monthly_trajectory.csv", index=False)
    pre_gap = result.loc[result["Month"] < "2024-01", "gap"]
    post_gap = result.loc[result["Month"] >= "2024-01", "gap"]
    treated_ratio = float(np.mean(post_gap**2) / np.mean(pre_gap**2))
    placebo_records = []
    for target in complete_controls:
        other = complete_controls.drop(target)
        target_series = donor_frame[target].to_numpy()
        other_frame = donor_frame[other]
        try:
            placebo_weights = _weights(target_series[pre], other_frame.loc[pre].to_numpy())
        except RuntimeError:
            continue
        placebo_gap = target_series - other_frame.to_numpy() @ placebo_weights
        ratio = float(np.mean(placebo_gap[~pre] ** 2) / np.mean(placebo_gap[pre] ** 2))
        if np.isfinite(ratio):
            placebo_records.append({"ticker": target, "post_pre_mspe_ratio": ratio})
    placebo_ratios = [record["post_pre_mspe_ratio"] for record in placebo_records]
    pd.DataFrame(
        placebo_records
        + [{"ticker": "TREATED_PROXY", "post_pre_mspe_ratio": treated_ratio}]
    ).to_csv(PYTHON_RESULTS / "scm_placebo_ratios.csv", index=False)
    sensitivity = []
    top_donors = [
        ticker for ticker, weight in sorted(zip(complete_controls, weights), key=lambda item: -item[1])[:5]
        if weight > 0
    ]
    for excluded in top_donors:
        reduced = donor_frame.drop(columns=excluded)
        reduced_weights = _weights(treated_series[pre].to_numpy(), reduced.loc[pre].to_numpy())
        reduced_gap = treated_series.to_numpy() - reduced.to_numpy() @ reduced_weights
        sensitivity.append(
            {
                "excluded_donor": excluded,
                "post_mean_gap": float(np.mean(reduced_gap[~pre])),
                "pre_rmspe": float(np.sqrt(np.mean(reduced_gap[pre] ** 2))),
            }
        )
    pd.DataFrame(sensitivity).to_csv(PYTHON_RESULTS / "scm_leave_one_out_sensitivity.csv", index=False)
    payload = {
        "method": "aggregate synthetic control on monthly abnormal returns",
        "treated_firms": len(complete_treated),
        "donor_firms": len(complete_controls),
        "pre_rmspe": float(np.sqrt(np.mean(pre_gap**2))),
        "post_rmspe": float(np.sqrt(np.mean(post_gap**2))),
        "post_mean_gap": float(post_gap.mean()),
        "post_pre_mspe_ratio": treated_ratio,
        "placebo_p_value": float(np.mean(np.array(placebo_ratios) >= treated_ratio)),
        "placebo_count": len(placebo_ratios),
        "top_weights": [
            {"ticker": ticker, "weight": float(weight)}
            for ticker, weight in sorted(zip(complete_controls, weights), key=lambda item: -item[1])[:10]
        ],
        "max_donor_weight": float(np.max(weights)),
        "donor_weight_hhi": float(np.sum(weights**2)),
        "leave_one_out_post_gap_min": float(min(item["post_mean_gap"] for item in sensitivity)),
        "leave_one_out_post_gap_max": float(max(item["post_mean_gap"] for item in sensitivity)),
        "formal_causal_claim": False,
        "assumptions_and_limitations": {
            "pre_treatment_fit": "SCM credibility depends on the synthetic control reproducing the treated proxy before implementation.",
            "no_spillovers": "Control firms may still be indirectly affected by Pillar Two or correlated regulatory shocks.",
            "donor_pool": "Only firms with complete monthly abnormal-return histories are used; missing returns are not imputed as zero.",
            "aggregate_treatment": "The treated outcome is a cohort average, which can conceal firm-level heterogeneity.",
        },
        "limitation": "Treatment is a revenue-threshold exposure proxy, not observed top-up tax liability.",
    }
    write_json(PYTHON_RESULTS / "scm.json", payload)
    return payload
