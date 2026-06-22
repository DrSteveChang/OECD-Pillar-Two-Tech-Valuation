from __future__ import annotations

import pandas as pd

from ..config import ANALYTICAL_GOLD, DATA_SCIENCE_RESULTS
from ..utils import write_json


FEATURES = ["threshold_distance_log", "Firm_Size", "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio", "pre_policy_tax_exposure_score"]


def run_exploratory_data_science() -> dict:
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")
    profile = design[["Ticker", "eligible_for_main_design", "pillar_two_four_year_scope_proxy", *FEATURES]].copy()
    profile.to_csv(DATA_SCIENCE_RESULTS / "fact_feature_profile.csv", index=False)
    correlation = design[FEATURES].corr().rename_axis("feature").reset_index()
    correlation.to_csv(DATA_SCIENCE_RESULTS / "fact_feature_correlation.csv", index=False)
    payload = {
        "evidence_role": "exploratory_only",
        "causal_claim_allowed": False,
        "features": FEATURES,
        "eligible_firms": int(design["eligible_for_main_design"].sum()),
        "limitation": "Feature profiles and correlations describe observed attributes; they do not estimate causal effects.",
    }
    write_json(DATA_SCIENCE_RESULTS / "exploratory_data_science_metadata.json", payload)
    return payload
