from __future__ import annotations

from .did import run_market_did, run_revenue_mechanism
from .event_study import run_event_study
from .exposure import run_exposure_event_study, run_weighted_did
from .scm import run_scm
from .data_science import run_exploratory_data_science
from ..config import PYTHON_RESULTS
from ..utils import write_json


def run_all_analysis() -> dict:
    results = {
        "event_study": run_event_study(),
        "exposure_event_study": run_exposure_event_study(),
        "weighted_did": run_weighted_did(),
        "market_did": run_market_did(),
        "revenue_mechanism": run_revenue_mechanism(),
        "scm": run_scm(),
        "data_science_exploratory": run_exploratory_data_science(),
    }
    write_json(PYTHON_RESULTS / "python_model_results.json", results)
    return results
