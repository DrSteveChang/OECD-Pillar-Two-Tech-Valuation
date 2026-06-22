from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.skipif(
    not Path("data/silver").exists(),
    reason="requires the locally regenerated Silver data layer",
)
def test_silver_dimension_primary_keys_and_fact_grains():
    contracts = {
        "dim_firm.csv": ["firm_id"],
        "dim_date.csv": ["date_id"],
        "dim_policy_event.csv": ["event_id"],
        "dim_jurisdiction.csv": ["jurisdiction_id"],
        "fact_firm_financial_year.csv": ["firm_id", "FiscalYear"],
        "fact_firm_revenue_year.csv": ["firm_id", "FiscalYear"],
        "fact_market_daily.csv": ["firm_id", "Date"],
        "fact_market_monthly.csv": ["firm_id", "Month"],
        "fact_cbcr_jurisdiction_year.csv": ["REF_AREA", "COUNTERPART_AREA", "TIME_PERIOD"],
    }
    for name, keys in contracts.items():
        frame = pd.read_csv(Path("data/silver") / name)
        assert not frame.duplicated(keys).any(), f"{name} violates grain {keys}"


def test_gold_contract_tables_exist():
    required = [
        "data/gold/analytical/dim_analysis_firm.csv",
        "data/gold/analytical/dim_model.csv",
        "data/gold/analytical/dim_model_assumption.csv",
        "data/gold/analytical/dim_event_window.csv",
        "data/gold/analytical/fact_scope_classification.csv",
        "data/gold/analytical/fact_exposure_score.csv",
        "data/gold/analytical/fact_event_firm_car.csv",
        "data/gold/statistical/python/fact_model_estimate.csv",
        "data/gold/statistical/python/fact_did_result.csv",
        "data/gold/statistical/python/fact_event_study_result.csv",
        "data/gold/statistical/python/fact_assumption_diagnostic.csv",
        "data/gold/statistical/python/fact_covariate_balance.csv",
        "data/gold/statistical/verified/fact_verified_estimate.csv",
        "data/gold/statistical/verified/fact_verified_assumption_diagnostic.csv",
        "data/gold/statistical/verified/fact_python_r_comparison.csv",
        "data/gold/statistical/verified/dim_verified_model.csv",
    ]
    assert all(Path(path).exists() for path in required)


def test_ai_corpus_uses_bronze_documents_and_verified_gold_only():
    corpus = pd.read_csv("data/serving/ai/rag_corpus.csv")
    model_paths = corpus.loc[corpus["document_type"].str.startswith("model"), "source_path"].astype(str)
    assert not model_paths.str.contains("gold/statistical/python").any()
    assert not model_paths.str.contains("gold/statistical/r_validation").any()
    allowed_remediation = (
        "fact_modern_method_applicability.csv",
        "fact_event_confound_screen.csv",
        "fact_jurisdiction_policy_timing.csv",
        "fact_overlap_restricted_sample.csv",
    )
    allowed = model_paths.str.contains("gold/statistical/verified|gold/scoreboards|gold/figures")
    allowed |= model_paths.map(lambda value: value.endswith(allowed_remediation))
    assert allowed.all()
