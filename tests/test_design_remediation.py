import json
from pathlib import Path

import pandas as pd


def test_overlap_restricted_sample_exists_and_keeps_treated_control_boundary():
    path = Path("data/gold/analytical/fact_overlap_restricted_sample.csv")
    assert path.exists()
    sample = pd.read_csv(path)

    required = {
        "Ticker",
        "pillar_two_in_scope_proxy",
        "eligible_for_main_design",
        "core_covariate_count",
        "propensity_score",
        "within_common_support",
        "restricted_sample_flag",
        "exclusion_reason",
    }
    assert required.issubset(sample.columns)
    assert set(sample.loc[sample["eligible_for_main_design"].eq(True), "pillar_two_in_scope_proxy"]) == {0, 1}
    assert sample["core_covariate_count"].ge(0).all()
    assert sample["restricted_sample_flag"].isin([True, False]).all()


def test_restricted_sample_attrition_and_result_are_diagnostic_only():
    result_path = Path("data/gold/statistical/python/restricted_sample_did.json")
    assert result_path.exists()
    result = json.loads(result_path.read_text())

    assert result["evidence_role"] == "diagnostic_only"
    assert result["causal_upgrade"] is False
    assert result["attrition_counts"]
    assert {"initial_firms", "eligible_main_design", "core_covariates", "common_support"}.issubset(
        result["attrition_counts"]
    )

    sample = pd.read_csv("data/gold/analytical/fact_overlap_restricted_sample.csv")
    post_policy_components = [column for column in sample.columns if "Post" in column or "post_policy" in column]
    assert post_policy_components == []


def test_modern_method_applicability_table_covers_expected_methods():
    path = Path("data/gold/analytical/fact_modern_method_applicability.csv")
    assert path.exists()
    table = pd.read_csv(path)

    expected = {
        "bacon_decomposition",
        "cs_did",
        "sa_did",
        "gsynth",
        "honestdid",
        "grf_blp_rate",
    }
    assert expected.issubset(set(table["method_id"]))
    assert table["required_data_condition"].notna().all()
    assert table["observed_data_condition"].notna().all()
    assert table["paper_interpretation"].notna().all()
