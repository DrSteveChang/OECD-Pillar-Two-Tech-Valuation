from pathlib import Path

import pandas as pd
import pytest

from oecd_pillar_two.cleaning.exposure_design import SCOPE_YEARS, build_exposure_design


pytestmark = pytest.mark.skipif(
    not Path("data/bronze/yahoo/corporate_financials.csv").exists(),
    reason="requires locally regenerated Bronze financial statements",
)


def test_four_year_scope_rule_is_strict_and_pre_policy():
    design = build_exposure_design()
    eligible = design[design["eligible_for_main_design"]]
    assert eligible["four_years_observed"].eq(4).all()
    assert eligible.loc[eligible["pillar_two_four_year_scope_proxy"].eq(1), "years_above_threshold"].ge(2).all()
    assert max(SCOPE_YEARS) == 2023
    assert design["FiscalYear"].dropna().le(2022).all()


def test_exposure_score_is_bounded_and_not_post_policy():
    design = build_exposure_design()
    score = design["pre_policy_tax_exposure_score"].dropna()
    assert score.between(0, 1).all()
    assert design["pillar_two_exposure_intensity"].dropna().between(0, 1).all()


def test_weighting_outputs_are_valid_when_present():
    path = "data/gold/statistical/python/propensity_scores.csv"
    try:
        weights = pd.read_csv(path)
    except FileNotFoundError:
        return
    assert weights["propensity_score"].between(0, 1).all()
    assert weights["overlap_weight"].ge(0).all()
