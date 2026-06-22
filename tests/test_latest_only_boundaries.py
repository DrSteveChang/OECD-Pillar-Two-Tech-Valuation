from pathlib import Path

import pytest


def test_models_only_read_etl_contract_tables():
    python_analysis = "\n".join(path.read_text() for path in Path("src/oecd_pillar_two/analysis").glob("*.py"))
    r_analysis = "\n".join(path.read_text() for path in Path("analysis/r").glob("*.R"))
    assert 'pd.read_csv(PYTHON_RESULTS / "event_study_firm_cars.csv")' not in python_analysis
    assert 'data/gold/statistical/python/propensity_scores.csv' not in r_analysis
    assert 'data/gold/statistical/python/event_study_firm_cars.csv' not in r_analysis


@pytest.mark.skipif(
    not Path("outputs/ai_reports").exists(),
    reason="requires locally generated report artifacts",
)
def test_only_latest_ai_report_is_retained():
    reports = sorted(path.name for path in Path("outputs/ai_reports").glob("*.md"))
    assert reports == ["latest_verified_decision_support_report.md"]


def test_latest_scoreboard_and_figures_are_retained():
    assert Path("data/gold/scoreboards/scoreboard.csv").exists()
    assert Path("data/gold/scoreboards/assumption_scoreboard.csv").exists()
    assert Path("data/gold/scoreboards/scoreboard.md").exists()
    assert Path("data/gold/figures/figure_manifest.csv").exists()
