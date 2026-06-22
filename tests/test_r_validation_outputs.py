from pathlib import Path


def test_modern_r_validation_outputs_exist_after_validation_run():
    expected = [
        "r_bacon_decomposition.csv",
        "r_cs_did_att.csv",
        "r_cs_did_aggregate.json",
        "r_cs_did_event_study.csv",
        "r_sa_did_event_study.csv",
        "r_gsynth_att.csv",
        "r_gsynth_factors.csv",
        "r_honest_did_sensitivity.csv",
        "r_grf_blp.csv",
        "r_grf_rates.csv",
    ]

    missing = [
        name for name in expected
        if not (Path("data/gold/statistical/r_validation") / name).exists()
    ]
    assert missing == []
