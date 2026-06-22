from pathlib import Path

import pandas as pd

from oecd_pillar_two.cleaning import design_remediation


def test_no_firm_level_staggered_exposure_without_geographic_split():
    exposure_path = Path("data/gold/analytical/fact_firm_jurisdiction_pillar_two_exposure.csv")
    applicability_path = Path("data/gold/analytical/fact_modern_method_applicability.csv")

    assert not exposure_path.exists()
    applicability = pd.read_csv(applicability_path)
    staggered = applicability[applicability["method_id"].isin(["cs_did", "sa_did"])]

    assert not staggered.empty
    assert staggered["applicable"].eq(False).all()
    assert staggered["failure_reason"].str.contains("firm-jurisdiction", case=False).all()


def test_cs_sa_outputs_do_not_report_fake_att_when_not_applicable():
    for path in [
        Path("data/gold/statistical/r_validation/r_cs_did_att.csv"),
        Path("data/gold/statistical/r_validation/r_sa_did_event_study.csv"),
    ]:
        assert path.exists()
        table = pd.read_csv(path)
        serialized = table.astype(str).agg(" ".join, axis=1).str.cat(sep=" ").lower()
        assert "placeholder" in serialized or "not_applicable" in serialized


def test_invalid_existing_firm_jurisdiction_exposure_is_not_applicable(tmp_path):
    exposure = tmp_path / "fact_firm_jurisdiction_pillar_two_exposure.csv"

    pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "jurisdiction_code": "IRL",
                "FiscalYear": 2024,
                "exposure_weight": 0.5,
                "first_treat_year": 2024,
                "treated": 1,
            }
        ]
    ).to_csv(exposure, index=False)

    validation = design_remediation.validate_firm_jurisdiction_exposure(exposure)

    assert validation["valid"] is False
    assert "at least two treatment cohorts" in validation["failure_reason"].lower()


def test_empty_or_malformed_firm_jurisdiction_exposure_is_not_applicable(tmp_path):
    empty = tmp_path / "empty.csv"
    missing_columns = tmp_path / "missing_columns.csv"
    empty.write_text("", encoding="utf-8")
    pd.DataFrame([{"Ticker": "AAPL", "treated": 1}]).to_csv(missing_columns, index=False)

    empty_result = design_remediation.validate_firm_jurisdiction_exposure(empty)
    missing_result = design_remediation.validate_firm_jurisdiction_exposure(missing_columns)

    assert empty_result["valid"] is False
    assert missing_result["valid"] is False
    assert "missing required columns" in missing_result["failure_reason"].lower()


def test_modern_applicability_rejects_invalid_existing_exposure_file(tmp_path, monkeypatch):
    analytical = tmp_path / "gold" / "analytical"
    silver = tmp_path / "silver"
    analytical.mkdir(parents=True)
    silver.mkdir(parents=True)

    pd.DataFrame(
        [
            {"Ticker": "A", "pillar_two_in_scope_proxy": 1, "restricted_sample_flag": False},
            {"Ticker": "B", "pillar_two_in_scope_proxy": 0, "restricted_sample_flag": False},
        ]
    ).to_csv(analytical / "fact_overlap_restricted_sample.csv", index=False)
    pd.DataFrame(
        [
            {"Ticker": "A", "Month": "2023-01", "Post": 0},
            {"Ticker": "B", "Month": "2023-01", "Post": 0},
        ]
    ).to_csv(silver / "fact_market_monthly.csv", index=False)
    pd.DataFrame(
        [
            {
                "Ticker": "A",
                "jurisdiction_code": "IRL",
                "FiscalYear": 2024,
                "exposure_weight": 0.5,
                "first_treat_year": 2024,
                "treated": 1,
            }
        ]
    ).to_csv(analytical / "fact_firm_jurisdiction_pillar_two_exposure.csv", index=False)

    monkeypatch.setattr(design_remediation, "ANALYTICAL_GOLD", analytical)
    monkeypatch.setattr(design_remediation, "SILVER", silver)

    applicability = design_remediation.build_modern_method_applicability()
    staggered = applicability[applicability["method_id"].isin(["cs_did", "sa_did"])]

    assert staggered["applicable"].eq(False).all()
    assert staggered["failure_reason"].str.contains("at least two treatment cohorts", case=False).all()
