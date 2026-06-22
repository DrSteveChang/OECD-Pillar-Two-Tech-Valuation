from pathlib import Path

import pandas as pd

from oecd_pillar_two.cleaning import design_remediation


ADOPTION_COLUMNS = [
    "jurisdiction_code",
    "jurisdiction_name",
    "rule_type",
    "effective_date",
    "status",
    "source_url",
    "source_type",
    "notes",
]


def test_adoption_reference_table_schema_and_sources():
    path = Path("data/reference/pillar_two_jurisdiction_adoption.csv")
    assert path.exists()
    adoption = pd.read_csv(path).fillna("")

    assert list(adoption.columns) == ADOPTION_COLUMNS
    assert set(adoption["rule_type"]).issubset({"IIR", "QDMTT", "UTPR", "safe_harbor", "directive"})
    assert set(adoption["status"]).issubset({"enacted", "directive_transposed", "announced", "unknown"})
    known = adoption[adoption["status"].ne("unknown")]
    assert known["source_url"].str.startswith("http").all()
    assert set(known["source_type"]).issubset({"official", "oecd", "eu", "tax_authority"})


def test_cbcr_jurisdiction_policy_timing_context_is_nonempty():
    path = Path("data/gold/analytical/fact_jurisdiction_policy_timing.csv")
    assert path.exists()
    timing = pd.read_csv(path)

    required = {
        "jurisdiction_code",
        "jurisdiction_name",
        "TIME_PERIOD",
        "status",
        "source_type",
        "formal_analysis_eligible",
        "firm_level_treatment_allowed",
    }
    assert required.issubset(timing.columns)
    assert not timing.empty
    assert timing["firm_level_treatment_allowed"].eq(False).all()


def test_jurisdiction_policy_timing_joins_by_code_not_display_name(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    silver = tmp_path / "silver"
    analytical = tmp_path / "gold" / "analytical"
    reference.mkdir()
    silver.mkdir()
    analytical.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "jurisdiction_code": "CZE",
                "jurisdiction_name": "Czechia",
                "rule_type": "directive",
                "effective_date": "2024-01-01",
                "status": "announced",
                "source_url": "https://example.com/source",
                "source_type": "eu",
                "notes": "Name differs from CbCR display name.",
            }
        ]
    ).to_csv(reference / "pillar_two_jurisdiction_adoption.csv", index=False)
    pd.DataFrame(
        [
            {
                "REF_AREA": "CZE",
                "Reference area": "Czech Republic",
                "TIME_PERIOD": 2022,
                "cash_etr": 0.1,
                "accrued_etr": 0.2,
            }
        ]
    ).to_csv(silver / "fact_cbcr_jurisdiction_year.csv", index=False)

    monkeypatch.setattr(design_remediation, "REFERENCE", reference)
    monkeypatch.setattr(design_remediation, "SILVER", silver)
    monkeypatch.setattr(design_remediation, "ANALYTICAL_GOLD", analytical)

    timing = design_remediation.build_jurisdiction_policy_timing()

    assert timing.loc[0, "status"] == "announced"
    assert timing.loc[0, "cbcr_jurisdiction_name"] == "Czech Republic"
    assert timing.loc[0, "adoption_jurisdiction_name"] == "Czechia"
