# lineage.py
# Data lineage tracking: Bronze → Silver → Gold → Report
# Records the provenance chain for every Gold-layer analytical output.

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

from ..config import ANALYTICAL_GOLD, LINEAGE, OUTPUTS, R_RESULTS, PYTHON_RESULTS, VERIFIED_RESULTS, FIGURES, SCOREBOARDS


def _hash(path: Path) -> str:
    if not path.exists():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def build_lineage() -> pd.DataFrame:
    records = []

    # Python analysis outputs
    records.extend([
        {
            "output_file": "market_did.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/silver/fact_market_monthly.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/did.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json", "ai_reports"],
            "sha256": _hash(PYTHON_RESULTS / "market_did.json"),
        },
        {
            "output_file": "scm.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/silver/fact_market_monthly.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/scm.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json"],
            "sha256": _hash(PYTHON_RESULTS / "scm.json"),
        },
        {
            "output_file": "event_study.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/gold/analytical/fact_event_firm_car.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/event_study.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json"],
            "sha256": _hash(PYTHON_RESULTS / "event_study.json"),
        },
        {
            "output_file": "exposure_event_study.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/gold/analytical/fact_exposure_score.csv", "data/gold/analytical/fact_event_firm_car.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/exposure.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json"],
            "sha256": _hash(PYTHON_RESULTS / "exposure_event_study.json"),
        },
        {
            "output_file": "weighted_did.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/gold/analytical/fact_exposure_score.csv", "data/silver/fact_market_monthly.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/exposure.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json"],
            "sha256": _hash(PYTHON_RESULTS / "weighted_did.json"),
        },
        {
            "output_file": "revenue_mechanism.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/silver/fact_firm_financial_year.csv"],
            "transform_script": "src/oecd_pillar_two/analysis/did.py",
            "downstream_consumers": ["scoreboard.csv", "verified_model_results.json"],
            "sha256": _hash(PYTHON_RESULTS / "revenue_mechanism.json"),
        },
    ])

    # Design-remediation outputs
    records.extend([
        {
            "output_file": "fact_overlap_restricted_sample.csv",
            "output_path": "data/gold/analytical/",
            "source_data": ["data/gold/analytical/fact_exposure_score.csv"],
            "transform_script": "src/oecd_pillar_two/cleaning/design_remediation.py",
            "downstream_consumers": ["restricted_sample_did.json", "scoreboard.csv", "ai_reports"],
            "sha256": _hash(ANALYTICAL_GOLD / "fact_overlap_restricted_sample.csv"),
        },
        {
            "output_file": "fact_event_confound_screen.csv",
            "output_path": "data/gold/analytical/",
            "source_data": ["data/reference/policy_events.csv", "data/silver/fact_market_daily.csv"],
            "transform_script": "src/oecd_pillar_two/cleaning/design_remediation.py",
            "downstream_consumers": ["figure_manifest.csv", "scoreboard.csv", "ai_reports"],
            "sha256": _hash(ANALYTICAL_GOLD / "fact_event_confound_screen.csv"),
        },
        {
            "output_file": "fact_modern_method_applicability.csv",
            "output_path": "data/gold/analytical/",
            "source_data": ["data/silver/fact_market_monthly.csv", "data/gold/analytical/fact_overlap_restricted_sample.csv"],
            "transform_script": "src/oecd_pillar_two/cleaning/design_remediation.py",
            "downstream_consumers": ["figure_manifest.csv", "scoreboard.csv", "ai_reports"],
            "sha256": _hash(ANALYTICAL_GOLD / "fact_modern_method_applicability.csv"),
        },
        {
            "output_file": "fact_jurisdiction_policy_timing.csv",
            "output_path": "data/gold/analytical/",
            "source_data": ["data/reference/pillar_two_jurisdiction_adoption.csv", "data/silver/fact_cbcr_jurisdiction_year.csv"],
            "transform_script": "src/oecd_pillar_two/cleaning/design_remediation.py",
            "downstream_consumers": ["figure_manifest.csv", "scoreboard.csv", "ai_reports"],
            "sha256": _hash(ANALYTICAL_GOLD / "fact_jurisdiction_policy_timing.csv"),
        },
        {
            "output_file": "restricted_sample_did.json",
            "output_path": "data/gold/statistical/python/",
            "source_data": ["data/gold/analytical/fact_overlap_restricted_sample.csv", "data/silver/fact_market_monthly.csv"],
            "transform_script": "src/oecd_pillar_two/cleaning/design_remediation.py",
            "downstream_consumers": ["scoreboard.csv", "ai_reports"],
            "sha256": _hash(PYTHON_RESULTS / "restricted_sample_did.json"),
        },
    ])

    # R validation outputs (new modern estimators included)
    records.extend([
        {
            "output_file": f"r_{script}_results",
            "output_path": "data/gold/statistical/r_validation/",
            "source_data": ["data/silver/fact_market_monthly.csv"],
            "transform_script": f"analysis/r/{script}",
            "downstream_consumers": ["r_validation_results.json", "scoreboard.csv"],
            "sha256": "",
        }
        for script in [
            "02b_bacon_decomposition.R", "02c_cs_did.R", "02d_sa_did.R",
            "02e_honest_did.R", "04b_gsynth.R", "05_validate_heterogeneity.R",
        ]
    ])

    # Scoreboard
    records.append({
        "output_file": "scoreboard.csv",
        "output_path": "data/gold/scoreboards/",
        "source_data": ["data/gold/statistical/python/*.json", "data/gold/statistical/r_validation/*.json"],
        "transform_script": "src/oecd_pillar_two/reporting/scoreboard.py",
        "downstream_consumers": ["ai_reports", "verified_model_results.json"],
        "sha256": _hash(SCOREBOARDS / "scoreboard.csv"),
    })

    # Verified results
    records.append({
        "output_file": "verified_model_results.json",
        "output_path": "data/gold/statistical/verified/",
        "source_data": [
            "data/gold/statistical/python/python_model_results.json",
            "data/gold/statistical/r_validation/r_validation_results.json",
        ],
        "transform_script": "src/oecd_pillar_two/validation/results.py",
        "downstream_consumers": ["ai_reports", "citation_registry.csv"],
        "sha256": _hash(VERIFIED_RESULTS / "verified_model_results.json"),
    })

    # Figures manifest
    records.append({
        "output_file": "figure_manifest.csv",
        "output_path": "data/gold/figures/",
        "source_data": ["data/gold/statistical/*/*.csv", "data/gold/statistical/*/*.json"],
        "transform_script": "src/oecd_pillar_two/reporting/figures.py",
        "downstream_consumers": ["ai_reports", "citation_registry.csv"],
        "sha256": _hash(FIGURES / "figure_manifest.csv"),
    })

    # Final report
    records.append({
        "output_file": "latest_verified_decision_support_report.md",
        "output_path": "outputs/ai_reports/",
        "source_data": [
            "data/gold/statistical/verified/verified_model_results.json",
            "data/serving/ai/rag_corpus.csv",
            "data/serving/ai/citation_registry.csv",
        ],
        "transform_script": "src/oecd_pillar_two/reporting/report.py",
        "downstream_consumers": ["end_user"],
        "sha256": _hash(OUTPUTS / "ai_reports" / "latest_verified_decision_support_report.md"),
    })

    lineage = pd.DataFrame(records)
    LINEAGE.parent.mkdir(parents=True, exist_ok=True)
    lineage.to_csv(LINEAGE, index=False)

    # Also write as JSON for programmatic consumption
    json_path = LINEAGE.with_suffix(".json")
    lineage.to_json(json_path, orient="records", indent=2)

    return lineage
