from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
SERVING = DATA / "serving"
REFERENCE = DATA / "reference"
QUARANTINE = DATA / "quarantine"
PYTHON_RESULTS = GOLD / "statistical" / "python"
R_RESULTS = GOLD / "statistical" / "r_validation"
VERIFIED_RESULTS = GOLD / "statistical" / "verified"
DATA_SCIENCE_RESULTS = GOLD / "data_science" / "exploratory"
ANALYTICAL_GOLD = GOLD / "analytical"
FIGURES = GOLD / "figures"
SCOREBOARDS = GOLD / "scoreboards"
AI_SERVING = SERVING / "ai"
VECTOR_STORE = AI_SERVING / "vector_store"
OUTPUTS = ROOT / "outputs"
# New algorithm output paths for modern estimators
LINEAGE = REFERENCE / "data_lineage.csv"
LITERATURE_METADATA = BRONZE / "literature" / "literature_metadata.json"
EVIDENCE_GRAPH = AI_SERVING / "evidence_graph.json"

def load_config() -> dict:
    with (ROOT / "config" / "project.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directories() -> None:
    for path in (
        BRONZE / "yahoo",
        BRONZE / "sec",
        BRONZE / "oecd",
        BRONZE / "literature",
        REFERENCE,
        SILVER,
        PYTHON_RESULTS,
        R_RESULTS,
        VERIFIED_RESULTS,
        DATA_SCIENCE_RESULTS,
        ANALYTICAL_GOLD,
        FIGURES,
        SCOREBOARDS,
        AI_SERVING,
        VECTOR_STORE,
        OUTPUTS / "ai_reports",
    ):
        path.mkdir(parents=True, exist_ok=True)
