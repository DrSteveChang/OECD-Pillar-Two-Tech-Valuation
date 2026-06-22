from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("data/bronze").exists(),
    reason="requires locally regenerated Bronze, Silver, and vector-index directories",
)
def test_medallion_directories_are_physically_separated():
    required = [
        Path("data/bronze"),
        Path("data/silver"),
        Path("data/gold/statistical/python"),
        Path("data/gold/statistical/r_validation"),
        Path("data/gold/statistical/verified"),
        Path("data/gold/data_science/exploratory"),
        Path("data/gold/figures"),
        Path("data/gold/scoreboards"),
        Path("data/serving/ai/vector_store"),
    ]
    assert all(path.is_dir() for path in required)
    assert not Path("data/raw").exists()
    assert not Path("data/processed").exists()
    assert not Path("data/vector_store").exists()
    assert not Path("outputs/formal_results").exists()


def test_python_r_and_verified_results_have_distinct_ownership():
    assert Path("data/gold/statistical/python/python_model_results.json").exists()
    assert Path("data/gold/statistical/r_validation/r_validation_results.json").exists()
    assert Path("data/gold/statistical/verified/verified_model_results.json").exists()
    assert not Path("data/gold/statistical/python/r_validation_results.json").exists()
    assert not Path("data/gold/statistical/python/verified_model_results.json").exists()


@pytest.mark.skipif(
    not Path("data/serving/ai/vector_store/document_index.csv").exists(),
    reason="requires the locally rebuilt retrieval index",
)
def test_ai_serving_assets_are_not_silver_analysis_inputs():
    assert Path("data/serving/ai/rag_corpus.csv").exists()
    assert Path("data/serving/ai/citation_registry.csv").exists()
    assert Path("data/serving/ai/vector_store/document_index.csv").exists()
    assert not Path("data/silver/rag_corpus.csv").exists()
    assert not Path("data/silver/citation_registry.csv").exists()


def test_consumers_respect_layer_boundaries():
    analysis = "\n".join(path.read_text() for path in Path("src/oecd_pillar_two/analysis").glob("*.py"))
    rag_consumers = "\n".join(
        path.read_text()
        for path in [
            Path("src/oecd_pillar_two/rag/corpus.py"),
            Path("src/oecd_pillar_two/rag/indexer.py"),
            Path("src/oecd_pillar_two/rag/retriever.py"),
        ]
    )
    assert "BRONZE" not in analysis
    assert "QUARANTINE" not in analysis
    assert "SILVER" not in rag_consumers
