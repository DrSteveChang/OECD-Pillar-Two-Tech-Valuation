import json
import sys
import types

import pandas as pd
import pytest

from oecd_pillar_two.rag import indexer


def test_embedding_model_loads_on_supported_intel_mac_stack(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(indexer.metadata, "version", lambda package: "2.2.2")

    model = indexer._try_embedding_model()

    assert model is not None
    assert model.model_name == "all-MiniLM-L6-v2"


def test_build_vector_index_writes_fallback_status(tmp_path, monkeypatch):
    serving = tmp_path / "serving" / "ai"
    vector_store = serving / "vector_store"
    serving.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "citation_id": "MODEL-1111111111",
                "document_type": "model_result",
                "source": "x",
                "source_path": "data/gold/scoreboards/scoreboard.csv",
                "source_url": "",
                "page": "",
                "section": "rows",
                "ticker": "",
                "text": "Pillar Two valuation evidence and model result text",
                "sha256": "abc",
            }
        ]
    ).to_csv(serving / "rag_corpus.csv", index=False)

    monkeypatch.setattr(indexer, "AI_SERVING", serving)
    monkeypatch.setattr(indexer, "VECTOR_STORE", vector_store)
    monkeypatch.setattr(indexer, "_try_embedding_model", lambda: None)

    result = indexer.build_vector_index()

    status = json.loads((vector_store / "index_status.json").read_text(encoding="utf-8"))
    assert result["embedding_status"] == "fallback_tfidf_only"
    assert status["embedding_status"] == "fallback_tfidf_only"


def test_formal_index_build_rejects_silent_embedding_fallback(tmp_path, monkeypatch):
    serving = tmp_path / "serving" / "ai"
    vector_store = serving / "vector_store"
    serving.mkdir(parents=True)
    pd.DataFrame(
        [{
            "citation_id": "MODEL-1111111111",
            "document_type": "model_result",
            "source": "x",
            "source_path": "data/gold/scoreboards/scoreboard.csv",
            "source_url": "",
            "page": "",
            "section": "rows",
            "ticker": "",
            "text": "Pillar Two valuation evidence",
            "sha256": "abc",
        }]
    ).to_csv(serving / "rag_corpus.csv", index=False)
    monkeypatch.setattr(indexer, "AI_SERVING", serving)
    monkeypatch.setattr(indexer, "VECTOR_STORE", vector_store)
    monkeypatch.setattr(indexer, "_try_embedding_model", lambda: None)

    with pytest.raises(RuntimeError, match="embedding"):
        indexer.build_vector_index(require_embeddings=True)
