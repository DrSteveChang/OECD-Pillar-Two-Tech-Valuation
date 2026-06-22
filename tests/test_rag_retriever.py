import pandas as pd
import numpy as np

from oecd_pillar_two.rag import indexer, retriever


def test_retrieve_uses_tfidf_when_embeddings_are_absent(tmp_path, monkeypatch):
    serving = tmp_path / "serving" / "ai"
    vector_store = serving / "vector_store"
    serving.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "citation_id": "PDF-1111111111",
                "document_type": "literature",
                "source": "x.pdf",
                "source_path": "data/bronze/literature/x.pdf",
                "source_url": "",
                "page": 1,
                "section": "test",
                "ticker": "",
                "text": "Pillar Two global minimum tax valuation evidence",
                "sha256": "abc",
            }
        ]
    ).to_csv(serving / "rag_corpus.csv", index=False)

    monkeypatch.setattr(indexer, "AI_SERVING", serving)
    monkeypatch.setattr(indexer, "VECTOR_STORE", vector_store)
    monkeypatch.setattr(indexer, "_try_embedding_model", lambda: None)
    monkeypatch.setattr(retriever, "AI_SERVING", serving)
    monkeypatch.setattr(retriever, "VECTOR_STORE", vector_store)

    indexer.build_vector_index()
    results = retriever.retrieve("minimum tax", document_types={"literature"})

    assert results[0]["citation_id"] == "PDF-1111111111"


def test_hybrid_retrieval_loads_query_model_once_per_process(tmp_path, monkeypatch):
    serving = tmp_path / "serving" / "ai"
    vector_store = serving / "vector_store"
    serving.mkdir(parents=True)
    pd.DataFrame([
        {
            "citation_id": "PDF-1111111111", "document_type": "literature",
            "source": "a.pdf", "source_path": "a.pdf", "source_url": "",
            "page": 1, "section": "a", "ticker": "", "text": "minimum tax evidence",
            "sha256": "abc",
        },
        {
            "citation_id": "PDF-2222222222", "document_type": "literature",
            "source": "b.pdf", "source_path": "b.pdf", "source_url": "",
            "page": 2, "section": "b", "ticker": "", "text": "unrelated market text",
            "sha256": "def",
        },
    ]).to_csv(serving / "rag_corpus.csv", index=False)
    monkeypatch.setattr(indexer, "AI_SERVING", serving)
    monkeypatch.setattr(indexer, "VECTOR_STORE", vector_store)
    monkeypatch.setattr(indexer, "_try_embedding_model", lambda: None)
    monkeypatch.setattr(retriever, "AI_SERVING", serving)
    monkeypatch.setattr(retriever, "VECTOR_STORE", vector_store)
    indexer.build_vector_index()
    np.save(vector_store / "embedding_matrix.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))

    calls = []

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 0.0]])

    monkeypatch.setattr(
        retriever,
        "_try_embedding_model",
        lambda: calls.append("load") or FakeModel(),
    )
    retriever._load_query_embedding_model.cache_clear()

    first = retriever.retrieve("minimum tax")
    second = retriever.retrieve("minimum tax")

    assert first[0]["retrieval_mode"] == "hybrid"
    assert second[0]["retrieval_mode"] == "hybrid"
    assert calls == ["load"]
