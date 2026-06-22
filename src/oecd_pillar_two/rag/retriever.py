# retriever.py
# Dual-mode retrieval: TF-IDF (baseline) + semantic embedding (if available).
# Hybrid mode combines both scores with configurable weights.

from __future__ import annotations

from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..config import AI_SERVING, VECTOR_STORE
from .indexer import _try_embedding_model


@lru_cache(maxsize=1)
def _load_query_embedding_model(expected_dim: int):
    """Load the CPU query encoder once for all retrieval calls in a report run."""
    return _try_embedding_model()


def _try_load_embeddings():
    """Load embedding matrix if available. Returns (embeddings, None) or (None, None)."""
    emb_path = VECTOR_STORE / "embedding_matrix.npy"
    if not emb_path.exists():
        return None, None
    try:
        embeddings = np.load(emb_path)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            return None, None
        model = _load_query_embedding_model(int(embeddings.shape[1]))
        if model is None:
            return None, None
        return embeddings, model
    except Exception:
        return None, None


def retrieve(
    query: str,
    *,
    document_types: set[str] | None = None,
    top_k: int = 5,
    embedding_weight: float = 0.5,
) -> list[dict]:
    """Retrieve relevant documents with hybrid TF-IDF + embedding scoring.

    Args:
        query: Search query string.
        document_types: Optional set of document types to filter by.
        top_k: Number of results to return.
        embedding_weight: Weight for embedding scores in hybrid mode (0 = TF-IDF only).
    """
    corpus = pd.read_csv(AI_SERVING / "rag_corpus.csv").fillna("")
    vectorizer = joblib.load(VECTOR_STORE / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(VECTOR_STORE / "tfidf_matrix.joblib")

    indices = corpus.index
    if document_types:
        indices = corpus.index[corpus["document_type"].isin(document_types)]

    # TF-IDF scores
    tfidf_scores = cosine_similarity(
        vectorizer.transform([query]), tfidf_matrix[indices]
    ).ravel()

    # Embedding scores (if available)
    embeddings = None
    hybrid = False
    if embedding_weight > 0:
        embeddings, emb_model = _try_load_embeddings()
        if embeddings is not None and emb_model is not None:
            query_emb = emb_model.encode(
                [query], normalize_embeddings=True
            )
            emb_scores = cosine_similarity(query_emb, embeddings[indices]).ravel()
            hybrid = True
        else:
            embedding_weight = 0.0

    # Compute final scores
    if hybrid:
        scores = (1 - embedding_weight) * tfidf_scores + embedding_weight * emb_scores
    else:
        scores = tfidf_scores

    # Rank and return top_k
    ranked = indices[scores.argsort()[::-1][:top_k]]
    results = corpus.loc[ranked].copy()
    results["score"] = sorted(scores, reverse=True)[: len(results)]
    results["tfidf_score"] = tfidf_scores[scores.argsort()[::-1][:top_k]]
    if hybrid:
        results["embedding_score"] = emb_scores[scores.argsort()[::-1][:top_k]]
        results["retrieval_mode"] = "hybrid"
    else:
        results["retrieval_mode"] = "tfidf"

    return results.to_dict("records")
