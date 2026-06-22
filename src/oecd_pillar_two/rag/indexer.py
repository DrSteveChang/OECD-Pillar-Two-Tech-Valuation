# indexer.py
# Vector index builder — dual-mode: TF-IDF (baseline) + semantic embedding (optional).
# TF-IDF always runs; semantic embeddings are attempted if sentence-transformers is installed.

from __future__ import annotations

from importlib import metadata
import hashlib
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import AI_SERVING, VECTOR_STORE

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_REVISION = os.environ.get(
    "RAG_EMBEDDING_REVISION", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "8"))


def _try_embedding_model():
    """Load the pinned CPU embedding model when the runtime supports it."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
            revision=EMBEDDING_REVISION,
        )
        return model
    except Exception:
        return None


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def build_vector_index(*, require_embeddings: bool = False) -> dict:
    corpus = pd.read_csv(AI_SERVING / "rag_corpus.csv").fillna("")
    VECTOR_STORE.mkdir(parents=True, exist_ok=True)
    if corpus.empty:
        result = {"status": "empty", "documents": 0, "embedding_status": "not_built_empty_corpus"}
        (VECTOR_STORE / "index_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # 1. TF-IDF baseline (always runs)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus["text"])
    joblib.dump(vectorizer, VECTOR_STORE / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, VECTOR_STORE / "tfidf_matrix.joblib")
    corpus.drop(columns=["text"]).to_csv(VECTOR_STORE / "document_index.csv", index=False)

    result = {
        "status": "built",
        "documents": len(corpus),
        "tfidf_features": matrix.shape[1],
        "embedding_model": None,
        "embedding_dim": None,
        "embedding_status": "fallback_tfidf_only",
        "embedding_backend": None,
        "embedding_revision": None,
        "embedding_matrix_sha256": None,
        "embedding_encode_seconds": None,
        "hybrid_retrieval_active": False,
        "runtime_versions": {
            package: _package_version(package)
            for package in ("numpy", "torch", "transformers", "sentence-transformers")
        },
    }

    # 2. Semantic embedding. Exploratory callers may fall back; formal report builds may not.
    embedding_model = _try_embedding_model()
    if embedding_model is not None:
        try:
            texts = corpus["text"].tolist()
            started = time.perf_counter()
            embeddings = embedding_model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or len(embeddings) != len(corpus):
                raise ValueError("Embedding matrix shape does not match the corpus")
            if not np.isfinite(embeddings).all():
                raise ValueError("Embedding matrix contains non-finite values")
            matrix_path = VECTOR_STORE / "embedding_matrix.npy"
            np.save(matrix_path, embeddings)
            result["embedding_model"] = EMBEDDING_MODEL
            result["embedding_dim"] = int(embeddings.shape[1])
            result["embedding_status"] = "built"
            result["embedding_backend"] = "sentence-transformers-cpu"
            result["embedding_revision"] = EMBEDDING_REVISION
            result["embedding_matrix_sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
            result["embedding_encode_seconds"] = round(time.perf_counter() - started, 3)
            result["hybrid_retrieval_active"] = True
        except Exception as exc:
            result["embedding_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["embedding_error"] = "Embedding model could not be loaded"

    (VECTOR_STORE / "index_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if require_embeddings and result["embedding_status"] != "built":
        raise RuntimeError(
            "Formal report generation requires an active embedding index: "
            + result.get("embedding_error", "unknown embedding error")
        )
    return result
