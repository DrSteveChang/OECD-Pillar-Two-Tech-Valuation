import json
from pathlib import Path

import pytest

from oecd_pillar_two.reporting import report


pytestmark = pytest.mark.skipif(
    not Path("data/serving/ai/vector_store/index_status.json").exists(),
    reason="requires the locally rebuilt hybrid retrieval index",
)


def test_report_uses_top_k_retrieval_for_literature_and_local_evidence(monkeypatch):
    calls = []
    real_retrieve = report.retrieve

    def recording_retrieve(query, *, document_types=None, top_k=5, embedding_weight=0.5):
        calls.append(
            {
                "query": query,
                "document_types": set(document_types or set()),
                "top_k": top_k,
            }
        )
        return real_retrieve(
            query,
            document_types=document_types,
            top_k=top_k,
            embedding_weight=embedding_weight,
        )

    monkeypatch.setattr(report, "retrieve", recording_retrieve)
    text = report.generate_evidence_report()

    assert any(call["document_types"] == {"literature"} for call in calls)
    assert any(
        {"model_result", "model_table"}.issubset(call["document_types"])
        for call in calls
    )
    assert all(call["top_k"] > 0 for call in calls)
    assert "## Retrieval-Grounded Evidence" in text
    assert "## Retrieval Audit Register" in text
    assert "## Report Metadata and Retrieval Status" in text
    assert "## Executive Interpretation" in text

    audit_path = Path("outputs/ai_reports/retrieval_audit.json")
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["retrieval_invoked"] is True
    assert audit["query_count"] == len(calls)
    assert audit["embedding_status"] == "built"
    assert audit["retrieval_mode"] == "hybrid"
    assert audit["embedding_model"] == "all-MiniLM-L6-v2"
    assert all(run["top_k"] > 0 and run["returned_citation_ids"] for run in audit["runs"])
    assert all(run["retrieval_mode"] == "hybrid" for run in audit["runs"])
