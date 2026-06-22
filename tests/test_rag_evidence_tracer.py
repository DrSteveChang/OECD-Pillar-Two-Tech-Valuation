import json
from pathlib import Path

import pandas as pd

from oecd_pillar_two.rag import evidence_tracer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_claim_traceability_rejects_missing_citation(tmp_path, monkeypatch):
    serving = tmp_path / "serving" / "ai"
    verified = tmp_path / "gold" / "statistical" / "verified"
    serving.mkdir(parents=True)
    verified.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "citation_id": "MODEL-1111111111",
                "document_type": "model_result",
                "source": "verified_model_results.json",
            }
        ]
    ).to_csv(serving / "citation_registry.csv", index=False)
    _write_json(
        verified / "verified_model_results.json",
        {"python": {"market_did": {"estimate": 0.123}}},
    )
    report = tmp_path / "report.md"
    report.write_text("estimate 999999 [MODEL-0000000000]", encoding="utf-8")

    monkeypatch.setattr(evidence_tracer, "AI_SERVING", serving)
    monkeypatch.setattr(evidence_tracer, "VERIFIED_RESULTS", verified)

    result = evidence_tracer.validate_claim_traceability(str(report))

    assert result["valid"] is False
    assert result["missing_citations"] == ["MODEL-0000000000"]
    assert result["untraceable_claims"]


def test_evidence_graph_has_no_dangling_edges(tmp_path, monkeypatch):
    python_results = tmp_path / "gold" / "statistical" / "python"
    r_results = tmp_path / "gold" / "statistical" / "r_validation"
    verified = tmp_path / "gold" / "statistical" / "verified"
    evidence_graph = tmp_path / "serving" / "ai" / "evidence_graph.json"

    for name in [
        "market_did.json",
        "scm.json",
        "event_study.json",
        "exposure_event_study.json",
        "weighted_did.json",
        "revenue_mechanism.json",
    ]:
        _write_json(python_results / name, {"status": "ok"})
    _write_json(r_results / "r_cs_did_aggregate.json", {"status": "ok"})
    for name in [
        "r_gsynth_att.csv",
        "r_bacon_decomposition.csv",
        "r_honest_did_sensitivity.csv",
        "r_grf_blp.csv",
    ]:
        (r_results / name).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"note": "placeholder"}]).to_csv(r_results / name, index=False)
    _write_json(verified / "verified_model_results.json", {"status": "ok"})

    monkeypatch.setattr(evidence_tracer, "PYTHON_RESULTS", python_results)
    monkeypatch.setattr(evidence_tracer, "R_RESULTS", r_results)
    monkeypatch.setattr(evidence_tracer, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(evidence_tracer, "EVIDENCE_GRAPH", evidence_graph)

    graph = evidence_tracer.build_evidence_graph()

    node_ids = {node["node_id"] for node in graph["nodes"]}
    dangling = [
        edge for edge in graph["edges"]
        if edge["from"] not in node_ids or edge["to"] not in node_ids
    ]
    assert dangling == []


def test_evidence_graph_traces_remediation_tables(tmp_path, monkeypatch):
    python_results = tmp_path / "gold" / "statistical" / "python"
    r_results = tmp_path / "gold" / "statistical" / "r_validation"
    verified = tmp_path / "gold" / "statistical" / "verified"
    analytical = tmp_path / "gold" / "analytical"
    evidence_graph = tmp_path / "serving" / "ai" / "evidence_graph.json"

    _write_json(python_results / "market_did.json", {"status": "ok"})
    _write_json(r_results / "r_cs_did_aggregate.json", {"status": "not_applicable"})
    _write_json(verified / "verified_model_results.json", {"status": "ok"})
    for name in [
        "fact_modern_method_applicability.csv",
        "fact_event_confound_screen.csv",
        "fact_jurisdiction_policy_timing.csv",
        "fact_overlap_restricted_sample.csv",
    ]:
        analytical.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "diagnostic"}]).to_csv(analytical / name, index=False)

    monkeypatch.setattr(evidence_tracer, "PYTHON_RESULTS", python_results)
    monkeypatch.setattr(evidence_tracer, "R_RESULTS", r_results)
    monkeypatch.setattr(evidence_tracer, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(evidence_tracer, "ANALYTICAL_GOLD", analytical)
    monkeypatch.setattr(evidence_tracer, "EVIDENCE_GRAPH", evidence_graph)

    graph = evidence_tracer.build_evidence_graph()

    node_ids = {node["node_id"] for node in graph["nodes"]}
    assert "gold_analytical:fact_modern_method_applicability.csv" in node_ids
    assert "gold_analytical:fact_event_confound_screen.csv" in node_ids
    assert graph["dangling_edge_count"] == 0
    assert {
        "from": "model_result:r_cs_did_aggregate.json",
        "to": "gold_analytical:fact_modern_method_applicability.csv",
        "relation": "diagnosed_by",
    } in graph["edges"]
