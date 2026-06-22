# evidence_tracer.py
# Evidence chain tracing system.
# Builds a three-level provenance graph:
#   Report Claim → [citation_id] → Model Result (Gold) → Silver Table → Bronze Source

from __future__ import annotations

import json
import re

import pandas as pd

from ..config import AI_SERVING, ANALYTICAL_GOLD, EVIDENCE_GRAPH, R_RESULTS, PYTHON_RESULTS, VERIFIED_RESULTS


# Known evidence chain mappings
_EVIDENCE_CHAINS = [
    # (claim_pattern, citation_id_pattern, model_result_file, silver_table, bronze_source)
    ("DiD estimate", "MODEL-", "market_did.json",
     "fact_market_monthly.csv", "yahoo/daily_prices.csv"),
    ("SCM.*gap", "MODEL-", "scm.json",
     "fact_market_monthly.csv", "yahoo/daily_prices.csv"),
    ("event study", "MODEL-", "event_study.json",
     "fact_event_firm_car.csv", "yahoo/daily_prices.csv"),
    ("exposure.*event", "MODEL-", "exposure_event_study.json",
     "fact_exposure_score.csv", "sec/companyfacts/*.json"),
    ("overlap.weighted", "MODEL-", "weighted_did.json",
     "fact_exposure_score.csv", "sec/companyfacts/*.json"),
    ("revenue.*mechanism", "MODEL-", "revenue_mechanism.json",
     "fact_firm_financial_year.csv", "yahoo/corporate_financials.csv"),
    ("gsynth|generalized.*synthetic", "MODEL-", "r_gsynth_att.csv",
     "fact_market_monthly.csv", "yahoo/daily_prices.csv"),
    ("Bacon.*decomposition", "MODEL-", "r_bacon_decomposition.csv",
     "fact_market_monthly.csv", "yahoo/daily_prices.csv"),
    ("HonestDiD|sensitivity", "MODEL-", "r_honest_did_sensitivity.csv",
     "fact_market_monthly.csv", "yahoo/daily_prices.csv"),
    ("causal forest|heterogeneity", "MODEL-", "r_grf_blp.csv",
     "fact_firm_financial_year.csv", "yahoo/corporate_financials.csv"),
]
_REMEDIATION_TABLE_CHAINS = {
    "fact_modern_method_applicability.csv": [
        ("silver:fact_market_monthly.csv", "diagnostic_uses"),
        ("gold_analytical:fact_overlap_restricted_sample.csv", "diagnostic_uses"),
    ],
    "fact_event_confound_screen.csv": [
        ("silver:fact_market_daily.csv", "diagnostic_uses"),
        ("reference:policy_events.csv", "diagnostic_uses"),
    ],
    "fact_jurisdiction_policy_timing.csv": [
        ("silver:fact_cbcr_jurisdiction_year.csv", "context_uses"),
        ("reference:pillar_two_jurisdiction_adoption.csv", "context_uses"),
    ],
    "fact_overlap_restricted_sample.csv": [
        ("silver:fact_market_monthly.csv", "diagnostic_uses"),
        ("silver:fact_exposure_score.csv", "diagnostic_uses"),
    ],
}
_DIAGNOSTIC_RESULT_CHAINS = {
    "r_cs_did_aggregate.json": "fact_modern_method_applicability.csv",
    "r_cs_did_att.csv": "fact_modern_method_applicability.csv",
    "r_cs_did_event_study.csv": "fact_modern_method_applicability.csv",
    "r_sa_did_event_study.csv": "fact_modern_method_applicability.csv",
}


def build_evidence_graph() -> dict:
    """Construct the evidence chain graph and write to serving/ai/evidence_graph.json."""
    verified_path = VERIFIED_RESULTS / "verified_model_results.json"
    if not verified_path.exists():
        graph = {"status": "not_built", "reason": "verified_model_results.json not found"}
        EVIDENCE_GRAPH.parent.mkdir(parents=True, exist_ok=True)
        with EVIDENCE_GRAPH.open("w", encoding="utf-8") as handle:
            json.dump(graph, handle, indent=2, ensure_ascii=False)
        return graph

    # Build node registry
    nodes = []
    edges = []

    # Model result nodes
    for source, path_root in [
        ("python", PYTHON_RESULTS),
        ("r_validation", R_RESULTS),
    ]:
        for result_file in sorted([*path_root.glob("*.json"), *path_root.glob("*.csv")]):
            node_id = f"model_result:{result_file.name}"
            nodes.append({
                "node_id": node_id,
                "type": "model_result",
                "source": result_file.name,
                "runtime": source,
                "path": str(result_file.relative_to(path_root.parents[2])),
            })

    # Source table nodes referenced by evidence chains.
    table_names = sorted(
        {silver_table for *_, silver_table, _ in _EVIDENCE_CHAINS}
        | {"fact_market_daily.csv", "fact_cbcr_jurisdiction_year.csv"}
    )
    gold_analytical_tables = {"fact_event_firm_car.csv", "fact_exposure_score.csv"}
    for table in table_names:
        path = f"data/gold/analytical/{table}" if table in gold_analytical_tables else f"data/silver/{table}"
        nodes.append({
            "node_id": f"silver:{table}",
            "type": "gold_analytical_table" if table in gold_analytical_tables else "silver_table",
            "source": table,
            "path": path,
        })

    for table in sorted(_REMEDIATION_TABLE_CHAINS):
        table_path = ANALYTICAL_GOLD / table
        if table_path.exists():
            nodes.append({
                "node_id": f"gold_analytical:{table}",
                "type": "gold_analytical_table",
                "source": table,
                "path": f"data/gold/analytical/{table}",
            })

    for reference_table in ["policy_events.csv", "pillar_two_jurisdiction_adoption.csv"]:
        nodes.append({
            "node_id": f"reference:{reference_table}",
            "type": "reference_table",
            "source": reference_table,
            "path": f"data/reference/{reference_table}",
        })

    # Bronze source nodes
    source_descriptions = {
        "yahoo/daily_prices.csv": "Yahoo Finance daily prices",
        "yahoo/corporate_financials.csv": "Yahoo Finance corporate financials",
        "yahoo/candidate_corporate_financials.csv": "Yahoo Finance candidate financials",
        "sec/companyfacts/*.json": "SEC EDGAR Company Facts",
        "literature/*.pdf": "Literature PDFs (Big Four + Academic)",
    }
    for path in sorted({bronze_source for *_, bronze_source in _EVIDENCE_CHAINS}):
        nodes.append({
            "node_id": f"bronze:{path}",
            "type": "bronze_source",
            "source": path,
            "description": source_descriptions.get(path, path),
        })

    # Build edges from evidence chains
    for claim_pattern, cit_pattern, model_file, silver_table, bronze_source in _EVIDENCE_CHAINS:
        if not (PYTHON_RESULTS / model_file).exists() and not (R_RESULTS / model_file).exists():
            continue
        # model_result → silver_table
        edges.append({
            "from": f"model_result:{model_file}",
            "to": f"silver:{silver_table}",
            "relation": "derived_from",
        })
        # silver_table → bronze_source
        edges.append({
            "from": f"silver:{silver_table}",
            "to": f"bronze:{bronze_source}",
            "relation": "sourced_from",
        })

    for table, upstreams in _REMEDIATION_TABLE_CHAINS.items():
        if not (ANALYTICAL_GOLD / table).exists():
            continue
        for upstream_node, relation in upstreams:
            edges.append({
                "from": f"gold_analytical:{table}",
                "to": upstream_node,
                "relation": relation,
            })

    for model_file, table in _DIAGNOSTIC_RESULT_CHAINS.items():
        if (R_RESULTS / model_file).exists() and (ANALYTICAL_GOLD / table).exists():
            edges.append({
                "from": f"model_result:{model_file}",
                "to": f"gold_analytical:{table}",
                "relation": "diagnosed_by",
            })

    # Remove duplicate edges
    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge["from"], edge["to"], edge["relation"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    node_ids = {node["node_id"] for node in nodes}
    dangling_edges = [
        edge for edge in unique_edges
        if edge["from"] not in node_ids or edge["to"] not in node_ids
    ]

    graph = {
        "status": "built",
        "nodes": nodes,
        "edges": unique_edges,
        "verified_results_source": str(verified_path.relative_to(verified_path.parents[3])),
        "node_count": len(nodes),
        "edge_count": len(unique_edges),
        "dangling_edge_count": len(dangling_edges),
        "dangling_edges": dangling_edges,
    }

    EVIDENCE_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_GRAPH.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2, ensure_ascii=False)

    return graph


def validate_claim_traceability(report_path: str) -> dict:
    """Check that every numeric claim in the report can be traced to a model result.

    Returns dict with validation results: valid, claims_checked, untraceable_claims, errors.
    """
    import re
    from pathlib import Path

    rp = Path(report_path)
    if not rp.exists():
        return {"valid": False, "errors": [f"Report not found: {report_path}"]}

    text = rp.read_text(encoding="utf-8")

    # Extract numeric claims (estimate, p-value, std_error patterns)
    numeric_pattern = re.compile(
        r"(estimate|p.value|std.error|ci.low|ci.high)\b.*?(-?\d+\.?\d*)",
        re.IGNORECASE,
    )
    claims = numeric_pattern.findall(text)

    # Extract citation IDs
    citation_pattern = re.compile(r"\[((?:PDF|MODEL|SEC)-[0-9a-f]{10})\]")
    citations = citation_pattern.findall(text)

    # Check verified results for matching values
    verified_path = VERIFIED_RESULTS / "verified_model_results.json"
    verified_data = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    known_numbers = set()

    def collect_numbers(value):
        if isinstance(value, dict):
            for item in value.values():
                collect_numbers(item)
        elif isinstance(value, list):
            for item in value:
                collect_numbers(item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            known_numbers.add(round(float(value), 6))

    collect_numbers(verified_data)

    registry_path = AI_SERVING / "citation_registry.csv"
    if registry_path.exists():
        registry_ids = set(pd.read_csv(registry_path).fillna("")["citation_id"].astype(str))
    else:
        registry_ids = set()

    errors = []
    missing_citations = sorted(set(citations) - registry_ids)
    untraceable_claims = []
    for metric, value in claims:
        number = round(float(value), 6)
        if number not in known_numbers:
            untraceable_claims.append({"metric": metric, "value": value})

    if not citations:
        errors.append("No citation IDs found in report — all claims are untraceable.")
    if missing_citations:
        errors.append("Citation IDs not found in citation registry: " + ", ".join(missing_citations))
    if untraceable_claims:
        errors.append("Numeric claims not found in verified results.")

    return {
        "valid": len(errors) == 0,
        "claims_checked": len(claims),
        "citations_found": len(citations),
        "missing_citations": missing_citations,
        "untraceable_claims": untraceable_claims,
        "errors": errors,
    }
