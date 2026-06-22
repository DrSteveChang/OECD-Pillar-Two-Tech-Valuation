from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import pandas as pd

from ..config import AI_SERVING, OUTPUTS, VERIFIED_RESULTS
from ..rag.retriever import retrieve
from ..utils import write_json
from .citations import citation_label, validate_report_citations


REPORT_RETRIEVAL_QUERIES = [
    {
        "query_id": "literature_incentives",
        "query": "Pillar Two investment incentives tax holidays intellectual property regimes",
        "document_types": {"literature"},
        "top_k": 3,
    },
    {
        "query_id": "literature_implementation",
        "query": "global minimum tax treaty implementation challenges",
        "document_types": {"literature"},
        "top_k": 3,
    },
    {
        "query_id": "literature_profit_shifting",
        "query": "Pillar Two effective tax rate multinational profit shifting",
        "document_types": {"literature"},
        "top_k": 3,
    },
    {
        "query_id": "literature_valuation",
        "query": "Pillar Two valuation earnings technology intangible assets",
        "document_types": {"literature"},
        "top_k": 3,
    },
    {
        "query_id": "persistent_effect_assumptions",
        "query": "monthly abnormal return DiD pretrend overlap treatment proxy assumptions",
        "document_types": {"model_result", "model_table"},
        "top_k": 5,
    },
    {
        "query_id": "event_and_scm_diagnostics",
        "query": "event study Holm concurrent news synthetic control placebo donor validity",
        "document_types": {"model_result", "model_table", "model_figure"},
        "top_k": 5,
    },
    {
        "query_id": "modern_method_applicability",
        "query": "firm jurisdiction exposure timing common support modern estimator applicability",
        "document_types": {"model_result", "model_table", "model_figure"},
        "top_k": 5,
    },
]


def _first_evidence_citation(corpus: pd.DataFrame, source: str, document_type: str) -> str:
    rows = corpus[(corpus["document_type"] == document_type) & (corpus["source"] == source)]
    if rows.empty:
        raise ValueError(f"Missing model evidence for {source}")
    return f"[{rows.iloc[0]['citation_id']}]"


def _literature_evidence(retrieval_runs: list[dict]) -> list[dict]:
    selected = {}
    for run in retrieval_runs:
        for row in run["results"]:
            if row["document_type"] == "literature":
                selected[row["citation_id"]] = row
    return list(selected.values())[:10]


def generate_evidence_report() -> str:
    verified = json.loads((VERIFIED_RESULTS / "verified_model_results.json").read_text())
    corpus = pd.read_csv(AI_SERVING / "rag_corpus.csv").fillna("")
    index_status = json.loads((AI_SERVING / "vector_store" / "index_status.json").read_text())
    if index_status.get("embedding_status") != "built":
        raise RuntimeError("Formal AI report requires an active embedding index")
    retrieval_runs = []
    for spec in REPORT_RETRIEVAL_QUERIES:
        results = retrieve(
            spec["query"],
            document_types=spec["document_types"],
            top_k=spec["top_k"],
        )
        retrieval_runs.append({**spec, "results": results})

    retrieval_audit = {
        "retrieval_invoked": True,
        "query_count": len(retrieval_runs),
        "embedding_status": index_status["embedding_status"],
        "embedding_model": index_status.get("embedding_model"),
        "retrieval_mode": "hybrid",
        "runs": [
            {
                "query_id": run["query_id"],
                "query": run["query"],
                "document_types": sorted(run["document_types"]),
                "top_k": run["top_k"],
                "returned_citation_ids": [row["citation_id"] for row in run["results"]],
                "returned_sources": [row["source"] for row in run["results"]],
                "scores": [round(float(row["score"]), 8) for row in run["results"]],
                "retrieval_mode": run["results"][0]["retrieval_mode"] if run["results"] else "none",
            }
            for run in retrieval_runs
        ],
    }
    write_json(OUTPUTS / "ai_reports" / "retrieval_audit.json", retrieval_audit)

    literature = _literature_evidence(retrieval_runs)
    if len(literature) < 3:
        raise ValueError("Insufficient PDF evidence for cited report")

    did = verified["python"]["market_did"]
    scm = verified["python"]["scm"]
    revenue = verified["python"]["revenue_mechanism"]
    exposure_events = verified["python"]["exposure_event_study"]["events"]
    weighted = verified["python"]["weighted_did"]
    events = verified["python"]["event_study"]["events"]
    negative_events = [
        event for event in events
        if event["difference"] < 0 and event["statistically_significant_holm_5pct"]
    ]
    positive_events = [
        event for event in events
        if event["difference"] > 0 and event["statistically_significant_holm_5pct"]
    ]

    model_sources = sorted(
        corpus.loc[corpus["document_type"].eq("model_result"), "source"].unique()
    )
    model_refs = {
        source: _first_evidence_citation(corpus, source, "model_result")
        for source in model_sources
    }
    verified_ref = model_refs["verified_model_results.json"]
    for source in (
        "market_did.json", "did_assumption_diagnostics.json", "r_validation_results.json",
        "event_study.json", "python_model_results.json", "scm.json", "revenue_mechanism.json",
        "exposure_event_study.json", "weighted_did.json",
    ):
        model_refs[source] = verified_ref
    table_sources = sorted(
        corpus.loc[corpus["document_type"].eq("model_table"), "source"].unique()
    )
    table_refs = {
        source: _first_evidence_citation(corpus, source, "model_table")
        for source in table_sources
    }
    figure_sources = sorted(
        corpus.loc[corpus["document_type"].eq("model_figure"), "source"].unique()
    )
    figure_refs = {
        source: _first_evidence_citation(corpus, source, "model_figure")
        for source in figure_sources
    }
    pdf_refs = [f"[{item['citation_id']}]" for item in literature]
    retrieval_grounding = ""
    for run in retrieval_runs:
        citations = " ".join(f"[{row['citation_id']}]" for row in run["results"])
        retrieval_grounding += (
            f"The `{run['query_id']}` query retrieved {len(run['results'])} of the requested "
            f"top {run['top_k']} records from {', '.join(sorted(run['document_types']))}. "
            f"These ranked records delimit the evidence inspected for this report {citations}.\n\n"
        )

    report = f"""# Pillar Two Evidence-Synthesis Report

## Report Metadata and Retrieval Status

- Generated: {date.today().isoformat()}.
- Corpus records: {len(corpus)}; indexed documents: {index_status['documents']}.
- Retrieval backend: active hybrid TF-IDF and {index_status['embedding_model']} embeddings.
- Retrieval execution: {len(retrieval_runs)} top-k queries with {sum(len(run['results']) for run in retrieval_runs)} ranked positions.
- External LLM rewrite: conditional and inactive unless separately authorized; the active report is deterministic and evidence-bound.

## Executive Interpretation

The Python and R implementations reproduce the declared calculations, but the available data do not support a robust general causal effect of Pillar Two on technology-firm valuation. Treatment is a revenue-threshold scope proxy rather than observed firm-level top-up-tax liability. Failed or weak identifying conditions therefore limit the permitted interpretation to diagnostic-qualified associations. The AI layer retrieves, organizes, and cites evidence; it does not repair treatment measurement, overlap, pretrend, or omitted-variable problems {verified_ref}.

This decision-support report combines the complete set of locally executed model reports with retrieved PDF literature. Treatment is a revenue-threshold **Pillar Two in-scope proxy**, not observed firm-level top-up-tax liability.

## Empirical Findings

The persistent monthly abnormal-return DiD estimate is **{did['estimate']:.6f}**, with standard error **{did['std_error']:.6f}** and p-value **{did['p_value']:.4f}**. It is not statistically significant at the 5% level. However, the annual pretrend joint test rejects equality before implementation, and baseline covariate overlap is weak; the estimate is therefore treated as an association rather than a strong causal effect {model_refs['market_did.json']} {model_refs['did_assumption_diagnostics.json']} {model_refs['r_validation_results.json']} {model_refs['verified_model_results.json']}.

The main continuous-exposure event study uses a pre-policy score based on low-tax ETR gaps, R&D intensity, and intangible-asset intensity. Across {len(exposure_events)} event-window estimates, {sum(event['statistically_significant_holm_5pct'] for event in exposure_events)} remain significant after Holm adjustment. This score is a proxy rather than observed top-up tax, and concurrent firm news remains a limitation {model_refs['exposure_event_study.json']}.

The overlap-weighted DiD estimate is **{weighted['estimate']:.6f}** with p-value **{weighted['p_value']:.4f}**. Its common-support share is **{weighted['assumption_diagnostics']['common_support_share']:.3f}** and weighted maximum absolute standardized difference is **{weighted['assumption_diagnostics']['max_absolute_smd_weighted']:.3f}**. Because common support fails, this is retained only as a failed-design diagnostic and supplementary sensitivity result {model_refs['weighted_did.json']}.

After Holm adjustment for multiple testing, the event study identifies {len(negative_events)} statistically significant negative short-window comparisons and {len(positive_events)} statistically significant positive short-window comparisons. Concurrent news and cross-firm dependence remain limitations {model_refs['event_study.json']} {model_refs['python_model_results.json']}.

The synthetic-control analysis reports a post/pre MSPE ratio of **{scm['post_pre_mspe_ratio']:.3f}** and placebo p-value of **{scm['placebo_p_value']:.4f}**. This remains proxy-treatment evidence and is not conventionally significant at 5% {model_refs['scm.json']} {model_refs['r_validation_results.json']}.

The revenue mechanism estimate is **{revenue['estimate']:.6f}** with p-value **{revenue['p_value']:.4f}**. It is exploratory because only one usable pre-implementation fiscal year is available {model_refs['revenue_mechanism.json']}.

## Figure-Grounded Analysis

The sample and coverage figure documents numerically similar firm counts and the expansion to stable market-panel coverage. Similar counts do not imply covariate balance, which is assessed separately and found to be weak {figure_refs['Figure01_sample_and_market_coverage.png']}.

The descriptive abnormal-return trend and cumulative-return figures show visible group differences, but neither substitutes for fixed-effects DiD inference. They must be interpreted together with the statistically insignificant Python/R DiD estimate {figure_refs['Figure02_market_parallel_trends.png']} {figure_refs['Figure03_cumulative_abnormal_returns.png']} {figure_refs['Figure04_python_r_did_validation.png']}.

The event-study forest plot shows that confidence intervals and directions vary across events and windows. The firm-level CAR distributions provide the corresponding cross-sectional dispersion for the short window, reinforcing that the event evidence is heterogeneous rather than uniformly negative {figure_refs['Figure05_event_study_forest.png']} {figure_refs['Figure06_event_car_distributions.png']}.

The SCM trajectory and monthly-gap figures visualize divergence from the synthetic control, while the placebo distribution shows that the treated proxy is not conventionally significant at the 5% level. Donor weights document which control firms construct the synthetic benchmark {figure_refs['Figure07_scm_cumulative_trajectory.png']} {figure_refs['Figure08_scm_monthly_gap.png']} {figure_refs['Figure09_scm_placebo_distribution.png']} {figure_refs['Figure10_scm_donor_weights.png']}.

The revenue-trend figure is explicitly exploratory. It visualizes level differences and post-2024 movement but cannot establish a causal operating effect because there is only one usable pre-implementation fiscal year {figure_refs['Figure11_exploratory_revenue_trends.png']}.

The assumption-diagnostic figures show that the annual pretrend test rejects, baseline covariates are materially imbalanced, and the DiD estimate varies under trend and baseline-covariate adjustments. The SCM leave-one-donor-out diagnostic shows the range of estimates when influential donors are removed {figure_refs['Figure12_did_dynamic_pretrend_diagnostic.png']} {figure_refs['Figure13_did_baseline_covariate_balance.png']} {figure_refs['Figure14_did_robustness_specifications.png']} {figure_refs['Figure15_scm_leave_one_out_sensitivity.png']}.

The redesigned-sample figures document the strict four-year scope rule, the exposure-score distribution, the continuous-exposure event estimates, and the failed common-support and balance diagnostics. The weighted pretrend result cannot rescue the design when treated and control firms do not overlap on baseline size and threshold distance {figure_refs['Figure16_four_year_scope_threshold.png']} {figure_refs['Figure17_pre_policy_exposure_distribution.png']} {figure_refs['Figure18_exposure_event_study.png']} {figure_refs['Figure19_propensity_common_support.png']} {figure_refs['Figure20_weighted_covariate_balance.png']} {figure_refs['Figure21_weighted_did_pretrend.png']}.

The remediation figures make the method boundary explicit. The applicability matrix, treatment-cohort chart, gsynth attrition chart, and HonestDiD support chart show why modern staggered estimators remain diagnostic or not applicable without firm-jurisdiction exposure timing {figure_refs['Figure22_modern_method_applicability_matrix.png']} {figure_refs['Figure23_treatment_timing_cohort_support.png']} {figure_refs['Figure24_gsynth_balanced_panel_attrition.png']} {figure_refs['Figure25_honestdid_preperiod_support.png']}.

The GRF support chart, restricted-sample DiD comparison, event-confound screen, and CbCR low-tax context figure are also diagnostics. They help write the paper's limitations and robustness sections, but they do not convert the revenue-threshold proxy into observed top-up-tax exposure {figure_refs['Figure26_grf_sample_and_covariate_support.png']} {figure_refs['Figure27_restricted_sample_did_comparison.png']} {figure_refs['Figure28_event_confound_screen.png']} {figure_refs['Figure29_cbcr_low_tax_context.png']}.

## Literature-Grounded Interpretation

The retrieved literature emphasizes that Pillar Two changes the treatment of tax incentives and can alter the value of low-tax structures, but effects depend on incentive design and jurisdiction-specific implementation {pdf_refs[0]} {pdf_refs[1]}.

The literature also identifies legal, treaty, and implementation frictions that can delay or weaken observable responses. These mechanisms are consistent with finding event-specific reactions without a statistically detectable persistent monthly effect {pdf_refs[2]} {pdf_refs[3]} {pdf_refs[4]}.

Research discussing profit shifting, effective tax rates, and tax competition supports using the OECD CbCR panel as policy-background evidence, but it does not convert the firm's revenue-threshold proxy into observed top-up-tax exposure {pdf_refs[5]} {pdf_refs[6]}.

## Retrieval-Grounded Evidence

{retrieval_grounding}
Retrieval ranks candidate evidence; it does not determine the permitted interpretation. Formal estimates still come from verified structured results, and assumption scoreboards still govern causal language.

## Evidence Boundary

- Main persistent-effect association: monthly abnormal-return DiD, independently reproduced in Python and R, but subject to material identifying-assumption concerns.
- Event-study findings: short-window market associations, sensitive to event and window selection.
- SCM and SDiD: complementary evidence using genuine pre-periods without fabricated years.
- Revenue evidence is exploratory; GRF output is an applicability and support diagnostic.
- PDF literature: contextual and interpretive evidence, not a substitute for model estimation.
- AI/reporting layer: may summarize evidence but may not calculate new treatment effects or significance.

## Local Model Report Register

"""
    for source in model_sources:
        report += f"- `{source}` {model_refs[source]}\n"

    report += "\n## Local Result Table Register\n\n"
    for source in table_sources:
        report += f"- `{source}` {table_refs[source]}\n"

    report += "\n## Local Figure Evidence Register\n\n"
    for source in figure_sources:
        report += f"- `{source}` {figure_refs[source]}\n"

    report += "\n## PDF Evidence Register\n\n"
    for row in literature:
        report += f"- {citation_label(row)}\n"

    report += "\n## Retrieval Audit Register\n\n"
    for run in retrieval_runs:
        hits = ", ".join(row["citation_id"] for row in run["results"])
        report += (
            f"- `{run['query_id']}`: top_k={run['top_k']}; "
            f"document_types={','.join(sorted(run['document_types']))}; hits={hits}\n"
        )

    validation = validate_report_citations(report)
    cited_model_sources = set(
        corpus.loc[
            corpus["citation_id"].isin(validation["citations"])
            & corpus["document_type"].eq("model_result"),
            "source",
        ]
    )
    cited_table_sources = set(
        corpus.loc[
            corpus["citation_id"].isin(validation["citations"])
            & corpus["document_type"].eq("model_table"),
            "source",
        ]
    )
    cited_figure_sources = set(
        corpus.loc[
            corpus["citation_id"].isin(validation["citations"])
            & corpus["document_type"].eq("model_figure"),
            "source",
        ]
    )
    missing_model_sources = sorted(set(model_sources) - cited_model_sources)
    missing_table_sources = sorted(set(table_sources) - cited_table_sources)
    missing_figure_sources = sorted(set(figure_sources) - cited_figure_sources)
    if missing_model_sources or missing_table_sources or missing_figure_sources:
        validation["valid"] = False
        if missing_model_sources:
            validation["errors"].append(f"Uncited local model reports: {missing_model_sources}")
        if missing_table_sources:
            validation["errors"].append(f"Uncited local result tables: {missing_table_sources}")
        if missing_figure_sources:
            validation["errors"].append(f"Uncited local figures: {missing_figure_sources}")
    validation["figure_citations"] = len(cited_figure_sources)
    validation["required_figures"] = len(figure_sources)
    output = OUTPUTS / "ai_reports" / "latest_verified_decision_support_report.md"
    output.write_text(report, encoding="utf-8")
    write_json(OUTPUTS / "ai_reports" / "citation_validation.json", validation)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    return report
