from __future__ import annotations

import argparse
import subprocess

from .analysis.pipeline import run_all_analysis
from .cleaning.design_remediation import build_design_remediation
from .cleaning.firm_panel import build_firm_year_panel
from .cleaning.exposure_design import build_exposure_design
from .cleaning.lineage import build_lineage
from .cleaning.manifest import build_manifest
from .cleaning.market import build_market_panels
from .cleaning.oecd import build_cbcr_panel
from .cleaning.warehouse import build_gold_analytical_schema, build_silver_star_schema
from .config import ROOT, ensure_directories
from .ingestion.sec import download_sec_data
from .ingestion.yahoo import download_prices
from .ingestion.universe import download_candidate_financials, write_candidate_universe
from .rag.corpus import build_corpus
from .rag.evidence_tracer import build_evidence_graph
from .rag.indexer import build_vector_index
from .rag.literature import build_literature_metadata
from .reporting.ai_platform import generate_ai_report
from .reporting.gold_tables import build_python_gold_facts, build_verified_gold_facts
from .reporting.cleanup import clear_previous_analysis_outputs, clear_previous_deliverables
from .validation.results import verify_results


def run_stage(stage: str, force: bool = False) -> None:
    ensure_directories()
    if stage in {"ingest", "all"}:
        write_candidate_universe()
        download_candidate_financials(force=force)
        build_exposure_design()
        download_prices(force=force)
        download_sec_data(include_filings=True)
    if stage in {"prepare", "all"}:
        build_exposure_design()
        build_firm_year_panel()
        build_market_panels()
        build_cbcr_panel()
        build_silver_star_schema()
        build_gold_analytical_schema()
        build_design_remediation()
        build_lineage()
        build_corpus()
        build_vector_index()
        build_manifest()
    if stage in {"analyze", "all"}:
        clear_previous_analysis_outputs()
        build_design_remediation()
        run_all_analysis()
        build_python_gold_facts()
    if stage in {"validate", "all"}:
        subprocess.run(["Rscript", "analysis/r/run_all.R"], cwd=ROOT, check=True)
        verify_results()
        build_verified_gold_facts()
    if stage in {"report", "all"}:
        from .reporting.figures import generate_figures
        from .reporting.scoreboard import generate_scoreboard
        clear_previous_deliverables()
        generate_figures()
        generate_scoreboard()
        build_literature_metadata()
        build_corpus()
        build_vector_index(require_embeddings=True)
        generate_ai_report()
        build_evidence_graph()
        build_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser("pipeline")
    pipeline.add_argument("--stage", choices=["ingest", "prepare", "analyze", "validate", "report", "all"], default="all")
    pipeline.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "pipeline":
        run_stage(args.stage, force=args.force)


if __name__ == "__main__":
    main()
