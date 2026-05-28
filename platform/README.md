# OECD Pillar Two Tech Valuation Platform

This project serves as a comprehensive research framework designed to analyze the impact of the OECD Pillar Two (GloBE Rules) on the global technology sector. By integrating large-scale econometric modeling with a Retrieval-Augmented Generation (RAG) system, this platform quantifies valuation compression and identifies systemic tax-structuring shifts within multinational technology enterprises.

## Project Architecture

The platform is designed as a three-stage pipeline, moving from raw data ingestion to causal inference and, finally, to an AI-driven synthesis of reports.

### Pipeline Overview

*   **Macro ETL & ETR Analysis:** Processes aggregated OECD Country-by-Country Reporting (CbCR) data to map global profit exposure and identify "low-tax" jurisdiction hotspots.
*   **Micro Network Engineering:** Maps corporate subsidiary networks via OpenCorporates API integration, calculating the Jurisdictional Blending Ratio (JBR) to quantify network complexity.
*   **Causal Inference Engine:** Utilizes Synthetic Difference-in-Differences (SDiD) for macro impacts and Causal Forest models to isolate micro-level valuation compression.
*   **RAG-Powered Synthesis:** An intelligent orchestrator retrieves relevant academic literature and SEC 10-K filings from a ChromaDB-backed vector store to provide contextual depth to quantitative findings.

## Directory Structure

```text
platform/
├── config/            # Configuration files and prompt templates
├── data/              # Data storage
│   ├── processed/     # Cleaned econometric outputs and feature matrices
│   ├── raw/           # Source OECD CbCR and academic PDFs
│   └── vector_db/     # ChromaDB persistent storage
├── docs/              # RAG knowledge base (SEC filings, Bank reports, OECD rules)
├── output/            # Final generated AI analysis reports
├── src/               # Core logic
│   ├── app_orchestrator.py                # Master orchestration and RAG synthesis
│   ├── econometric_engine.py              # Causal inference and statistical modeling
│   ├── oecd_macro_etl.py                  # OECD CbCR data processing
│   ├── opencorporates_network_mapper.py   # Network feature extraction
│   └── vector_indexer.py                  # Vector embedding and retrieval
├── requirements.txt   # Python dependencies
└── README.md