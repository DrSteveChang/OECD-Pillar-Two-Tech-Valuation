OECD Pillar Two Tech Valuation Platform
This project serves as a comprehensive research framework designed to analyze the impact of the OECD Pillar Two (GloBE Rules) on the global technology sector. By integrating large-scale econometric modeling with a Retrieval-Augmented Generation (RAG) system, this platform quantifies valuation compression and identifies systemic tax-structuring shifts within multinational technology enterprises.

1. Project Architecture
The platform is designed as a three-stage pipeline, moving from raw data ingestion to causal inference and, finally, to an AI-driven synthesis of reports.

Pipeline Overview
Macro ETL & ETR Analysis: Processes aggregated OECD Country-by-Country Reporting (CbCR) data to map global profit exposure and identify "low-tax" jurisdiction hotspots.

Micro Network Engineering: Maps corporate subsidiary networks via OpenCorporates API integration, calculating the Jurisdictional Blending Ratio (JBR) to quantify network complexity.

Causal Inference Engine: Utilizes Synthetic Difference-in-Differences (SDiD) for macro impacts and Causal Forest models to isolate micro-level valuation compression.

RAG-Powered Synthesis: An intelligent orchestrator retrieves relevant academic literature and SEC 10-K filings from a ChromaDB-backed vector store to provide contextual depth to quantitative findings.

2. Directory Structure

platform/
├── config/             # Configuration files and prompt templates
├── data/               # Data storage
│   ├── processed/      # Cleaned econometric outputs and feature matrices
│   ├── raw/            # Source OECD CbCR and academic PDFs
│   └── vector_db/      # ChromaDB persistent storage
├── docs/               # RAG knowledge base (SEC filings, Bank reports, OECD rules)
├── output/             # Final generated AI analysis reports
├── src/                # Core logic
│   ├── app_orchestrator.py      # Master orchestration and RAG synthesis
│   ├── econometric_engine.py    # Causal inference and statistical modeling
│   ├── oecd_macro_etl.py        # OECD CbCR data processing
│   ├── opencorporates_network_mapper.py  # Network feature extraction
│   └── vector_indexer.py        # Vector embedding and retrieval
├── requirements.txt    # Python dependencies
└── README.md

3. Methodology
This project employs a mixed-method approach to investigate policy efficacy:

Quantitative Modeling:

SDiD: Used to create a synthetic counterfactual for low-tax jurisdictions to estimate the causal impact of Pillar Two on profit shifting.

Causal Forest: Applied to micro-panel data to estimate the heterogeneous treatment effect (HTE) of tax framework exposure on firm valuation.

Qualitative Synthesis (RAG):

Embeds technical documentation and financial reports using SentenceTransformer (all-MiniLM-L6-v2).

The app_orchestrator dynamically retrieves these insights to contextualize econometric results in reports for policymakers and corporate executives.

4. Setup and Execution
Prerequisites
Python 3.10+

API Keys:

GEMINI_API_KEY: Required for report synthesis.

OPENCORPORATES_API_KEY: Required for network mapping.

Installation
Clone the repository and navigate to the project directory.

Set up the virtual environment:
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt

Execution
The entry point for the entire platform is the app_orchestrator.py. Ensure your GEMINI_API_KEY is exported in your environment before running:
export GEMINI_API_KEY="your_api_key_here"
python src/app_orchestrator.py

5. Academic Context
This platform is developed as part of a Master’s Thesis (TFM) project at EAE Business School. It adheres to rigorous data-handling standards and neutral, empirically-grounded reporting for international policy and institutional equity research.

Author: Boyan Zhang

Field: Big Data Analytics

Date: 2026