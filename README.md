# OECD Pillar Two and Technology-Firm Valuation

An auditable empirical research pipeline for evaluating whether OECD Pillar Two is associated with changes in technology-firm valuation. The project combines Python estimation, independent R reproduction, governed evidence publication, and an evidence-bound AI reporting layer.

The central empirical conclusion is deliberately limited: the Python and R implementations reproduce the declared calculations, but the available data do not identify a robust, general causal effect of Pillar Two on technology-firm valuation.

## Research boundary

The treatment indicator is a revenue-threshold **Pillar Two in-scope proxy**. It is not observed firm-level top-up-tax liability or jurisdiction-specific exposure. Results must therefore be read subject to treatment measurement, common support, pretrend, interference, sample-size, and omitted-variable limitations.

The reporting layer distinguishes four statements that are not interchangeable:

1. an estimate is statistically insignificant;
2. an identifying assumption is not supported;
3. a model is not applicable to the available design; and
4. there is no economic effect.

Only the first three can be supported directly by the current evidence. The fourth cannot be inferred from a null estimate or a failed design.

## Architecture

```text
Source APIs and local literature
              |
              v
Bronze --> Silver --> Gold --> Serving
 raw       canonical   verified   AI-ready
 data      panels      results    evidence
  |           |           |          |
  +------ lineage, validation, and publication controls ------+
```

- **Bronze** stores immutable SEC, Yahoo Finance, OECD, and literature inputs.
- **Silver** provides canonical firm-year, firm-month, market, and jurisdiction tables.
- **Gold** contains model outputs, assumption diagnostics, scoreboards, publication figures, and independently verified results.
- **Serving** contains the RAG corpus, citation registry, evidence graph, and rebuildable retrieval index.
- **Quarantine** isolates artifacts that fail provenance, schema, or methodological controls.

Raw Bronze and canonical Silver data are not distributed through GitHub. The repository retains a compact Gold/reference/Serving evidence snapshot so that published estimates and their interpretation boundaries can be inspected.

## Empirical workflow

The executable pipeline has five stages:

```bash
pillar-two pipeline --stage ingest
pillar-two pipeline --stage prepare
pillar-two pipeline --stage analyze
pillar-two pipeline --stage validate
pillar-two pipeline --stage report
```

`--stage all` executes the complete sequence. Ingestion requires network access and a compliant SEC identity:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
pillar-two pipeline --stage all
```

Python produces the formal estimates and diagnostics. `analysis/r/run_all.R` independently reproduces the declared calculations and writes comparison artifacts. Agreement between Python and R establishes computational reproducibility; it does not establish causal validity.

## Hybrid retrieval implementation

The active RAG component is a **local hybrid retrieval index**, not ChromaDB and not a separately deployed vector database.

`build_vector_index()` creates two persistent representations of the same governed corpus:

- a unigram/bigram TF-IDF sparse matrix;
- normalized 384-dimensional embeddings from `all-MiniLM-L6-v2`.

`retrieve()` embeds each query, computes exact cosine similarity against the local embedding matrix, computes TF-IDF cosine similarity, combines both scores with equal default weights, applies optional document-type filters, and returns Top-k evidence records. At the current corpus size, exact scanning is simpler and operationally adequate. ChromaDB would become useful only when scale, incremental indexing, concurrent serving, or richer metadata-index requirements justify a database service.

The retrieval layer ranks candidate context. It does not determine the permitted interpretation:

- the **citation registry** resolves every evidence identifier;
- the **evidence graph** traces claims to model, table, figure, and literature sources;
- assumption scoreboards govern whether a result can be described causally;
- formal report generation fails if the embedding index is unavailable or citation validation fails.

The deterministic AI report executes seven declared Top-k queries over local evidence. External LLM rewriting is off by default and requires both explicit authorization and an API key:

```bash
export ALLOW_EXTERNAL_LLM_REWRITE=1
export GEMINI_API_KEY="..."
pillar-two pipeline --stage report
```

Without both variables, no manuscript, report, or evidence corpus is uploaded to an external model.

## Installation

Python 3.11 and R are required. The tested Python environment is managed with `uv`:

```bash
uv sync --extra rag --extra dev
source .venv/bin/activate
```

The embedding stack is pinned for compatibility with the tested Intel macOS environment. Other platforms may need a platform-appropriate PyTorch build before running the formal report stage.

Run the Python tests with:

```bash
PYTHONPATH=src pytest -q
```

Run the independent R validation with:

```bash
Rscript analysis/r/run_all.R
```

## Repository data and reproducibility

The GitHub snapshot includes:

- source code, R validation scripts, configuration, and technical documentation;
- Gold analytical and statistical outputs;
- sample, event, lineage, and provenance registries;
- the compact RAG corpus, citation registry, and evidence graph.

It excludes raw and canonical datasets, downloaded model files, vector matrices, report renders, thesis materials, and machine-local credentials. A clean clone can inspect the published evidence and run unit-level contracts. Rebuilding ingestion, estimation, and the final report requires regenerating the excluded data layers from their registered sources.

## Methodological limitations

- Revenue-threshold scope is an imperfect proxy for actual Pillar Two liability.
- Firm-jurisdiction exposure and jurisdiction-specific implementation timing are incomplete.
- Baseline balance and common support are weak in important specifications.
- Some pretrend diagnostics reject the identifying restriction required for a strong DiD interpretation.
- Event studies remain exposed to concurrent firm news and cross-firm dependence.
- Revenue evidence has insufficient pre-implementation history for a strong causal design.
- Unobserved tax planning, geographic profit allocation, managerial decisions, macroeconomic shocks, and investor expectations may confound observed associations.

Modern staggered-treatment and heterogeneous-treatment estimators are treated as diagnostic or not applicable when their design requirements are absent. The pipeline does not restore legacy K-means, DoubleML, or causal-forest mechanism claims.

## Project layout

```text
analysis/r/                 Independent R validation
config/                     Project configuration
data/gold/                  Verified results and scoreboards
data/reference/             Sample, event, and provenance registries
data/serving/ai/            RAG corpus and traceability controls
docs/                       Architecture and data-contract documentation
src/oecd_pillar_two/        Python pipeline, analysis, RAG, and reporting
tests/                      Core behavioral and evidence-boundary tests
```

## License

See [LICENSE](LICENSE).
