from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

from ..config import AI_SERVING, ANALYTICAL_GOLD, BRONZE, FIGURES, OUTPUTS, SCOREBOARDS, VERIFIED_RESULTS
from .chunker import chunk_dataframe_rows, chunk_json_entries, chunk_text


CORPUS_COLUMNS = [
    "citation_id", "document_type", "source", "source_path", "source_url",
    "page", "section", "ticker", "text", "sha256",
]
REMEDIATION_TABLES = [
    "fact_modern_method_applicability.csv",
    "fact_event_confound_screen.csv",
    "fact_jurisdiction_policy_timing.csv",
    "fact_overlap_restricted_sample.csv",
]


def _citation_id(prefix: str, source: str, locator: str) -> str:
    stable = sha256(f"{prefix}|{source}|{locator}".encode()).hexdigest()[:10]
    return f"{prefix}-{stable}"


def _record(
    document_type: str,
    source: str,
    source_path: str,
    text: str,
    *,
    page: str | int = "",
    section: str = "",
    source_url: str = "",
    ticker: str = "",
) -> dict:
    locator = f"page={page}|section={section}|ticker={ticker}"
    prefix = "PDF" if document_type == "literature" else "MODEL" if document_type.startswith("model") else "SEC"
    return {
        "citation_id": _citation_id(prefix, source, locator),
        "document_type": document_type,
        "source": source,
        "source_path": source_path,
        "source_url": source_url,
        "page": page,
        "section": section,
        "ticker": ticker,
        "text": text,
        "sha256": sha256(text.encode()).hexdigest(),
    }


def _flatten_json(value, prefix: str = "") -> list[str]:
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            lines.extend(_flatten_json(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            lines.extend(_flatten_json(item, f"{prefix}[{index}]"))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def _load_literature_metadata() -> dict:
    metadata_path = BRONZE / "literature" / "literature_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _metadata_prefix(metadata: dict) -> str:
    if not metadata:
        return ""
    topics = ", ".join(metadata.get("key_topics", []))
    jurisdictions = ", ".join(metadata.get("jurisdictions", []))
    lines = [
        f"title: {metadata.get('title', '')}",
        f"institution: {metadata.get('institution', '')}",
        f"institution_type: {metadata.get('institution_type', '')}",
        f"publication_date: {metadata.get('publication_date', '')}",
        f"key_topics: {topics}",
        f"jurisdictions: {jurisdictions}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def _json_chunks(value) -> list[dict]:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return chunk_json_entries(value, group_size=10)
    lines = [{"path": index, "value": line} for index, line in enumerate(_flatten_json(value), start=1)]
    return chunk_json_entries(lines, group_size=35)


def _read_strict_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            return pd.DataFrame(columns=columns)
        for row in reader:
            if None in row:
                continue
            rows.append({column: row.get(column, "") for column in columns})
    return pd.DataFrame(rows, columns=columns)


def _source_path(path: Path) -> str:
    try:
        return str(path.relative_to(OUTPUTS.parent))
    except ValueError:
        return str(path)


def build_corpus() -> pd.DataFrame:
    records = []
    literature_metadata = _load_literature_metadata()
    try:
        import fitz

        for path in sorted((BRONZE / "literature").glob("*.pdf")):
            metadata_text = _metadata_prefix(literature_metadata.get(path.name, {}))
            document = fitz.open(path)
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if len(text) < 100:
                    continue
                enriched_text = f"{metadata_text}\n\n{text}".strip() if metadata_text else text
                chunks = chunk_text(enriched_text, page=page_number, source=path.name) or [
                    {"text": enriched_text, "page": page_number, "section": f"page_{page_number}_chunk_0"}
                ]
                for chunk in chunks:
                    section = chunk.get("section") or f"page_{page_number}_chunk_{chunk.get('chunk_index', 0)}"
                    records.append(
                        _record(
                            "literature", path.name, str(path.relative_to(path.parents[2])),
                            chunk["text"], page=chunk.get("page", page_number), section=section,
                        )
                    )
    except ImportError:
        pass

    sec_path = BRONZE / "sec" / "tax_disclosures.jsonl"
    if sec_path.exists():
        for line in sec_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            text = item.get("text", "").strip()
            if not text:
                continue
            source = f"{item['ticker']}_{item['form']}_{item['filing_date']}"
            records.append(
                _record(
                    "sec_filing", source, "data/bronze/sec/tax_disclosures.jsonl", text,
                    source_url=item["source_url"], ticker=item["ticker"],
                )
            )

    approved_gold_roots = (VERIFIED_RESULTS, SCOREBOARDS)
    for path in sorted(path for root in approved_gold_roots for path in root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for chunk in _json_chunks(value):
            records.append(
                _record(
                    "model_result", path.name, _source_path(path),
                    chunk["text"], section=chunk["section"],
                )
            )

    for path in sorted(path for root in approved_gold_roots for path in root.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            records.append(
                _record(
                    "model_result", path.name, _source_path(path),
                    text, section="complete_report",
                )
            )

    for path in sorted(path for root in approved_gold_roots for path in root.rglob("*.csv")):
        frame = pd.read_csv(path)
        for chunk in chunk_dataframe_rows(frame, chunk_size=40, source=path.name):
            records.append(
                _record(
                    "model_table", path.name, _source_path(path),
                    chunk["text"], section=chunk["section"],
                )
            )

    for path in sorted(ANALYTICAL_GOLD / name for name in REMEDIATION_TABLES if (ANALYTICAL_GOLD / name).exists()):
        frame = pd.read_csv(path)
        for chunk in chunk_dataframe_rows(frame, chunk_size=40, source=path.name):
            records.append(
                _record(
                    "model_table", path.name, _source_path(path),
                    chunk["text"], section=chunk["section"],
                )
            )

    figure_manifest = FIGURES / "figure_manifest.csv"
    if figure_manifest.exists():
        figure_columns = ["figure_id", "title", "evidence_role", "interpretation_boundary", "png_path", "pdf_path"]
        figures = _read_strict_csv(figure_manifest, figure_columns).fillna("")
        for figure in figures.to_dict("records"):
            source = Path(figure["png_path"]).name
            text = "\n".join(
                [
                    f"figure_id: {figure['figure_id']}",
                    f"title: {figure['title']}",
                    f"evidence_role: {figure['evidence_role']}",
                    f"interpretation_boundary: {figure['interpretation_boundary']}",
                    f"png_path: {figure['png_path']}",
                    f"pdf_path: {figure['pdf_path']}",
                ]
            )
            records.append(
                _record(
                    "model_figure", source, figure["png_path"], text,
                    section=figure["evidence_role"],
                )
            )

    corpus = pd.DataFrame(records).drop_duplicates("citation_id") if records else pd.DataFrame(columns=CORPUS_COLUMNS)
    corpus.to_csv(AI_SERVING / "rag_corpus.csv", index=False)
    corpus.drop(columns=["text"]).to_csv(AI_SERVING / "citation_registry.csv", index=False)
    return corpus
