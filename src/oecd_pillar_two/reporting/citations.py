from __future__ import annotations

import re
import pandas as pd

from ..config import AI_SERVING


CITATION_PATTERN = re.compile(r"\[(PDF|MODEL|SEC)-[0-9a-f]{10}\]")


def validate_report_citations(report: str, *, require_pdf: bool = True, require_model: bool = True) -> dict:
    registry = pd.read_csv(AI_SERVING / "citation_registry.csv").fillna("")
    valid = set(registry["citation_id"])
    cited = {match.group(0)[1:-1] for match in CITATION_PATTERN.finditer(report)}
    missing = sorted(cited - valid)
    pdf_count = sum(item.startswith("PDF-") for item in cited)
    model_count = sum(item.startswith("MODEL-") for item in cited)
    errors = []
    if missing:
        errors.append(f"Unknown citation IDs: {missing}")
    if require_pdf and pdf_count == 0:
        errors.append("Report must cite at least one PDF source")
    if require_model and model_count == 0:
        errors.append("Report must cite at least one local model source")
    return {
        "valid": not errors,
        "citations": sorted(cited),
        "pdf_citations": pdf_count,
        "model_citations": model_count,
        "errors": errors,
    }


def citation_label(row: dict) -> str:
    page = row.get("page", "")
    if isinstance(page, float) and page.is_integer():
        page = int(page)
    locator = f", p. {page}" if str(page) else f", {row.get('section', '')}"
    return f"[{row['citation_id']}] {row['source']}{locator}"
