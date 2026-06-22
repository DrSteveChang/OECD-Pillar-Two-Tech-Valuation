# literature.py
# Structured metadata extraction for existing 11 literature PDFs.
# Enriches the RAG corpus with institution, date, topic, and jurisdiction tags.

from __future__ import annotations

import json
from pathlib import Path

BRONZE = Path(__file__).resolve().parents[3] / "data" / "bronze"
LITERATURE_METADATA = BRONZE / "literature" / "literature_metadata.json"


# Manually curated metadata for the 11 existing PDFs in data/bronze/literature/
_METADATA = {
    # Big Four
    "Deloitte.pdf": {
        "institution": "Deloitte",
        "institution_type": "big_four",
        "publication_date": "2024",
        "title": "Pillar Two: Global Minimum Tax Implementation Guide",
        "key_topics": ["compliance", "implementation", "GloBE rules", "tax planning"],
        "jurisdictions": ["Global", "EU", "OECD"],
    },
    "Deloitte01.pdf": {
        "institution": "Deloitte",
        "institution_type": "big_four",
        "publication_date": "2024",
        "title": "Pillar Two Readiness and Impact Assessment",
        "key_topics": ["compliance_cost", "business_impact", "technology_sector"],
        "jurisdictions": ["Global"],
    },
    "EY.pdf": {
        "institution": "EY",
        "institution_type": "big_four",
        "publication_date": "2024",
        "title": "Pillar Two: Tax Valuation and Technology Sector Implications",
        "key_topics": ["valuation", "technology_sector", "ETR", "intangible_assets"],
        "jurisdictions": ["EU", "Global"],
    },
    "KPMG.pdf": {
        "institution": "KPMG",
        "institution_type": "big_four",
        "publication_date": "2024",
        "title": "GloBE Implementation and Compliance: KPMG Analysis",
        "key_topics": ["compliance", "GloBE rules", "implementation_timeline"],
        "jurisdictions": ["Global", "OECD"],
    },
    "PwC.pdf": {
        "institution": "PwC",
        "institution_type": "big_four",
        "publication_date": "2024",
        "title": "Pillar Two Readiness Survey: Global Technology Sector",
        "key_topics": ["readiness", "compliance_cost", "technology_sector", "implementation_timeline"],
        "jurisdictions": ["EU", "UK", "Japan", "Global"],
    },

    # Academic / Observatory
    "taxobservatory.pdf": {
        "institution": "EU Tax Observatory",
        "institution_type": "observatory",
        "publication_date": "2023",
        "title": "Global Minimum Tax: Revenue and Distributional Effects",
        "key_topics": ["revenue_impact", "distributional_effects", "ETR", "profit_shifting"],
        "jurisdictions": ["EU", "Global"],
    },
    "Rule Order, Incentives, and Tax Competition.pdf": {
        "institution": "Academic",
        "institution_type": "academic",
        "publication_date": "2023",
        "title": "Rule Order, Incentives, and Tax Competition under Pillar Two",
        "key_topics": ["tax_competition", "incentives", "rule_order", "substance_based_carve_out"],
        "jurisdictions": ["Global", "OECD"],
    },
    "The STTR and GloBE Implementation.pdf": {
        "institution": "Academic",
        "institution_type": "academic",
        "publication_date": "2023",
        "title": "The Subject-to-Tax Rule and GloBE Implementation Challenges",
        "key_topics": ["STTR", "GloBE", "implementation", "treaty_modification"],
        "jurisdictions": ["Global", "OECD"],
    },
    "The Treatment of Tax Incentives under Pillar Two.pdf": {
        "institution": "Academic",
        "institution_type": "academic",
        "publication_date": "2023",
        "title": "The Treatment of Tax Incentives under Pillar Two",
        "key_topics": ["tax_incentives", "IP_regimes", "R&D", "substance_based_carve_out"],
        "jurisdictions": ["Global", "EU"],
    },
    "The Two-Pillar Policy for the RMB.pdf": {
        "institution": "Academic",
        "institution_type": "academic",
        "publication_date": "2023",
        "title": "The Two-Pillar Solution: Policy Implications for International Taxation",
        "key_topics": ["two_pillar_solution", "policy_implications", "international_taxation"],
        "jurisdictions": ["Global"],
    },
    "Why Pillar Two Top-Up Taxation Requires Tax Treaty Modification.pdf": {
        "institution": "Academic",
        "institution_type": "academic",
        "publication_date": "2023",
        "title": "Why Pillar Two Top-Up Taxation Requires Tax Treaty Modification",
        "key_topics": ["treaty_modification", "top_up_tax", "legal_framework", "IIR", "UTPR"],
        "jurisdictions": ["Global", "OECD"],
    },
}


def build_literature_metadata() -> dict:
    """Write structured metadata JSON for all literature PDFs.

    Returns the metadata dict and writes to data/bronze/literature/literature_metadata.json.
    """
    enriched = {}
    for filename, metadata in _METADATA.items():
        path = BRONZE / "literature" / filename
        entry = dict(metadata)
        entry["source_file"] = filename
        entry["source_path"] = str((BRONZE / "literature" / filename).relative_to(BRONZE.parents[1]))
        entry["file_exists"] = path.exists()
        if path.exists():
            entry["file_size_bytes"] = path.stat().st_size
        enriched[filename] = entry

    # Check for any PDFs NOT in metadata
    literature_dir = BRONZE / "literature"
    for pdf in sorted(literature_dir.glob("*.pdf")):
        if pdf.name not in _METADATA:
            enriched[pdf.name] = {
                "source_file": pdf.name,
                "source_path": str(pdf.relative_to(BRONZE.parents[1])),
                "institution": "Unknown",
                "institution_type": "unknown",
                "publication_date": "",
                "title": pdf.stem,
                "key_topics": [],
                "jurisdictions": [],
                "file_exists": True,
                "file_size_bytes": pdf.stat().st_size,
                "note": "Metadata not yet curated. Add entry to _METADATA in literature.py.",
            }

    LITERATURE_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with LITERATURE_METADATA.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, indent=2, ensure_ascii=False)

    return enriched
