from __future__ import annotations

import json
import hashlib
import os

from ..config import VERIFIED_RESULTS, OUTPUTS
from .report import generate_evidence_report
from .citations import validate_report_citations


def generate_ai_report() -> None:
    """Generate an optional LLM narrative without allowing the model to alter evidence."""
    cited_report = generate_evidence_report()
    api_key = os.environ.get("GEMINI_API_KEY")
    rewrite_authorized = os.environ.get("ALLOW_EXTERNAL_LLM_REWRITE") == "1"
    manifest_path = OUTPUTS / "ai_reports" / "llm_rewrite_manifest.json"
    manifest = {
        "status": "inactive",
        "provider": None,
        "model": None,
        "authorization_required": True,
        "authorization_present": rewrite_authorized,
        "api_key_present": bool(api_key),
        "source_report_sha256": hashlib.sha256(cited_report.encode("utf-8")).hexdigest(),
    }
    if not api_key or not rewrite_authorized:
        manifest["reason"] = (
            "External rewrite was not explicitly authorized"
            if not rewrite_authorized
            else "GEMINI_API_KEY was not configured"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return
    from google import genai

    verified = json.loads((VERIFIED_RESULTS / "verified_model_results.json").read_text())
    prompt = f"""Rewrite the cited evidence report below into a concise decision-support interpretation.
Do not calculate new effects, do not claim actual top-up-tax exposure, and do not describe a result as
statistically significant unless its structured field says so. Explicitly distinguish event-study results,
monthly DiD, exploratory revenue mechanisms, and SCM limitations. Preserve the distinction between
computational replication and causal-assumption credibility. State failed, weak, and untestable assumptions
explicitly, and do not upgrade associational evidence into a causal claim. Preserve every citation ID exactly.
Do not add citation IDs that are absent from the source report.

SOURCE CITED REPORT:
{cited_report}
"""
    response = genai.Client(api_key=api_key).models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    validation = validate_report_citations(response.text)
    if not validation["valid"]:
        raise ValueError("Gemini report citation validation failed: " + "; ".join(validation["errors"]))
    (OUTPUTS / "ai_reports" / "latest_verified_decision_support_report.md").write_text(response.text, encoding="utf-8")
    manifest.update({
        "status": "active",
        "provider": "Google Gemini",
        "model": "gemini-2.5-flash",
        "citation_validation": "passed",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
