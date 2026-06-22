from __future__ import annotations

import pandas as pd

from ..config import DATA, REFERENCE
from ..utils import file_sha256, utc_now


def build_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file() or path.name == "source_manifest.csv":
            continue
        relative = path.relative_to(DATA)
        quarantined = relative.parts[0] == "quarantine"
        if quarantined:
            purpose = "audit_only"
        elif relative.parts[0] == "bronze":
            purpose = "immutable_source"
        elif relative.parts[0] == "silver":
            purpose = "standardized_reusable_data"
        elif relative.parts[0] == "gold":
            purpose = "analysis_bound_or_verified_evidence"
        elif relative.parts[0] == "serving":
            purpose = "rebuildable_retrieval_index"
        else:
            purpose = "reference_or_metadata"
        rows.append(
            {
                "path": str(relative),
                "layer": relative.parts[0],
                "source_type": "legacy_invalid" if quarantined else "api_or_source_file",
                "trusted_for_formal_analysis": not quarantined,
                "purpose": purpose,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "catalogued_at": utc_now(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(REFERENCE / "source_manifest.csv", index=False)
    return manifest
