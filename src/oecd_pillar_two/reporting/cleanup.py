from __future__ import annotations

import shutil

from ..config import (
    AI_SERVING, DATA_SCIENCE_RESULTS, FIGURES, OUTPUTS, PYTHON_RESULTS,
    R_RESULTS, SCOREBOARDS, VERIFIED_RESULTS, ensure_directories,
)


def clear_previous_analysis_outputs() -> None:
    """Ensure model and validation outputs belong only to the current run."""
    for directory in (PYTHON_RESULTS, R_RESULTS, VERIFIED_RESULTS, DATA_SCIENCE_RESULTS):
        if directory.exists():
            shutil.rmtree(directory)
    ensure_directories()


def clear_previous_deliverables() -> None:
    """Remove prior presentation and AI artifacts before generating the latest run."""
    for directory in (FIGURES, SCOREBOARDS, OUTPUTS / "ai_reports", AI_SERVING):
        if directory.exists():
            shutil.rmtree(directory)
    ensure_directories()
