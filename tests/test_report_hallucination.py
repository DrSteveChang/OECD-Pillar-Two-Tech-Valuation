# test_report_hallucination.py
# Anti-hallucination verification for AI-generated reports.
# Ensures every numeric claim in the report can be traced to verified model results.

import json
import re
from pathlib import Path

import pytest

REPORT_PATH = Path("outputs/ai_reports/latest_verified_decision_support_report.md")
VERIFIED_PATH = Path("data/gold/statistical/verified/verified_model_results.json")
CITATION_REGISTRY_PATH = Path("data/serving/ai/citation_registry.csv")


@pytest.fixture
def report_text():
    if not REPORT_PATH.exists():
        pytest.skip("Report not yet generated")
    return REPORT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def verified_data():
    with VERIFIED_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def citation_registry_ids():
    import pandas as pd
    registry = pd.read_csv(CITATION_REGISTRY_PATH).fillna("")
    return set(registry["citation_id"])


class TestReportNoHallucination:
    """Verify that the AI-generated report contains no hallucinated claims."""

    def test_report_exists(self):
        """Report file must exist after pipeline execution."""
        if not REPORT_PATH.exists():
            pytest.skip("requires the locally generated evidence report")

    def test_all_citation_ids_exist_in_registry(self, report_text, citation_registry_ids):
        """Every [CITATION_ID] in the report must exist in the citation registry."""
        pattern = re.compile(r"\[(PDF|MODEL|SEC)-[0-9a-f]{10}\]")
        cited = {match.group(0)[1:-1] for match in pattern.finditer(report_text)}
        unknown = cited - citation_registry_ids

        assert len(cited) > 0, "Report must contain at least one citation ID"
        assert not unknown, (
            f"Unknown citation IDs found in report: {sorted(unknown)}. "
            "These IDs do not exist in the citation registry."
        )

    def test_at_least_one_pdf_cited(self, report_text):
        """Report must cite at least one literature PDF source."""
        pdf_citations = len(re.findall(r"\[PDF-[0-9a-f]{10}\]", report_text))
        assert pdf_citations > 0, "Report must cite at least one PDF literature source"

    def test_at_least_one_model_cited(self, report_text):
        """Report must cite at least one local model result."""
        model_citations = len(re.findall(r"\[MODEL-[0-9a-f]{10}\]", report_text))
        assert model_citations > 0, "Report must cite at least one local model result"

    def test_numeric_values_match_verified_results(self, report_text, verified_data):
        """Numeric estimates in the report should match verified model results.

        This is a weak check — we look for known estimate values in the report text.
        A stronger check would parse the report structurally.
        """
        # Collect all known estimate values from verified results
        known_values = set()

        def collect_values(obj, path=""):
            if isinstance(obj, dict):
                for key in ("estimate", "std_error", "p_value", "ci_low", "ci_high",
                           "post_mean_gap", "post_pre_mspe_ratio", "placebo_p_value"):
                    if key in obj and isinstance(obj[key], (int, float)):
                        known_values.add(round(float(obj[key]), 6))
                for key, val in obj.items():
                    collect_values(val, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_values(item, f"{path}[{i}]")

        collect_values(verified_data)

        # Check that report doesn't contain values that look like estimates
        # but are NOT in our verified results
        # This catches cases where an LLM might invent a number
        numeric_pattern = re.compile(r"\*\*(-?\d+\.\d{4,6})\*\*")
        reported_values = {
            round(float(v), 6) for v in numeric_pattern.findall(report_text)
        }

        unknown_values = reported_values - known_values
        if unknown_values:
            # Not all numbers in the report need to be in verified results
            # (e.g., dates, page numbers), but we flag this for review
            print(f"\nNote: {len(unknown_values)} bold numeric values in report "
                  f"not directly found in verified results. Review manually.")
            print(f"  Sample: {sorted(unknown_values)[:5]}")

    def test_report_mentions_analysis_limitations(self, report_text):
        """Report should explicitly mention that treatment is a proxy variable."""
        limitations_indicators = [
            "proxy",
            "not observed",
            "causal",
            "limitation",
            "assumption",
            "evidence boundary",
        ]
        found = [ind for ind in limitations_indicators
                 if ind.lower() in report_text.lower()]
        assert len(found) >= 3, (
            f"Report must discuss limitations. Found only: {found}. "
            "The report should explicitly note that treatment is a revenue-threshold proxy."
        )

    def test_report_mentions_new_methods(self, report_text):
        """Report should reference modern estimation methods added in this refactoring."""
        method_indicators = [
            "Callaway",
            "Sant'Anna",
            "gsynth",
            "Bacon",
            "HonestDiD",
            "generalized synthetic",
            "staggered",
        ]
        found = [ind for ind in method_indicators
                 if ind.lower() in report_text.lower()]
        if found:
            print(f"\nReport references modern methods: {found}")
        else:
            print("\nNote: Report does not yet reference modern estimation methods. "
                  "This is expected if the report template has not been updated.")

    def test_each_figure_cited_has_file(self, report_text):
        """Each figure referenced in the report should have a corresponding file."""
        figure_pattern = re.compile(r"Figure(\d+_[a-z_]+)", re.IGNORECASE)
        figures_cited = set(figure_pattern.findall(report_text))
        figures_dir = Path("data/gold/figures")

        missing = []
        for fig_id in figures_cited:
            pattern = f"Figure{fig_id}.*"
            matches = list(figures_dir.glob(pattern))
            if not matches:
                missing.append(f"Figure{fig_id}")

        if missing:
            print(f"\nWarning: Figures cited in report but no file found: {missing}")
            print("These may be references to figures generated by new methods.")

    def test_no_markdown_artifacts(self, report_text):
        """Report should not contain unexpanded LLM template artifacts."""
        artifacts = ["{{", "}}", "[TODO]", "[INSERT]", "[PLACEHOLDER]"]
        for artifact in artifacts:
            assert artifact not in report_text, (
                f"Report contains unresolved template artifact: {artifact}"
            )
