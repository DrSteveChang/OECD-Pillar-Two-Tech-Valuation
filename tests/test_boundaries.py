from pathlib import Path


def test_formal_code_does_not_read_quarantine():
    for path in Path("src/oecd_pillar_two").rglob("*.py"):
        if path.name == "manifest.py":
            continue
        assert "legacy_invalid" not in path.read_text(encoding="utf-8")


def test_no_hardcoded_treatment_effects():
    text = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/oecd_pillar_two").rglob("*.py"))
    assert "macro_tau =" not in text
    assert "micro_att =" not in text


def test_cited_report_has_pdf_and_model_citations():
    from oecd_pillar_two.reporting.citations import validate_report_citations

    path = Path("outputs/ai_reports/latest_verified_decision_support_report.md")
    if path.exists():
        result = validate_report_citations(path.read_text(encoding="utf-8"))
        assert result["valid"], result["errors"]
        assert result["pdf_citations"] >= 3
        assert result["model_citations"] >= 5
        corpus = __import__("pandas").read_csv("data/serving/ai/rag_corpus.csv")
        figure_ids = set(corpus.loc[corpus["document_type"].eq("model_figure"), "citation_id"])
        assert len(figure_ids) >= 10
        assert all(f"[{citation_id}]" in path.read_text(encoding="utf-8") for citation_id in figure_ids)
        verified_report = path.read_text(encoding="utf-8")
        assert all(f"[{citation_id}]" in verified_report for citation_id in figure_ids)


def test_formal_outputs_are_centralized():
    outputs = Path("outputs")
    assert not (outputs / "model_results").exists()
    assert not (outputs / "tables").exists()
    assert not (outputs / "figures").exists()
    assert not (outputs / "formal_results").exists()
    assert Path("data/gold/statistical/python").exists()


def test_publication_figures_are_complete_and_centralized():
    figures = Path("data/gold/figures")
    if figures.exists():
        png_stems = {path.stem for path in figures.glob("*.png")}
        pdf_stems = {path.stem for path in figures.glob("*.pdf")}
        assert len(png_stems) >= 10
        assert png_stems == pdf_stems
        assert Path("data/gold/figures/figure_manifest.csv").exists()


def test_assumption_diagnostics_are_required_outputs():
    import json

    formal = Path("data/gold/statistical/python")
    required = [
        "did_assumption_diagnostics.json",
        "did_dynamic_coefficients.csv",
        "did_baseline_balance.csv",
        "did_robustness_specifications.csv",
        "scm_leave_one_out_sensitivity.csv",
    ]
    assert all((formal / name).exists() for name in required)
    assert Path("data/gold/scoreboards/assumption_scoreboard.csv").exists()
    verified = json.loads(Path("data/gold/statistical/verified/verified_model_results.json").read_text(encoding="utf-8"))
    assert verified["status"] == "computationally_replicated"
    assert verified["causal_assumption_status"] == "material_concerns"
