import json

import pytest

from oecd_pillar_two.rag import corpus as corpus_module


def test_build_corpus_enriches_literature_with_metadata(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    bronze = tmp_path / "data" / "bronze"
    literature = bronze / "literature"
    serving = tmp_path / "data" / "serving" / "ai"
    verified = tmp_path / "data" / "gold" / "statistical" / "verified"
    scoreboards = tmp_path / "data" / "gold" / "scoreboards"
    figures = tmp_path / "data" / "gold" / "figures"
    outputs = tmp_path / "outputs"
    for path in [literature, serving, verified, scoreboards, figures, outputs]:
        path.mkdir(parents=True)

    pdf_path = literature / "Example.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Pillar Two metadata test paragraph with enough content to be indexed. "
        "This page describes valuation, implementation, and tax technology impacts. "
        "The paragraph is intentionally long enough for corpus inclusion.",
    )
    document.save(pdf_path)
    document.close()

    (literature / "literature_metadata.json").write_text(
        json.dumps(
            {
                "Example.pdf": {
                    "institution": "Example Institute",
                    "institution_type": "academic",
                    "publication_date": "2026",
                    "title": "Example Pillar Two Study",
                    "key_topics": ["valuation", "implementation"],
                    "jurisdictions": ["EU", "Global"],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(corpus_module, "BRONZE", bronze)
    monkeypatch.setattr(corpus_module, "AI_SERVING", serving)
    monkeypatch.setattr(corpus_module, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(corpus_module, "SCOREBOARDS", scoreboards)
    monkeypatch.setattr(corpus_module, "FIGURES", figures)
    monkeypatch.setattr(corpus_module, "OUTPUTS", outputs)

    corpus = corpus_module.build_corpus()

    literature_rows = corpus[corpus["document_type"] == "literature"]
    assert not literature_rows.empty
    text = literature_rows.iloc[0]["text"]
    assert "institution: Example Institute" in text
    assert "key_topics: valuation, implementation" in text


def test_build_corpus_skips_malformed_figure_manifest_rows(tmp_path, monkeypatch):
    bronze = tmp_path / "data" / "bronze"
    serving = tmp_path / "data" / "serving" / "ai"
    verified = tmp_path / "data" / "gold" / "statistical" / "verified"
    scoreboards = tmp_path / "data" / "gold" / "scoreboards"
    figures = tmp_path / "data" / "gold" / "figures"
    outputs = tmp_path / "outputs"
    for path in [bronze / "literature", serving, verified, scoreboards, figures, outputs]:
        path.mkdir(parents=True)

    (figures / "figure_manifest.csv").write_text(
        "figure_id,title,evidence_role,interpretation_boundary,png_path,pdf_path\n"
        "Figure_arch,Architecture,architecture,Data pipeline feeding analysis, RAG, and reports,path.svg,path.svg\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(corpus_module, "BRONZE", bronze)
    monkeypatch.setattr(corpus_module, "AI_SERVING", serving)
    monkeypatch.setattr(corpus_module, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(corpus_module, "SCOREBOARDS", scoreboards)
    monkeypatch.setattr(corpus_module, "FIGURES", figures)
    monkeypatch.setattr(corpus_module, "OUTPUTS", outputs)

    corpus = corpus_module.build_corpus()

    assert "model_figure" not in set(corpus.get("document_type", []))


def test_build_corpus_includes_remediation_gold_tables(tmp_path, monkeypatch):
    bronze = tmp_path / "data" / "bronze"
    serving = tmp_path / "data" / "serving" / "ai"
    verified = tmp_path / "data" / "gold" / "statistical" / "verified"
    scoreboards = tmp_path / "data" / "gold" / "scoreboards"
    figures = tmp_path / "data" / "gold" / "figures"
    analytical = tmp_path / "data" / "gold" / "analytical"
    outputs = tmp_path / "outputs"
    for path in [bronze / "literature", serving, verified, scoreboards, figures, analytical, outputs]:
        path.mkdir(parents=True)

    for name in [
        "fact_modern_method_applicability.csv",
        "fact_event_confound_screen.csv",
        "fact_jurisdiction_policy_timing.csv",
        "fact_overlap_restricted_sample.csv",
    ]:
        (analytical / name).write_text("column\nvalue\n", encoding="utf-8")

    monkeypatch.setattr(corpus_module, "BRONZE", bronze)
    monkeypatch.setattr(corpus_module, "AI_SERVING", serving)
    monkeypatch.setattr(corpus_module, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(corpus_module, "SCOREBOARDS", scoreboards)
    monkeypatch.setattr(corpus_module, "FIGURES", figures)
    monkeypatch.setattr(corpus_module, "ANALYTICAL_GOLD", analytical)
    monkeypatch.setattr(corpus_module, "OUTPUTS", outputs)

    corpus = corpus_module.build_corpus()

    remediation_sources = set(corpus.loc[corpus["document_type"].eq("model_table"), "source"])
    assert {
        "fact_modern_method_applicability.csv",
        "fact_event_confound_screen.csv",
        "fact_jurisdiction_policy_timing.csv",
        "fact_overlap_restricted_sample.csv",
    }.issubset(remediation_sources)
