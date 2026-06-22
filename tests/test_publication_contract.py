from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_github_snapshot_excludes_private_and_large_artifacts():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    required_patterns = {
        "data/bronze/",
        "data/silver/",
        "data/serving/ai/vector_store/",
        "outputs/",
        "legacy/",
        ".kunsdd/",
        "docs/TFM_*.pdf",
        "scripts/build_tfm_thesis.py",
        "scripts/generate_architecture_diagrams.py",
        "scripts/render_ai_report_artifacts.py",
        "scripts/update_tfm_navigation_pages.py",
        "tests/test_tfm_*.py",
        "tests/test_ai_report_artifacts.py",
        "tests/test_architecture_diagrams.py",
    }

    assert required_patterns.issubset(set(ignore.splitlines()))


def test_public_dependencies_do_not_include_thesis_rendering_packages():
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "python-docx" not in manifest
    assert '"Pillow' not in manifest
    assert "PyMuPDF" in manifest


def test_readme_describes_the_actual_local_hybrid_index():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_text = {
        "local hybrid retrieval index",
        "not ChromaDB",
        "all-MiniLM-L6-v2",
        "exact cosine similarity",
        "citation registry",
        "evidence graph",
        "ALLOW_EXTERNAL_LLM_REWRITE",
    }

    assert all(text in readme for text in required_text)
