from oecd_pillar_two import cli


def test_analyze_stage_rebuilds_remediation_outputs_after_cleanup(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "ensure_directories", lambda: calls.append("ensure"))
    monkeypatch.setattr(cli, "clear_previous_analysis_outputs", lambda: calls.append("clear"))
    monkeypatch.setattr(cli, "build_design_remediation", lambda: calls.append("remediation"))
    monkeypatch.setattr(cli, "run_all_analysis", lambda: calls.append("analysis"))
    monkeypatch.setattr(cli, "build_python_gold_facts", lambda: calls.append("python_gold"))

    cli.run_stage("analyze")

    assert calls == ["ensure", "clear", "remediation", "analysis", "python_gold"]


def test_report_stage_builds_metadata_and_evidence_graph(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "ensure_directories", lambda: calls.append("ensure"))
    monkeypatch.setattr(cli, "clear_previous_deliverables", lambda: calls.append("clear"))
    monkeypatch.setattr(cli, "build_literature_metadata", lambda: calls.append("literature"))
    monkeypatch.setattr(cli, "build_corpus", lambda: calls.append("corpus"))
    monkeypatch.setattr(
        cli,
        "build_vector_index",
        lambda **kwargs: calls.append(("index", kwargs)),
    )
    monkeypatch.setattr(cli, "generate_ai_report", lambda: calls.append("ai_report"))
    monkeypatch.setattr(cli, "build_evidence_graph", lambda: calls.append("evidence"))
    monkeypatch.setattr(cli, "build_manifest", lambda: calls.append("manifest"))

    import oecd_pillar_two.reporting.figures as figures
    import oecd_pillar_two.reporting.scoreboard as scoreboard

    monkeypatch.setattr(figures, "generate_figures", lambda: calls.append("figures"))
    monkeypatch.setattr(scoreboard, "generate_scoreboard", lambda: calls.append("scoreboard"))

    cli.run_stage("report")

    assert calls == [
        "ensure",
        "clear",
        "figures",
        "scoreboard",
        "literature",
        "corpus",
        ("index", {"require_embeddings": True}),
        "ai_report",
        "evidence",
        "manifest",
    ]
