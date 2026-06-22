import json

import pandas as pd

from oecd_pillar_two.reporting import scoreboard as scoreboard_module


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scoreboard_modern_method_rows_follow_applicability_table(tmp_path, monkeypatch):
    analytical = tmp_path / "gold" / "analytical"
    python_results = tmp_path / "gold" / "statistical" / "python"
    r_results = tmp_path / "gold" / "statistical" / "r_validation"
    verified = tmp_path / "gold" / "statistical" / "verified"
    scoreboards = tmp_path / "gold" / "scoreboards"
    for path in [analytical, python_results, r_results, verified, scoreboards]:
        path.mkdir(parents=True)

    _write_json(
        python_results / "python_model_results.json",
        {
            "market_did": {"estimate": 0.0, "std_error": 0.1, "p_value": 0.9, "assumption_diagnostics": {}},
            "scm": {},
            "revenue_mechanism": {},
            "exposure_event_study": {"events": []},
            "weighted_did": {"assumption_diagnostics": {}},
            "event_study": {"events": []},
        },
    )
    _write_json(python_results / "restricted_sample_did.json", {"status": "not_estimated"})
    _write_json(r_results / "r_validation_results.json", {"market_did": {}})
    pd.DataFrame([{"specification": "baseline", "estimate": 0.0, "p_value": 1.0}]).to_csv(
        python_results / "did_robustness_specifications.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "method_id": "cs_did",
                "method_label": "Callaway and Sant'Anna DiD",
                "applicable": False,
                "failure_reason": "Missing firm-jurisdiction exposure.",
                "paper_interpretation": "Do not estimate.",
            },
            {
                "method_id": "gsynth",
                "method_label": "Generalized synthetic control",
                "applicable": False,
                "failure_reason": "No stable factor fit.",
                "paper_interpretation": "Diagnostic only.",
            },
        ]
    ).to_csv(analytical / "fact_modern_method_applicability.csv", index=False)

    monkeypatch.setattr(scoreboard_module, "ANALYTICAL_GOLD", analytical)
    monkeypatch.setattr(scoreboard_module, "PYTHON_RESULTS", python_results)
    monkeypatch.setattr(scoreboard_module, "R_RESULTS", r_results)
    monkeypatch.setattr(scoreboard_module, "VERIFIED_RESULTS", verified)
    monkeypatch.setattr(scoreboard_module, "SCOREBOARDS", scoreboards)

    scoreboard = scoreboard_module.generate_scoreboard()
    modern = scoreboard[scoreboard["analysis"].isin(["Callaway and Sant'Anna DiD", "Generalized synthetic control"])]

    assert not modern.empty
    assert modern["evidence_role"].eq("not_applicable_diagnostic").all()
    assert set(modern["assumption_status"]) == {"Missing firm-jurisdiction exposure.", "No stable factor fit."}
    assert "main_model" not in set(modern["evidence_role"])
