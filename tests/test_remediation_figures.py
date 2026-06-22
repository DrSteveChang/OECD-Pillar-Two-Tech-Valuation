from pathlib import Path

import pandas as pd


NEW_FIGURES = {
    "Figure22_modern_method_applicability_matrix",
    "Figure23_treatment_timing_cohort_support",
    "Figure24_gsynth_balanced_panel_attrition",
    "Figure25_honestdid_preperiod_support",
    "Figure26_grf_sample_and_covariate_support",
    "Figure27_restricted_sample_did_comparison",
    "Figure28_event_confound_screen",
    "Figure29_cbcr_low_tax_context",
}


def test_remediation_figures_exist_as_png_and_pdf():
    for figure_id in NEW_FIGURES:
        assert Path(f"data/gold/figures/{figure_id}.png").exists()
        assert Path(f"data/gold/figures/{figure_id}.pdf").exists()


def test_figure_manifest_registers_remediation_boundaries():
    manifest = pd.read_csv("data/gold/figures/figure_manifest.csv")

    assert NEW_FIGURES.issubset(set(manifest["figure_id"]))
    new_rows = manifest[manifest["figure_id"].isin(NEW_FIGURES)]
    assert new_rows["interpretation_boundary"].notna().all()
    assert new_rows["interpretation_boundary"].str.len().gt(20).all()
