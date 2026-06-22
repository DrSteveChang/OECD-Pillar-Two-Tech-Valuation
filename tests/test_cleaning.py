from pathlib import Path

import pandas as pd
import pytest

from oecd_pillar_two.cleaning.firm_panel import build_firm_year_panel
from oecd_pillar_two.cleaning.oecd import build_cbcr_panel


pytestmark = pytest.mark.skipif(
    not (
        Path("data/bronze/oecd/oecd_cbcr_raw.csv").exists()
        and Path("data/bronze/yahoo/corporate_financials.csv").exists()
    ),
    reason="requires the locally regenerated Bronze data layer",
)


def test_firm_year_panel_is_unique():
    panel = build_firm_year_panel()
    assert not panel.duplicated(["Ticker", "FiscalYear"]).any()
    assert panel["FiscalYear"].max() < 2026


def test_cbcr_panel_is_unique():
    panel = build_cbcr_panel()
    keys = ["REF_AREA", "COUNTERPART_AREA", "TIME_PERIOD"]
    assert not panel.duplicated(keys).any()
