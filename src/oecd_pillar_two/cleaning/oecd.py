from __future__ import annotations

import pandas as pd

from ..config import BRONZE, SILVER


MEASURES = {
    "Profit (loss) before income tax": "profit_before_tax",
    "Income tax paid (on cash basis)": "cash_tax_paid",
    "Income tax accrued - current year": "current_tax_accrued",
    "Total revenues": "total_revenues",
}


def build_cbcr_panel() -> pd.DataFrame:
    source = BRONZE / "oecd" / "oecd_cbcr_raw.csv"
    columns = [
        "REF_AREA", "Reference area", "COUNTERPART_AREA", "Counterpart area",
        "MEASURE", "Measure", "PROFIT_GROUPING", "TIME_PERIOD", "OBS_VALUE",
    ]
    frame = pd.read_csv(source, usecols=columns, low_memory=False)
    frame = frame[
        frame["Measure"].isin(MEASURES) & frame["PROFIT_GROUPING"].eq("_T")
    ].copy()
    frame["metric"] = frame["Measure"].map(MEASURES)
    frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    keys = ["REF_AREA", "Reference area", "COUNTERPART_AREA", "Counterpart area", "TIME_PERIOD"]
    if frame.duplicated(keys + ["metric"]).any():
        raise ValueError("OECD CbCR metric keys are not unique")
    panel = frame.pivot(index=keys, columns="metric", values="OBS_VALUE").reset_index()
    panel["cash_etr"] = (panel["cash_tax_paid"] / panel["profit_before_tax"]).where(
        panel["profit_before_tax"] > 0
    )
    panel["accrued_etr"] = (
        panel["current_tax_accrued"] / panel["profit_before_tax"]
    ).where(panel["profit_before_tax"] > 0)
    panel.to_csv(SILVER / "fact_cbcr_jurisdiction_year.csv", index=False)
    return panel
