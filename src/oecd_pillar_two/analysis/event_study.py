from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..config import ANALYTICAL_GOLD, PYTHON_RESULTS
from ..utils import write_json


WINDOWS = (1, 3, 5)


def run_event_study() -> dict:
    estimates = pd.read_csv(ANALYTICAL_GOLD / "fact_event_firm_car.csv")
    summaries = []
    for (event_id, window), group in estimates.groupby(["event_id", "window"]):
        treated = group.loc[group["pillar_two_in_scope_proxy"].eq(1), "car"].dropna()
        control = group.loc[group["pillar_two_in_scope_proxy"].eq(0), "car"].dropna()
        test = stats.ttest_ind(treated, control, equal_var=False)
        diff = treated.mean() - control.mean()
        std_error = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        summaries.append(
            {
                "event_id": event_id,
                "window": window,
                "treated_n": len(treated),
                "control_n": len(control),
                "treated_mean_car": treated.mean(),
                "control_mean_car": control.mean(),
                "difference": diff,
                "std_error": std_error,
                "ci_low": diff - 1.96 * std_error,
                "ci_high": diff + 1.96 * std_error,
                "p_value": test.pvalue,
                "statistically_significant_5pct": bool(test.pvalue < 0.05),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["p_value_bh"] = stats.false_discovery_control(summary["p_value"].to_numpy(), method="bh")
    summary["p_value_holm"] = multipletests(summary["p_value"].to_numpy(), method="holm")[1]
    summary["statistically_significant_bh_5pct"] = summary["p_value_bh"] < 0.05
    summary["statistically_significant_holm_5pct"] = summary["p_value_holm"] < 0.05
    summary.to_csv(PYTHON_RESULTS / "event_study_summary.csv", index=False)
    payload = {
        "method": "market-adjusted event study",
        "events": summary.to_dict("records"),
        "assumptions_and_limitations": {
            "event_exogeneity": "Policy dates are externally dated, but concurrent market news within each window is not fully controlled.",
            "benchmark_model": "Abnormal return is stock log return minus QQQ log return; alternative expected-return models may differ.",
            "cross_firm_dependence": "Welch tests do not fully model cross-firm dependence from common shocks.",
            "multiple_testing": "Benjamini-Hochberg and Holm adjusted p-values are reported across all event-window comparisons.",
        },
    }
    write_json(PYTHON_RESULTS / "event_study.json", payload)
    return payload
