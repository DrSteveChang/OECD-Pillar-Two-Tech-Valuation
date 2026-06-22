from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oecd-pillar-two-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ANALYTICAL_GOLD, FIGURES as GOLD_FIGURES, PYTHON_RESULTS, SILVER, R_RESULTS


FIGURES = GOLD_FIGURES
BLUE = "#2F5597"
RED = "#C00000"
GOLD = "#BF9000"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9E2F3"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _policy_line(ax: plt.Axes) -> None:
    ax.axvline(pd.Timestamp("2024-01-01"), color=RED, linestyle="--", linewidth=1)


def _sample_coverage(monthly: pd.DataFrame) -> None:
    membership = monthly.groupby("Ticker")["pillar_two_in_scope_proxy"].first()
    coverage = monthly.groupby("Month")["Ticker"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = membership.value_counts().reindex([1, 0], fill_value=0)
    axes[0].bar(["In-scope proxy", "Control"], counts, color=[RED, BLUE], width=0.6)
    axes[0].set_ylabel("Number of firms")
    axes[0].set_title("A. Sample composition")
    for index, value in enumerate(counts):
        axes[0].text(index, value + 1, str(value), ha="center")
    dates = pd.to_datetime(coverage.index)
    axes[1].plot(dates, coverage, color=BLUE, linewidth=1.8)
    axes[1].set_ylabel("Firms with monthly observations")
    axes[1].set_title("B. Market-panel coverage")
    _policy_line(axes[1])
    _save(fig, "Figure01_sample_and_market_coverage")


def _market_trends(monthly: pd.DataFrame) -> None:
    trends = (
        monthly.groupby(["Month", "pillar_two_in_scope_proxy"], as_index=False)["abnormal_return"]
        .mean()
        .sort_values("Month")
    )
    trends["Date"] = pd.to_datetime(trends["Month"])
    trends["rolling_mean"] = trends.groupby("pillar_two_in_scope_proxy")["abnormal_return"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, label, color in [(1, "In-scope proxy", RED), (0, "Control", BLUE)]:
        part = trends[trends["pillar_two_in_scope_proxy"].eq(group)]
        ax.plot(part["Date"], part["rolling_mean"], label=label, color=color, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    _policy_line(ax)
    ax.set_ylabel("Three-month mean abnormal return")
    ax.set_title("Descriptive Abnormal-Return Trends by Exposure Group")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "Figure02_market_parallel_trends")


def _cumulative_market_returns(monthly: pd.DataFrame) -> None:
    trends = monthly.groupby(["Month", "pillar_two_in_scope_proxy"], as_index=False)["abnormal_return"].mean()
    trends = trends.sort_values("Month")
    trends["Date"] = pd.to_datetime(trends["Month"])
    trends["cumulative_abnormal_return"] = trends.groupby("pillar_two_in_scope_proxy")["abnormal_return"].cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, label, color in [(1, "In-scope proxy", RED), (0, "Control", BLUE)]:
        part = trends[trends["pillar_two_in_scope_proxy"].eq(group)]
        ax.plot(part["Date"], part["cumulative_abnormal_return"], label=label, color=color, linewidth=2)
    _policy_line(ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative mean abnormal log return")
    ax.set_title("Cumulative Abnormal Returns by Exposure Group")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "Figure03_cumulative_abnormal_returns")


def _did_validation() -> None:
    python = json.loads((PYTHON_RESULTS / "market_did.json").read_text())
    r = json.loads((R_RESULTS / "r_validation_results.json").read_text())["market_did"]
    estimates = [python["estimate"], r["estimate"]]
    errors = [python["std_error"], r["std_error"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(2)
    ax.errorbar(estimates, y, xerr=np.array(errors) * 1.96, fmt="o", color=BLUE, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, ["Python FE DiD", "R independent validation"])
    ax.set_xlabel("Monthly abnormal-return DiD estimate (95% CI)")
    ax.set_title("Python and R DiD Validation")
    ax.invert_yaxis()
    _save(fig, "Figure04_python_r_did_validation")


def _event_forest(event: pd.DataFrame) -> None:
    event = event.copy()
    event["label"] = event["event_id"].str.replace("_", " ").str.title() + " " + event["window"]
    event = event.sort_values(["event_id", "window"])
    colors = [RED if value else BLUE for value in event["statistically_significant_holm_5pct"]]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(event))
    for index, row in event.reset_index(drop=True).iterrows():
        ax.errorbar(
            row["difference"], index,
            xerr=[[row["difference"] - row["ci_low"]], [row["ci_high"] - row["difference"]]],
            fmt="o", color=colors[index], capsize=3,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, event["label"])
    ax.set_xlabel("Treated-minus-control CAR difference (95% CI)")
    ax.set_title("Event-Study Estimates across Policy Events and Windows")
    ax.invert_yaxis()
    _save(fig, "Figure05_event_study_forest")


def _event_car_distributions(cars: pd.DataFrame) -> None:
    short = cars[cars["window"].eq("[-1,+1]")].copy()
    events = sorted(short["event_id"].unique())
    positions, values, colors, labels = [], [], [], []
    for index, event_id in enumerate(events):
        for offset, group, color in [(-0.18, 0, BLUE), (0.18, 1, RED)]:
            positions.append(index + offset)
            values.append(short.loc[(short["event_id"].eq(event_id)) & (short["pillar_two_in_scope_proxy"].eq(group)), "car"])
            colors.append(color)
        labels.append(event_id.replace("_", " ").title())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    boxes = ax.boxplot(values, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(events)), labels, rotation=15, ha="right")
    ax.set_ylabel("Firm-level [-1,+1] CAR")
    ax.set_title("Short-Window CAR Distributions by Event and Exposure Group")
    ax.plot([], [], color=BLUE, linewidth=8, alpha=0.65, label="Control")
    ax.plot([], [], color=RED, linewidth=8, alpha=0.65, label="In-scope proxy")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "Figure06_event_car_distributions")


def _scm_figures(scm: pd.DataFrame) -> None:
    scm = scm.copy()
    scm["Date"] = pd.to_datetime(scm["Month"])
    scm["treated_cumulative"] = scm["treated_abnormal_return"].cumsum()
    scm["synthetic_cumulative"] = scm["synthetic_abnormal_return"].cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(scm["Date"], scm["treated_cumulative"], color=RED, linewidth=2, label="Treated proxy")
    ax.plot(scm["Date"], scm["synthetic_cumulative"], color=BLUE, linewidth=2, linestyle="--", label="Synthetic control")
    _policy_line(ax)
    ax.set_ylabel("Cumulative abnormal log return")
    ax.set_title("Synthetic-Control Market Trajectory")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "Figure07_scm_cumulative_trajectory")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(scm["Date"], scm["gap"], color=GOLD, linewidth=1.7)
    ax.fill_between(scm["Date"], 0, scm["gap"], color=GOLD, alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.8)
    _policy_line(ax)
    ax.set_ylabel("Treated-minus-synthetic abnormal return")
    ax.set_title("Synthetic-Control Monthly Gap")
    _save(fig, "Figure08_scm_monthly_gap")


def _scm_placebos_and_weights() -> None:
    placebos = pd.read_csv(PYTHON_RESULTS / "scm_placebo_ratios.csv")
    treated = placebos.loc[placebos["ticker"].eq("TREATED_PROXY"), "post_pre_mspe_ratio"].iloc[0]
    donors = placebos.loc[~placebos["ticker"].eq("TREATED_PROXY"), "post_pre_mspe_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(donors, bins=12, color=LIGHT_GRAY, edgecolor=BLUE)
    ax.axvline(treated, color=RED, linewidth=2, label=f"Treated proxy = {treated:.2f}")
    ax.set_xlabel("Post/pre MSPE ratio")
    ax.set_ylabel("Number of donor placebos")
    ax.set_title("SCM Placebo-Inference Distribution")
    ax.legend(frameon=False)
    _save(fig, "Figure09_scm_placebo_distribution")

    scm = json.loads((PYTHON_RESULTS / "scm.json").read_text())
    weights = pd.DataFrame(scm["top_weights"]).sort_values("weight")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(weights["ticker"], weights["weight"], color=BLUE)
    ax.set_xlabel("Synthetic-control weight")
    ax.set_title("Largest SCM Donor Weights")
    _save(fig, "Figure10_scm_donor_weights")


def _revenue_trends() -> None:
    panel = pd.read_csv(SILVER / "fact_firm_financial_year.csv").dropna(subset=["Log_Revenue"])
    summary = panel.groupby(["FiscalYear", "pillar_two_in_scope_proxy"])["Log_Revenue"].agg(["mean", "sem", "count"]).reset_index()
    summary = summary[summary["count"] >= 10]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for group, label, color in [(1, "In-scope proxy", RED), (0, "Control", BLUE)]:
        part = summary[summary["pillar_two_in_scope_proxy"].eq(group)]
        ax.errorbar(part["FiscalYear"], part["mean"], yerr=part["sem"] * 1.96, marker="o", color=color, capsize=3, label=label)
    ax.axvline(2024, color=RED, linestyle="--", linewidth=1)
    ax.set_xticks(sorted(summary["FiscalYear"].unique()))
    ax.set_ylabel("Mean log revenue (95% CI)")
    ax.set_title("Exploratory Revenue Trends by Exposure Group")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "Figure11_exploratory_revenue_trends")


def _assumption_diagnostic_figures() -> None:
    dynamic = pd.read_csv(PYTHON_RESULTS / "did_dynamic_coefficients.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        dynamic["calendar_year"], dynamic["estimate"],
        yerr=[dynamic["estimate"] - dynamic["ci_low"], dynamic["ci_high"] - dynamic["estimate"]],
        fmt="o-", color=BLUE, capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(2023, color=GRAY, linestyle=":", linewidth=1)
    ax.axvline(2024, color=RED, linestyle="--", linewidth=1)
    ax.set_xticks(dynamic["calendar_year"])
    ax.set_ylabel("Exposure-group differential vs. 2023 (95% CI)")
    ax.set_title("DiD Dynamic Coefficients and Pretrend Diagnostic")
    _save(fig, "Figure12_did_dynamic_pretrend_diagnostic")

    balance = pd.read_csv(PYTHON_RESULTS / "did_baseline_balance.csv").sort_values("standardized_mean_difference")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = [RED if abs(value) > 0.25 else BLUE for value in balance["standardized_mean_difference"]]
    ax.barh(balance["variable"], balance["standardized_mean_difference"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-0.25, color=GRAY, linestyle="--", linewidth=0.8)
    ax.axvline(0.25, color=GRAY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Standardized mean difference at 2022 baseline")
    ax.set_title("Baseline Covariate Balance and Overlap Diagnostic")
    _save(fig, "Figure13_did_baseline_covariate_balance")

    specs = pd.read_csv(PYTHON_RESULTS / "did_robustness_specifications.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(specs))
    ax.errorbar(specs["estimate"], y, xerr=1.96 * specs["std_error"], fmt="o", color=BLUE, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, specs["specification"].str.replace("_", " ").str.title())
    ax.set_xlabel("DiD estimate (95% CI)")
    ax.set_title("DiD Robustness across Benchmarks and Specifications")
    ax.invert_yaxis()
    _save(fig, "Figure14_did_robustness_specifications")

    sensitivity = pd.read_csv(PYTHON_RESULTS / "scm_leave_one_out_sensitivity.csv")
    scm = json.loads((PYTHON_RESULTS / "scm.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(sensitivity["excluded_donor"], sensitivity["post_mean_gap"], color=BLUE)
    ax.axhline(scm["post_mean_gap"], color=RED, linestyle="--", label="Full donor pool")
    ax.set_ylabel("Post-period mean treated-minus-synthetic gap")
    ax.set_title("SCM Leave-One-Donor-Out Sensitivity")
    ax.legend(frameon=False)
    _save(fig, "Figure15_scm_leave_one_out_sensitivity")


def _exposure_design_figures() -> None:
    design = pd.read_csv(ANALYTICAL_GOLD / "fact_exposure_score.csv")
    eligible = design[design["eligible_for_main_design"].eq(True)].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    eligible = eligible.sort_values("median_pre_policy_revenue_eur")
    colors = np.where(eligible["pillar_two_four_year_scope_proxy"].eq(1), RED, BLUE)
    ax.scatter(eligible["median_pre_policy_revenue_eur"] / 1e9, np.arange(len(eligible)), c=colors, s=20)
    ax.axvline(0.75, color=GOLD, linestyle="--", label="EUR 750m threshold")
    ax.set_xlabel("Median 2020-2023 revenue (EUR bn)")
    ax.set_ylabel("Firms ordered by revenue")
    ax.set_title("Four-Year Scope Classification around the Pillar Two Threshold")
    ax.legend(frameon=False)
    _save(fig, "Figure16_four_year_scope_threshold")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, label, color in [(1, "Four-year in-scope proxy", RED), (0, "Out-of-scope proxy", BLUE)]:
        values = eligible.loc[eligible["pillar_two_four_year_scope_proxy"].eq(group), "pre_policy_tax_exposure_score"].dropna()
        ax.hist(values, bins=12, alpha=0.55, color=color, label=label)
    ax.set_xlabel("Pre-policy tax-exposure score")
    ax.set_ylabel("Number of firms")
    ax.set_title("Distribution of Constructed Pre-Policy Tax Exposure")
    ax.legend(frameon=False)
    _save(fig, "Figure17_pre_policy_exposure_distribution")

    event = pd.read_csv(PYTHON_RESULTS / "exposure_event_study.csv")
    if not event.empty:
        event["label"] = event["event_id"].str.replace("_", " ").str.title() + " " + event["window"]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        y = np.arange(len(event))
        ax.errorbar(event["estimate"], y, xerr=[event["estimate"] - event["ci_low"], event["ci_high"] - event["estimate"]], fmt="o", color=RED, capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, event["label"])
        ax.set_xlabel("CAR change per unit of exposure score (95% CI)")
        ax.set_title("Main Model: Continuous Exposure-Intensity Event Study")
        ax.invert_yaxis()
        _save(fig, "Figure18_exposure_event_study")

    propensity = pd.read_csv(PYTHON_RESULTS / "propensity_scores.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, label, color in [(1, "Four-year in-scope proxy", RED), (0, "Out-of-scope proxy", BLUE)]:
        values = propensity.loc[propensity["pillar_two_four_year_scope_proxy"].eq(group), "propensity_score"]
        ax.hist(values, bins=12, alpha=0.5, color=color, label=label)
    ax.set_xlabel("Estimated propensity score")
    ax.set_title("Common-Support Diagnostic before Overlap Weighting")
    ax.legend(frameon=False)
    _save(fig, "Figure19_propensity_common_support")

    balance = pd.read_csv(PYTHON_RESULTS / "weighted_did_balance.csv").sort_values("smd_unweighted")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(balance))
    ax.scatter(balance["smd_unweighted"], y + 0.12, color=GRAY, label="Unweighted")
    ax.scatter(balance["smd_overlap_weighted"], y - 0.12, color=BLUE, label="Overlap weighted")
    ax.axvline(-0.1, color=GRAY, linestyle="--", linewidth=0.8)
    ax.axvline(0.1, color=GRAY, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, balance["variable"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate Balance before and after Overlap Weighting")
    ax.legend(frameon=False)
    _save(fig, "Figure20_weighted_covariate_balance")

    dynamic = pd.read_csv(PYTHON_RESULTS / "weighted_did_dynamic.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(dynamic["calendar_year"], dynamic["estimate"], yerr=[dynamic["estimate"] - dynamic["ci_low"], dynamic["ci_high"] - dynamic["estimate"]], fmt="o-", color=BLUE, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(2024, color=RED, linestyle="--")
    ax.set_xticks(dynamic["calendar_year"])
    ax.set_ylabel("Weighted group differential vs. 2023")
    ax.set_title("Overlap-Weighted DiD Parallel-Trends Diagnostic")
    _save(fig, "Figure21_weighted_did_pretrend")


def _remediation_figures() -> None:
    applicability = pd.read_csv(ANALYTICAL_GOLD / "fact_modern_method_applicability.csv")
    methods = applicability.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = np.where(methods["applicable"], BLUE, GRAY)
    ax.barh(methods["method_label"], np.ones(len(methods)), color=colors)
    for index, row in enumerate(methods.itertuples(index=False)):
        label = "Applicable" if row.applicable else "Not applicable"
        ax.text(0.03, index, label, va="center", ha="left", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Modern Method Applicability under Current Data")
    ax.set_xlabel("Blue = currently supported; gray = diagnostic/not applicable")
    _save(fig, "Figure22_modern_method_applicability_matrix")

    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv")
    firm_groups = monthly.groupby("Ticker", as_index=False)["pillar_two_in_scope_proxy"].first()
    firm_groups["cohort"] = np.where(firm_groups["pillar_two_in_scope_proxy"].eq(1), "Proxy treated in 2024", "Control")
    counts = firm_groups["cohort"].value_counts().reindex(["Proxy treated in 2024", "Control"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(counts.index, counts.values, color=[RED, BLUE], width=0.55)
    for index, value in enumerate(counts.values):
        ax.text(index, value + 0.5, str(int(value)), ha="center")
    ax.set_ylabel("Firms")
    ax.set_title("Treatment-Timing Support: One Proxy Cohort Only")
    ax.text(
        0.5,
        0.88,
        "No firm-jurisdiction staggered exposure file",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
    )
    _save(fig, "Figure23_treatment_timing_cohort_support")

    months_required = monthly["Month"].nunique()
    complete = monthly.groupby("Ticker")["Month"].nunique().eq(months_required)
    restricted = pd.read_csv(ANALYTICAL_GOLD / "fact_overlap_restricted_sample.csv")
    attrition = pd.Series(
        {
            "Market-panel firms": monthly["Ticker"].nunique(),
            "Balanced firms": int(complete.sum()),
            "Eligible design firms": int(restricted["eligible_for_main_design"].sum()),
            "Overlap-restricted firms": int(restricted["restricted_sample_flag"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(attrition.index, attrition.values, color=[BLUE, BLUE, GOLD, RED])
    ax.set_ylabel("Firms")
    ax.set_title("Balanced-Panel and Overlap Attrition for gsynth Diagnostics")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "Figure24_gsynth_balanced_panel_attrition")

    dynamic = pd.read_csv(PYTHON_RESULTS / "did_dynamic_coefficients.csv")
    pre_count = int(dynamic["calendar_year"].lt(2024).sum())
    post_count = int(dynamic["calendar_year"].ge(2024).sum())
    sa_supported = int((ANALYTICAL_GOLD / "fact_firm_jurisdiction_pillar_two_exposure.csv").exists())
    support = pd.Series({"Pre-period coefficients": pre_count, "Post coefficients": post_count, "Staggered exposure file": sa_supported})
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(support.index, support.values, color=[BLUE, GOLD, GRAY])
    ax.set_ylabel("Available count")
    ax.set_title("HonestDiD Input Support")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "Figure25_honestdid_preperiod_support")

    covariates = ["Firm_Size", "ETR", "RD_Intensity"]
    missing = restricted[covariates].isna().sum().sort_values()
    group_counts = restricted.groupby("pillar_two_in_scope_proxy")["Ticker"].nunique().reindex([1, 0], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].bar(["Treated proxy", "Control"], group_counts.values, color=[RED, BLUE])
    axes[0].set_title("A. Treatment Groups")
    axes[0].set_ylabel("Firms")
    axes[1].barh(missing.index, missing.values, color=GOLD)
    axes[1].set_title("B. Missing Core Covariates")
    axes[1].set_xlabel("Firms")
    fig.suptitle("GRF Sample and Covariate Support")
    _save(fig, "Figure26_grf_sample_and_covariate_support")

    full = json.loads((PYTHON_RESULTS / "market_did.json").read_text())
    restricted_result = json.loads((PYTHON_RESULTS / "restricted_sample_did.json").read_text())
    labels = ["Full sample DiD", "Restricted overlap DiD"]
    estimates = [full.get("estimate", np.nan), restricted_result.get("estimate", np.nan)]
    errors = [full.get("std_error", np.nan), restricted_result.get("std_error", np.nan)]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = np.arange(len(labels))
    for index, (estimate, error) in enumerate(zip(estimates, errors)):
        if pd.notna(estimate) and pd.notna(error):
            ax.errorbar(estimate, index, xerr=1.96 * error, fmt="o", color=BLUE if index == 0 else RED, capsize=4)
        else:
            ax.text(0, index, restricted_result.get("status", "not_estimated"), va="center", ha="center", color=GRAY)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Monthly abnormal-return DiD estimate (95% CI)")
    ax.set_title("Full Sample vs Restricted Overlap-Sample DiD")
    _save(fig, "Figure27_restricted_sample_did_comparison")

    screen = pd.read_csv(ANALYTICAL_GOLD / "fact_event_confound_screen.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    event_labels = screen["event_id"].str.replace("_", " ").str.title()
    colors = np.where(screen["market_wide_volatility_flag"], RED, BLUE)
    ax.bar(event_labels, screen["abnormal_return_dispersion"], color=colors)
    ax.set_ylabel("Abnormal-return dispersion in [-3,+3]")
    ax.set_title("Policy-Event Confound Screen")
    ax.tick_params(axis="x", rotation=18)
    _save(fig, "Figure28_event_confound_screen")

    cbcr = pd.read_csv(SILVER / "fact_cbcr_jurisdiction_year.csv")
    low_tax = cbcr[
        cbcr["profit_before_tax"].gt(0)
        & cbcr["cash_etr"].between(0, 0.15)
    ].copy()
    context = (
        low_tax.groupby("Counterpart area", as_index=False)
        .agg(mean_cash_etr=("cash_etr", "mean"), positive_profit_rows=("profit_before_tax", "count"))
        .sort_values(["positive_profit_rows", "mean_cash_etr"], ascending=[False, True])
        .head(12)
        .sort_values("positive_profit_rows")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(context["Counterpart area"], context["positive_profit_rows"], color=BLUE)
    ax.set_xlabel("Positive-profit CbCR rows with cash ETR below 15%")
    ax.set_title("CbCR Low-Tax Jurisdiction Context")
    _save(fig, "Figure29_cbcr_low_tax_context")


def _write_manifest() -> None:
    records = [
        ("Figure01_sample_and_market_coverage", "Sample composition and data coverage", "descriptive", "Shows sample balance and usable observations."),
        ("Figure02_market_parallel_trends", "Abnormal-return trends", "diagnostic", "Descriptive trends; not a formal parallel-trends test."),
        ("Figure03_cumulative_abnormal_returns", "Cumulative abnormal returns", "descriptive", "Unadjusted group means; formal inference is in DiD."),
        ("Figure04_python_r_did_validation", "Python/R DiD validation", "formal", "Displays the main persistent-effect estimate and 95% confidence intervals."),
        ("Figure05_event_study_forest", "Event-study forest plot", "formal", "Short-window associations across events and windows."),
        ("Figure06_event_car_distributions", "Event CAR distributions", "diagnostic", "Firm-level distribution for the [-1,+1] window."),
        ("Figure07_scm_cumulative_trajectory", "SCM cumulative trajectory", "complementary", "Treatment is a revenue-threshold exposure proxy."),
        ("Figure08_scm_monthly_gap", "SCM monthly gap", "complementary", "Treatment is a revenue-threshold exposure proxy."),
        ("Figure09_scm_placebo_distribution", "SCM placebo inference", "formal_complementary", "Compares treated proxy MSPE ratio with donor placebos."),
        ("Figure10_scm_donor_weights", "SCM donor weights", "diagnostic", "Documents synthetic-control composition."),
        ("Figure11_exploratory_revenue_trends", "Revenue trends", "exploratory", "Only one usable pre-implementation fiscal year."),
        ("Figure12_did_dynamic_pretrend_diagnostic", "DiD dynamic pretrend diagnostic", "assumption_diagnostic", "Pretrend rejection weakens causal interpretation."),
        ("Figure13_did_baseline_covariate_balance", "DiD baseline covariate balance", "assumption_diagnostic", "Large standardized differences indicate poor comparability."),
        ("Figure14_did_robustness_specifications", "DiD robustness specifications", "robustness", "Specifications remain associational when identifying assumptions are weak."),
        ("Figure15_scm_leave_one_out_sensitivity", "SCM leave-one-donor-out sensitivity", "assumption_diagnostic", "Assesses dependence on the largest donor weights."),
        ("Figure16_four_year_scope_threshold", "Four-year scope classification", "design_diagnostic", "Scope requires all four years and at least two years above EUR 750m."),
        ("Figure17_pre_policy_exposure_distribution", "Pre-policy exposure distribution", "design_diagnostic", "Exposure is a constructed proxy, not observed top-up tax."),
        ("Figure18_exposure_event_study", "Continuous exposure event study", "main_model", "Conditional short-window association; concurrent news remains possible."),
        ("Figure19_propensity_common_support", "Propensity common support", "assumption_diagnostic", "Shows whether treated and controls are comparable on observed covariates."),
        ("Figure20_weighted_covariate_balance", "Weighted covariate balance", "assumption_diagnostic", "Weighting addresses observed covariates only."),
        ("Figure21_weighted_did_pretrend", "Weighted DiD pretrend", "assumption_diagnostic", "Failure to reject does not prove parallel trends."),
        ("Figure22_modern_method_applicability_matrix", "Modern method applicability", "design_diagnostic", "Shows which modern estimators are supported by current data and which remain not applicable."),
        ("Figure23_treatment_timing_cohort_support", "Treatment-timing cohort support", "design_diagnostic", "Current proxy has one implementation cohort and no firm-jurisdiction staggered exposure."),
        ("Figure24_gsynth_balanced_panel_attrition", "gsynth balanced-panel attrition", "design_diagnostic", "Documents sample attrition before interpreting any generalized synthetic-control output."),
        ("Figure25_honestdid_preperiod_support", "HonestDiD pre-period support", "design_diagnostic", "HonestDiD requires credible event-study coefficients; this chart reports input support only."),
        ("Figure26_grf_sample_and_covariate_support", "GRF sample and covariate support", "heterogeneity_diagnostic", "GRF results are diagnostics because treatment proxy and overlap limitations remain."),
        ("Figure27_restricted_sample_did_comparison", "Restricted-sample DiD comparison", "diagnostic_sensitivity", "Restricted overlap estimates are not promoted to the main causal result."),
        ("Figure28_event_confound_screen", "Event confound screen", "event_diagnostic", "High-volatility flags are diagnostics only and do not delete policy events."),
        ("Figure29_cbcr_low_tax_context", "CbCR low-tax context", "context", "CbCR jurisdiction context does not create firm-level Pillar Two treatment exposure."),
    ]
    frame = pd.DataFrame(records, columns=["figure_id", "title", "evidence_role", "interpretation_boundary"])
    frame["png_path"] = frame["figure_id"].map(lambda value: f"data/gold/figures/{value}.png")
    frame["pdf_path"] = frame["figure_id"].map(lambda value: f"data/gold/figures/{value}.pdf")
    frame.to_csv(FIGURES / "figure_manifest.csv", index=False)


def generate_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _style()
    monthly = pd.read_csv(SILVER / "fact_market_monthly.csv")
    monthly = monthly[monthly["trading_days"] >= 10].copy()
    _sample_coverage(monthly)
    _market_trends(monthly)
    _cumulative_market_returns(monthly)
    _did_validation()
    _event_forest(pd.read_csv(PYTHON_RESULTS / "event_study_summary.csv"))
    _event_car_distributions(pd.read_csv(ANALYTICAL_GOLD / "fact_event_firm_car.csv"))
    _scm_figures(pd.read_csv(PYTHON_RESULTS / "scm_monthly_trajectory.csv"))
    _scm_placebos_and_weights()
    _revenue_trends()
    _assumption_diagnostic_figures()
    _exposure_design_figures()
    _remediation_figures()
    _write_manifest()
    for legacy_name in ("event_study_car.png", "scm_market_trajectory.png"):
        legacy_path = PYTHON_RESULTS / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
