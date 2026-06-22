from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BRONZE, REFERENCE, SILVER, load_config


def build_market_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(BRONZE / "yahoo" / "daily_prices.csv", parse_dates=["Date"])
    registry = REFERENCE / "analysis_registry.csv"
    sample = pd.read_csv(registry if registry.exists() else REFERENCE / "sample_registry.csv").rename(
        columns={"Treatment_Group": "pillar_two_in_scope_proxy"}
    )
    prices = prices.dropna(subset=["Date", "Ticker", "Adj_Close"]).copy()
    prices = prices.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"])
    prices["log_return"] = prices.groupby("Ticker")["Adj_Close"].transform(
        lambda values: np.log(values).diff()
    )
    daily = prices.copy()
    for ticker, suffix in (("QQQ", "qqq"), ("SPY", "spy"), ("XLK", "xlk")):
        benchmark = prices[prices["Ticker"].eq(ticker)][["Date", "log_return"]].rename(
            columns={"log_return": f"benchmark_return_{suffix}"}
        )
        daily = daily.merge(benchmark, on="Date", how="left")
        daily[f"abnormal_return_{suffix}"] = daily["log_return"] - daily[f"benchmark_return_{suffix}"]
    daily["benchmark_return"] = daily["benchmark_return_qqq"]
    daily["abnormal_return"] = daily["abnormal_return_qqq"]
    daily = daily.merge(
        sample[["Ticker", "pillar_two_in_scope_proxy", "pre_policy_tax_exposure_score", "pillar_two_exposure_intensity"]],
        on="Ticker", how="left"
    )
    daily["firm_id"] = "FIRM_" + daily["Ticker"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    daily.to_csv(SILVER / "fact_market_daily.csv", index=False)

    monthly = (
        daily.dropna(subset=["pillar_two_in_scope_proxy"])
        .assign(Month=lambda frame: frame["Date"].dt.to_period("M").astype(str))
        .groupby(
            ["Ticker", "Month", "pillar_two_in_scope_proxy", "pre_policy_tax_exposure_score", "pillar_two_exposure_intensity"],
            as_index=False,
        )
        .agg(
            monthly_return=("log_return", "sum"),
            benchmark_return=("benchmark_return", "sum"),
            abnormal_return=("abnormal_return", "sum"),
            abnormal_return_spy=("abnormal_return_spy", "sum"),
            abnormal_return_xlk=("abnormal_return_xlk", "sum"),
            trading_days=("Date", "count"),
        )
    )
    post = load_config()["project"]["policy_post_month"]
    monthly["Post"] = (monthly["Month"] >= post).astype(int)
    monthly["DiD"] = monthly["pillar_two_in_scope_proxy"] * monthly["Post"]
    monthly["Exposure_Post"] = monthly["pillar_two_exposure_intensity"] * monthly["Post"]
    monthly["firm_id"] = "FIRM_" + monthly["Ticker"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    monthly.to_csv(SILVER / "fact_market_monthly.csv", index=False)
    return daily, monthly
