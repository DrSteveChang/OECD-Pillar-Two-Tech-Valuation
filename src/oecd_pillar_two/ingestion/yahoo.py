from __future__ import annotations

import time
import pandas as pd
from yahooquery import Ticker

from ..config import BRONZE, REFERENCE, load_config
from ..utils import utc_now, write_json


def download_prices(force: bool = False) -> pd.DataFrame:
    output = BRONZE / "yahoo" / "daily_prices.csv"
    if output.exists() and not force:
        return pd.read_csv(output, parse_dates=["Date"])

    config = load_config()["project"]
    registry = REFERENCE / "analysis_registry.csv"
    sample = pd.read_csv(registry if registry.exists() else REFERENCE / "sample_registry.csv")
    tickers = sample["Ticker"].dropna().unique().tolist() + config["benchmarks"]
    failures: list[str] = []
    frames: list[pd.DataFrame] = []

    for start in range(0, len(tickers), 20):
        batch = tickers[start : start + 20]
        try:
            result = Ticker(batch, asynchronous=True).history(
                start=config["price_start"], end=config["price_end"], interval="1d"
            )
            if not isinstance(result, pd.DataFrame) or result.empty:
                raise RuntimeError("YahooQuery returned no rows")
            result = result.reset_index().rename(
                columns={
                    "symbol": "Ticker",
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "adjclose": "Adj_Close",
                    "volume": "Volume",
                    "dividends": "Dividends",
                    "splits": "Stock_Splits",
                }
            )
            frames.append(result)
            failures.extend(sorted(set(batch) - set(result["Ticker"].unique())))
        except Exception:
            failures.extend(batch)
        time.sleep(1)

    if not frames:
        raise RuntimeError("Yahoo price download returned no data")
    prices = pd.concat(frames, ignore_index=True)
    if prices.empty:
        raise RuntimeError("Yahoo price download returned empty frames")
    prices.columns = [str(column).replace(" ", "_") for column in prices.columns]
    prices.to_csv(output, index=False)
    write_json(
        BRONZE / "yahoo" / "daily_prices_request.json",
        {
            "source": "Yahoo Finance via yfinance",
            "retrieved_at": utc_now(),
            "start": config["price_start"],
            "end": config["price_end"],
            "tickers": tickers,
            "failures": sorted(set(failures)),
        },
    )
    return prices
