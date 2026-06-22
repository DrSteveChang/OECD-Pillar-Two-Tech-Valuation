from pathlib import Path

import pandas as pd


def test_each_policy_event_has_trading_window_risk_record():
    events = pd.read_csv("data/reference/policy_events.csv")
    screen_path = Path("data/gold/analytical/fact_event_confound_screen.csv")
    assert screen_path.exists()
    screen = pd.read_csv(screen_path)

    assert set(events["event_id"]).issubset(set(screen["event_id"]))
    assert screen["window_start_trading_day"].eq(-3).all()
    assert screen["window_end_trading_day"].eq(3).all()
    assert screen["firm_observations"].gt(0).all()
    assert screen["abnormal_return_dispersion"].notna().all()
    assert screen["market_wide_volatility_flag"].isin([True, False]).all()


def test_event_confound_flags_are_diagnostics_only():
    screen = pd.read_csv("data/gold/analytical/fact_event_confound_screen.csv")
    events = pd.read_csv("data/reference/policy_events.csv")

    assert screen["diagnostic_only"].eq(True).all()
    assert screen["event_retained_for_analysis"].eq(True).all()
    assert len(screen["event_id"].unique()) == len(events["event_id"].unique())
