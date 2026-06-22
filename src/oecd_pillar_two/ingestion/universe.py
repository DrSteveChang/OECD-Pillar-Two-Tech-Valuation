from __future__ import annotations

import time

import pandas as pd
from yahooquery import Ticker

from ..config import BRONZE, REFERENCE
from ..utils import utc_now, write_json


CANDIDATE_TICKERS = """
AAPL MSFT GOOGL AMZN META NVDA AVGO CSCO ADBE ORCL CRM AMD QCOM INTC TXN INTU IBM
NOW AMAT ADI MU LRCX PANW SNPS CDNS KLAC FTNT APH ROP MSI COGN MCHP TEL ACN SAP
ASML TSM WIT INFY NXPI STM TEAM WDAY SHOP SQ PYPL LOGI CHKP NET DDOG SNOW CRWD ZS
MDB HUBS PLTR UBER ABNB FSLY BOX PD ESTC SMAR YEXT PRO SPSC E2OPEN RAMP QNST MODN
TCX SREV APPF BASE BL CALX INOV MITK SCWX SCSC SMSI TLS SCOR INTT LIVE SEAC DSS
ISDR BCOV AGYS ASYS ATEN AUDC AWRE AXTI AZPN BAND FORM CEVA CIEN CLSK CMBM COHU
CPSI DCO DGII DMRC ZUO DOMO VRNS TENB RPD DOCN GTLB ASAN AMPL FIVN SUMO NCNO PRVA
WKME KLTR COUR UDMY APPN CXM XMTR LZ BIRD CRSR HEAR EGHT API NEWR WK PAYC PCTY RNG
DBD FARO GPRO IRBT KAMN MESA ALRM ENVX U MNDY TWLO PEGA VNT ALIT ZI LAW MCW
""".split()


def write_candidate_universe() -> pd.DataFrame:
    current = pd.read_csv(REFERENCE / "sample_registry.csv")["Ticker"].dropna().astype(str)
    tickers = sorted(set(CANDIDATE_TICKERS) | set(current))
    frame = pd.DataFrame({"Ticker": tickers, "universe_source": "expanded_curated_global_technology_pool"})
    frame.to_csv(REFERENCE / "candidate_universe.csv", index=False)
    return frame


def _fetch_statement(tickers: list[str], statement: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for start in range(0, len(tickers), 15):
        batch = tickers[start : start + 15]
        for attempt in range(3):
            try:
                client = Ticker(batch, asynchronous=True)
                result = getattr(client, statement)(frequency="a")
                if isinstance(result, pd.DataFrame) and not result.empty:
                    result = result.reset_index()
                    frames.append(result)
                    failures.extend(sorted(set(batch) - set(result["symbol"].astype(str))))
                    break
            except Exception:
                if attempt == 2:
                    failures.extend(batch)
                time.sleep(2 ** attempt)
        time.sleep(1)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), sorted(set(failures))


def download_candidate_financials(force: bool = False) -> dict:
    universe = write_candidate_universe()
    outputs = {
        "income_statement": BRONZE / "yahoo" / "candidate_corporate_financials.csv",
        "balance_sheet": BRONZE / "yahoo" / "candidate_balance_sheets.csv",
    }
    failures: dict[str, list[str]] = {}
    for statement, output in outputs.items():
        if output.exists() and not force:
            failures[statement] = []
            continue
        frame, statement_failures = _fetch_statement(universe["Ticker"].tolist(), statement)
        if frame.empty:
            raise RuntimeError(f"Yahoo candidate {statement} download returned no data")
        frame.to_csv(output, index=False)
        failures[statement] = statement_failures
    write_json(
        BRONZE / "yahoo" / "candidate_financials_request.json",
        {
            "source": "Yahoo Finance via yahooquery",
            "retrieved_at": utc_now(),
            "candidate_count": len(universe),
            "failures": failures,
        },
    )
    return {"candidate_count": len(universe), "failures": failures}
