from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

from ..config import BRONZE, REFERENCE
from ..utils import utc_now, write_json


BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict[str, str]:
    identity = os.environ.get("SEC_USER_AGENT", "Academic research contact@example.com")
    return {"User-Agent": identity, "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str, output: Path) -> dict:
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(response.text, encoding="utf-8")
    time.sleep(0.15)
    return response.json()


def download_sec_data(include_filings: bool = True) -> None:
    mapping_path = BRONZE / "sec" / "company_tickers.json"
    mapping = _get_json("https://www.sec.gov/files/company_tickers.json", mapping_path)
    ticker_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in mapping.values()}
    pd.DataFrame(
        [{"Ticker": k, "CIK": v} for k, v in ticker_map.items()]
    ).to_csv(REFERENCE / "ticker_cik_mapping.csv", index=False)

    sample = pd.read_csv(REFERENCE / "sample_registry.csv")
    text_records = []
    failures = []
    for ticker in sample["Ticker"]:
        cik = ticker_map.get(ticker)
        if not cik:
            failures.append({"ticker": ticker, "stage": "mapping", "reason": "missing CIK"})
            continue
        try:
            _get_json(
                f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                BRONZE / "sec" / "companyfacts" / f"{ticker}.json",
            )
        except requests.RequestException as error:
            failures.append({"ticker": ticker, "stage": "companyfacts", "reason": str(error)})
        try:
            submissions = _get_json(
                f"{BASE}/submissions/CIK{cik}.json",
                BRONZE / "sec" / "submissions" / f"{ticker}.json",
            )
        except requests.RequestException as error:
            failures.append({"ticker": ticker, "stage": "submissions", "reason": str(error)})
            continue
        if not include_filings:
            continue
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for index, form in enumerate(forms):
            if form not in {"10-K", "20-F"}:
                continue
            accession = recent["accessionNumber"][index].replace("-", "")
            document = recent["primaryDocument"][index]
            url = f"{ARCHIVES}/{int(cik)}/{accession}/{document}"
            response = requests.get(url, headers=_headers(), timeout=30)
            if response.ok:
                text = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
                tax_blocks = re.findall(r"(?i)(?:.{0,500}\\btax\\b.{0,1500})", text)
                text_records.append(
                    {
                        "ticker": ticker,
                        "form": form,
                        "filing_date": recent["filingDate"][index],
                        "source_url": url,
                        "text": "\n".join(tax_blocks[:20]),
                    }
                )
            time.sleep(0.15)
            break
    output = BRONZE / "sec" / "tax_disclosures.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for record in text_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        BRONZE / "sec" / "request_metadata.json",
        {"retrieved_at": utc_now(), "failures": failures, "tax_disclosures": len(text_records)},
    )
