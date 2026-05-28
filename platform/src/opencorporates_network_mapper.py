# ==============================================================================
# Script: opencorporates_network_mapper.py
# Purpose: Extract Subsidiary Networks & Engineer Jurisdictional Blending Features
# ==============================================================================

import os
import requests
import pandas as pd
import time
import json
import pycountry
import re
import logging

# Configure logging for tracking API successes and failures
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
MICRO_CSV = os.path.join(BASE_DIR, "data/processed/micro_panel_enriched.csv")
MACRO_JSON = os.path.join(BASE_DIR, "data/processed/macro_baseline.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "data/processed/micro_panel_with_network_features.csv")

class OpenCorporatesClient:
    def __init__(self, api_token):
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Boyan Zhang (Academic Research) - boyan@example.com'
        })
        self.base_url = "https://api.opencorporates.com/v0.4/companies/search"

    def get_core_brand_name(self, official_name, ticker):
        """Standardize company name for better search matching."""
        tech_giants = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOG': 'Google', 
                       'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Facebook',
                       'NVDA': 'Nvidia', 'TSLA': 'Tesla', 'NFLX': 'Netflix', 'INTC': 'Intel'}
        
        if ticker in tech_giants:
            return tech_giants[ticker]

        # Strip suffixes via regex
        clean_name = re.sub(r'(?i)\b(inc|corp|corporation|ltd|limited|llc|plc|group|holdings|company|co)\b\.?', '', official_name)
        clean_name = re.sub(r'[,.\(\)]', '', clean_name).strip()
        words = clean_name.split()
        return " ".join(words[:2]) if len(words) > 2 else clean_name

    def search_subsidiaries(self, official_name, ticker, tax_haven_codes):
        """Perform search with fallback strategies."""
        search_terms = [self.get_core_brand_name(official_name, ticker), official_name]
        
        for term in search_terms:
            params = {'q': term, 
                      'per_page': 100, 
                      'normalise_company_name': 'true',
                      'api_token': self.api_token}
            try:
                response = self.session.get(self.base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    companies = data.get('results', {}).get('companies', [])
                    if companies:
                        return self._calculate_metrics(companies, tax_haven_codes)
            except requests.exceptions.RequestException as e:
                logging.warning(f"Connection error for {ticker}: {e}")
                continue
        return 0, 0

    def _calculate_metrics(self, companies, tax_haven_codes):
        total = len(companies)
        haven = sum(1 for c in companies if c.get('company', {}).get('jurisdiction_code', '').split('_')[0].lower() in tax_haven_codes)
        return total, haven

class NetworkFeatureEngine:
    def __init__(self):
        self.client = OpenCorporatesClient()
        self.tax_havens = self._load_tax_havens()
        self.ticker_map = self._load_sec_mapping()

    def _load_tax_havens(self):
        """Load OECD baselines."""
        with open(MACRO_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return set(item['Jurisdiction'] for item in data.get("high_risk_jurisdictions_sample", [])) # Simplified mapping

    def _load_sec_mapping(self):
        """Fetch SEC ticker directory with error handling and proper headers."""
        sec_url = "https://www.sec.gov/files/company_tickers.json"
        
        try:
            # Use the global HEADERS defined at the top of your script
            resp = requests.get(sec_url, headers=HEADERS, timeout=10)
            
            if resp.status_code != 200:
                logging.error(f"Failed to fetch SEC data. Status Code: {resp.status_code}")
                # Print a snippet of the response to see if it's an HTML error page
                logging.error(f"Response snippet: {resp.text[:200]}")
                return {}
                
            # Parse the JSON response
            data = resp.json()
            return {v['ticker']: v['title'] for k, v in data.items()}
            
        except requests.exceptions.JSONDecodeError:
            logging.error("Received response is not valid JSON. The SEC might be blocking this request.")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching SEC mapping: {e}")
            return {}

    def run(self):
        df = pd.read_csv(MICRO_CSV)
        df['Jurisdictional_Blending_Ratio'] = 0.0
        
        tickers = df['Ticker'].unique()
        logging.info(f"Starting feature engineering for {len(tickers)} companies.")
        
        for i, ticker in enumerate(tickers):
            name = self.ticker_map.get(ticker, ticker)
            logging.info(f"Processing [{i+1}/{len(tickers)}]: {ticker}")
            
            total, haven = self.client.search_subsidiaries(name, ticker, self.tax_havens)
            jbr = (haven / total) if total > 0 else 0.0
            
            df.loc[df['Ticker'] == ticker, 'Jurisdictional_Blending_Ratio'] = round(jbr, 4)
            time.sleep(1.5) # API Compliance
            
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Pipeline complete. Data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    engine = NetworkFeatureEngine()
    engine.run()