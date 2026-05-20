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

BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
MICRO_CSV = os.path.join(BASE_DIR, "data/processed/micro_panel_enriched.csv")
MACRO_JSON = os.path.join(BASE_DIR, "data/processed/macro_baseline.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "data/processed/micro_panel_with_network_features.csv")

HEADERS = {
    'User-Agent': 'Boyan Zhang (Academic Research Project) boyan@example.com'
}

# ------------------------------------------------------------------------------
# 1. DYNAMIC TAX HAVEN MAPPING (Connecting Battle 2 to Battle 3)
# ------------------------------------------------------------------------------
def get_dynamic_tax_havens():
    """Reads the 20 OECD tax havens from Battle 2 and converts them to ISO-2 codes."""
    if not os.path.exists(MACRO_JSON):
        raise FileNotFoundError(f"Macro baseline not found at {MACRO_JSON}")
        
    with open(MACRO_JSON, 'r', encoding='utf-8') as f:
        macro_data = json.load(f)
        
    jurisdictions = macro_data.get("high_risk_jurisdictions_sample", [])
    
    iso_codes = []
    
    # [Optimization] Precise overrides for special political/economic entities 
    # to bypass pycountry fuzzy search limitations.
    name_overrides = {
        "United Arab Emirates": "AE",
        "Türkiye": "TR",
        "Slovak Republic": "SK",
        "Korea": "KR",
        "Macau (China)": "MO",
        "Macau": "MO",
        "Hong Kong (China)": "HK",
        "Hong Kong": "HK",
        "Cayman Islands": "KY",
        "Bermuda": "BM",
        "British Virgin Islands": "VG"
    }

    print("[1/4] Loading dynamic tax havens from OECD Baseline...")
    for item in jurisdictions:
        country_name = item.get("Jurisdiction")
        if not country_name:
            continue
            
        # 1. Prioritize override dictionary
        if country_name in name_overrides:
            code = name_overrides[country_name].lower()
            iso_codes.append(code)
            print(f"      -> Mapped (Override): {country_name} -> {code}")
            continue
            
        # 2. Fallback to fuzzy matching
        try:
            country = pycountry.countries.search_fuzzy(country_name)[0]
            iso_codes.append(country.alpha_2.lower())
            print(f"      -> Mapped (Fuzzy): {country_name} -> {country.alpha_2.lower()}")
        except LookupError:
            print(f"      [!] Could not map jurisdiction: {country_name}")
            
    return set(iso_codes)

# ------------------------------------------------------------------------------
# 2. TICKER TO COMPANY NAME TRANSLATION
# ------------------------------------------------------------------------------
def get_sec_ticker_to_name_mapping():
    """Fetches official SEC company titles because OpenCorporates needs names, not tickers."""
    print("\n[2/4] Fetching SEC Ticker-to-Name mapping...")
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception("Failed to fetch SEC mapping.")
    
    return {v['ticker']: v['title'] for k, v in response.json().items()}

def get_core_brand_name(official_name, ticker):
    """Forcefully strip legal suffixes to extract core brand terms."""
    
    # 1. Hard-code core brand names for high-frequency tech tax structures
    tech_giants_map = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOG': 'Google', 
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'META': 'Facebook', 
        'NVDA': 'Nvidia',
        'TSLA': 'Tesla',
        'NFLX': 'Netflix',
        'INTC': 'Intel'
    }
    
    if ticker in tech_giants_map:
        return tech_giants_map[ticker]

    # 2. Regex cleaning: remove common legal suffixes case-insensitively
    clean_name = re.sub(
        r'(?i)\b(inc|corp|corporation|ltd|limited|llc|plc|group|holdings|company|co)\b\.?', 
        '', 
        official_name
    )
    
    # 3. Strip commas, periods, parentheses, and extra whitespace
    clean_name = re.sub(r'[,.\(\)]', '', clean_name).strip()
    
    # 4. If the name is long, keep only the first two words to avoid over-constrained searching
    words = clean_name.split()
    if len(words) > 2:
        clean_name = " ".join(words[:2])
        
    return clean_name

# ------------------------------------------------------------------------------
# 3. OPENCORPORATES API QUERY
# ------------------------------------------------------------------------------
def fetch_subsidiary_network(official_name, ticker, tax_haven_codes):
    """Queries OpenCorporates for subsidiary data based on cleaned company name."""
    api_url = f"https://api.opencorporates.com/v0.4/companies/search"
    
    # Use the violence-cleaned brand name for the API search
    search_keyword = get_core_brand_name(official_name, ticker)
    
    params = {
        'q': search_keyword,
        'per_page': 100, 
        'normalise_company_name': 'true'
    }

    try:
        response = requests.get(api_url, params=params, headers=HEADERS)
        if response.status_code != 200:
            return 0, 0
            
        data = response.json()
        companies = data.get('results', {}).get('companies', [])
        
        total_subs = len(companies)
        haven_subs = 0
        
        for item in companies:
            # Extract 2-letter jurisdiction code
            jur_code = item.get('company', {}).get('jurisdiction_code', '').lower()
            country_prefix = jur_code.split('_')[0] if jur_code else ""
            
            if country_prefix in tax_haven_codes:
                haven_subs += 1
                
        return total_subs, haven_subs

    except Exception:
        return 0, 0

# ------------------------------------------------------------------------------
# 4. MAIN FEATURE ENGINEERING PIPELINE
# ------------------------------------------------------------------------------
def engineer_network_features():
    print("="*60)
    print(" STARTING OPENCORPORATES NETWORK FEATURE PIPELINE ")
    print("="*60)

    try:
        tax_haven_codes = get_dynamic_tax_havens()
        ticker_to_name = get_sec_ticker_to_name_mapping()
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return

    print("\n[3/4] Loading Micro Panel Data...")
    if not os.path.exists(MICRO_CSV):
        print(f"[ERROR] Micro panel not found at {MICRO_CSV}")
        return

    df = pd.read_csv(MICRO_CSV)
    
    if 'Ticker' not in df.columns:
        print("[ERROR] CSV must contain a 'Ticker' column.")
        return

    df['Total_Subsidiaries'] = 0
    df['Haven_Subs_Count'] = 0
    df['Jurisdictional_Blending_Ratio'] = 0.0

    print("\n[4/4] Querying OpenCorporates & Calculating Complexity (JBR)...")
    target_tickers = df['Ticker'].unique() 
    
    for i, ticker in enumerate(target_tickers):
        official_name = ticker_to_name.get(ticker, ticker)
        print(f"      [{i+1}/{len(target_tickers)}] Mapping offshore network for: {ticker} ({official_name})...")
        
        total_subs, haven_subs = fetch_subsidiary_network(official_name, ticker, tax_haven_codes)
        jbr = (haven_subs / total_subs) if total_subs > 0 else 0.0
        
        mask = df['Ticker'] == ticker
        df.loc[mask, 'Total_Subsidiaries'] = total_subs
        df.loc[mask, 'Haven_Subs_Count'] = haven_subs
        df.loc[mask, 'Jurisdictional_Blending_Ratio'] = round(jbr, 4)
        
        time.sleep(1.5) # Compliance with API rate limits

    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*60)
    print(f"[SUCCESS] Network Features Engineered! Saved to: {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    engineer_network_features()