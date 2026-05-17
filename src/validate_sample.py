import os
import pandas as pd
import numpy as np
from yahooquery import Ticker
import time

# Expanded candidate pool of 160+ global technology tickers to ensure a strict balanced 50/50 matrix structure
CANDIDATE_POOL = [
    # --- Large/Mega Caps (Potential Treatment Group, > €750M) ---
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "CSCO", "ADBE", "ORCL",
    "CRM", "AMD", "QCOM", "INTC", "TXN", "INTU", "IBM", "NOW", "AMAT", "ADI",
    "MU", "LRCX", "PANW", "SNPS", "CDNS", "KLAC", "FTNT", "APH", "ROP", "MSI",
    "COGN", "MCHP", "TEL", "ACN", "SAP", "ASML", "TSM", "WIT", "INFY", "NXPI",
    "STM", "TEAM", "WDAY", "SHOP", "SQ", "PYPL", "LOGI", "CHKP", "NET", "DDOG",
    "SNOW", "CRWD", "ZS", "MDB", "HUBS", "PLTR", "UBER", "ABNB",

    # --- Mid/Small Caps (Potential Control Group / Donor Pool, < €750M) ---
    "FSLY", "BOX", "PD", "ESTC", "SMAR", "YEXT", "PRO", "SPSC", "E2OPEN", "RAMP",
    "QNST", "MODN", "TCX", "SREV", "APPF", "BASE", "BL", "CALX", "INOV", "MITK",
    "SCWX", "SCSC", "SMSI", "TLS", "SCOR", "INTT", "LIVE", "SEAC", "DSS", "ISDR",
    "BCOV", "AGYS", "ASYS", "ATEN", "AUDC", "AWRE", "AXTI", "AZPN", "BAND", "FORM",
    "CEVA", "CIEN", "CLSK", "CMBM", "COHU", "CPSI", "DCO", "DGII", "DMRC",
    "ZUO", "DOMO", "VRNS", "TENB", "RPD", "DOCN", "GTLB", "ASAN", "AMPL", "FIVN",
    "SUMO", "NCNO", "PRVA", "WKME", "KLTR", "COUR", "UDMY", "APPN", "CXM", "XMTR",
    "LZ", "BIRD", "CRSR", "HEAR", "EGHT", "API", "NEWR", "WK", "PAYC", "PCTY",
    "RNG", "DBD", "FARO", "GPRO", "IRBT", "KAMN", "MESA", "ALRM", "ENVX", "U",
    "MNDY", "TWLO", "PEGA", "VNT", "ALIT", "ZI", "LAW", "MCW"
]

# Standardized annualized exchange rates to EUR for FY2022 regulatory baseline definition
EXCHANGE_RATES_TO_EUR = {
    'EUR': 1.0,
    'USD': 0.950,  # Historical annual average exchange rate for USD/EUR in 2022
    'TWD': 0.032,  # New Taiwan Dollar
    'INR': 0.012,  # Indian Rupee
    'CHF': 0.995   # Swiss Franc
}

def validate_and_classify_samples(tickers):
    print(f"Starting execution: Auditing {len(tickers)} tickers against OECD Pillar Two scope...")
    
    # Segment tickers into smaller chunks to bypass Yahoo Finance rate-limiting policies
    chunk_size = 20
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_income_statements = []
    
    # Process batches sequentially with cooling thresholds
    for idx, chunk in enumerate(ticker_chunks):
        print(f"Processing batch {idx + 1}/{len(ticker_chunks)} ({len(chunk)} tickers)...")
        try:
            t = Ticker(chunk, asynchronous=True)
            df_chunk = t.income_statement(frequency='a')
            
            if df_chunk is not None and isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                all_income_statements.append(df_chunk)
                print(f"--> Batch {idx + 1} successfully retrieved. Rows: {len(df_chunk)}")
            else:
                print(f"--> Warning: Batch {idx + 1} returned empty data structure.")
        except Exception as e:
            print(f"--> Critical failure fetching batch {idx + 1}: {str(e)}")
        
        # Introduce latency buffer to mitigate systemic IP blocking / anti-scraping triggers
        time.sleep(1.5)
        
    if not all_income_statements:
        raise RuntimeError("CRITICAL: All batch ingestion attempts failed. API access completely restricted.")
        
    # Consolidate chunked dataframes into a unified historical panel
    df_income = pd.concat(all_income_statements)
    print(f"\nTotal aggregated raw panel rows fetched: {len(df_income)}")
    
    # Standardize data index to expose categorical variables explicitly
    df_income = df_income.reset_index()
    
    if 'symbol' not in df_income.columns and 'index' in df_income.columns:
        df_income = df_income.rename(columns={'index': 'symbol'})
    if 'symbol' not in df_income.columns and 'level_0' in df_income.columns:
        df_income = df_income.rename(columns={'level_0': 'symbol'})
        
    # Standardize and isolate temporal variables
    df_income['asOfDate'] = pd.to_datetime(df_income['asOfDate'])
    df_income['Year'] = df_income['asOfDate'].dt.year
    
    print(f"Available historical cross-sections detected: {list(df_income['Year'].unique())}")
    
    # Isolate FY2022 as the rigorous pre-implementation statutory baseline benchmark
    BASELINE_YEAR = 2022
    df_baseline = df_income[df_income['Year'] == BASELINE_YEAR].copy()
    print(f"Total entries filtered for benchmark year {BASELINE_YEAR}: {len(df_baseline)}")
    
    treatment_group = []
    control_group = []
    
    # Execute statutory audits over the isolated cross-sectional dataframe
    for _, row in df_baseline.iterrows():
        ticker = row.get('symbol')
        if pd.isna(ticker):
            continue
            
        try:
            raw_revenue = row.get('TotalRevenue', np.nan)
            currency = row.get('currencyCode', 'USD') 
            
            if pd.isna(currency) or not isinstance(currency, str):
                currency = 'USD'
            
            if pd.isna(raw_revenue) or raw_revenue <= 0:
                continue
            
            # Execute currency harmonization into EUR values
            fx_rate = EXCHANGE_RATES_TO_EUR.get(currency, 0.950)
            revenue_in_eur = raw_revenue * fx_rate
            
            # Establish statutory OECD GloBE Rule Chapter 1 revenue requirement
            statutory_threshold = 750000000
            
            record = {
                "Ticker": ticker,
                "Revenue_EUR": revenue_in_eur,
                "Currency": currency
            }
            
            if revenue_in_eur >= statutory_threshold:
                treatment_group.append(record)
            else:
                control_group.append(record)
                
        except Exception as e:
            print(f"Skipping tracking for asset [{ticker}]: {str(e)}")

    # Secure DataFrame building routines
    df_treat = pd.DataFrame(treatment_group).drop_duplicates(subset=['Ticker']).sort_values(by="Revenue_EUR", ascending=False) if treatment_group else pd.DataFrame(columns=["Ticker", "Revenue_EUR", "Currency"])
    df_ctrl = pd.DataFrame(control_group).drop_duplicates(subset=['Ticker']).sort_values(by="Revenue_EUR", ascending=False) if control_group else pd.DataFrame(columns=["Ticker", "Revenue_EUR", "Currency"])
    
    print(f"\nFinal Filtered Matrices - Treatment Candidates: {len(df_treat)}, Control Candidates: {len(df_ctrl)}")
    
    # Truncate and balance both dimensions to maintain strict 50/50 distribution density
    final_treatment = df_treat.head(50).copy()
    final_control = df_ctrl.head(50).copy()
    
    return final_treatment, final_control

if __name__ == "__main__":
    treat, ctrl = validate_and_classify_samples(CANDIDATE_POOL)
    
    if len(treat) < 50 or len(ctrl) < 50:
        print(f"\n[EXECUTION WARNING]: Imbalance in cohorts detected (Treatment: {len(treat)}, Control: {len(ctrl)}).")
    
    # Map econometric dummy markers (Treatment=1, Control=0)
    treat['Treatment_Group'] = 1
    ctrl['Treatment_Group'] = 0
    
    # Merge filtered cohorts into a singular master dataframe
    final_100 = pd.concat([treat, ctrl])
    
    # Structural output serialization 
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/verified_100_constituents.csv"
    final_100.to_csv(output_path, index=False)
    
    print(f"\n--- Statutory Alignment Report ---")
    print(f"Successfully locked {len(treat)} Treatment MNEs (Revenue >= €750M).")
    print(f"Successfully locked {len(ctrl)} Control MNEs (Revenue < €750M).")
    print(f"Master sample registration sheet locked at: {output_path}")