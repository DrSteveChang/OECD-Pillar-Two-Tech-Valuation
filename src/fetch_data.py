import os
import time
import pandas as pd
from yahooquery import Ticker

def load_verified_constituents(file_path="data/verified_100_constituents.csv"):
    """
    Dynamically loads the 100 methodologically verified MNE tickers 
    from the statutory alignment phase.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CRITICAL: Master sample registry not found at {file_path}. Run validate_sample.py first.")
    
    df_meta = pd.read_csv(file_path)
    tickers = df_meta['Ticker'].tolist()
    print(f"Successfully loaded {len(tickers)} verified constituents from local registry.")
    return tickers

def fetch_and_stage_data(tickers):
    """
    Fetches raw financial statements and valuation metrics using yahooquery async API.
    Implements chunk-based batching to bypass rate-limiting protocols.
    """
    print(f"\nInitializing multidimensional data ingestion pipeline...")
    
    chunk_size = 20
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_income = []
    all_balance = []
    all_valuation = []
    
    for idx, chunk in enumerate(ticker_chunks):
        print(f"Fetching batch {idx + 1}/{len(ticker_chunks)} ({len(chunk)} tickers)...")
        try:
            # Initialize async requests
            t = Ticker(chunk, asynchronous=True)
            
            # 1. Fetch Income Statement Dimensions
            df_inc = t.income_statement(frequency='a')
            if df_inc is not None and isinstance(df_inc, pd.DataFrame) and not df_inc.empty:
                all_income.append(df_inc)
                
            # 2. Fetch Balance Sheet Dimensions
            df_bal = t.balance_sheet(frequency='a')
            if df_bal is not None and isinstance(df_bal, pd.DataFrame) and not df_bal.empty:
                all_balance.append(df_bal)
                
            # 3. Fetch High-frequency Valuation Metrics
            df_val = t.valuation_measures
            if df_val is not None and isinstance(df_val, pd.DataFrame) and not df_val.empty:
                all_valuation.append(df_val)
                
            print(f"--> Batch {idx + 1} dimensions successfully retrieved.")
            
        except Exception as e:
            print(f"--> Warning: Batch {idx + 1} encountered an exception: {str(e)}")
            
        # 2.0 second latency buffer to prevent IP ban
        time.sleep(2.0)
        
    return all_income, all_balance, all_valuation

def consolidate_and_save(all_income, all_balance, all_valuation):
    """
    Consolidates batched dataframes, standardizes index structures, 
    and flushes the raw panels into local CSV storage.
    """
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    print("\nExecuting data consolidation and local staging...")
    
    def save_panel(data_list, filename):
        if not data_list:
            print(f"[ERROR]: No data aggregated for {filename}")
            return
            
        # Concatenate batches
        df_combined = pd.concat(data_list)
        
        # Standardize categorical variables from indices
        if 'symbol' not in df_combined.columns:
            df_combined = df_combined.reset_index()
        if 'index' in df_combined.columns and 'symbol' not in df_combined.columns:
            df_combined = df_combined.rename(columns={'index': 'symbol'})
            
        output_path = f"{output_dir}/{filename}"
        df_combined.to_csv(output_path, index=False)
        print(f"Successfully staged: {output_path} (Total Rows: {len(df_combined)})")

    # Execute storage protocols for each structural dimension
    save_panel(all_income, "raw_corporate_financials.csv")
    save_panel(all_balance, "raw_balance_sheets.csv")
    save_panel(all_valuation, "raw_market_dimensions.csv")

if __name__ == "__main__":
    try:
        target_tickers = load_verified_constituents()
        inc_data, bal_data, val_data = fetch_and_stage_data(target_tickers)
        consolidate_and_save(inc_data, bal_data, val_data)
        
        print("\n===============================================================")
        print("   DATA INGESTION PHASE COMPLETED SUCCESSFULLY                 ")
        print("===============================================================")
    except Exception as e:
        print(f"\nCRITICAL PIPELINE FAILURE: {str(e)}")