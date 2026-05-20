# ==============================================================================
# Script: oecd_macro_etl.py
# Purpose: Clean and Process OECD CbCR Data with precise SDMX schema mapping
# ==============================================================================

import os
import pandas as pd
import json

BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
RAW_CSV = os.path.join(BASE_DIR, "data/raw/oecd_cbcr_raw.csv")
CLEAN_JSON = os.path.join(BASE_DIR, "data/processed/macro_baseline.json")

def process_oecd_cbcr_data():
    print("="*60)
    print(" STARTING OECD CbCR MACRO DATA ETL PIPELINE ")
    print("="*60)

    if not os.path.exists(RAW_CSV):
        print(f"[ERROR] Raw OECD data not found at {RAW_CSV}")
        return

    print("[1/3] Loading raw OECD CbCR dataset (Memory Optimized)...")
    
    # Extract only the 5 core dimensions required for research, 
    # significantly reducing memory usage for the 400MB CSV
    target_cols = [
        'Reference area', 
        'Measure', 
        'AGGREGATION_TYPE', 
        'PROFIT_GROUPING', 
        'OBS_VALUE'
    ]
    
    df = pd.read_csv(RAW_CSV, usecols=target_cols, low_memory=False)
    
    print("[2/3] Cleaning data and filtering for Macro ETR metrics...")
    
    # Cleaning: Remove invalid numerical rows
    df = df.dropna(subset=['OBS_VALUE'])
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['OBS_VALUE'])

    # Precise mapping to OECD statistical standards:
    # Filter for macro aggregate level (avoiding quantile confusion) 
    # and select the 'Aggregate totals by jurisdiction' panel.
    # Given the definitions in the 'Measure' column, perform precise slicing using keywords:
    
    # 1. Extract total pre-tax profit rows
    profit_condition = (
        df['Measure'].str.contains('Profit', na=False, case=False) & 
        ~df['Measure'].str.contains('relative to', na=False, case=False) # Exclude ratio metrics
    )
    profits_df = df[profit_condition]
    
    # 2. Extract total tax paid rows
    tax_condition = (
        df['Measure'].str.contains('Tax Paid', na=False, case=False) &
        ~df['Measure'].str.contains('relative to', na=False, case=False)
    )
    taxes_df = df[tax_condition]

    # Aggregate totals by jurisdiction
    profits_agg = profits_df.groupby('Reference area')['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Profit'})
    taxes_agg = taxes_df.groupby('Reference area')['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Tax_Paid'})

    # Merge macro panel
    macro_df = pd.merge(profits_agg, taxes_agg, on='Reference area', how='inner')
    
    # Filter out invalid or negative profit data; calculate the true Macro Effective Tax Rate (Macro ETR)
    macro_df = macro_df[macro_df['Profit'] > 0]
    macro_df['Macro_ETR'] = macro_df['Tax_Paid'] / macro_df['Profit']
    
    # Lock onto Pillar Two core objectives: jurisdictions with Macro ETR below 15%
    tax_havens = macro_df[macro_df['Macro_ETR'] < 0.15]
    total_exposed_profit = tax_havens['Profit'].sum()

    print(f"      -> Successfully processed {len(macro_df)} valid jurisdictions.")
    print(f"      -> Identified {len(tax_havens)} tax jurisdictions with Macro ETR < 15%.")

    # --------------------------------------------------------------------------
    # 3. EXPORT CLEANED BASELINE FOR AGENT COMPLIANCE
    # --------------------------------------------------------------------------
    print("[3/3] Exporting Macro Baseline for Econometric Engine...")
    
    baseline_metrics = {
        "dataset_source": "OECD Anonymised and Aggregated CbCR Data Table I",
        "global_metrics": {
            "total_jurisdictions_analyzed": len(macro_df),
            "jurisdictions_below_15_percent_etr": len(tax_havens),
            "aggregate_profit_exposed_to_pillar_two": float(total_exposed_profit),
            "global_average_etr": float(macro_df['Macro_ETR'].mean())
        },
        "high_risk_jurisdictions_sample": tax_havens[['Reference area', 'Macro_ETR']].sort_values(by='Macro_ETR').rename(columns={'Reference area': 'Jurisdiction'}).to_dict(orient='records'),
        "methodological_note": "Calculated strictly from aggregate pre-treatment multi-national group variables. Baseline targets with ETR < 0.15 serve as the causal reference foundation."
    }

    # Write to processed directory
    os.makedirs(os.path.dirname(CLEAN_JSON), exist_ok=True)
    with open(CLEAN_JSON, 'w', encoding='utf-8') as f:
        json.dump(baseline_metrics, f, indent=4)

    print(f"\n[SUCCESS] Baseline structure persisted to: {CLEAN_JSON}")
    print("="*60)

if __name__ == "__main__":
    process_oecd_cbcr_data()