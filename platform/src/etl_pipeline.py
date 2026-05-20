# ==============================================================================
# Script: etl_pipeline.py
# Purpose: Enterprise Data Platform Ingestion & First-Difference Aggregation (ETL)
# Location: platform/src/etl_pipeline.py
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd

# 1. HARDCODED DIRECTORY MAPPING
BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/external/analytical_panel_dataset.csv")
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, "data/processed/micro_panel_fd.csv")
PROCESSED_JSON_PATH = os.path.join(BASE_DIR, "data/processed/macro_metrics.json")

def run_uniqueness_etl():
    print("--- STARTING ETL PIPELINE: INGESTION & STRUCTURAL CLEANING ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"CRITICAL ERROR: Input dataset not found at {RAW_DATA_PATH}")
        
    # Read raw tracking data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Sanitize dataframe schema
    df.columns = df.columns.str.strip()
    
    # Handle logarithmic revenue boundaries securely to avoid infinite mathematical limits
    df['Log_Revenue'] = np.log(np.where(df['TotalRevenue'] <= 0, 1, df['TotalRevenue']))
    
    # --------------------------------------------------------------------------
    # CORE FIX: EXPLICIT UNIQUE AGGREGATION
    # Resolves the many-to-many relationship leakage that compromises Honest Splits
    # --------------------------------------------------------------------------
    df_secure = df.groupby(['Ticker', 'Year']).agg({
        'Log_Revenue': 'mean',
        'Leverage': 'mean',
        'RD_Intensity': 'mean',
        'ETR': 'mean',
        'Treatment_Group': 'max'
    }).reset_index()
    
    # --------------------------------------------------------------------------
    # TIME HORIZON ALIGNMENT & EXTRACTION (2022 vs 2025)
    # --------------------------------------------------------------------------
    # Extract baseline pre-treatment year (FY2022)
    df_2022 = df_secure[df_secure['Year'] == 2022][['Ticker', 'Log_Revenue']].rename(
        columns={'Log_Revenue': 'Log_Revenue_2022'}
    )
    
    # Extract peak post-treatment impact year (FY2025)
    df_2025 = df_secure[df_secure['Year'] == 2025][[
        'Ticker', 'Log_Revenue', 'Leverage', 'RD_Intensity', 'ETR', 'Treatment_Group'
    ]].rename(columns={'Log_Revenue': 'Log_Revenue_2025'})
    
    # Execute strict 1:1 single-key inner join to link timelines without data inflation
    df_ml = pd.merge(df_2025, df_2022, on='Ticker', how='inner')
    
    # Compute the pure First-Difference growth rate metric (Cancels individual size premium)
    df_ml['Delta_Log_Revenue'] = df_ml['Log_Revenue_2025'] - df_ml['Log_Revenue_2022']
    
    # Drop rows containing unresolved null structures in key covariates
    df_final = df_ml.dropna(subset=['Delta_Log_Revenue', 'Leverage', 'RD_Intensity', 'ETR', 'Treatment_Group'])
    
    print(f"[ETL SUCCESS] Perfect 1:1 cross-sectional grid established.")
    print(f"[ETL SUCCESS] Retained unique full-history firms: {len(df_final)}")
    
    # Export clean micro panel for Causal Forest ingestion
    df_final.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"[EXPORT] Micro Golden Data exported to: {PROCESSED_CSV_PATH}")
    
    # --------------------------------------------------------------------------
    # PERSIST STATIC METRIC FACT WAREHOUSE (For AI Agent Rational Layer)
    # Saves compute latency by packaging static macro outputs from Script 02
    # --------------------------------------------------------------------------
    macro_metrics = {
        "Global_Econometric_Framework": "Synthetic Difference-in-Differences (SDiD) & SCM Placebo Permutation",
        "Temporal_Horizon": "2018 - 2025 Golden Horizon Grid Optimization",
        "SDiD_ATT_Estimate": -0.0101,
        "SDiD_Jackknife_Standard_Error": 0.0518,
        "SDiD_T_Statistic": -0.1950,
        "SDiD_Statistical_Significance": "Highly non-significant at conventional alpha boundaries (平滑效应抹杀长尾冲击)",
        "SCM_Placebo_Empirical_P_Value": 0.7755,
        "Causal_Forest_Unbiased_ATT": -0.0477,
        "Methodological_Note": "Macro aggregated estimates are structurally neutralized due to time-smoothing filters in FY23-24, justifying the transition to micro-level First-Difference Causal Forests."
    }
    
    with open(PROCESSED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(macro_metrics, f, indent=4, ensure_ascii=False)
    print(f"[EXPORT] Macro Facts Warehouse exported to: {PROCESSED_JSON_PATH}")
    print("--- ETL PIPELINE EXECUTION SUCCESSFUL ---")

if __name__ == "__main__":
    run_uniqueness_etl()