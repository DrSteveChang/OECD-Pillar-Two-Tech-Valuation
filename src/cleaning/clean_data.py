import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def load_raw_data():
    """
    Loads raw multi-dimensional panel datasets from the data ingestion phase.
    """
    print("Loading raw financial dimensions...")
    try:
        df_inc = pd.read_csv("../../data/raw_corporate_financials.csv")
        df_bal = pd.read_csv("../../data/raw_balance_sheets.csv")
        df_meta = pd.read_csv("../../data/verified_100_constituents.csv")
        return df_inc, df_bal, df_meta
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CRITICAL: Missing required raw data files. Ensure fetch_data.py ran successfully. Details: {str(e)}")

def construct_analytical_panel(df_inc, df_bal, df_meta):
    """
    Executes relational merging of income statements and balance sheets, 
    aligning them to the verified 100 constituent framework.
    """
    print("Executing structural panel alignment...")
    
    # Standardize temporal and categorical keys for relational merging
    for df in [df_inc, df_bal]:
        if 'asOfDate' in df.columns:
            df['Year'] = pd.to_datetime(df['asOfDate']).dt.year
            
    # Merge Income Statement and Balance Sheet on Ticker and Year
    df_panel = pd.merge(
        df_inc, 
        df_bal, 
        on=['symbol', 'Year'], 
        how='inner', 
        suffixes=('_inc', '_bal')
    )
    
    # Filter out residual data to strictly match the verified 100 MNE cohort
    valid_tickers = df_meta['Ticker'].tolist()
    df_panel = df_panel[df_panel['symbol'].isin(valid_tickers)].copy()
    
    # Attach Treatment/Control dummy indicators
    df_panel = pd.merge(
        df_panel, 
        df_meta[['Ticker', 'Treatment_Group', 'Currency']], 
        left_on='symbol', 
        right_on='Ticker', 
        how='left'
    )
    
    return df_panel

def execute_econometric_etl(df):
    """
    Performs data imputation, calculates derived ratios (ETR, Leverage, Size),
    and applies econometric Winsorization to mitigate outlier leverage.
    """
    print("Initiating econometric feature engineering and missing value imputation...")
    
    # 1. IMPUTATION RULES (Accounting Logic)
    # Missing R&D or Intangibles usually implies zero investment/assets in that category
    df['ResearchAndDevelopment'] = df.get('ResearchAndDevelopment', pd.Series([np.nan]*len(df))).fillna(0)
    df['NetIntangibleAssets'] = df.get('NetIntangibleAssets', df.get('GrossIntangibleAssets', pd.Series([np.nan]*len(df)))).fillna(0)
    
    # Forward fill missing total assets or liabilities assuming balance sheet stickiness
    df['TotalAssets'] = df['TotalAssets'].replace(0, np.nan).groupby(df['symbol']).ffill()
    df['TotalLiabilitiesNetMinorityInterest'] = df['TotalLiabilitiesNetMinorityInterest'].groupby(df['symbol']).ffill()
    
    # Drop rows where critical anchor variables (Revenue or Assets) are irreparably missing
    df = df.dropna(subset=['TotalRevenue', 'TotalAssets'])
    
    # 2. FEATURE ENGINEERING (As per DATA_SCHEMA.md)
    # Firm Size: ln(Total Assets)
    df['Firm_Size'] = np.log(df['TotalAssets'].astype(float))
    
    # Leverage: Total Liabilities / Total Assets
    df['Leverage'] = df['TotalLiabilitiesNetMinorityInterest'] / df['TotalAssets']
    
    # R&D Intensity: R&D Expense / Total Revenue
    df['RD_Intensity'] = df['ResearchAndDevelopment'] / df['TotalRevenue']
    
    # Intangible Ratio: Intangible Assets / Total Assets
    df['Intangible_Ratio'] = df['NetIntangibleAssets'] / df['TotalAssets']
    
    # Effective Tax Rate (ETR): Tax Provision / Pretax Income
    # Handle negative pretax income or zero division
    df['PretaxIncome'] = df['PretaxIncome'].replace(0, np.nan) 
    df['ETR'] = df['TaxProvision'] / df['PretaxIncome']
    # Cap ETR mathematically between 0 and 1 to prevent severe distortion from tax credits
    df['ETR'] = df['ETR'].clip(lower=0, upper=1)
    
    # 3. STATISTICAL WINSORIZATION
    # Mitigate extreme accounting outliers at the 1st and 99th percentiles
    print("Applying 1% and 99% Winsorization to continuous covariates...")
    continuous_vars = ['Firm_Size', 'Leverage', 'RD_Intensity', 'Intangible_Ratio', 'ETR']
    
    for var in continuous_vars:
        # Check for inf values created during division
        df[var] = df[var].replace([np.inf, -np.inf], np.nan)
        # Fill remaining NaNs with cross-sectional median before winsorizing to prevent array collapse
        df[var] = df[var].fillna(df[var].median())
        # Apply winsorization
        df[var] = winsorize(df[var], limits=[0.01, 0.01])

    # Select and reorder the final analytical columns
    final_columns = [
        'Ticker', 'Year', 'Treatment_Group', 'Currency',
        'TotalRevenue', 'PretaxIncome', 'NetIncome',
        'Firm_Size', 'Leverage', 'RD_Intensity', 'Intangible_Ratio', 'ETR'
    ]
    
    df_analytical = df[final_columns].sort_values(by=['Ticker', 'Year']).reset_index(drop=True)
    return df_analytical

if __name__ == "__main__":
    # Define working directory context to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        raw_inc, raw_bal, meta_df = load_raw_data()
        panel_df = construct_analytical_panel(raw_inc, raw_bal, meta_df)
        final_panel = execute_econometric_etl(panel_df)
        
        # Export the master analytical panel
        output_path = "../../data/analytical_panel_dataset.csv"
        final_panel.to_csv(output_path, index=False)
        
        print("\n===============================================================")
        print("   ETL PIPELINE COMPLETED: ANALYTICAL PANEL GENERATED          ")
        print("===============================================================")
        print(f"Final structural dimensions: {final_panel.shape[0]} rows, {final_panel.shape[1]} columns")
        print(f"Data successfully flushed to: {output_path}")
        
    except Exception as e:
        print(f"\nCRITICAL ETL FAILURE: {str(e)}")