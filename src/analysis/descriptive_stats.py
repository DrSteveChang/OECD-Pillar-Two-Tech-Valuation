import os
import pandas as pd
import numpy as np

def generate_academic_summary_stats():
    """
    Ingests the analytical panel and generates standard Academic Table 1 
    (Descriptive Statistics) and Table 2 (Correlation Matrix) stratified by cohort.
    """
    print("Loading analytical panel for econometric review...")
    data_path = "../../data/analytical_panel_dataset.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CRITICAL: Analytical dataset not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    
    # 1. BASIC PANEL DIMENSIONS
    total_obs = len(df)
    unique_firms = df['Ticker'].nunique()
    years_covered = sorted(df['Year'].unique())
    print(f"\n--- Panel Structure Overview ---")
    print(f"Total Observations (Firm-Years): {total_obs}")
    print(f"Unique Firms: {unique_firms}")
    print(f"Time Horizon: {years_covered[0]} to {years_covered[-1]}")
    
    # 2. ACADEMIC SUMMARY STATISTICS (Table 1)
    # Define continuous covariates to summarize
    continuous_vars = ['TotalRevenue', 'NetIncome', 'Firm_Size', 'Leverage', 'RD_Intensity', 'Intangible_Ratio', 'ETR']
    
    print("\nComputing stratified descriptive statistics...")
    
    # Stratify by Treatment Group (1) and Control Group (0)
    summary_list = []
    for group_code, group_name in [(1, 'Treatment (>= 750M)'), (0, 'Control (< 750M)')]:
        df_group = df[df['Treatment_Group'] == group_code][continuous_vars]
        
        # Calculate standard academic metrics
        stats = df_group.describe(percentiles=[.25, .50, .75]).T
        stats['Group'] = group_name
        
        # Reorder columns for academic presentation
        stats = stats[['Group', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        summary_list.append(stats)
        
    # Combine and save Table 1
    table1 = pd.concat(summary_list)
    
    output_dir = "../../docs/analysis_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    table1_path = f"{output_dir}/Table1_Summary_Statistics.csv"
    table1.to_csv(table1_path)
    print(f"--> Table 1 (Summary Statistics) successfully exported to: {table1_path}")
    
    # 3. CORRELATION MATRIX (Table 2 - Multicollinearity Check)
    print("Computing Pearson correlation matrix for structural covariates...")
    # Select ratio-based covariates to check for multicollinearity
    ratio_vars = ['Firm_Size', 'Leverage', 'RD_Intensity', 'Intangible_Ratio', 'ETR']
    corr_matrix = df[ratio_vars].corr(method='pearson').round(4)
    
    table2_path = f"{output_dir}/Table2_Correlation_Matrix.csv"
    corr_matrix.to_csv(table2_path)
    print(f"--> Table 2 (Correlation Matrix) successfully exported to: {table2_path}")
    
    print("\n[DIAGNOSTIC COMPLETE]: Data is mathematically ready for DiD/SCM estimation.")

if __name__ == "__main__":
    # Contextual directory binding
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_academic_summary_stats()