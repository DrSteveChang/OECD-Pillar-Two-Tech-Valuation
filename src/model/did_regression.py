import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def load_data():
    """
    Loads the cleaned and standardized analytical panel dataset.
    """
    data_path = "../../data/analytical_panel_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("CRITICAL: analytical_panel_dataset.csv not found. Ensure the ETL pipeline has been executed.")
    return pd.read_csv(data_path)

def execute_did_regression(df):
    """
    Executes the Difference-in-Differences (DiD) regression analysis
    with firm-level clustered standard errors.
    """
    print("--- Initializing Difference-in-Differences (DiD) Econometric Model ---")
    
    # 1. Variable Derivation and Logarithmic Transformation
    # Exclude non-positive revenue observations to prevent log transformation errors
    df = df[df['TotalRevenue'] > 0].copy()
    df['Log_Revenue'] = np.log(df['TotalRevenue'])
    
    # 2. Define Post-Treatment Dummy
    # Based on the OECD implementation timeline and the anticipation effect 
    # verified by our SCM model, 2023 onwards is defined as the post-shock window.
    POLICY_YEAR = 2023
    df['Post'] = (df['Year'] >= POLICY_YEAR).astype(int)
    
    # 3. Construct DiD Interaction Term (Treatment * Post)
    # This captures the causal impact of the Pillar Two policy.
    df['DiD_Estimator'] = df['Treatment_Group'] * df['Post']
    
    # 4. Formulate Regression Equation
    # Dependent Variable: Log_Revenue
    # Two-Way Fixed Effects (TWFE) Proxy: C(Year) absorbs time fixed effects, 
    # Treatment_Group absorbs structural group baseline differences.
    formula = (
        "Log_Revenue ~ DiD_Estimator + Treatment_Group + "
        "Leverage + RD_Intensity + C(Year)"
    )
    
    print("\nExecuting OLS panel regression with clustered standard errors at the firm level...")
    
    # 5. Model Fitting
    # cov_type='cluster' ensures robust statistical inference against heteroskedasticity 
    # and serial correlation inherent in longitudinal panel data.
    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['Ticker']})
    
    # 6. Extract and Print Academic Standard Output
    print("\n==============================================================================")
    print("                      DIFFERENCE-IN-DIFFERENCES (DiD) RESULTS                 ")
    print("==============================================================================")
    print(results.summary().tables[1])
    print("==============================================================================")
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Number of Observations: {int(results.nobs)}")
    
    # Export the pure text regression output for academic paper integration
    output_dir = "../../docs/analysis_reports"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/Table3_DiD_Regression_Results.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())
        
    print(f"\n[SUCCESS]: DiD regression report successfully exported to: {report_path}")

if __name__ == "__main__":
    # Bind context to the current directory to ensure relative paths resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        df_panel = load_data()
        execute_did_regression(df_panel)
    except Exception as e:
        print(f"\nCRITICAL REGRESSION FAILURE: {str(e)}")