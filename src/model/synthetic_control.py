import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_data():
    data_path = "../../data/analytical_panel_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("CRITICAL: analytical_panel_dataset.csv not found.")
    return pd.read_csv(data_path)

def optimize_weights(X_treated, X_donor):
    J = X_donor.shape[1]
    def objective(W, X_t, X_d):
        diff = X_t - np.dot(X_d, W)
        return np.sum(diff**2)
    
    cons = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1})
    bounds = [(0, 1) for _ in range(J)]
    initial_W = np.ones(J) / J
    
    result = minimize(
        objective, initial_W, args=(X_treated, X_donor),
        method='SLSQP', bounds=bounds, constraints=cons,
        options={'disp': False, 'maxiter': 1000}
    )
    return result.x

def estimate_scm_trajectory(df, target_ticker, pre_policy_year=2022):
    """
    Silent trajectory estimator for batch processing. 
    Returns the indexed time-series panel for a given target.
    """
    df_pre = df[df['Year'] <= pre_policy_year].copy()
    predictors = ['Leverage', 'RD_Intensity', 'Intangible_Ratio', 'ETR']
    
    treated_data = df_pre[df_pre['Ticker'] == target_ticker]
    if treated_data.empty:
        return None
    X1 = treated_data[predictors].mean().values
    
    donors = df_pre[df_pre['Treatment_Group'] == 0]['Ticker'].unique()
    donor_matrix = []
    valid_donor_tickers = []
    
    for donor in donors:
        donor_data = df_pre[df_pre['Ticker'] == donor]
        if len(donor_data) > 0:
            donor_means = donor_data[predictors].mean()
            if not donor_means.isna().any():
                donor_matrix.append(donor_means.values)
                valid_donor_tickers.append(donor)
                
    if not donor_matrix:
        return None
        
    X0 = np.column_stack(donor_matrix)
    optimal_weights = optimize_weights(X1, X0)
    
    # Generate long-run trajectory
    years = sorted(df['Year'].unique())
    df_base = df[df['Year'] == pre_policy_year]
    
    true_base_val = df_base[df_base['Ticker'] == target_ticker]['TotalRevenue'].values[0]
    syn_base_val = sum(df_base[df_base['Ticker'] == t]['TotalRevenue'].values[0] * optimal_weights[i] for i, t in enumerate(valid_donor_tickers) if df_base[df_base['Ticker'] == t].size > 0)
    
    if true_base_val <= 0 or syn_base_val <= 0:
        return None
        
    records = []
    for year in years:
        df_year = df[df['Year'] == year]
        t_row = df_year[df_year['Ticker'] == target_ticker]
        if t_row.empty: continue
        y_t_raw = t_row['TotalRevenue'].values[0]
        
        y_s_raw = sum(df_year[df_year['Ticker'] == t]['TotalRevenue'].values[0] * optimal_weights[i] for i, t in enumerate(valid_donor_tickers) if df_year[df_year['Ticker'] == t].size > 0)
        
        idx_true = (y_t_raw / true_base_val) * 100
        idx_syn = (y_s_raw / syn_base_val) * 100
        
        records.append({
            "Ticker": target_ticker,
            "Year": year,
            "Index_True": idx_true,
            "Index_Syn": idx_syn,
            "Causal_Gap_Pts": idx_true - idx_syn
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        df_panel = load_data()
        treatment_firms = df_panel[df_panel['Treatment_Group'] == 1]['Ticker'].unique()
        print(f"Loaded analytical panel. Detected {len(treatment_firms)} MNEs in the Treatment Group.")
        
        all_trajectories = []
        
        print("\nExecuting batch SCM estimation across full treatment cohort...")
        for firm in treatment_firms:
            df_firm_traj = estimate_scm_trajectory(df_panel, firm, pre_policy_year=2022)
            if df_firm_traj is not None and not df_firm_traj.empty:
                all_trajectories.append(df_firm_traj)
                
        # Consolidate all factual vs counterfactual trajectories
        df_global_scm = pd.concat(all_trajectories)
        
        # Calculate the SCM-based ATT (Average Treatment Effect on the Treated) per year
        df_att = df_global_scm.groupby('Year')[['Index_True', 'Index_Syn', 'Causal_Gap_Pts']].mean().reset_index()
        
        # Output paths
        output_dir = "../../data/derived_results"
        os.makedirs(output_dir, exist_ok=True)
        
        df_global_scm.to_csv(f"{output_dir}/full_cohort_scm_trajectories.csv", index=False)
        df_att.to_csv(f"{output_dir}/global_scm_att_results.csv", index=False)
        
        print("\n===============================================================")
        print("   BATCH SCM PROCESSING COMPLETED SUCCESSFULLY                ")
        print("===============================================================")
        print("Global Aggregated ATT Trajectory (Base 2022 = 100):")
        print(df_att.round(2).to_string(index=False))
        print(f"\n[SUCCESS]: Full panels flushed to {output_dir}/")
        
    except Exception as e:
        print(f"\nCRITICAL BATCH SCM FAILURE: {str(e)}")