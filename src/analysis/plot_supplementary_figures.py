import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

def set_academic_plot_style():
    """
    Configures standardized academic plotting aesthetics mimicking STATA/LaTeX output.
    Forces Adobe Core 14 standard fonts and fixes the unicode minus sign bug (?).
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['pdf.use14corefonts'] = True  
    
    # -------------------------------------------------------------------------
    # CRITICAL FIX: COMPATIBILITY LAYER FOR MINUS SIGN CORRUPTION (?)
    # -------------------------------------------------------------------------
    # Disabling unicode_minus forces matplotlib to use standard ASCII hyphens (-) 
    # instead of Unicode minus glyphs (\u2212), eliminating '?' rendering bugs 
    # inside certain macOS/Adobe PDF viewer environments.
    plt.rcParams['axes.unicode_minus'] = False  

    try:
        sns.set_theme(style="whitegrid")
    except AttributeError:
        try:
            sns.set_style("whitegrid")
        except AttributeError:
            if 'seaborn-v0_8-whitegrid' in plt.style.available:
                plt.style.use('seaborn-v0_8-whitegrid')
            else:
                plt.style.use('ggplot')

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14
    })

def generate_did_event_study(panel_path, output_dir):
    """
    Executes a dynamic TWFE Event Study regression omitting 2022 as the reference year,
    extracts the interaction coefficients with 95% CIs, and renders the plot.
    """
    print("\n[1/2] Computing Dynamic TWFE Event Study Regression...")
    df = pd.read_csv(panel_path)
    df = df[df['TotalRevenue'] > 0].copy()
    df['Log_Revenue'] = np.log(df['TotalRevenue'])
    
    # Formulate Event Study using 2022 as the excluded reference baseline year
    formula = "Log_Revenue ~ Treatment_Group * C(Year, Treatment(2022)) + Leverage + RD_Intensity"
    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['Ticker']})
    
    years = sorted(df['Year'].unique())
    plot_records = []
    
    for y in years:
        if y == 2022:
            plot_records.append({"Year": y, "Coef": 0.0, "Lower_CI": 0.0, "Upper_CI": 0.0})
        else:
            target_term = [p for p in results.params.index if 'Treatment_Group' in p and str(y) in p]
            if target_term:
                term = target_term[0]
                coef = results.params[term]
                se = results.bse[term]
                plot_records.append({
                    "Year": y,
                    "Coef": coef,
                    "Lower_CI": coef - (1.96 * se),
                    "Upper_CI": coef + (1.96 * se)
                })
                
    df_plot = pd.DataFrame(plot_records)
    
    # Canvas Construction
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Plot coefficients and 95% error bars
    ax.errorbar(df_plot['Year'], df_plot['Coef'], 
                yerr=[df_plot['Coef'] - df_plot['Lower_CI'], df_plot['Upper_CI'] - df_plot['Coef']],
                fmt='-o', color='#1f77b4', linewidth=2, elinewidth=1.5, capsize=4, 
                label='Point Estimate (95% Clustered CI)')
    
    # Structural baselines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax.axvline(x=2022, color='#7f7f7f', linestyle=':', linewidth=1.5)
    
    # Dynamic text positioning relative to Y-axis scale limits
    y_max_padded = ax.get_ylim()[1] * 0.7
    ax.text(2022.05, y_max_padded, 'Reference Year (2022)', rotation=90, color='#555555', fontsize=10)
    
    # Titles and axis mechanics
    ax.set_xlabel('Fiscal Year', labelpad=12)
    ax.set_ylabel('DiD Impact Coefficient ($\delta_k$)', labelpad=12)
    ax.set_title('DiD Event Study: Dynamic Timeline of Pillar Two Shock on Revenue Scale', pad=18)
    ax.set_xticks(df_plot['Year'].astype(int))
    ax.set_xticklabels(df_plot['Year'].astype(int))
    
    # Externalized horizontal legend footer placement
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure2_DiD_Event_Study.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/Figure2_DiD_Event_Study.png", format='png', dpi=300, bbox_inches='tight')
    print("[SUCCESS]: Figure2 (Event Study Plot) fixed and generated successfully.")
    plt.close()

def generate_scm_spaghetti_plot(cohort_path, att_path, output_dir):
    """
    Ingests full cohort long-run SCM data to plot individual firm trajectories 
    as an underlying distribution layer, overlaying the bold aggregated ATT gap.
    """
    print("\n[2/2] Generating Full-Sample SCM Heterogeneity Spaghetti Plot...")
    df_cohort = pd.read_csv(cohort_path)
    df_att = pd.read_csv(att_path)
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    tickers = df_cohort['Ticker'].unique()
    for i, ticker in enumerate(tickers):
        df_firm = df_cohort[df_cohort['Ticker'] == ticker]
        lbl = 'Individual Treated MNE Gaps' if i == 0 else ""
        ax.plot(df_firm['Year'], df_firm['Causal_Gap_Pts'], color='#cccccc', alpha=0.4, linewidth=1.0, label=lbl)
        
    ax.plot(df_att['Year'], df_att['Causal_Gap_Pts'], marker='D', color='#d62728', linewidth=3.0, label='Aggregated SCM ATT (Cohort Mean)')
    
    # Structural baselines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax.axvline(x=2022, color='#7f7f7f', linestyle=':', linewidth=1.5)
    
    y_max_padded_scm = ax.get_ylim()[1] * 0.4
    ax.text(2022.05, y_max_padded_scm, 'Policy Baseline (2022)', rotation=90, color='#555555', fontsize=10)
    
    # Labels and canvas mechanics
    ax.set_xlabel('Fiscal Year', labelpad=12)
    ax.set_ylabel('Causal Gap in Index Points ($\Delta$ Index)', labelpad=12)
    ax.set_title('SCM Heterogeneity Spectrum: Individual Gaps vs. Consolidated ATT', pad=18)
    ax.set_xticks(df_att['Year'].astype(int))
    ax.set_xticklabels(df_att['Year'].astype(int))
    
    # Externalized horizontal legend footer placement
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure3_SCM_Distribution_Gaps.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/Figure3_SCM_Distribution_Gaps.png", format='png', dpi=300, bbox_inches='tight')
    print("[SUCCESS]: Figure3 (SCM Spaghetti Plot) fixed and generated successfully.")
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    set_academic_plot_style()
    
    PANEL_DATA = "../../data/analytical_panel_dataset.csv"
    COHORT_SCM = "../../data/derived_results/full_cohort_scm_trajectories.csv"
    ATT_SCM = "../../data/derived_results/global_scm_att_results.csv"
    OUTPUT_DIR = "../../docs/analysis_reports/figures"
    
    try:
        generate_did_event_study(PANEL_DATA, OUTPUT_DIR)
        generate_scm_spaghetti_plot(COHORT_SCM, ATT_SCM, OUTPUT_DIR)
        print("\n===============================================================")
        print("   ALL SUPPLEMENTARY ACADEMIC FIGURES COPIED & FLUSHED         ")
        print("===============================================================")
    except Exception as e:
        print(f"\nCRITICAL PIPELINE FAILURE: {str(e)}")