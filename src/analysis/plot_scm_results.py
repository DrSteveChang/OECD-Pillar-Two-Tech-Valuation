import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_plot_style():
    """
    Configures standard academic plotting aesthetics.
    Forces Adobe Core 14 standard fonts (Helvetica) and standardizes
    Type 42 font embedding to ensure seamless LaTeX integration.
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['pdf.use14corefonts'] = True  

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

def plot_aggregated_scm_att():
    """
    Ingests the global SCM ATT results and renders a high-resolution, 
    publication-quality trajectory plot with an externalized legend footer.
    """
    print("Loading aggregated SCM ATT dataset for visualization...")
    data_path = "../../data/derived_results/global_scm_att_results.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CRITICAL: Aggregated SCM results not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    
    # Initialize the figure canvas
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Plot True vs Synthetic trajectories
    ax.plot(df['Year'], df['Index_True'], marker='o', color='#1f77b4', linewidth=2.5, label='True Treated MNEs (Observed)')
    ax.plot(df['Year'], df['Index_Syn'], marker='s', linestyle='--', color='#d62728', linewidth=2.5, label='Synthetic Counterfactual (Donor Aggregate)')
    
    # -------------------------------------------------------------------------
    # POLICY TIMELINE ANNOTATIONS (EXTERNAL LEGEND FIX)
    # -------------------------------------------------------------------------
    ax.axvline(x=2022, color='#7f7f7f', linestyle=':', linewidth=1.5)
    
    # FIX: Centered the text vertically at Y=120 and set alignment to 'center'.
    # This places the label beautifully along the baseline axis without crowding headers.
    ax.text(2022.05, 120, 'Baseline Alignment (2022)', 
            rotation=90, 
            horizontalalignment='left', 
            verticalalignment='center', 
            color='#555555', 
            fontsize=10)
    
    # Draw shaded area for Policy Anticipation & Implementation (2023 - 2025)
    ax.axvspan(2023, 2025, color='#e5e5e5', alpha=0.5, label='Pillar Two Treatment Window')
    
    # Labels and scales configuration
    ax.set_xlabel('Fiscal Year (Reporting Cycle)', labelpad=12)
    ax.set_ylabel('Normalized Revenue Index (Base 2022 = 100)', labelpad=12)
    ax.set_title('Global SCM Estimates of OECD Pillar Two Impact on Technology MNEs', pad=18)
    
    # Force X-axis ticks to display as explicit chronological integer blocks
    ax.set_xticks(df['Year'].astype(int))
    ax.set_xticklabels(df['Year'].astype(int))
    
    # Set explicit coordinate limits to lock padding space
    ax.set_ylim(95, 145)
    ax.set_xlim(2021.7, 2025.3)
    
    # -------------------------------------------------------------------------
    # JOURNAL LAYOUT FIX: EXTERNAL HORIZONTAL LEGEND
    # -------------------------------------------------------------------------
    # bbox_to_anchor=(0.5, -0.16) moves the legend box safely below the X-axis line.
    # ncol=3 forces the three legend tokens to align horizontally in a single row.
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.16), 
              ncol=3, 
              frameon=True, 
              facecolor='white', 
              edgecolor='none')
    
    # Layout adjustment optimization
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # MULTI-FORMAT STRUCTURAL OUTPUT PROTOCOL
    # -------------------------------------------------------------------------
    output_dir = "../../docs/analysis_reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = f"{output_dir}/Figure1_SCM_Global_ATT.pdf"
    png_path = f"{output_dir}/Figure1_SCM_Global_ATT.png"
    
    # bbox_inches='tight' is critical here to ensure the newly appended bottom legend
    # and top titles are fully captured within the dynamic saving bounding box.
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"[SUCCESS]: High-resolution figures successfully exported to:\n  - {pdf_path}\n  - {png_path}")
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    set_academic_plot_style()
    try:
        plot_aggregated_scm_att()
    except Exception as e:
        print(f"\nCRITICAL VISUALIZATION FAILURE: {str(e)}")