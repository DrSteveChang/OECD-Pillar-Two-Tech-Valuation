# ==============================================================================
# Script: app_orchestrator.py
# Purpose: Master Platform Orchestrator (Runtime RAG & Econometric Synthesis)
# ==============================================================================

import os
import json
import pandas as pd
import chromadb
from google import genai

# 1. PATH CONFIGURATIONS
BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
ECON_JSON = os.path.join(BASE_DIR, "data/processed/econometric_outputs.json")
NETWORK_CSV = os.path.join(BASE_DIR, "data/processed/micro_panel_with_network_features.csv")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data/vector_db") 

POLICYMAKER_REPORT_OUT = os.path.join(BASE_DIR, "output/policymaker_global_impact_report.txt")
CFO_REPORT_OUT = os.path.join(BASE_DIR, "output/cfo_corporate_valuation_briefing.txt")

# 2. RUNTIME RETRIEVAL-AUGMENTED GENERATION (THE REAL RAG)
def retrieve_qualitative_context(query_text, n_results=4):
    """Performs semantic search over the real SEC 10-K and academic database."""
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"[WARNING] Vector store missing at {CHROMA_DB_DIR}. Proceeding with zero-context.")
        return ""
        
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        # [Fix 1] Remove mandatory embedding function to use the default one persisted in the collection
        collection = client.get_collection(name="tfm_knowledge_base")
        results = collection.query(query_texts=[query_text], n_results=n_results)
        
        documents = results.get('documents', [[]])[0]
        return "\n\n---\n\n".join(documents)
    except Exception as e:
        print(f"[!] Critical Vector Query Failure: {e}")
        return ""

# 3. REPORT COMPILATION ORCHESTRATOR
def generate_platform_outputs():
    print("="*60)
    print(" EXECUTING MASTER PLATFORM ORCHESTRATOR: COMPLETE SYSTEM INTEGRATION ")
    print("="*60)

    # --------------------------------------------------------------------------
    # STEP 1: INGEST VERIFIED QUANTITATIVE DATA
    # --------------------------------------------------------------------------
    print("[1/4] Loading real econometric estimates and network feature matrices...")
    if not os.path.exists(ECON_JSON) or not os.path.exists(NETWORK_CSV):
        print("[ERROR] Micro network data or econometric outputs missing. Run prerequisites first.")
        return

    with open(ECON_JSON, 'r', encoding='utf-8') as f:
        econ_metrics = json.load(f)
        
    network_df = pd.read_csv(NETWORK_CSV)
    
    avg_jbr = network_df['Jurisdictional_Blending_Ratio'].mean()
    max_jbr_row = network_df.loc[network_df['Jurisdictional_Blending_Ratio'].idxmax()]

    macro_p2 = econ_metrics["macro_analysis"]
    micro_p2 = econ_metrics["micro_analysis"]

    # --------------------------------------------------------------------------
    # STEP 2: COMPILE ANALYSIS FOR POLICYMAKERS
    # --------------------------------------------------------------------------
    print("[2/4] Triggering Runtime RAG & Generating Policymaker Global Impact Report...")
    
    policymaker_rag_context = retrieve_qualitative_context(
        "OECD Pillar Two GloBE rules statutory implementation friction DTA deferred tax asset"
    )

    policymaker_system_prompt = f"""
    You are a Senior Sovereign Tax Policy Analyst writing an official, neutral, and empirically grounded report for international policymakers.
    Your output must be structured, professional, and strictly free from AI-clichés. Output exclusively in English.

    [GROUND TRUTH ECONOMETRIC GROUNDING]
    - Data Source: {macro_p2['data_source']}
    - Total Jurisdictions Analyzed: {macro_p2['total_jurisdictions_analyzed']}
    - Flagged Low-Tax Jurisdictions (ETR < 15%): {macro_p2['jurisdictions_below_15_percent_etr']}
    - Global Pre-Treatment Exposed Profit: USD {macro_p2['aggregate_profit_exposed_to_pillar_two_usd']:,}
    - Macro SDiD ATT (Tau): {macro_p2['sdid_estimates']['tau']} (p-value: {macro_p2['sdid_estimates']['placebo_permutation_p_value']})
    - Macro Statistical Status: {macro_p2['sdid_estimates']['statistical_status']}

    [RETRIEVED SEC 10-K & ACADEMIC GROUNDING (RAG)]
    {policymaker_rag_context}

    [REQUIRED STRUCTURAL HEADINGS]
    1. Executive Summary
    2. Global Macro Allocation and Revenue Exposure
    3. Empirical Synthetic Difference-in-Differences (SDiD) Interpretation
    4. Statutory Implementation Frictions and Policy Recommendations
    """

    # --------------------------------------------------------------------------
    # STEP 3: COMPILE ANALYSIS FOR CORPORATE CFOS
    # --------------------------------------------------------------------------
    print("[3/4] Triggering Runtime RAG & Generating Corporate CFO Valuation Briefing...")
    
    cfo_rag_context = retrieve_qualitative_context(
        "SEC 10-K financial notes valuation allowance corporate tax restructuring technology sector"
    )

    cfo_system_prompt = f"""
    You are an Elite Institutional Equity Research Director advising Tech Sector Chief Financial Officers (CFOs).
    Your tone must be highly objective, rigorous, and professional. Output exclusively in English.

    [GROUND TRUTH MICRO METRICS GROUNDING]
    - Total Tech Cohorts Analyzed: {micro_p2['analyzed_firm_cohorts']}
    - Sample Mean Jurisdictions Blending Ratio (JBR): {avg_jbr:.4f}
    - Maximum Corporate Blending Risk Profile: {max_jbr_row['Ticker']} (JBR: {max_jbr_row['Jurisdictional_Blending_Ratio']})
    - First-Difference Causal Forest ATT (Valuation Impact): {micro_p2['causal_forest_estimates']['unbiased_att']}
    - Statistical Status: {micro_p2['causal_forest_estimates']['statistical_status']}

    [RETRIEVED SEC 10-K & ACADEMIC GROUNDING (RAG)]
    {cfo_rag_context}

    [REQUIRED STRUCTURAL HEADINGS]
    1. Strategic C-Suite Briefing
    2. Corporate Network Complexity & Jurisdictional Blending Risk
    3. Causal Forest Micro Valuation Compression Interpretation
    4. Balance Sheet Defense & Tax Accounting Restructuring
    """

    # --------------------------------------------------------------------------
    # STEP 4: GENERATE VIA GEMINI INFRASTRUCTURE & PERSIST
    # --------------------------------------------------------------------------
    print("[4/4] Executing Gemini inference models (google.genai SDK) and saving complete files...")
    
    # Suppress environment variable conflicts
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
        print("      -> Suppressed conflicting GOOGLE_API_KEY in runtime environment.")
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY not found in environment variables.")
        return
        
    client = genai.Client(api_key=gemini_key)
    os.makedirs(os.path.dirname(POLICYMAKER_REPORT_OUT), exist_ok=True)
    
    # Establish robust fallback routing strategy
    # Automatically iterate through supported model aliases to prevent 404 errors
    available_models = [
        'gemini-2.5-pro', 
        'gemini-1.5-pro-latest', 
        'gemini-2.5-flash', 
        'gemini-1.5-flash'
    ]
    
    successful_model = None
    response_policy = None
    response_cfo = None
    
    for model_name in available_models:
        try:
            print(f"      -> Attempting to connect to model engine: {model_name} ...")
            # Generate Policy Report
            response_policy = client.models.generate_content(
                model=model_name,
                contents=policymaker_system_prompt
            )
            # Generate CFO Report
            response_cfo = client.models.generate_content(
                model=model_name,
                contents=cfo_system_prompt
            )
            successful_model = model_name
            print(f"      -> {model_name} connection and inference successful!")
            break 
            
        except Exception as e:
            error_msg = str(e).upper()
            # Capture 404, 503, 429, 500 errors to trigger automatic fallback
            if any(err in error_msg for err in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500"]):
                print(f"         [!] Model {model_name} currently unavailable (Reason: Server high load or permissions), automatically falling back...")
                continue 
            else:
                raise e # If it is a fatal logic error, raise it

    if not successful_model:
        print("[ERROR] All pre-set model aliases were rejected by the server. Please check API key permissions.")
        return

    # Persist results
    with open(POLICYMAKER_REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(response_policy.text)
        
    with open(CFO_REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write(response_cfo.text)

    print("\n" + "="*60)
    print(f"[SUCCESS] Orchestration complete! Reports successfully generated by {successful_model}.")
    print(f"   -> Policymaker Report saved to: {POLICYMAKER_REPORT_OUT}")
    print(f"   -> CFO Briefing saved to: {CFO_REPORT_OUT}")
    print("="*60)

if __name__ == "__main__":
    generate_platform_outputs()