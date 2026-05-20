# ==============================================================================
# Script: data_ingestion_pipeline.py
# Purpose: Automated Batch ETL for Panel SEC Filings & Market Consensus
# ==============================================================================

import os
import requests
from bs4 import BeautifulSoup
import feedparser
import chromadb
import time
import urllib.parse
import pandas as pd

BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
DB_PATH = os.path.join(BASE_DIR, "data/vector_db")
CSV_PATH = os.path.join(BASE_DIR, "data/processed/micro_panel_enriched.csv")

# SEC requires a valid User-Agent declaring identity and purpose
HEADERS = {
    'User-Agent': 'Boyan Zhang (Academic Research Project) boyan@example.com'
}

# ------------------------------------------------------------------------------
# 1. DYNAMIC TICKER TO CIK MAPPING
# ------------------------------------------------------------------------------
def get_ticker_to_cik_mapping():
    print("[1/4] Fetching SEC Ticker-to-CIK mapping...")
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(mapping_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception("Failed to fetch SEC mapping.")
    
    mapping_data = response.json()
    # Build dictionary: {'AAPL': '0000320193', 'MSFT': '00000789019', ...}
    ticker_to_cik = {
        item['ticker']: str(item['cik_str']).zfill(10) 
        for item in mapping_data.values()
    }
    return ticker_to_cik

# ------------------------------------------------------------------------------
# 2. BATCH FETCH SEC 10-K DATA
# ------------------------------------------------------------------------------
def fetch_sec_10k_tax_section(cik, ticker):
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=HEADERS)
    
    if response.status_code != 200:
        return None

    data = response.json()
    filings = data.get('filings', {}).get('recent', {})
    
    if not filings:
        return None

    for idx, form in enumerate(filings.get('form', [])):
        # Look for 10-K (US domestic) or 20-F (Foreign private issuers)
        if form in ['10-K', '20-F']:
            accession_number = filings['accessionNumber'][idx].replace('-', '')
            primary_doc = filings['primaryDocument'][idx]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_doc}"
            
            try:
                doc_response = requests.get(doc_url, headers=HEADERS)
                soup = BeautifulSoup(doc_response.content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                paragraphs = text.split('  ')
                
                # Extract core paragraphs containing the word 'tax'
                tax_context = [p for p in paragraphs if 'tax' in p.lower() and len(p) > 100]
                # Truncate to the first 3000 characters to optimize for vectorization
                return " ".join(tax_context)[:3000] 
            except Exception as e:
                print(f"      [!] Error parsing {ticker}: {e}")
                return None
    return None

# ------------------------------------------------------------------------------
# 3. FETCH REAL MARKET CONSENSUS
# ------------------------------------------------------------------------------
def fetch_market_consensus_rss():
    print("\n[3/4] Fetching real-time Macro Market Consensus via RSS...")
    query = "Pillar Two Global Minimum Tax impact on technology companies effective tax rate"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    consensus_texts = []
    
    for entry in feed.entries[:5]:  
        title = entry.title
        published = entry.get('published', 'Recent')
        consensus_texts.append(f"Source: {title} (Date: {published})")
        
    return "REAL MARKET CONSENSUS SUMMARY:\n" + "\n".join(consensus_texts)

# ------------------------------------------------------------------------------
# 4. AUTOMATED BATCH INGESTION PIPELINE
# ------------------------------------------------------------------------------
def run_batch_pipeline():
    print("="*60)
    print(" STARTING BATCH DATA INGESTION PIPELINE (SEC & RSS) ")
    print("="*60)

    # 1. Fetch Ticker-to-CIK mapping
    ticker_cik_map = get_ticker_to_cik_mapping()
    
    # 2. Read target companies from local data warehouse
    print(f"\n[2/4] Reading Target Companies from Enriched Panel...")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Panel data not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    target_tickers = df['Ticker'].unique().tolist()
    print(f"      -> Found {len(target_tickers)} unique tickers to process.")

    # 3. Prepare Vector Database
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma_client.delete_collection(name="tfm_knowledge_base") # Reset old collection
    except:
        pass
    collection = chroma_client.create_collection(name="tfm_knowledge_base")

    # 4. Batch fetch SEC filings and prepare for ingestion
    docs = []
    metas = []
    doc_ids = []

    # To prevent the first run from taking too long, we slice the list.
    # If you want to process the full dataset, change 'target_tickers[:10]' to 'target_tickers'
    for i, ticker in enumerate(target_tickers): 
        print(f"      [{i+1}/{len(target_tickers)}] Processing {ticker}...")
        
        cik = ticker_cik_map.get(ticker)
        if not cik:
            print(f"      [!] Skipping {ticker}: No CIK mapping found.")
            continue
            
        tax_text = fetch_sec_10k_tax_section(cik, ticker)
        
        if tax_text:
            docs.append(tax_text)
            metas.append({"source": f"SEC_{ticker}_REAL", "type": "cfo_footnote", "ticker": ticker})
            doc_ids.append(f"doc_{ticker}_10k")
        else:
            print(f"      [!] No sufficient tax data found for {ticker}.")
            
        # [CRITICAL] Mandatory 0.5s sleep to comply with SEC API rate limits and avoid IP bans
        time.sleep(0.5) 

    # 5. Fetch market consensus
    market_consensus_text = fetch_market_consensus_rss()
    docs.append(market_consensus_text)
    metas.append({"source": "RSS_NEWS_API", "type": "market_consensus", "ticker": "GLOBAL"})
    doc_ids.append("doc_market_consensus_real")

    # 6. Batch ingest into ChromaDB
    print(f"\n[4/4] Ingesting {len(docs)} documents into Vector Database...")
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=doc_ids
    )
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Batch Pipeline Complete! {len(docs)-1} companies and 1 Macro Consensus ingested.")
    print("="*60)

if __name__ == "__main__":
    run_batch_pipeline()