# ==============================================================================
# Script: vector_indexer.py
# Purpose: Tri-dimensional Knowledge Base (OECD + 10-K + Bank Reports)
# Location: platform/src/vector_indexer.py
# ==============================================================================

import os
import chromadb
from chromadb.utils import embedding_functions

# Define base paths for document storage and the vector database
BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
DOCS_OECD_DIR = os.path.join(BASE_DIR, "docs/oecd_rules")
DOCS_FIRM_DIR = os.path.join(BASE_DIR, "docs/firm_footnotes")
DOCS_BANK_DIR = os.path.join(BASE_DIR, "docs/bank_reports")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data/processed/chroma_db")

def _generate_mock_documents():
    """
    Populates directory structure with synthetic data samples 
    for initial system verification and testing.
    """
    os.makedirs(DOCS_OECD_DIR, exist_ok=True)
    os.makedirs(DOCS_FIRM_DIR, exist_ok=True)
    os.makedirs(DOCS_BANK_DIR, exist_ok=True)

    with open(os.path.join(DOCS_OECD_DIR, "OECD_Rules.txt"), "w", encoding="utf-8") as f:
        f.write("OECD Pillar Two GloBE Rules Mandate: The minimum effective tax rate (ETR) is set at 15%. "
                "If an MNE has an ETR below 15%, a Top-up Tax will be applied.")
                
    with open(os.path.join(DOCS_FIRM_DIR, "Tech_10K.txt"), "w", encoding="utf-8") as f:
        f.write("Corporate Tax Footnote 2025: Due to OECD Pillar Two, our deferred tax assets "
                "related to offshore IP are subject to intense revaluation.")
                
    with open(os.path.join(DOCS_BANK_DIR, "Goldman_Sachs_Report.txt"), "w", encoding="utf-8") as f:
        f.write("Goldman Sachs Equity Research: The Pillar Two 15% minimum tax will primarily compress "
                "earnings per share (EPS) for tech firms with high intangible asset concentration. "
                "However, market valuation impact is partially priced in by institutional investors.")
        print("[AUTO-HEAL] Generated mock Investment Bank Report.")

def ingest_documents(directory, doc_type):
    """
    Reads text files from a directory, splits content into semantic chunks 
    (based on periods), and prepares them for embedding.
    """
    documents, metadatas, ids = [], [], []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                # Basic chunking strategy: split by period and filter short segments
                chunks = [chunk.strip() for chunk in f.read().split(". ") if len(chunk) > 10]
                for i, chunk in enumerate(chunks):
                    documents.append(chunk + ".")
                    metadatas.append({"source": filename, "type": doc_type})
                    ids.append(f"{doc_type}_{filename}_chunk_{i}")
    return documents, metadatas, ids

def build_vector_database():
    """
    Initializes the ChromaDB persistent client, sets up the embedding model, 
    and adds processed document chunks to the vector collection.
    """
    print("--- UPDATING VECTOR INDEXER: TRI-DIMENSIONAL KNOWLEDGE BASE ---")
    _generate_mock_documents()
    
    # Initialize persistent client to save index to disk
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Load SentenceTransformer model (all-MiniLM-L6-v2) for high-performance semantic embedding
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Attempt to reset existing collection for a clean update
    try: 
        client.delete_collection(name="tax_policy_collection")
    except Exception: 
        pass
        
    # Create the collection with the configured embedding function
    collection = client.create_collection(name="tax_policy_collection", embedding_function=sentence_transformer_ef)
    
    # Iterate through various knowledge sources
    for doc_dir, doc_type in [(DOCS_OECD_DIR, "policymaker_rule"), 
                              (DOCS_FIRM_DIR, "cfo_footnote"), 
                              (DOCS_BANK_DIR, "market_consensus")]:
        docs, meta, ids = ingest_documents(doc_dir, doc_type)
        if docs:
            collection.add(documents=docs, metadatas=meta, ids=ids)
            print(f"[INDEX SUCCESS] Embedded {len(docs)} chunks for {doc_type}.")
            
    print("--- VECTOR INDEXER UPDATE SUCCESSFUL ---")

def retrieve_context(query, filter_type, n_results=1):
    """
    Performs a semantic similarity search within the vector database, 
    applying a metadata filter to isolate specific knowledge domains.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="tax_policy_collection", embedding_function=sentence_transformer_ef)
    
    # Query with metadata filter to ensure domain-specific retrieval
    results = collection.query(query_texts=[query], n_results=n_results, where={"type": filter_type})
    return " ".join(results['documents'][0]) if results['documents'][0] else "No context found."

if __name__ == "__main__":
    build_vector_database()