# ==============================================================================
# Script: academic_pdf_ingestion.py
# Purpose: Extract, Chunk, and Ingest Academic PDFs & Big 4 Reports
# ==============================================================================

import os
import fitz  # PyMuPDF
import chromadb

BASE_DIR = "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/platform"
PDF_DIR = os.path.join(BASE_DIR, "data/raw/academic_pdfs")
DB_PATH = os.path.join(BASE_DIR, "data/vector_db")

# ------------------------------------------------------------------------------
# 1. TEXT EXTRACTION & CHUNKING
# ------------------------------------------------------------------------------
def extract_and_chunk_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Extracts text from a PDF and splits it into overlapping chunks.
    This ensures that the AI gets complete context without hitting token limits.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    
    # Simple sliding window chunking mechanism
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end].strip())
        start += (chunk_size - overlap) # Move forward, but keep overlap
        
    return chunks

# ------------------------------------------------------------------------------
# 2. INCREMENTAL INGESTION PIPELINE
# ------------------------------------------------------------------------------
def run_academic_ingestion():
    print("="*60)
    print(" STARTING ACADEMIC PDF INGESTION PIPELINE ")
    print("="*60)

    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        print(f"[ERROR] No PDFs found in {PDF_DIR}. Please download some reports first.")
        return

    # Connect to the EXISTING Vector Database
    print("[1/3] Connecting to existing Vector Database...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(name="tfm_knowledge_base")

    docs = []
    metas = []
    doc_ids = []

    print(f"[2/3] Processing PDFs in {PDF_DIR}...")
    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"      -> Extracting and chunking: {filename}")
        
        chunks = extract_and_chunk_pdf(pdf_path)
        
        for i, chunk in enumerate(chunks):
            # We filter out empty or overly short chunks (likely table artifacts)
            if len(chunk) > 150: 
                docs.append(chunk)
                # Tag it specifically as academic/expert consensus
                metas.append({"source": filename, "type": "market_consensus", "chunk_idx": i})
                doc_ids.append(f"doc_pdf_{filename.replace('.pdf', '')}_chunk_{i}")

    if not docs:
        print("[!] No valid text extracted.")
        return

    print(f"\n[3/3] Ingesting {len(docs)} high-value knowledge chunks into ChromaDB...")
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=doc_ids
    )

    print("\n" + "="*60)
    print(f"[SUCCESS] Academic Integration Complete! Added {len(docs)} chunks from {len(os.listdir(PDF_DIR))} PDFs.")
    print("="*60)

if __name__ == "__main__":
    run_academic_ingestion()