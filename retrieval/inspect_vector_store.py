#!/usr/bin/env python3
"""
Simple script to inspect the vector store contents
"""

import argparse
import jsonlines
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def inspect_vector_store():
    """Show basic stats about the vector store."""
    root = Path(__file__).resolve().parents[1]
    idx_path = root / "data" / "faiss" / "index_flatip.faiss"
    ids_path = root / "data" / "faiss" / "ids.jsonl"
    chunks_path = root / "data" / "chunks.jsonl"
    
    print("=== VECTOR STORE INSPECTION ===")
    print()
    
    # Check files exist
    print("Files:")
    print(f"  FAISS Index: {idx_path.exists()} ({idx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  IDs File: {ids_path.exists()} ({ids_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Chunks File: {chunks_path.exists()} ({chunks_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    
    if not idx_path.exists() or not ids_path.exists():
        print("ERROR: Vector store files not found!")
        return
    
    # Load FAISS index
    index = faiss.read_index(str(idx_path))
    print(f"FAISS Index Stats:")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")
    print(f"  Index type: {type(index).__name__}")
    print()
    
    # Load IDs
    with jsonlines.open(ids_path, "r") as r:
        ids = list(r)
    
    print(f"IDs File Stats:")
    print(f"  Total records: {len(ids)}")
    
    # Sample some IDs
    print(f"  Sample records:")
    for i, record in enumerate(ids[:5]):
        paper_id = record.get('paper_id', 'Unknown')
        chunk_idx = record.get('chunk_index', 'Unknown')
        title = record.get('title', 'No title')[:50]
        print(f"    {i+1}. {paper_id} #{chunk_idx} - {title}...")
    print()
    
    # Check chunks if available
    if chunks_path.exists():
        with jsonlines.open(chunks_path, "r") as r:
            chunks = list(r)
        print(f"Chunks File Stats:")
        print(f"  Total chunks: {len(chunks)}")
        
        # Sample chunks
        print(f"  Sample chunk content:")
        for i, chunk in enumerate(chunks[:2]):
            paper_id = chunk.get('paper_id', 'Unknown')
            chunk_idx = chunk.get('chunk_index', 'Unknown')
            content = chunk.get('chunk_text', 'No content')[:100]
            print(f"    {i+1}. {paper_id} #{chunk_idx}: {content}...")
        print()
    
    # Show unique papers
    unique_papers = set(record.get('paper_id') for record in ids if record.get('paper_id'))
    print(f"Unique Papers: {len(unique_papers)}")
    print(f"Sample paper IDs: {list(unique_papers)[:10]}")

def search_example():
    """Show a simple search example."""
    root = Path(__file__).resolve().parents[1]
    idx_path = root / "data" / "faiss" / "index_flatip.faiss"
    ids_path = root / "data" / "faiss" / "ids.jsonl"
    chunks_path = root / "data" / "chunks.jsonl"
    
    if not idx_path.exists() or not ids_path.exists():
        print("ERROR: Vector store files not found!")
        return
    
    # Load everything
    index = faiss.read_index(str(idx_path))
    with jsonlines.open(ids_path, "r") as r:
        ids = list(r)
    
    chunk_map = {}
    if chunks_path.exists():
        with jsonlines.open(chunks_path, "r") as reader:
            for rec in reader:
                try:
                    key = (rec.get("paper_id"), int(rec.get("chunk_index", -1)))
                    chunk_map[key] = rec.get("chunk_text", "")
                except Exception:
                    continue
    
    # Load model and search
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    query = "machine learning in healthcare"
    
    vec = model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    q = np.asarray(vec, dtype=np.float32)
    
    D, I = index.search(q, 3)
    
    print(f"\n=== SEARCH EXAMPLE: '{query}' ===")
    print()
    
    for rank, (score, ridx) in enumerate(zip(D[0], I[0]), 1):
        if ridx < 0:
            continue
        
        meta = ids[int(ridx)]
        paper_id = meta.get('paper_id', 'Unknown')
        chunk_idx = meta.get('chunk_index', 'Unknown')
        title = meta.get('title', 'No title')[:100]
        
        key = (paper_id, int(chunk_idx))
        content = chunk_map.get(key, "")
        
        print(f"[{rank}] Score: {score:.3f}")
        print(f"Paper ID: {paper_id}")
        print(f"Chunk: #{chunk_idx}")
        print(f"Title: {title}...")
        if content:
            print(f"Content: {content[:150]}...")
        else:
            print(f"Content: [No content available]")
        print()

def main():
    parser = argparse.ArgumentParser(description="Inspect vector store contents")
    parser.add_argument("--search", action="store_true", help="Show search example")
    args = parser.parse_args()
    
    if args.search:
        search_example()
    else:
        inspect_vector_store()

if __name__ == "__main__":
    main()
