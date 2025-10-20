#!/usr/bin/env python3
"""
Quick script to create embeddings and FAISS index for the Mistral RAG system.
Uses lightweight all-MiniLM-L6-v2 model for fast processing.
"""

import jsonlines, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Paths
ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "database" / "data" / "chunks_oai.jsonl"
EMB_DIR = ROOT / "database" / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = EMB_DIR / "faiss_index_bge.bin"
META_PATH = EMB_DIR / "metadata_bge.jsonl"

print("Loading lightweight model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading chunks...")
texts, metas = [], []
with jsonlines.open(CHUNKS, "r") as reader:
    for rec in reader:
        texts.append(rec["chunk_text"])
        metas.append({
            "paper_id": rec["paper_id"],
            "chunk_index": rec["chunk_index"],
            "title": rec.get("title", ""),
            "authors": rec.get("authors", ""),
            "token_count": rec.get("token_count", 0),
        })
        if len(texts) >= 1000:  # Limit for speed
            break

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype=np.float32)

print("Creating FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("Saving files...")
faiss.write_index(index, str(INDEX_PATH))
with jsonlines.open(META_PATH, "w") as w:
    for m in metas:
        w.write(m)

print("Done! Ready to run Mistral script.")
