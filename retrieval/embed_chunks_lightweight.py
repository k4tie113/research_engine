#!/usr/bin/env python3
"""
embed_chunks_lightweight.py
---------------------------
Lightweight embedding script using all-MiniLM-L6-v2 (much faster than BGE).
Creates embeddings and FAISS index for the Mistral RAG script.
"""

import os, json, jsonlines, numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch, faiss

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "database" / "data" / "chunks_oai.jsonl"
EMB_DIR = ROOT / "database" / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# === MODEL CONFIG ===
MODEL_NAME = "all-MiniLM-L6-v2"  # Much lighter than BGE
DEVICE = "cpu"  # Keep on CPU for simplicity
BATCH_SIZE = 128  # Larger batch since model is smaller
MAX_CHUNKS = 1000  # Limit for testing

# === FAISS CONFIG ===
FAISS_PATH = EMB_DIR / "faiss_index_bge.bin"  # Keep same name for compatibility
METADATA_PATH = EMB_DIR / "metadata_bge.jsonl"  # Keep same name for compatibility

def main(max_chunks: int = MAX_CHUNKS):
    assert CHUNKS.exists(), f"Missing {CHUNKS}; run the PDF pipeline first."

    print(f"Loading lightweight model {MODEL_NAME} (device={DEVICE})...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded (dim={dim})")

    # Warmup check
    try:
        print("Warming up encoder on 2 texts...")
        warm = model.encode(["hello", "world"], normalize_embeddings=True)
        warm = np.asarray(warm, dtype=np.float32)
        print(f"Warmup OK: {warm.shape}")
    except Exception as e:
        print(f"Warmup failed: {e}")
        raise

    texts, metas = [], []
    with jsonlines.open(CHUNKS, "r") as reader:
        for i, rec in enumerate(reader):
            if max_chunks and i >= max_chunks:
                break
            texts.append(rec["chunk_text"])
            metas.append({
                "paper_id": rec["paper_id"],
                "chunk_index": rec["chunk_index"],
                "title": rec.get("title", ""),
                "authors": rec.get("authors", ""),
                "token_count": rec.get("token_count", 0),
            })

    print(f"Encoding {len(texts)} chunks using {MODEL_NAME}...")
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i+BATCH_SIZE]
        with torch.inference_mode():
            vecs = model.encode(batch, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        embs.append(vecs)
    embs = np.vstack(embs)

    # === SAVE EMBEDDINGS + METADATA ===
    np.save(EMB_DIR / "embeddings_bge.npy", embs)
    with jsonlines.open(METADATA_PATH, "w") as w:
        for m in metas:
            w.write(m)

    print(f"Saved embeddings: {embs.shape} -> {EMB_DIR/'embeddings_bge.npy'}")
    print(f"Saved metadata: {METADATA_PATH}")

    # === CREATE AND SAVE FAISS INDEX ===
    print("Building FAISS index (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)  # inner product = cosine when normalized
    index.add(embs)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index saved -> {FAISS_PATH}")
    print("Done! Your lightweight embeddings are ready for retrieval.")

if __name__ == "__main__":
    import sys
    max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_CHUNKS
    main(max_chunks=max_chunks)
