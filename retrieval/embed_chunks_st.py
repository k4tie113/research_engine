#!/usr/bin/env python3
import os, jsonlines, numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# Config via env
MODEL_NAME = os.environ.get("ST_MODEL", "intfloat/e5-base-v2")
BATCH = int(os.environ.get("ST_BATCH", "64"))
MAX_CHUNKS = int(os.environ.get("ST_MAX_CHUNKS", "0"))  # 0 = no limit

def main():
    assert CHUNKS.exists(), f"Missing {CHUNKS}; run the PDF pipeline first."
    print(f"Loading sentence-transformers model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts, metas = [], []
    with jsonlines.open(CHUNKS, "r") as reader:
        for i, rec in enumerate(reader):
            if MAX_CHUNKS and i >= MAX_CHUNKS:
                break
            # Many E5-style models expect instruction/queries; for passages, prepend nothing
            texts.append(rec["chunk_text"]) 
            metas.append({
                "paper_id": rec["paper_id"],
                "chunk_index": rec["chunk_index"],
                "title": rec.get("title", ""),
                "authors": rec.get("authors", ""),
                "token_count": rec.get("token_count", 0),
            })

    print(f"Encoding {len(texts)} chunks with batch_size={BATCH} on CPU…")
    embs = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i:i+BATCH]
        vecs = model.encode(batch, batch_size=BATCH, device="cpu", show_progress_bar=False, normalize_embeddings=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        # L2-normalize for cosine via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = (vecs / norms).astype(np.float32)
        embs.append(vecs)

    X = np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
    # Persist
    np.save(EMB_DIR / "embeddings.npy", X)
    with jsonlines.open(EMB_DIR / "metadata.jsonl", "w") as w:
        for m in metas:
            w.write(m)
    print(f"Wrote {X.shape} -> {EMB_DIR/'embeddings.npy'}")
    print(f"Metadata -> {EMB_DIR/'metadata.jsonl'}")

if __name__ == "__main__":
    main()


