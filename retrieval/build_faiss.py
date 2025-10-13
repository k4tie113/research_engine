#!/usr/bin/env python3
import faiss, numpy as np, json, jsonlines
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "data" / "faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    emb_path = EMB_DIR / "embeddings.npy"
    meta_path = EMB_DIR / "metadata.jsonl"
    assert emb_path.exists() and meta_path.exists(), "Run embed_chunks_gritlm.py first."

    X = np.load(emb_path).astype(np.float32)  # already L2-normalized
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)            # cosine via inner product on normalized vectors
    index.add(X)
    faiss.write_index(index, str(OUT_DIR / "index_flatip.faiss"))

    with jsonlines.open(meta_path) as r:
        metas = list(r)
    with open(OUT_DIR / "ids.jsonl", "w", encoding="utf-8") as f:
        for i, m in enumerate(metas):
            f.write(json.dumps({"row_id": i, **m}, ensure_ascii=False) + "\n")

    print(f"Indexed {X.shape[0]} vectors (dim={dim}) -> {OUT_DIR/'index_flatip.faiss'}")
    print(f"IDs map -> {OUT_DIR/'ids.jsonl'}")

if __name__ == "__main__":
    main()
