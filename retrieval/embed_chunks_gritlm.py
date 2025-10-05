#!/usr/bin/env python3
import os, json, jsonlines, numpy as np
from pathlib import Path
from tqdm import tqdm
from gritlm import GritLM

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("GRITLM_MODEL", "GritLM/GritLM-7B")

def gritlm_instruction(instr: str) -> str:
    return "<|user|>\n" + instr + "\n<|embed|>\n" if instr else "<|embed|>\n"

def main(batch_size: int = 16, max_chunks: int | None = None):
    assert CHUNKS.exists(), f"Missing {CHUNKS}; run the PDF pipeline first."
    print(f"Loading model {MODEL_NAME} (embedding mode)…")
    model = GritLM(MODEL_NAME, torch_dtype="auto", mode="embedding")

    texts, metas = [], []
    with jsonlines.open(CHUNKS, "r") as reader:
        for i, rec in enumerate(reader):
            if max_chunks and i >= max_chunks: break
            texts.append(rec["chunk_text"])
            metas.append({
                "paper_id": rec["paper_id"],
                "chunk_index": rec["chunk_index"],
                "title": rec.get("title", ""),
                "authors": rec.get("authors", ""),
                "token_count": rec.get("token_count", 0),
            })

    print(f"Encoding {len(texts)} chunks…")
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, instruction=gritlm_instruction(""))
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        embs.append(vecs)
    embs = np.vstack(embs)

    np.save(EMB_DIR / "embeddings.npy", embs)
    with jsonlines.open(EMB_DIR / "metadata.jsonl", "w") as w:
        for m in metas: w.write(m)

    print(f"Wrote {embs.shape} → {EMB_DIR/'embeddings.npy'}")
    print(f"Metadata → {EMB_DIR/'metadata.jsonl'}")

if __name__ == "__main__":
    main()

