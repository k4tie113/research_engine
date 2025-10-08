#!/usr/bin/env python3
import argparse, os, jsonlines, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
IDX_PATH = ROOT / "data" / "faiss" / "index_flatip.faiss"
IDS_PATH = ROOT / "data" / "faiss" / "ids.jsonl"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
ST_MODEL = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")

def noop_instruction(_: str) -> str:
    return ""

def load_ids():
    with jsonlines.open(IDS_PATH) as r:
        return [row for row in r]

def load_chunks_map():
    """Return dict keyed by (paper_id, chunk_index) -> chunk_text."""
    mp = {}
    if not CHUNKS_PATH.exists():
        return mp
    with jsonlines.open(CHUNKS_PATH, "r") as reader:
        for rec in reader:
            try:
                mp[(rec.get("paper_id"), int(rec.get("chunk_index", -1)))] = rec.get("chunk_text", "")
            except Exception:
                continue
    return mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="natural-language query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--instruction", type=str,
                    default="Retrieve relevant passages from NLP research papers.")
    ap.add_argument("--show_chars", type=int, default=180,
                    help="print up to N characters of the matched chunk text")
    args = ap.parse_args()

    assert IDX_PATH.exists() and IDS_PATH.exists(), "Build the FAISS index first."
    index = faiss.read_index(str(IDX_PATH))
    ids = load_ids()
    chunk_map = load_chunks_map()
    if not chunk_map:
        print(f"(info) No chunk text map available at {CHUNKS_PATH}; showing titles only.\n")

    model = SentenceTransformer(ST_MODEL, device="cpu")
    # Ignore instruction for ST models; encode plain text
    q_vec = model.encode([args.query], batch_size=1, device="cpu", show_progress_bar=False, normalize_embeddings=False)
    q_vec = np.asarray(q_vec, dtype=np.float32)
    q_vec /= (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    D, I = index.search(q_vec, args.k)
    print(f"\nTop {args.k} for: {args.query}\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        meta = ids[idx]
        title = meta.get("title", "")[:150]
        key = (meta.get("paper_id"), int(meta.get("chunk_index", -1)))
        snippet = chunk_map.get(key, "")
        if snippet:
            snippet = snippet.replace("\n", " ")[:args.show_chars]
        print(f"[{rank}] score={score:.3f}  {meta['paper_id']}  #{meta['chunk_index']}  {title}")
        if snippet:
            print(f"      └ {snippet}")
        else:
            print(f"      └ [no chunk text found in chunks.jsonl]")

if __name__ == "__main__":
    main()
