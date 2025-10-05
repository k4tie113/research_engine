#!/usr/bin/env python3
import argparse, os, jsonlines, numpy as np, faiss
from pathlib import Path
from gritlm import GritLM

ROOT = Path(__file__).resolve().parents[1]
IDX_PATH = ROOT / "data" / "faiss" / "index_flatip.faiss"
IDS_PATH = ROOT / "data" / "faiss" / "ids.jsonl"
MODEL_NAME = os.environ.get("GRITLM_MODEL", "GritLM/GritLM-7B")

def gritlm_instruction(instr: str) -> str:
    return "<|user|>\n" + instr + "\n<|embed|>\n" if instr else "<|embed|>\n"

def load_ids():
    with jsonlines.open(IDS_PATH) as r:
        return [row for row in r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="natural-language query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--instruction", type=str,
                    default="Retrieve relevant passages from NLP research papers.")
    args = ap.parse_args()

    assert IDX_PATH.exists() and IDS_PATH.exists(), "Build the FAISS index first."
    index = faiss.read_index(str(IDX_PATH))
    ids = load_ids()

    model = GritLM(MODEL_NAME, torch_dtype="auto", mode="embedding")
    q_vec = model.encode([args.query], instruction=gritlm_instruction(args.instruction))
    q_vec = np.asarray(q_vec, dtype=np.float32)
    q_vec /= (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    D, I = index.search(q_vec, args.k)
    print(f"\nTop {args.k} for: {args.query}\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        meta = ids[idx]
        title = meta.get("title", "")[:150]
        print(f"[{rank}] score={score:.3f}  {meta['paper_id']}  #{meta['chunk_index']}  {title}")

if __name__ == "__main__":
    main()
