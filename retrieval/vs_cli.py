#!/usr/bin/env python3
import argparse, os, numpy as np
from pathlib import Path
from gritlm import GritLM
from .vector_store import FaissVectorStore

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings" / "embeddings.npy"
META = ROOT / "data" / "embeddings" / "metadata.jsonl"

def get_vs() -> FaissVectorStore:
    return FaissVectorStore.default_paths(ROOT)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="build FAISS index from embeddings + metadata")
    sub.add_parser("stats", help="print index size/dim")

    sp = sub.add_parser("search", help="search with a text query")
    sp.add_argument("query", type=str)
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--instruction", default="<|embed|>\n")

    args = ap.parse_args()
    vs = get_vs()

    if args.cmd == "build":
        vs.build_from_files(EMB, META)
        print(f"✅ Built vector store → {vs.size} vectors, dim={vs.dim}")

    elif args.cmd == "stats":
        vs.open()
        print(f"📦 index: {vs.size} vectors, dim={vs.dim}")
        print(f"    paths: {vs.paths.index_path} | {vs.paths.ids_path}")

    elif args.cmd == "search":
        vs.open()
        model = GritLM(os.environ.get("GRITLM_MODEL", "GritLM/GritLM-7B"),
                       torch_dtype="auto", mode="embedding")
        q = model.encode([args.query], instruction=args.instruction)[0]
        hits = vs.search(np.asarray(q, dtype=np.float32), k=args.k)
        print(f"\nTop {args.k} for: {args.query}\n")
        for r, h in enumerate(hits, 1):
            m = h["metadata"]
            title = (m.get("title") or "")[:150]
            print(f"[{r}] cos={h['score']:+.3f}  {m.get('paper_id')}  "
                  f"#{m.get('chunk_index')}  {title}")

if __name__ == "__main__":
    main()
