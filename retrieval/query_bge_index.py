#!/usr/bin/env python3
import faiss
import numpy as np
import jsonlines
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests
import xml.etree.ElementTree as ET

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "database" / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "faiss_index_bge.bin"
META_PATH = EMB_DIR / "metadata_bge.jsonl"
CHUNKS_PATH = ROOT / "database" / "data" / "chunks_oai.jsonl"

# === LOAD MODEL + INDEX ===
print(f"🔍 Loading FAISS index from {INDEX_PATH}...")
index = faiss.read_index(str(INDEX_PATH))
meta = [m for m in jsonlines.open(META_PATH)]
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# === LOAD CHUNKS ===
chunks = {}
with jsonlines.open(CHUNKS_PATH, "r") as reader:
    for rec in reader:
        chunks[(rec["paper_id"], rec["chunk_index"])] = rec["chunk_text"]

# === HELPER: Fetch publication date from arXiv ===
def get_arxiv_date(paper_id: str) -> str:
    """
    Fetch publication date using the arXiv API given a paper_id (e.g. 'cs_0407005').
    Returns 'Unknown' if not found or invalid.
    """
    # Clean paper_id format for arXiv (replace underscore with slash)
    arxiv_id = paper_id.replace("_", "/")
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return "Unknown"
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return "Unknown"
        published = entry.find("atom:published", ns)
        if published is None:
            return "Unknown"
        return published.text.split("T")[0]  # e.g. '2005-11-24'
    except Exception as e:
        return f"Unknown ({e.__class__.__name__})"

# === QUERY ===
query = input("Enter your query: ")
q_emb = model.encode(query, normalize_embeddings=True)
k = 3
D, I = index.search(np.array([q_emb], dtype="float32"), k)

# === DISPLAY RESULTS ===
print("\nTop Retrieved Chunks:\n")
for rank, idx in enumerate(I[0]):
    m = meta[idx]
    pid, cidx = m["paper_id"], m["chunk_index"]
    full_chunk = chunks.get((pid, cidx), "[Chunk text missing]")
    date_published = get_arxiv_date(pid)

    print(f"{rank+1}. [{pid}] {m.get('title', '(no title)')}")
    print(f"   → Chunk {cidx} ({m['token_count']} tokens)")
    print(f"   Published: {date_published}")
    print("──────────────────────────────────────────────────────────────")
    print(full_chunk.strip())
    print("──────────────────────────────────────────────────────────────\n")
