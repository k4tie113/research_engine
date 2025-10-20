#!/usr/bin/env python3
"""
generate_with_context_hfapi.py
------------------------------
Retrieves top-k chunks from your FAISS index and sends the context
to the Hugging Face Inference API for generation using Mistral-7B-Instruct.
"""

import faiss, numpy as np, jsonlines, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# === CONFIG ===
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K = 15
MAX_NEW_TOKENS = 600

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "database" / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "faiss_index_bge.bin"
META_PATH = EMB_DIR / "metadata_bge.jsonl"
CHUNKS_PATH = ROOT / "database" / "data" / "chunks_oai.jsonl"

# === LOAD RETRIEVAL ASSETS ===
print(f"Loading FAISS index from {INDEX_PATH}...")
index = faiss.read_index(str(INDEX_PATH))
meta = [m for m in jsonlines.open(META_PATH)]

print("Loading lightweight embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load all chunks (so we can display and use them)
chunks = {}
with jsonlines.open(CHUNKS_PATH, "r") as reader:
    for rec in reader:
        chunks[(rec["paper_id"], rec["chunk_index"])] = rec["chunk_text"]

# === CONNECT TO HUGGING FACE INFERENCE API ===
print(f"Connecting to Hugging Face Inference API ({MODEL_ID})...")
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token="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"
)

# === MAIN LOOP ===
import sys
if len(sys.argv) > 1:
    # Run with command line query
    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query}")
else:
    # Interactive mode
    while True:
        print("\nEnter your query (or 'exit' to quit): ", end="")
        query = input().strip()
        if query.lower() in {"exit", "quit"}:
            break

# ---- RETRIEVE ----
q_emb = embed_model.encode(query, normalize_embeddings=True)
D, I = index.search(np.array([q_emb], dtype="float32"), TOP_K)

retrieved_texts = []
print("\nTop Retrieved Contexts:\n")
for rank, idx in enumerate(I[0]):
    m = meta[idx]
    pid, cidx = m["paper_id"], m["chunk_index"]
    snippet = textwrap.shorten(chunks.get((pid, cidx), ""), width=250, placeholder="...")
    print(f"{rank+1}. [{pid}] {m.get('title','(no title)')}")
    print(f"   -> Chunk {cidx} ({m['token_count']} tokens)")
    try:
        print(f"   {snippet}\n")
    except UnicodeEncodeError:
        # Handle unicode characters that can't be displayed
        safe_snippet = snippet.encode('ascii', errors='ignore').decode('ascii')
        print(f"   {safe_snippet}\n")
    retrieved_texts.append(snippet)

context = "\n\n".join(retrieved_texts)
prompt = f"""You are a helpful research assistant.
Use the following context to answer the user's question clearly and concisely.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

print("Generating answer from Mistral (Hugging Face API)...\n")

response = client.chat_completion(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Use the following context to answer clearly and concisely:\n\n{context}\n\nQuestion: {query}"}
    ],
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9
)

print("Answer:\n")
print(response.choices[0].message["content"])
print("\n" + "="*90)

