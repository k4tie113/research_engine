#!/usr/bin/env python3
"""
generate_with_context_openai.py
------------------------------
Retrieves top-k chunks from your FAISS index and sends the context
to OpenAI's GPT-4o mini for generation.
"""

import faiss, numpy as np, jsonlines, textwrap
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === CONFIG ===
MODEL_ID = "gpt-4o-mini"
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

# === CONNECT TO OPENAI API ===
print(f"Connecting to OpenAI API ({MODEL_ID})...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

print("Generating answer from GPT-4o mini...\n")

try:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant. Use the provided context to answer questions clearly and concisely. If the context doesn't contain enough information, say so."},
            {"role": "user", "content": f"Context from academic papers:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above."}
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7
    )

    print("Answer:\n")
    print(response.choices[0].message.content)
    print("\n" + "="*90)

except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    print("Make sure your OPENAI_API_KEY is set in the .env file")
