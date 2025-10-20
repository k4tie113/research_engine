from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import faiss
import numpy as np
import jsonlines
import textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the retrieval directory to the path
sys.path.append(str(Path(__file__).resolve().parents[2] / "retrieval"))

app = Flask(__name__)
CORS(app)

# === RAG SYSTEM INITIALIZATION ===
print("Initializing RAG system...")

# Paths
ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "database" / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "faiss_index_bge.bin"
META_PATH = EMB_DIR / "metadata_bge.jsonl"
CHUNKS_PATH = ROOT / "database" / "data" / "chunks_oai.jsonl"

# Load FAISS index and metadata
try:
    index = faiss.read_index(str(INDEX_PATH))
    meta = [m for m in jsonlines.open(META_PATH)]
    print(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    index = None
    meta = []

# Load embedding model
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded embedding model")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embed_model = None

# Load chunks
try:
    chunks = {}
    with jsonlines.open(CHUNKS_PATH, "r") as reader:
        for rec in reader:
            chunks[(rec["paper_id"], rec["chunk_index"])] = rec["chunk_text"]
    print(f"Loaded {len(chunks)} chunks")
except Exception as e:
    print(f"Error loading chunks: {e}")
    chunks = {}

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Initialized OpenAI client")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

print("RAG system initialization complete!")

def get_rag_response(query, top_k=10):
    """Get RAG response using local embeddings and GPT-4o mini"""
    if not all([index, meta, embed_model, chunks, openai_client]):
        return "RAG system not properly initialized. Please check the logs."
    
    try:
        # Retrieve relevant chunks
        q_emb = embed_model.encode(query, normalize_embeddings=True)
        D, I = index.search(np.array([q_emb], dtype="float32"), top_k)
        
        retrieved_texts = []
        sources = []
        
        for rank, idx in enumerate(I[0]):
            m = meta[idx]
            pid, cidx = m["paper_id"], m["chunk_index"]
            snippet = textwrap.shorten(chunks.get((pid, cidx), ""), width=300, placeholder="...")
            retrieved_texts.append(snippet)
            sources.append({
                "paper_id": pid,
                "title": m.get('title', 'No title'),
                "chunk_index": cidx,
                "rank": rank + 1
            })
        
        context = "\n\n".join(retrieved_texts)
        
        # Generate response with GPT-4o mini
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Use the provided context from academic papers to answer questions clearly and concisely. If the context doesn't contain enough information, say so. Always cite the relevant papers when possible."},
                {"role": "user", "content": f"Context from academic papers:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above."}
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # Format sources
        source_text = "\n\n**Sources:**\n"
        for source in sources[:5]:  # Show top 5 sources
            source_text += f"{source['rank']}. [{source['paper_id']}] {source['title']}\n"
        
        return answer + source_text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please enter a message."})

    # Use RAG system instead of Semantic Scholar API
    reply = get_rag_response(message)
    
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=5000)