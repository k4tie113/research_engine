#!/usr/bin/env python3
"""
rag_service.py
--------------
A service module for RAG operations that can be imported by app.py.
Extracts reusable functions from generate_with_context_openai.py
"""

import faiss
import numpy as np
import jsonlines
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

# === CONFIG ===
MODEL_ID = "gpt-4o-mini"
DEFAULT_TOP_K = 15
DEFAULT_MAX_TOKENS = 600

# === PATHS ===
ROOT = Path(__file__).resolve().parents[2]  # Go up to research_engine root
EMB_DIR = ROOT / "database" / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "faiss_index_minilm.bin"
META_PATH = EMB_DIR / "metadata_minilm.jsonl"
CHUNKS_PATH = ROOT / "database" / "data" / "chunks_oai.jsonl"

# Global variables for loaded resources
_index = None
_meta = None
_embed_model = None
_chunks = None
_openai_client = None


def initialize_rag_system():
    """Initialize the RAG system by loading all required resources."""
    global _index, _meta, _embed_model, _chunks, _openai_client
    
    if _index is not None:
        # Already initialized
        return True
    
    try:
        print(f"Loading FAISS index from: {INDEX_PATH}")
        print(f"Index file exists: {INDEX_PATH.exists()}")
        _index = faiss.read_index(str(INDEX_PATH))
        
        print("Loading metadata...")
        _meta = [m for m in jsonlines.open(META_PATH)]
        
        print("Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Loading chunks...")
        _chunks = {}
        with jsonlines.open(CHUNKS_PATH, "r") as reader:
            for rec in reader:
                _chunks[(rec["paper_id"], rec["chunk_index"])] = rec["chunk_text"]
        
        print("Initializing OpenAI client...")
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"RAG system initialized - {_index.ntotal} vectors, {len(_chunks)} chunks loaded")
        return True
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False


def get_rag_response(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = DEFAULT_MAX_TOKENS, debug: bool = False, conversation_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
    """
    Get a RAG response for a query with optional conversation history.
    
    Args:
        query: The user query
        top_k: Number of top chunks to retrieve
        max_tokens: Maximum tokens for the response
        debug: Whether to print debug information
        conversation_history: List of previous messages in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        Tuple of (answer_text, sources_list)
    """
    if not initialize_rag_system():
        return "RAG system not properly initialized. Please check the logs.", []
    
    if not all([_index, _meta, _embed_model, _chunks, _openai_client]):
        return "RAG system not properly initialized. Please check the logs.", []
    
    try:
        # Retrieve relevant chunks
        q_emb = _embed_model.encode(query, normalize_embeddings=True)
        D, I = _index.search(np.array([q_emb], dtype="float32"), top_k)
        
        retrieved_texts = []
        sources = []
        
        for rank, idx in enumerate(I[0]):
            m = _meta[idx]
            pid, cidx = m["paper_id"], m["chunk_index"]
            full_chunk = _chunks.get((pid, cidx), "")
            
            retrieved_texts.append(full_chunk)
            sources.append({
                "paper_id": pid,
                "title": m.get('title', 'No title'),
                "chunk_index": cidx,
                "rank": rank + 1,
                "similarity_score": float(D[0][rank])
            })
        
        context = "\n\n".join(retrieved_texts)
        
        # Build messages array with conversation history
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Use the provided context from academic papers to answer questions clearly and concisely. If the context doesn't contain enough information, say so. Always cite the relevant papers when possible. Use conversation history to provide context-aware responses."}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            # Add only recent history (last 4 messages to avoid token limits)
            for msg in conversation_history[-4:]:
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user", 
            "content": f"Context from academic papers:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above."
        })
        
        # Print the full query being sent to GPT
        if debug:
            print("\n" + "=" * 80)
            print("MESSAGES BEING SENT TO GPT:")
            print("=" * 80)
            for i, msg in enumerate(messages, 1):
                print(f"\n--- Message {i} ({msg['role']}) ---")
                print(msg['content'])
                print(f"--- End of Message {i} ---\n")
            print("=" * 80)
        
        # Generate response with GPT-4o mini
        response = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        return answer, sources
        
    except Exception as e:
        return f"Error generating response: {str(e)}", []


def format_sources(sources: List[Dict], max_sources: int = 5) -> str:
    """Format sources into a readable string."""
    if not sources:
        return ""
    
    source_text = "\n\n**Sources:**\n"
    for source in sources[:max_sources]:
        source_text += f"{source['rank']}. [{source['paper_id']}] {source['title']}\n"
    
    return source_text


def get_system_status() -> Dict:
    """Get the status of the RAG system."""
    return {
        "initialized": _index is not None,
        "index_size": _index.ntotal if _index else 0,
        "metadata_size": len(_meta) if _meta else 0,
        "chunks_loaded": len(_chunks) if _chunks else 0,
        "embedding_model": _embed_model.__class__.__name__ if _embed_model else None,
        "openai_initialized": _openai_client is not None
    }
