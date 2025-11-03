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
import json
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


def _rewrite_query_with_history(query: str, conversation_history: Optional[List[Dict]], debug: bool = False) -> str:
    """Rewrite the current user question into a standalone, retrieval-optimized query using recent conversation history.

    Falls back to the original query on any error.
    """
    if not conversation_history:
        return query

    # Use only the most recent few turns to avoid excessive token use
    recent_history = conversation_history[-6:]

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into standalone search queries. "
                    "Preserve all specific entities, acronyms, methods, and constraints. "
                    "Resolve pronouns (e.g., 'it', 'they', 'this method') to their explicit referents from the conversation. "
                    "Output only the rewritten query without commentary."
                ),
            }
        ]
        # Provide compressed context of recent dialogue
        for msg in recent_history:
            # Keep only role and content; ignore other fields if present
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        # Add the current question
        messages.append({
            "role": "user",
            "content": (
                "Rewrite the following question into a standalone query for document retrieval.\n\n"
                f"Question: {query}"
            ),
        })

        if debug:
            print("\n" + "-" * 80)
            print("[Retrieval] Query rewrite PROMPT (messages to LLM):")
            print("-" * 80)
            for i, m in enumerate(messages, 1):
                role = m.get("role", "user")
                content = m.get("content", "").strip()
                print(f"[{i}] {role}:\n{content}\n")
            print("-" * 80)

        response = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=120,
            temperature=0.0,
        )

        rewritten = response.choices[0].message.content.strip()
        if debug:
            print("[Retrieval] Rewritten query:", rewritten)

        # Basic sanity check; fall back if the model returned something empty
        if not rewritten:
            return query
        return rewritten
    except Exception as e:
        if debug:
            print(f"[Retrieval] Query rewrite failed, falling back. Error: {e}")
        return query


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
        # Build a context-aware retrieval query
        retrieval_query = _rewrite_query_with_history(query, conversation_history, debug=debug)
        # Optionally enrich with the original question to broaden recall slightly
        combined_query = retrieval_query if retrieval_query == query else f"{retrieval_query} \n\nOriginal question: {query}"

        # Retrieve relevant chunks using the context-aware query
        q_emb = _embed_model.encode(combined_query, normalize_embeddings=True)
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
            {"role": "system", "content": "You are a helpful research assistant. Use the provided context from academic papers to answer questions clearly and concisely. If the context doesn't contain enough information, say so. Always cite the relevant papers when possible. Use conversation history to provide context-aware responses. When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) and display math with \\[ ... \\]. Do not wrap LaTeX in code blocks and do not escape backslashes. Do NOT use headings, tables, or section numbering. Keep formatting simple: short paragraphs, bullet lists ( - item ), and optional bold/italics only."}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            # Add only recent history (last 4 messages to avoid token limits)
            for msg in conversation_history[-4:]:
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user", 
            "content": f"Context from academic papers:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above. If you include math, use LaTeX with \\(inline\\) or \\[display\\] delimiters, not code fences."
        })
        if debug:
            print("\n" + "-" * 80)
            print("[Generation] Final user message to LLM (answer prompt):")
            print("-" * 80)
            print(messages[-1]["content"])
            print("-" * 80)
        
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


def stream_rag_response(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = DEFAULT_MAX_TOKENS, debug: bool = False, conversation_history: List[Dict] = None):
    """Generator that streams the LLM answer tokens and finally emits sources.

    Yields SSE-like lines: "data: {json}\n\n" where json has either a
    {"event":"delta","text":"..."} shape or a final
    {"event":"done","sources":[...]}.
    """
    if not initialize_rag_system() or not all([_index, _meta, _embed_model, _chunks, _openai_client]):
        yield f"data: {json.dumps({'event': 'error', 'message': 'RAG not initialized'})}\n\n"
        return

    try:
        # Build a context-aware retrieval query
        retrieval_query = _rewrite_query_with_history(query, conversation_history, debug=debug)
        combined_query = retrieval_query if retrieval_query == query else f"{retrieval_query} \n\nOriginal question: {query}"

        # Retrieve relevant chunks
        q_emb = _embed_model.encode(combined_query, normalize_embeddings=True)
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

        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Use the provided context from academic papers to answer questions clearly and concisely. If the context doesn't contain enough information, say so. Always cite the relevant papers when possible. Use conversation history to provide context-aware responses. When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) and display math with \\[ ... \\]. Do not wrap LaTeX in code blocks and do not escape backslashes."}
        ]
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append(msg)
        messages.append({
            "role": "user",
            "content": f"Context from academic papers:\n\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context above. Keep formatting minimal (no headings/tables): use plain paragraphs, bullet points where helpful, and bold/italics only. If you include math, use LaTeX with \\(inline\\) or \\[display\\] delimiters, not code fences."
        })

        if debug:
            print("\n" + "=" * 80)
            print("STREAMING: Sending messages to GPT")
            print("=" * 80)
            for i, msg in enumerate(messages, 1):
                print(f"\n--- Message {i} ({msg['role']}) ---")
                print(msg['content'])
                print(f"--- End of Message {i} ---\n")

        stream = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )

        # Stream deltas
        for event in stream:
            try:
                delta = event.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield f"data: {json.dumps({'event': 'delta', 'text': delta})}\n\n"

        # Emit final sources
        yield f"data: {json.dumps({'event': 'done', 'sources': sources})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
