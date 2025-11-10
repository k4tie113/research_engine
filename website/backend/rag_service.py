#!/usr/bin/env python3
"""
rag_service.py
--------------
A service module for RAG operations that can be imported by app.py.
Uses Elasticsearch for retrieval instead of FAISS.
"""

import os
import json
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

# === CONFIG ===
MODEL_ID = "gpt-4o-mini"
DEFAULT_TOP_K = 15
DEFAULT_MAX_TOKENS = 600

# === ELASTICSEARCH CONFIG ===
ES_URL = os.getenv("ES_URL", "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_INDEX = "paper_chunks"  # Index name in Elasticsearch cluster

# Global variables for loaded resources
_es_client = None
_openai_client = None


def initialize_rag_system():
    """Initialize the RAG system by connecting to Elasticsearch and OpenAI."""
    global _es_client, _openai_client
    
    if _es_client is not None:
        # Already initialized
        return True
    
    try:
        if not ES_API_KEY:
            print("ERROR: ES_API_KEY not found in environment variables. Please set it in .env file.")
            return False
        
        print(f"Connecting to Elasticsearch at: {ES_URL}")
        # Initialize Elasticsearch client with API key authentication and increased timeouts
        _es_client = Elasticsearch(
            [ES_URL],
            api_key=ES_API_KEY,
            request_timeout=60,  # Increase timeout to 60 seconds
            max_retries=3,      # Retry up to 3 times
            retry_on_timeout=True
        )
        
        # Test connection
        if not _es_client.ping():
            print("Failed to connect to Elasticsearch")
            return False
        
        # Check if index exists
        if not _es_client.indices.exists(index=ES_INDEX):
            print(f"Warning: Index '{ES_INDEX}' does not exist. Please create it first.")
            # Don't fail here, let the search function handle it
        
        # Get index stats if it exists
        try:
            stats = _es_client.count(index=ES_INDEX)
            print(f"Elasticsearch index '{ES_INDEX}' contains {stats['count']:,} documents")
        except Exception:
            pass
        
        print("Initializing OpenAI client...")
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print("RAG system initialized with Elasticsearch")
        return True
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
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
    
    if not all([_es_client, _openai_client]):
        return "RAG system not properly initialized. Please check the logs.", []
    
    try:
        # Build a context-aware retrieval query
        retrieval_query = _rewrite_query_with_history(query, conversation_history, debug=debug)
        
        # Retrieve relevant chunks using Elasticsearch
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text^3", "title^2", "authors"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text", "title", "authors"],
                                "type": "phrase",
                                "boost": 2.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count"]
        }
        
        if debug:
            print(f"\n[Retrieval] Searching Elasticsearch with query: {retrieval_query}")
            print(f"[Retrieval] Index: {ES_INDEX}, Top K: {top_k}")
        
        # Elasticsearch 8.x uses different API - pass query directly instead of body
        try:
            # Try new API first (Elasticsearch 8.x) - pass parameters directly
            response = _es_client.search(
                index=ES_INDEX,
                query=search_body["query"],
                size=search_body["size"],
                _source=search_body["_source"],
                timeout="60s"  # Add explicit timeout to search request
            )
        except (TypeError, KeyError):
            # Fall back to old API (Elasticsearch 7.x)
            try:
                search_body_with_timeout = search_body.copy()
                search_body_with_timeout["timeout"] = "60s"
                response = _es_client.search(index=ES_INDEX, body=search_body_with_timeout)
            except Exception as e:
                if debug:
                    print(f"[Retrieval] Error with both API methods: {e}")
                raise
        
        if debug:
            print(f"[Retrieval] Elasticsearch returned {len(response.get('hits', {}).get('hits', []))} results")

        retrieved_texts = []
        sources = []
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            if debug:
                print("[Retrieval] WARNING: No results found in Elasticsearch!")
            # Still continue with empty context - let GPT handle it
        
        for rank, hit in enumerate(hits, 1):
            source_data = hit.get("_source", {})
            chunk_text = source_data.get("chunk_text", "")
            retrieved_texts.append(chunk_text)
            
            paper_id = source_data.get("paper_id", "unknown")
            title = source_data.get("title", "").strip()
            authors = source_data.get("authors", "").strip()
            
            # Construct arXiv URL if paper_id looks like an arXiv ID
            arxiv_url = None
            if paper_id and not paper_id.startswith("http"):
                # Check if it's an arXiv ID (format: YYMM.NNNN or YYMM.NNNNvN)
                if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                    arxiv_url = f"https://arxiv.org/abs/{paper_id}"
            
            # Use what's in Elasticsearch - no fallback fetching
            sources.append({
                "paper_id": paper_id,
                "title": title if title else f"Paper {paper_id}",
                "authors": authors if authors else None,
                "chunk_index": source_data.get("chunk_index", 0),
                "rank": rank,
                "similarity_score": float(hit.get("_score", 0.0)),
                "url": arxiv_url
            })
        
        if debug:
            print(f"[Retrieval] Retrieved {len(retrieved_texts)} chunks, total context length: {sum(len(t) for t in retrieved_texts)} chars")

        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."
        
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
        paper_id = source.get('paper_id', 'unknown')
        title = source.get('title', 'No title')
        authors = source.get('authors')
        
        # Build source line
        source_line = f"{source.get('rank', '?')}. [{paper_id}] {title}"
        
        # Add authors if available
        if authors:
            source_line += f" - {authors}"
        
        # Add URL if available
        url = source.get('url')
        if url:
            source_line += f" ({url})"
        
        source_text += source_line + "\n"
    
    return source_text


def get_system_status() -> Dict:
    """Get the status of the RAG system."""
    index_count = 0
    if _es_client is not None:
        try:
            stats = _es_client.count(index=ES_INDEX)
            index_count = stats['count']
        except Exception:
            pass
    
    return {
        "initialized": _es_client is not None,
        "elasticsearch_connected": _es_client.ping() if _es_client else False,
        "index_name": ES_INDEX,
        "index_size": index_count,
        "openai_initialized": _openai_client is not None
    }


def stream_rag_response(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = DEFAULT_MAX_TOKENS, debug: bool = False, conversation_history: List[Dict] = None):
    """Generator that streams the LLM answer tokens and finally emits sources.

    Yields SSE-like lines: "data: {json}\n\n" where json has either a
    {"event":"delta","text":"..."} shape or a final
    {"event":"done","sources":[...]}.
    """
    if not initialize_rag_system() or not all([_es_client, _openai_client]):
        yield f"data: {json.dumps({'event': 'error', 'message': 'RAG not initialized'})}\n\n"
        return

    try:
        # Build a context-aware retrieval query
        retrieval_query = _rewrite_query_with_history(query, conversation_history, debug=debug)

        # Retrieve relevant chunks using Elasticsearch
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text^3", "title^2", "authors"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text", "title", "authors"],
                                "type": "phrase",
                                "boost": 2.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count"]
        }
        
        if debug:
            print(f"\n[Retrieval] Searching Elasticsearch with query: {retrieval_query}")
            print(f"[Retrieval] Index: {ES_INDEX}, Top K: {top_k}")
        
        # Elasticsearch 8.x uses different API - pass query directly instead of body
        try:
            # Try new API first (Elasticsearch 8.x) - pass parameters directly
            response = _es_client.search(
                index=ES_INDEX,
                query=search_body["query"],
                size=search_body["size"],
                _source=search_body["_source"],
                timeout="60s"  # Add explicit timeout to search request
            )
        except (TypeError, KeyError):
            # Fall back to old API (Elasticsearch 7.x)
            try:
                search_body_with_timeout = search_body.copy()
                search_body_with_timeout["timeout"] = "60s"
                response = _es_client.search(index=ES_INDEX, body=search_body_with_timeout)
            except Exception as e:
                if debug:
                    print(f"[Retrieval] Error with both API methods: {e}")
                raise
        
        if debug:
            print(f"[Retrieval] Elasticsearch returned {len(response.get('hits', {}).get('hits', []))} results")

        retrieved_texts = []
        sources = []
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            if debug:
                print("[Retrieval] WARNING: No results found in Elasticsearch!")
            # Still continue with empty context - let GPT handle it
        
        for rank, hit in enumerate(hits, 1):
            source_data = hit.get("_source", {})
            chunk_text = source_data.get("chunk_text", "")
            retrieved_texts.append(chunk_text)
            
            paper_id = source_data.get("paper_id", "unknown")
            title = source_data.get("title", "").strip()
            authors = source_data.get("authors", "").strip()
            
            # Construct arXiv URL if paper_id looks like an arXiv ID
            arxiv_url = None
            if paper_id and not paper_id.startswith("http"):
                # Check if it's an arXiv ID (format: YYMM.NNNN or YYMM.NNNNvN)
                if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                    arxiv_url = f"https://arxiv.org/abs/{paper_id}"
            
            # Use what's in Elasticsearch - no fallback fetching
            sources.append({
                "paper_id": paper_id,
                "title": title if title else f"Paper {paper_id}",
                "authors": authors if authors else None,
                "chunk_index": source_data.get("chunk_index", 0),
                "rank": rank,
                "similarity_score": float(hit.get("_score", 0.0)),
                "url": arxiv_url
            })
        
        if debug:
            print(f"[Retrieval] Retrieved {len(retrieved_texts)} chunks, total context length: {sum(len(t) for t in retrieved_texts)} chars")

        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

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
        if debug:
            print("[Streaming] Starting to stream response from OpenAI...")
        
        has_yielded = False
        for event in stream:
            try:
                delta = event.choices[0].delta.content
                if delta:
                    has_yielded = True
                    yield f"data: {json.dumps({'event': 'delta', 'text': delta})}\n\n"
            except Exception as e:
                if debug:
                    print(f"[Streaming] Error processing delta: {e}")
                delta = None
        
        if debug:
            print(f"[Streaming] Finished streaming. Has yielded: {has_yielded}")

        # Emit final sources
        yield f"data: {json.dumps({'event': 'done', 'sources': sources})}\n\n"
    except Exception as e:
        import traceback
        error_msg = str(e)
        if debug:
            print(f"[ERROR] Exception in stream_rag_response: {error_msg}")
            traceback.print_exc()
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg})}\n\n"
