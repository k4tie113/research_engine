#!/usr/bin/env python3
"""
rag_service.py
--------------
A service module for RAG operations that can be imported by app.py.
Uses Elasticsearch for retrieval instead of FAISS.
"""

import os
import json
import re
import requests
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
from query_analyzer import analyze_query
from collections import defaultdict

# Load environment variables
load_dotenv()

# === CONFIG ===
MODEL_ID = "gpt-4o-mini"
SUMMARIZATION_MODEL_ID = "gpt-4o-mini"  # Use same model as rest of system for consistency
DEFAULT_TOP_K = 15
DEFAULT_MAX_TOKENS = 600

# === ELASTICSEARCH CONFIG ===
ES_URL = os.getenv("ES_URL", "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_INDEX = "chunks"  # Index name in Elasticsearch cluster
ES_METADATA_INDEX = None  # Will be discovered dynamically

# Global variables for loaded resources
_es_client = None
_openai_client = None
_metadata_index_cache = None  # Cache the discovered metadata index name


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
        
        # Try to discover metadata index
        _discover_metadata_index()
        
        return True
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False


def _discover_metadata_index() -> Optional[str]:
    """Discover the metadata index by trying common index names."""
    global _metadata_index_cache, _es_client
    
    if _metadata_index_cache:
        return _metadata_index_cache
    
    if not _es_client:
        return None
    
    # Common metadata index names to try
    potential_names = [
        "paper_metadata",
        "papers_metadata", 
        "metadata",
        "papers",
        "paper_meta",
        "papers_meta"
    ]
    
    for index_name in potential_names:
        try:
            if _es_client.indices.exists(index=index_name):
                # Verify it has paper_id field by checking mapping or sample doc
                try:
                    result = _es_client.search(index=index_name, size=1)
                    if result.get("hits", {}).get("hits"):
                        sample = result["hits"]["hits"][0]["_source"]
                        # Check if it has paper_id (likely a metadata index)
                        if "paper_id" in sample:
                            _metadata_index_cache = index_name
                            print(f"Discovered metadata index: {index_name}")
                            return index_name
                except Exception:
                    continue
        except Exception:
            continue
    
    print("Warning: Could not discover metadata index. Paper titles may not be available.")
    return None


def _normalize_authors(authors: Any) -> str:
    """
    Normalize authors field to a string.
    Handles both string and list formats.
    """
    if not authors:
        return ""
    if isinstance(authors, list):
        return ", ".join(str(a).strip() for a in authors if a)
    return str(authors).strip()


def _get_paper_metadata(paper_ids: List[str], debug: bool = False) -> Dict[str, Dict]:
    """
    Fetch paper metadata (title, authors, etc.) from the metadata index.
    
    Args:
        paper_ids: List of paper IDs to fetch metadata for
        debug: Whether to print debug information
    
    Returns:
        Dictionary mapping paper_id -> metadata dict
    """
    global _es_client, _metadata_index_cache
    
    if not _es_client:
        return {}
    
    # Discover metadata index if not already discovered
    metadata_index = _discover_metadata_index()
    if not metadata_index:
        if debug:
            print("[Metadata] No metadata index available")
        return {}
    
    if not paper_ids:
        return {}
    
    # Remove duplicates
    unique_paper_ids = list(set(paper_ids))
    
    try:
        results = {}
        
        # First, try to get documents by paper_id as the document ID (using mget)
        try:
            mget_response = _es_client.mget(
                index=metadata_index,
                body={"ids": unique_paper_ids},
                _source=["paper_id", "title", "authors", "year", "categories"]
            )
            
            for doc in mget_response.get("docs", []):
                if doc.get("found"):
                    source = doc.get("_source", {})
                    # If paper_id is in source, use it; otherwise use the document ID
                    paper_id = source.get("paper_id") or doc.get("_id")
                    if paper_id:
                        results[paper_id] = {
                            "title": source.get("title", ""),
                            "authors": source.get("authors", ""),
                            "year": source.get("year"),
                            "categories": source.get("categories", [])
                        }
            
            # If we got all results from mget, return early
            if len(results) == len(unique_paper_ids):
                if debug:
                    print(f"[Metadata] Fetched metadata for {len(results)}/{len(unique_paper_ids)} papers via mget")
                return results
        except Exception as e:
            if debug:
                print(f"[Metadata] mget failed, trying search query: {e}")
        
        # Fall back to querying by paper_id field
        query = {
            "query": {
                "terms": {
                    "paper_id": unique_paper_ids
                }
            },
            "size": len(unique_paper_ids),
            "_source": ["paper_id", "title", "authors", "year", "categories"]
        }
        
        try:
            response = _es_client.search(
                index=metadata_index,
                query=query["query"],
                size=query["size"],
                _source=query["_source"],
                timeout="10s"
            )
        except (TypeError, KeyError):
            # Fall back to old API
            response = _es_client.search(index=metadata_index, body=query, timeout="10s")
        
        hits = response.get("hits", {}).get("hits", [])
        
        for hit in hits:
            source = hit.get("_source", {})
            # Try to get paper_id from source, or from document ID
            paper_id = source.get("paper_id") or hit.get("_id")
            if paper_id:
                results[paper_id] = {
                    "title": source.get("title", ""),
                    "authors": source.get("authors", ""),
                    "year": source.get("year"),
                    "categories": source.get("categories", [])
                }
        
        if debug:
            print(f"[Metadata] Fetched metadata for {len(results)}/{len(unique_paper_ids)} papers")
        
        return results
        
    except Exception as e:
        if debug:
            print(f"[Metadata] Error fetching metadata: {e}")
        return {}


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
            "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count", "year"]
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
        
        # Collect paper IDs for metadata lookup
        paper_ids_for_metadata = []
        for hit in hits:
            source_data = hit.get("_source", {})
            paper_id = source_data.get("paper_id", "unknown")
            if paper_id and paper_id != "unknown":
                paper_ids_for_metadata.append(paper_id)
        
        # Fetch metadata from metadata index
        metadata_dict = _get_paper_metadata(paper_ids_for_metadata, debug=debug)
        
        for rank, hit in enumerate(hits, 1):
            source_data = hit.get("_source", {})
            chunk_text = source_data.get("chunk_text", "")
            retrieved_texts.append(chunk_text)
            
            paper_id = source_data.get("paper_id", "unknown")
            
            # Try to get title from metadata index first, fall back to chunk data
            title = ""
            authors = ""
            if paper_id in metadata_dict:
                metadata = metadata_dict[paper_id]
                title = metadata.get("title", "")
                if title:
                    title = str(title).strip()
                authors = _normalize_authors(metadata.get("authors", ""))
            
            # Fall back to chunk data if metadata not available
            if not title:
                title = source_data.get("title", "")
                if title:
                    title = str(title).strip()
            if not authors:
                authors = _normalize_authors(source_data.get("authors", ""))
            
            # Construct arXiv URL if paper_id looks like an arXiv ID
            arxiv_url = None
            if paper_id and not paper_id.startswith("http"):
                # Check if it's an arXiv ID (format: YYMM.NNNN or YYMM.NNNNvN)
                if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                    arxiv_url = f"https://arxiv.org/abs/{paper_id}"
            
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
            {"role": "system", "content": "You are a paper finder assistant. Your primary role is to provide a very concise, high-level summary of what the retrieved academic papers are about. Keep the user's query in mind, but focus on giving a brief overview (one to two paragraphs maximum) of the key topics and contributions across the papers. Do NOT cite papers inline (no [Paper Title] or similar citations). The papers are already listed separately as sources. Use conversation history to provide context-aware responses. When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) and display math with \\[ ... \\]. Do not wrap LaTeX in code blocks and do not escape backslashes. Do NOT use headings, tables, or section numbering. Keep formatting simple: short paragraphs, bullet lists ( - item ), and optional bold/italics only."}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            # Add only recent history (last 4 messages to avoid token limits)
            for msg in conversation_history[-4:]:
                messages.append(msg)
        
        # Add current question with context
        messages.append({
            "role": "user", 
            "content": f"Context from academic papers:\n\n{context}\n\nUser's query: {query}\n\nProvide a very concise, high-level summary (one to two paragraphs maximum) of what these papers are about, keeping the user's query in mind. Focus on the key topics and contributions across the papers rather than detailed explanations. Do NOT cite papers inline - the papers are already listed separately as sources. If you include math, use LaTeX with \\(inline\\) or \\[display\\] delimiters, not code fences."
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
        
        # Return sources first, then summary
        sources_text = format_sources(sources, max_sources=5)
        return sources_text + "\n\n" + answer, sources
        
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


def _augment_query_with_analysis(base: str, analysis: dict) -> str:
    """
    Attach gentle steering hints from analysis to the query.
    This helps the RAG system focus on relevant aspects.
    """
    lines = []

    # Add relevance criteria as keywords
    criteria = analysis.get("relevance_criteria", [])
    if criteria:
        names = [c.get("name") for c in criteria if isinstance(c, dict) and c.get("name")]
        if names:
            lines.append("Key topics: " + ", ".join(names))

    # Add author preferences
    authors = analysis.get("authors", [])
    if authors:
        lines.append("Authors: " + ", ".join(authors))

    # Add venue preferences
    venues = analysis.get("venues", [])
    if venues:
        lines.append("Venues: " + ", ".join(venues))

    # Add time range
    tr = analysis.get("time_range", {})
    if tr and (tr.get("start") or tr.get("end")):
        start = tr.get("start", "")
        end = tr.get("end", "")
        if start and end:
            lines.append(f"Years: {start}-{end}")
        elif start:
            lines.append(f"After: {start}")
        elif end:
            lines.append(f"Before: {end}")

    # Combine base query with hints
    if not lines:
        return base
    
    return base + "\n\nFilters:\n" + "\n".join(lines)


def _generate_reworded_queries(query: str, conversation_history: Optional[List[Dict]], num_queries: int = 10, debug: bool = False) -> List[str]:
    """Generate multiple reworded versions of the query using GPT, with conversation history context."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research query rewriter. Generate diverse, context-aware KEYWORD queries optimized for document retrieval. "
                    "Each query should be a concise set of keywords and key phrases (not full sentences) that will help a search engine "
                    "find the most relevant academic paper chunks. Focus on: technical terms, method names, concepts, domain-specific vocabulary, "
                    "and synonyms. Use the conversation history to understand context and resolve references. "
                    "Return exactly 10 keyword queries, one per line, without numbering or bullets. "
                    "The FIRST query must be the single best representation of the user's core intent and should be the one most likely to retrieve the perfect answer. "
                    "Each query should be optimized for retrieval - use keywords and phrases, not complete sentences."
                ),
            }
        ]
        
        # Add conversation history for context
        recent_history = []
        if conversation_history:
            recent_history = conversation_history[-6:]  # Use last 6 messages
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        
        # Add the current query
        messages.append({
            "role": "user",
            "content": (
                f"Generate {num_queries} diverse KEYWORD queries based on this research question. "
                f"Each should be a concise set of keywords and key phrases (not full sentences) optimized for document retrieval. "
                f"Focus on extracting the most important technical terms, concepts, and synonyms that will help find relevant academic paper chunks.\n\n"
                f"The FIRST query must be the single best representation of the user's main intent and should capture the key idea precisely.\n\n"
                f"Original query: {query}\n\n"
                f"Return exactly {num_queries} keyword queries, one per line. Use keywords and phrases, not complete sentences."
            ),
        })
        
        if debug:
            print("\n" + "="*60)
            print("GENERATING REWORDED QUERIES")
            print("="*60)
            print(f"Original query: {query}")
            if conversation_history:
                print(f"Using {len(recent_history)} messages from conversation history")
        
        response = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=400,
            temperature=0.7,  # Higher temperature for diversity
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Parse the output - split by newlines and clean
        queries = [q.strip() for q in raw_output.split('\n') if q.strip()]
        
        # If we got fewer than requested, pad with variations
        if len(queries) < num_queries:
            queries.append(query)  # Always include original
            # Generate more if needed
            while len(queries) < num_queries:
                queries.append(f"{query} (alternative perspective)")
        
        # Limit to exactly num_queries
        queries = queries[:num_queries]
        
        if debug:
            print(f"Generated {len(queries)} reworded queries:")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
        
        return queries
        
    except Exception as e:
        if debug:
            print(f"Error generating reworded queries: {e}")
        # Fallback: return original query repeated
        return [query] * num_queries


def _search_elasticsearch(query: str, top_k: int = 20, debug: bool = False) -> List[Dict]:
    """Search Elasticsearch and return hits."""
    search_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_text^3", "title^2", "authors"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
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
        "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count", "year"]
    }
    
    try:
        # Try new API first (Elasticsearch 8.x)
        response = _es_client.search(
            index=ES_INDEX,
            query=search_body["query"],
            size=search_body["size"],
            _source=search_body["_source"],
            timeout="60s"
        )
    except (TypeError, KeyError):
        # Fall back to old API (Elasticsearch 7.x)
        search_body_with_timeout = search_body.copy()
        search_body_with_timeout["timeout"] = "60s"
        response = _es_client.search(index=ES_INDEX, body=search_body_with_timeout)
    
    hits = response.get("hits", {}).get("hits", [])
    return hits
def _summarize_paper(full_text: str, target_ratio: float = 0.15, debug: bool = False) -> str:
    """
    Summarize a full paper to exactly 500 words as a single paragraph using a lightweight LLM.
    
    Args:
        full_text: The full text of the paper
        target_ratio: Deprecated - kept for backward compatibility, not used (always targets 500 words)
        debug: Whether to print debug information
    
    Returns:
        Summarized text as a single paragraph (approximately 500 words)
    """
    global _openai_client
    
    if not _openai_client:
        if debug:
            print("[Summarization] OpenAI client not initialized")
        return full_text
    
    if not full_text or len(full_text) < 200:
        # Too short to summarize meaningfully
        return full_text
    
    # Fixed target: 500 words (single paragraph)
    target_word_count = 500
    
    # Estimate word count for better prompting
    word_count = len(full_text.split())
    
    if debug:
        print(f"[Summarization] Original: {word_count} words, Target: {target_word_count} words (single paragraph)")
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic paper summarizer. Create a concise summary as ONE SINGLE PARAGRAPH "
                    "(one continuous block of text, no line breaks or multiple paragraphs). The summary must be "
                    f"exactly {target_word_count} words long. Write as a flowing single paragraph that summarizes "
                    "the key information, main arguments, methodologies, findings, and conclusions. Keep technical "
                    "terms and important details. Output ONLY the summary text with no additional formatting, headings, "
                    "bullets, or structure markers. Do NOT use multiple paragraphs - write everything as one continuous paragraph."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Summarize the following academic paper in exactly {target_word_count} words as ONE SINGLE PARAGRAPH. "
                    "Write everything as one continuous block of text with no line breaks or paragraph separations. "
                    "No headings, bullets, or formatting - just one flowing paragraph summarizing the key content:\n\n"
                    f"{full_text}"
                )
            }
        ]
        
        # Use gpt-4o-mini for summarization (consistent with rest of system)
        # Calculate max_tokens: roughly 1.5 tokens per word, add buffer for safety
        estimated_tokens = int(target_word_count * 1.5)  # 1.5 tokens per word with buffer
        max_tokens = min(4000, max(800, estimated_tokens))  # At least 800 tokens for 500 words, max 4000
        
        response = _openai_client.chat.completions.create(
            model=SUMMARIZATION_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temperature for more consistent summarization
        )
        
        # Extract response content
        if not response.choices or len(response.choices) == 0:
            if debug:
                print(f"[Summarization] ERROR: No choices in response. Response: {response}")
            return full_text
        
        choice = response.choices[0]
        finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None
        
        if debug:
            print(f"[Summarization] Finish reason: {finish_reason}")
            print(f"[Summarization] Response object: {response}")
            print(f"[Summarization] Choice object: {choice}")
        
        if not choice.message:
            if debug:
                print(f"[Summarization] ERROR: No message in choice. Finish reason: {finish_reason}")
                print(f"[Summarization] Full response: {response}")
            return full_text
        
        content = choice.message.content
        if content is None:
            if debug:
                print(f"[Summarization] ERROR: Content is None. Finish reason: {finish_reason}")
                print(f"[Summarization] Full response: {response}")
                print(f"[Summarization] Message object: {choice.message}")
            return full_text
        
        summarized_text = content.strip() if isinstance(content, str) else str(content).strip()
        
        if not summarized_text:
            if debug:
                print(f"[Summarization] ERROR: Summarized text is empty after strip. Content was: {repr(content)}")
                print(f"[Summarization] Finish reason: {finish_reason}")
            return full_text
        
        if debug:
            summary_word_count = len(summarized_text.split())
            actual_ratio = summary_word_count / word_count if word_count > 0 else 0
            print(f"[Summarization] Summary: {summary_word_count} words (target: {target_word_count} words, {actual_ratio*100:.1f}% of original)")
        
        return summarized_text
        
    except Exception as e:
        if debug:
            print(f"[Summarization] Error summarizing paper: {e}")
            import traceback
            traceback.print_exc()
        # Return original text if summarization fails
        return full_text


def _remove_repeated_phrases(text: str, min_phrase_words: int = 10) -> str:
    """
    Post-processing step to remove repeated phrases that might have been missed.
    Looks for phrases of min_phrase_words or more that appear multiple times.
    
    Args:
        text: The text to clean
        min_phrase_words: Minimum number of words in a phrase to consider for removal
    
    Returns:
        Text with repeated phrases removed
    """
    if not text or len(text) < 100:  # Skip if text is too short
        return text
    
    words = text.split()
    if len(words) < min_phrase_words * 2:  # Need at least 2x the phrase length
        return text
    
    # Look for repeated phrases by checking sequences of words
    # Start from longer phrases and work down
    max_phrase_length = min(len(words) // 2, 100)  # Don't check phrases longer than half the text
    max_iterations = 5  # Limit iterations to avoid infinite loops
    iteration = 0
    
    result_text = text
    
    while iteration < max_iterations:
        iteration += 1
        found_repeat = False
        
        # Check phrases from longest to shortest
        for phrase_length in range(max_phrase_length, min_phrase_words - 1, -1):
            # Build a dictionary of phrases and their positions
            phrase_positions = {}
            
            # Check all possible phrases of this length
            for start_idx in range(len(words) - phrase_length + 1):
                phrase = words[start_idx:start_idx + phrase_length]
                phrase_text = ' '.join(phrase)
                
                # Skip if phrase is too short (character-wise)
                if len(phrase_text) < 50:
                    continue
                
                # Find all occurrences of this phrase in the result_text
                occurrences = []
                search_start = 0
                
                while True:
                    pos = result_text.find(phrase_text, search_start)
                    if pos == -1:
                        break
                    occurrences.append(pos)
                    search_start = pos + 1
                
                # If we found multiple occurrences, store it
                if len(occurrences) > 1:
                    phrase_positions[phrase_text] = sorted(occurrences)
                    found_repeat = True
                    break  # Process this phrase first
        
        # If we found repeats, remove them
        if found_repeat and phrase_positions:
            # Process the first phrase found (longest)
            phrase_text = list(phrase_positions.keys())[0]
            occurrences = phrase_positions[phrase_text]
            
            # Remove all but the first occurrence (work backwards to maintain positions)
            for i in range(len(occurrences) - 1, 0, -1):  # Start from last occurrence
                occ_pos = occurrences[i]
                # Remove this occurrence
                before = result_text[:occ_pos]
                after = result_text[occ_pos + len(phrase_text):]
                
                # Clean up any double spaces or awkward breaks
                before = before.rstrip()
                after = after.lstrip()
                
                # Add a space if needed for smooth flow
                if before and after:
                    if before[-1] not in '.!?;:\n' and after[0] not in '.!?;:,':
                        if not before.endswith(' '):
                            before += ' '
                
                result_text = before + after
            
            # Re-split words for next iteration
            words = result_text.split()
            if len(words) < min_phrase_words * 2:
                break
        else:
            # No more repeats found
            break
    
    return result_text


def _remove_overlap_between_chunks(chunks: List[str], min_overlap_length: int = 50) -> List[str]:
    """
    Remove overlapping text between consecutive chunks and ensure smooth flow.
    Handles cases where chunks overlap at both ends.
    
    Args:
        chunks: List of chunk texts
        min_overlap_length: Minimum character length of overlap to consider (to avoid false positives)
    
    Returns:
        List of chunks with overlaps removed and smooth transitions
    """
    if not chunks or len(chunks) <= 1:
        return chunks
    
    deduplicated = []
    
    for i, curr_chunk in enumerate(chunks):
        if i == 0:
            # First chunk is always included as-is
            deduplicated.append(curr_chunk)
            continue
        
        prev_chunk = deduplicated[-1]
        
        # Normalize whitespace for comparison, but preserve original for output
        prev_chunk_clean = prev_chunk.strip()
        curr_chunk_clean = curr_chunk.strip()
        
        if not prev_chunk_clean or not curr_chunk_clean:
            deduplicated.append(curr_chunk)
            continue
        
        # Find overlap by comparing end of prev_chunk with start of curr_chunk
        # Use word-boundary matching for accuracy
        prev_words = prev_chunk_clean.split()
        curr_words = curr_chunk_clean.split()
        
        if not prev_words or not curr_words:
            deduplicated.append(curr_chunk)
            continue
        
        # Find the longest matching word sequence
        overlap_found = False
        overlap_num_words = 0
        overlap_text = ""
        
        # Check from longest possible overlap down to minimum
        max_possible_words = min(len(prev_words), len(curr_words))
        # Increase the range - check all possible overlaps, not just those meeting min_overlap_length estimate
        for num_words in range(max_possible_words, 0, -1):
            prev_end_words = prev_words[-num_words:]
            curr_start_words = curr_words[:num_words]
            
            # Check if word sequences match exactly
            if prev_end_words == curr_start_words:
                overlap_text_words = ' '.join(curr_start_words)
                # Only consider if it meets minimum length requirement
                if len(overlap_text_words) >= min_overlap_length:
                    overlap_found = True
                    overlap_num_words = num_words
                    overlap_text = overlap_text_words
                    break
        
        # If word matching didn't work, try character-level matching with increased limits
        overlap_len_chars = 0
        if not overlap_found:
            prev_len = len(prev_chunk_clean)
            curr_len = len(curr_chunk_clean)
            # Increase limit significantly - overlaps can be very long
            max_possible_overlap = min(prev_len, curr_len, 2000)  # Increased from 500 to 2000
            
            # Check from end of prev_chunk, with smaller step size for better accuracy
            # Start from longer overlaps and work backwards
            for test_len in range(max_possible_overlap, min_overlap_length - 1, -10):  # Step by 10, backwards
                prev_suffix = prev_chunk_clean[-test_len:]
                curr_prefix = curr_chunk_clean[:test_len]
                
                if prev_suffix == curr_prefix:
                    overlap_found = True
                    overlap_len_chars = test_len
                    overlap_text = prev_suffix
                    break
            
            # If still not found with step size 10, try step size 1 for the last 500 chars
            if not overlap_found:
                fine_search_limit = min(500, max_possible_overlap)
                for test_len in range(fine_search_limit, min_overlap_length - 1, -1):
                    prev_suffix = prev_chunk_clean[-test_len:]
                    curr_prefix = curr_chunk_clean[:test_len]
                    
                    if prev_suffix == curr_prefix:
                        overlap_found = True
                        overlap_len_chars = test_len
                        overlap_text = prev_suffix
                        break
        
        # Remove overlap and ensure smooth transition
        if overlap_found:
            # Find where overlap starts in the original curr_chunk
            # Try multiple search strategies to handle whitespace differences
            overlap_pos = -1
            cut_position = None
            
            # First, try exact match
            overlap_pos = curr_chunk.find(overlap_text)
            
            # If not found, try searching in cleaned area
            if overlap_pos == -1:
                search_start = max(0, len(curr_chunk) - len(curr_chunk_clean) - 100)
                overlap_pos = curr_chunk.find(overlap_text, search_start)
            
            # If still not found, try normalized version (collapse whitespace)
            if overlap_pos == -1:
                overlap_text_normalized = re.sub(r'\s+', ' ', overlap_text).strip()
                curr_chunk_normalized = re.sub(r'\s+', ' ', curr_chunk).strip()
                normalized_pos = curr_chunk_normalized.find(overlap_text_normalized)
                if normalized_pos >= 0:
                    # Find corresponding position in original
                    # Count non-whitespace chars up to normalized_pos
                    char_count = 0
                    for i, char in enumerate(curr_chunk):
                        if char not in ' \n\t':
                            char_count += 1
                        if char_count > normalized_pos:
                            overlap_pos = i
                            break
            
            # If still not found but we have word-level match, try finding by first word
            if overlap_pos == -1 and overlap_num_words > 0:
                # Find where the first word of overlap appears in curr_chunk
                first_overlap_word = curr_words[0] if curr_words else ""
                if first_overlap_word:
                    # Try to find the word, accounting for word boundaries
                    word_pos = curr_chunk.find(first_overlap_word)
                    if word_pos >= 0:
                        # Verify this is at a word boundary
                        if word_pos == 0 or curr_chunk[word_pos - 1] in ' \n\t':
                            overlap_pos = word_pos
                            # Calculate cut position based on word count
                            # Find the end of the last overlapping word
                            words_seen = 1
                            pos = word_pos + len(first_overlap_word)
                            while pos < len(curr_chunk) and words_seen < overlap_num_words:
                                # Skip whitespace
                                while pos < len(curr_chunk) and curr_chunk[pos] in ' \n\t':
                                    pos += 1
                                # Find end of current word
                                while pos < len(curr_chunk) and curr_chunk[pos] not in ' \n\t':
                                    pos += 1
                                words_seen += 1
                            # Include trailing whitespace
                            while pos < len(curr_chunk) and curr_chunk[pos] in ' \n\t':
                                pos += 1
                            cut_position = pos
                        else:
                            # Not at word boundary, try next occurrence
                            next_word_pos = curr_chunk.find(first_overlap_word, word_pos + 1)
                            if next_word_pos >= 0 and (next_word_pos == 0 or curr_chunk[next_word_pos - 1] in ' \n\t'):
                                overlap_pos = next_word_pos
                                # Similar calculation as above
                                words_seen = 1
                                pos = next_word_pos + len(first_overlap_word)
                                while pos < len(curr_chunk) and words_seen < overlap_num_words:
                                    while pos < len(curr_chunk) and curr_chunk[pos] in ' \n\t':
                                        pos += 1
                                    while pos < len(curr_chunk) and curr_chunk[pos] not in ' \n\t':
                                        pos += 1
                                    words_seen += 1
                                while pos < len(curr_chunk) and curr_chunk[pos] in ' \n\t':
                                    pos += 1
                                cut_position = pos
            
            if overlap_pos >= 0:
                # Calculate cut position: after the overlap
                if cut_position is None:
                    # Fallback: use length-based calculation
                    cut_position = overlap_pos + len(overlap_text)
                
                # Include any trailing whitespace/newlines after the overlap
                while cut_position < len(curr_chunk) and curr_chunk[cut_position] in ' \n\t':
                    cut_position += 1
                
                # Get remaining text
                remaining_text = curr_chunk[cut_position:].strip()
                
                if remaining_text:
                    # Ensure smooth flow by checking spacing
                    prev_end = prev_chunk.rstrip()
                    if prev_end and not prev_end[-1] in '.!?;:\n':
                        # If prev doesn't end with punctuation, ensure proper spacing
                        if not remaining_text[0] in '.!?;:,' and not prev_end.endswith(' '):
                            remaining_text = ' ' + remaining_text
                    
                    deduplicated.append(remaining_text)
                # If nothing remains, the chunk was completely overlapped - skip it
            else:
                # Couldn't find exact position, append as-is
                deduplicated.append(curr_chunk)
        else:
            # No overlap found, append as-is but ensure smooth transition
            prev_end = prev_chunk.rstrip()
            if prev_end and curr_chunk_clean:
                # Ensure proper spacing between non-overlapping chunks
                if (prev_end[-1] not in '.!?;:\n' and 
                    curr_chunk_clean[0] not in '.!?;:,' and
                    not prev_end.endswith(' ')):
                    deduplicated.append(' ' + curr_chunk)
                else:
                    deduplicated.append(curr_chunk)
            else:
                deduplicated.append(curr_chunk)
    
    return deduplicated


def _get_all_chunks_for_papers(paper_ids: List[str], debug: bool = False) -> Dict[Tuple[str, int], Dict]:
    """
    Retrieve ALL chunks for given paper IDs from Elasticsearch.
    
    Args:
        paper_ids: List of paper IDs to retrieve chunks for
        debug: Whether to print debug information
    
    Returns:
        Dictionary mapping (paper_id, chunk_index) -> chunk data
    """
    global _es_client
    
    if not _es_client or not paper_ids:
        return {}
    
    unique_paper_ids = list(set(paper_ids))
    all_chunks = {}
    
    try:
        # Query Elasticsearch for all chunks matching these paper IDs
        # Note: We don't sort by chunk_index as it may not be sortable in the index
        # Instead, we'll sort the results in Python after retrieval
        query = {
            "query": {
                "terms": {
                    "paper_id": unique_paper_ids
                }
            },
            "size": 10000,  # Large size to get all chunks (adjust if needed)
            "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count", "year"]
            # Removed sort - will sort in Python instead
        }
        
        if debug:
            print(f"[Full Paper] Querying Elasticsearch for {len(unique_paper_ids)} papers: {unique_paper_ids}")
        
        try:
            # Try new API first (Elasticsearch 8.x) - without sort since chunk_index may not be sortable
            response = _es_client.search(
                index=ES_INDEX,
                query=query["query"],
                size=query["size"],
                _source=query["_source"],
                timeout="60s"
            )
        except (TypeError, KeyError):
            # Fall back to old API
            if debug:
                print("[Full Paper] Using fallback API (old Elasticsearch client)")
            response = _es_client.search(index=ES_INDEX, body=query, timeout="60s")
        
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", len(hits))
        else:
            total_count = total_hits if total_hits else len(hits)
        
        if debug:
            print(f"[Full Paper] Elasticsearch returned {len(hits)} hits (total available: {total_count})")
        
        # Handle pagination if we have more results than size limit
        if total_count > len(hits):
            if debug:
                print(f"[Full Paper] WARNING: More chunks available ({total_count}) than retrieved ({len(hits)}). Consider increasing size limit.")
        
        for hit in hits:
            source_data = hit.get("_source", {})
            paper_id = source_data.get("paper_id", "unknown")
            chunk_index = int(source_data.get("chunk_index", 0))
            key = (paper_id, chunk_index)
            
            all_chunks[key] = {
                "source_data": source_data,
                "score": float(hit.get("_score", 0.0)),
                "hit": hit
            }
        
        # Sort chunks by paper_id and chunk_index in Python
        # Convert to list, sort, then back to dict (or just keep as sorted list of items)
        sorted_chunk_items = sorted(all_chunks.items(), key=lambda x: (x[0][0], x[0][1]))
        all_chunks = dict(sorted_chunk_items)
        
        if debug:
            print(f"[Full Paper] Retrieved {len(all_chunks)} chunks for {len(unique_paper_ids)} papers")
            for paper_id in unique_paper_ids:
                paper_chunks = [k for k in all_chunks.keys() if k[0] == paper_id]
                print(f"[Full Paper]   Paper {paper_id}: {len(paper_chunks)} chunks")
        
        return all_chunks
        
    except Exception as e:
        if debug:
            print(f"[Full Paper] Error retrieving all chunks: {e}")
        return {}


def _extract_year_from_paper_id(paper_id: str) -> Optional[int]:
    """
    Extract year from paper_id if it's an arXiv ID.
    Supports two arXiv ID formats:
    1. Old format (pre-2007): YYYY.MMMM or YYYY.MMMMvN (e.g., 2004.0123 -> 2004)
    2. New format (post-2007): category/YYMMNNN or YYMM.NNNN (e.g., cs/0407005 -> 2004, 1701.01234 -> 2017)
    """
    if not paper_id:
        return None
    
    try:
        # Remove version suffix if present (e.g., "v2")
        paper_id_clean = paper_id.split('v')[0]
        
        # Check for old format: YYYY.MMMM (4 digits before dot)
        if '.' in paper_id_clean:
            parts = paper_id_clean.split('.')
            if len(parts) >= 1 and len(parts[0]) == 4 and parts[0].isdigit():
                year = int(parts[0])
                if 1990 <= year <= 2100:  # Reasonable year range
                    return year
        
        # Check for new format: YYMM.NNNN or category/YYMMNNN
        # Extract YYMM pattern (2 digits for year, 2 digits for month)
        # Match YYMM pattern (4 digits)
        match = re.search(r'(\d{2})(\d{2})', paper_id_clean)
        if match:
            yy = int(match.group(1))
            mm = int(match.group(2))
            # Validate month (1-12)
            if 1 <= mm <= 12:
                # Convert to full year: if YY < 50, assume 20YY, else assume 19YY
                if yy < 50:
                    return 2000 + yy
                else:
                    return 1900 + yy
    except (ValueError, IndexError, AttributeError):
        pass
    
    return None


def _get_chunk_year(chunk_data: Dict) -> Optional[int]:
    """Get year from chunk data, either from year field or extracted from paper_id."""
    source_data = chunk_data.get("source_data", {})
    
    # First try the year field
    year = source_data.get("year")
    if year is not None:
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2100:
                return year_int
        except (ValueError, TypeError):
            pass
    
    # Fall back to extracting from paper_id
    paper_id = source_data.get("paper_id", "")
    return _extract_year_from_paper_id(paper_id)


def _filter_chunks_by_analysis(all_chunks_dict: Dict, analysis: Dict[str, Any], debug: bool = False) -> Dict:
    """
    Filter chunks based on query analysis criteria (year, authors, venues).
    
    Args:
        all_chunks_dict: Dictionary mapping (paper_id, chunk_index) -> chunk data
        analysis: Result from analyze_query containing filters
        debug: Whether to print debug information
    
    Returns:
        Filtered dictionary with same structure
    """
    if not analysis or analysis.get("status") != "success":
        if debug:
            print("[Filtering] Analysis not available or failed, skipping filtering")
        return all_chunks_dict
    
    time_range = analysis.get("time_range", {})
    requested_authors = analysis.get("authors", [])
    requested_venues = analysis.get("venues", [])
    
    # Check if any filtering criteria are present
    has_year_filter = time_range.get("start") is not None or time_range.get("end") is not None
    has_author_filter = len(requested_authors) > 0
    has_venue_filter = len(requested_venues) > 0
    
    if not (has_year_filter or has_author_filter or has_venue_filter):
        if debug:
            print("[Filtering] No filtering criteria found in analysis, keeping all chunks")
        return all_chunks_dict
    
    if debug:
        print("\n" + "="*60)
        print("FILTERING CHUNKS BY ANALYSIS")
        print("="*60)
        print(f"Year filter: {time_range}")
        print(f"Author filter: {requested_authors}")
        print(f"Venue filter: {requested_venues}")
        print(f"Chunks before filtering: {len(all_chunks_dict)}")
    
    filtered_chunks = {}
    year_start = time_range.get("start")
    year_end = time_range.get("end")
    
    for key, chunk_data in all_chunks_dict.items():
        source_data = chunk_data.get("source_data", {})
        should_keep = True
        
        # Filter by year
        if has_year_filter:
            chunk_year = _get_chunk_year(chunk_data)
            if chunk_year is None:
                # If we can't determine the year, keep it (don't filter out)
                pass
            else:
                if year_start and chunk_year < year_start:
                    should_keep = False
                if year_end and chunk_year > year_end:
                    should_keep = False
        
        # Filter by authors
        if should_keep and has_author_filter:
            chunk_authors = source_data.get("authors", "")
            if chunk_authors:
                # Normalize for comparison (case-insensitive, handle comma-separated)
                chunk_authors_lower = chunk_authors.lower()
                requested_authors_lower = [a.lower().strip() for a in requested_authors]
                
                # Check if any requested author name is contained in chunk authors
                author_match = False
                for req_author in requested_authors_lower:
                    # Split chunk authors by comma and check each
                    chunk_author_list = [a.strip() for a in chunk_authors_lower.split(',')]
                    for chunk_author in chunk_author_list:
                        # Check if requested author is in chunk author or vice versa
                        if req_author in chunk_author or chunk_author in req_author:
                            author_match = True
                            break
                    if author_match:
                        break
                
                if not author_match:
                    should_keep = False
            else:
                # No authors in chunk, filter it out if author filter is active
                should_keep = False
        
        # Filter by venues (if venue data is available in chunks)
        # Note: Currently chunks don't seem to have venue data, but we'll check anyway
        if should_keep and has_venue_filter:
            # Venue filtering would go here if venue field exists in chunks
            # For now, we skip venue filtering as it's not in the chunk metadata
            pass
        
        if should_keep:
            filtered_chunks[key] = chunk_data
    
    if debug:
        print(f"Chunks after filtering: {len(filtered_chunks)}")
        print(f"Chunks removed: {len(all_chunks_dict) - len(filtered_chunks)}")
        print("="*60)
    
    return filtered_chunks


def _fetch_arxiv_papers_by_author(author_names: List[str], max_results_per_author: int = 5, debug: bool = False) -> List[Dict]:
    """
    Fetch papers from arXiv API for given author names.
    
    Args:
        author_names: List of author names to search for
        max_results_per_author: Maximum number of papers to fetch per author
        debug: Whether to print debug information
    
    Returns:
        List of dictionaries with paper information (paper_id, title, authors, chunk_text, etc.)
    """
    if not author_names:
        return []
    
    all_papers = []
    arxiv_ns = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
    
    for author_name in author_names:
        if not author_name or not author_name.strip():
            continue
        
        try:
            # Construct arXiv API query for author search
            # Use au:"Author Name" format for exact author matching
            search_query = f'au:"{author_name.strip()}"'
            url = f"http://export.arxiv.org/api/query"
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results_per_author,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            if debug:
                print(f"[arXiv] Fetching papers for author: {author_name}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            entries = root.findall("atom:entry", arxiv_ns)
            
            for entry in entries:
                # Extract paper ID (format: http://arxiv.org/abs/YYMM.NNNNvN -> YYMM.NNNNvN)
                paper_id_elem = entry.find("atom:id", arxiv_ns)
                if paper_id_elem is None:
                    continue
                
                paper_id = paper_id_elem.text.split("/")[-1] if paper_id_elem.text else None
                if not paper_id:
                    continue
                
                # Extract title
                title_elem = entry.find("atom:title", arxiv_ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""
                
                # Extract authors
                author_elems = entry.findall("atom:author", arxiv_ns)
                authors_list = []
                for author_elem in author_elems:
                    name_elem = author_elem.find("atom:name", arxiv_ns)
                    if name_elem is not None and name_elem.text:
                        authors_list.append(name_elem.text.strip())
                authors = ", ".join(authors_list) if authors_list else ""
                
                # Extract abstract/summary
                summary_elem = entry.find("atom:summary", arxiv_ns)
                abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                
                # Extract published date
                published_elem = entry.find("atom:published", arxiv_ns)
                published_date = published_elem.text.split("T")[0] if published_elem is not None and published_elem.text else None
                year = None
                if published_date and len(published_date) >= 4:
                    try:
                        year = int(published_date.split("-")[0])
                    except (ValueError, AttributeError):
                        year = None
                
                # Create chunk text from title and abstract
                chunk_text = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
                
                # Construct arXiv URL
                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                
                paper_data = {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "chunk_text": chunk_text,
                    "abstract": abstract,
                    "year": year,
                    "published_date": published_date,
                    "url": arxiv_url,
                    "chunk_index": 0,  # arXiv papers are treated as single chunks
                    "source": "arxiv_api"
                }
                
                all_papers.append(paper_data)
            
            if debug:
                print(f"[arXiv] Found {len(entries)} papers for author: {author_name}")
        
        except Exception as e:
            if debug:
                print(f"[arXiv] Error fetching papers for author '{author_name}': {e}")
            continue
    
    if debug:
        print(f"[arXiv] Total papers fetched from arXiv: {len(all_papers)}")
    
    return all_papers


def _reciprocal_rank_fusion(rank_lists: List[List[Tuple[str, int]]], k: int = 60) -> Dict[Tuple[str, int], float]:
    """
    Apply Reciprocal Rank Fusion to combine multiple ranked lists.
    
    Args:
        rank_lists: List of ranked lists, where each list contains tuples of (paper_id, chunk_index)
        k: RRF constant (default 60)
    
    Returns:
        Dictionary mapping (paper_id, chunk_index) to RRF score
    """
    scores: Dict[Tuple[str, int], float] = defaultdict(float)
    
    for ranks in rank_lists:
        for rank, (paper_id, chunk_index) in enumerate(ranks, start=1):
            key = (paper_id, chunk_index)
            scores[key] += 1.0 / (k + rank)
    
    return scores


def stream_rag_response(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = DEFAULT_MAX_TOKENS, debug: bool = False, conversation_history: List[Dict] = None, user_filters: Dict = None):
    """Generator that streams the LLM answer tokens and finally emits sources.

    Yields SSE-like lines: "data: {json}\n\n" where json has either a
    {"event":"delta","text":"..."} shape or a final
    {"event":"done","sources":[...]}.
    """
    if not initialize_rag_system() or not all([_es_client, _openai_client]):
        yield f"data: {json.dumps({'event': 'error', 'message': 'RAG not initialized'})}\n\n"
        return

    try:
        # Step 1: Generate 10 reworded queries using GPT with conversation history
        yield f"data: {json.dumps({'event': 'status', 'message': 'Generating reformulated queries...'})}\n\n"
        if debug:
            print("\n" + "="*60)
            print("STREAMING: Generating reworded queries")
            print("="*60)
        
        reworded_queries = _generate_reworded_queries(query, conversation_history, num_queries=10, debug=debug)
        
        # Step 2: For each query, retrieve top 20 chunks from Elasticsearch
        yield f"data: {json.dumps({'event': 'status', 'message': 'Retrieving best chunks...'})}\n\n"
        if debug:
            print("\n" + "="*60)
            print("RETRIEVING CHUNKS FOR EACH QUERY")
            print("="*60)
        
        all_rank_lists = []  # For RRF: list of ranked lists
        all_chunks_dict = {}  # Map (paper_id, chunk_index) -> chunk data
        
        for i, reworded_query in enumerate(reworded_queries, 1):
            if debug:
                print(f"\nQuery {i}/10: {reworded_query}")
            
            hits = _search_elasticsearch(reworded_query, top_k=20, debug=debug)
            
            if debug:
                print(f"  Retrieved {len(hits)} chunks")
            
            # Build ranked list for RRF and store chunk data
            rank_list = []
            for hit in hits:
                source_data = hit.get("_source", {})
                paper_id = source_data.get("paper_id", "unknown")
                chunk_index = int(source_data.get("chunk_index", 0))
                key = (paper_id, chunk_index)
                
                rank_list.append(key)
                
                # Store chunk data (will overwrite duplicates, which is fine)
                if key not in all_chunks_dict:
                    all_chunks_dict[key] = {
                        "source_data": source_data,
                        "score": float(hit.get("_score", 0.0)),
                        "hit": hit
                    }
            
            all_rank_lists.append(rank_list)
        
        # Step 3: Remove duplicates (already done by using dict, but count them)
        total_before_dedup = sum(len(rank_list) for rank_list in all_rank_lists)
        unique_chunks = len(all_chunks_dict)
        duplicates_removed = total_before_dedup - unique_chunks
        
        if debug:
            print(f"\n" + "="*60)
            print("DEDUPLICATION")
            print("="*60)
            print(f"Total chunks before deduplication: {total_before_dedup}")
            print(f"Unique chunks after deduplication: {unique_chunks}")
            print(f"Duplicates removed: {duplicates_removed}")
        
        if unique_chunks == 0:
            no_data_msg = "No relevant chunks could be retrieved for your request. Please try rephrasing the question."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return
        
        # Step 3.5: Analyze query and filter chunks by criteria (year, authors, venues)
        if debug:
            print("\n" + "="*60)
            print("ANALYZING QUERY FOR FILTERING")
            print("="*60)
        
        query_analysis = analyze_query(query)
        
        # Merge user-provided filters with analysis results
        if user_filters:
            print(f"[DEBUG] user_filters provided: {user_filters}")
            print(f"[DEBUG] fullPaperProcessing in user_filters: {user_filters.get('fullPaperProcessing', 'NOT FOUND')}")
            # Override or merge with user-provided filters
            if user_filters.get("yearStart") is not None or user_filters.get("yearEnd") is not None:
                if not query_analysis.get("time_range"):
                    query_analysis["time_range"] = {"start": None, "end": None}
                if user_filters.get("yearStart") is not None:
                    query_analysis["time_range"]["start"] = user_filters["yearStart"]
                if user_filters.get("yearEnd") is not None:
                    query_analysis["time_range"]["end"] = user_filters["yearEnd"]
            
            if user_filters.get("authors") and len(user_filters["authors"]) > 0:
                existing_authors = query_analysis.get("authors", [])
                print(f"[DEBUG] Merging authors - existing: {existing_authors}, from filters: {user_filters.get('authors')}")
                # Merge authors, avoiding duplicates
                merged_authors = list(set(existing_authors + user_filters["authors"]))
                query_analysis["authors"] = merged_authors
                print(f"[DEBUG] Merged authors: {merged_authors}")
            else:
                print(f"[DEBUG] No authors in user_filters or empty list")
            
            if user_filters.get("venues") and len(user_filters["venues"]) > 0:
                existing_venues = query_analysis.get("venues", [])
                # Merge venues, avoiding duplicates
                merged_venues = list(set(existing_venues + user_filters["venues"]))
                query_analysis["venues"] = merged_venues
            
            if user_filters.get("queryType"):
                query_analysis["query_type"] = user_filters["queryType"]
        else:
            print(f"[DEBUG] No user_filters provided")
        
        # Step 3.6: Fetch papers from arXiv if authors are specified (BEFORE filtering)
        # This ensures we get arXiv papers even if Elasticsearch chunks are filtered out
        arxiv_papers = []
        authors_list = query_analysis.get("authors", [])
        
        # Debug: Always log whether authors were found
        print(f"\n[DEBUG] Checking for authors in query_analysis...")
        print(f"[DEBUG] query_analysis.get('authors'): {query_analysis.get('authors')}")
        print(f"[DEBUG] authors_list: {authors_list}")
        print(f"[DEBUG] len(authors_list): {len(authors_list) if authors_list else 0}")
        
        if authors_list and len(authors_list) > 0:
            print("\n" + "="*60)
            print("FETCHING PAPERS FROM ARXIV BY AUTHOR")
            print("="*60)
            print(f"Authors found: {authors_list}")
            
            arxiv_papers = _fetch_arxiv_papers_by_author(
                authors_list, 
                max_results_per_author=5, 
                debug=debug
            )
            
            print(f"Fetched {len(arxiv_papers)} papers from arXiv")
            print("="*60)
        else:
            print(f"[DEBUG] No authors found - skipping arXiv API call")
            print(f"[DEBUG] authors_list is empty or None: {authors_list}")
        
        # Filter chunks based on merged analysis
        all_chunks_dict = _filter_chunks_by_analysis(all_chunks_dict, query_analysis, debug=debug)
        
        # Also filter rank_lists to only include chunks that passed the filter
        filtered_rank_lists = []
        for rank_list in all_rank_lists:
            filtered_rank_list = [key for key in rank_list if key in all_chunks_dict]
            if filtered_rank_list:  # Only add non-empty rank lists
                filtered_rank_lists.append(filtered_rank_list)
        
        all_rank_lists = filtered_rank_lists
        
        # If no chunks remain after filtering, but we have arXiv papers, continue with those
        if len(all_chunks_dict) == 0 and len(arxiv_papers) == 0:
            no_data_msg = "No relevant chunks matched the specified criteria (year, author, etc.). Please try adjusting your filters."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return
        
        # Step 4: Apply RRF reranking on the deduplicated and filtered chunks
        # Only do RRF if we have Elasticsearch chunks to rank
        sorted_chunks = []
        if len(all_chunks_dict) > 0 and len(all_rank_lists) > 0:
            if debug:
                print("\n" + "="*60)
                print("APPLYING RRF RERANKING")
                print("="*60)
            
            try:
                rrf_scores = _reciprocal_rank_fusion(all_rank_lists, k=60)
                sorted_chunks = sorted(
                    [(key, score) for key, score in rrf_scores.items() if key in all_chunks_dict],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if debug:
                    print(f"RRF reranked {len(sorted_chunks)} unique chunks")
            except Exception as e:
                if debug:
                    print(f"[ERROR] RRF reranking failed: {e}")
                sorted_chunks = []
        
        # If we have arXiv papers but no Elasticsearch chunks, use arXiv papers only
        if len(all_chunks_dict) == 0 and len(arxiv_papers) > 0:
            if debug:
                print("\n[DEBUG] No Elasticsearch chunks after filtering, but have arXiv papers. Using arXiv papers only.")
        
        # If we have neither chunks nor arXiv papers, return error
        if len(sorted_chunks) == 0 and len(arxiv_papers) == 0:
            no_data_msg = "No relevant chunks matched the specified criteria and no papers found from authors. Please try adjusting your filters."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return
        
        # Step 5: Build final sources and context from top 5 results
        retrieved_texts = []
        sources = []
        best_query = reworded_queries[0] if reworded_queries else query
        context_chunk_count = 0  # Initialize to avoid UnboundLocalError
        
        # Collect paper IDs for metadata lookup and final sources
        paper_ids_for_metadata = []
        final_paper_ids = []  # The 5 final paper IDs
        
        # Get paper IDs from sorted_chunks (Elasticsearch results)
        if len(all_chunks_dict) > 0 and len(sorted_chunks) > 0:
            for (paper_id, chunk_index), _ in sorted_chunks[:5]:
                if paper_id and paper_id != "unknown":
                    paper_ids_for_metadata.append(paper_id)
                    if paper_id not in final_paper_ids:
                        final_paper_ids.append(paper_id)
        
        # Also add arXiv paper IDs (limit to 5 total papers)
        for arxiv_paper in arxiv_papers[:5]:
            paper_id = arxiv_paper.get("paper_id")
            if paper_id and paper_id != "unknown":
                paper_ids_for_metadata.append(paper_id)
                if paper_id not in final_paper_ids and len(final_paper_ids) < 5:
                    final_paper_ids.append(paper_id)
        
        # Limit to 5 papers total
        final_paper_ids = final_paper_ids[:5]
        
        # Check if full paper processing is enabled
        full_paper_processing = user_filters and user_filters.get("fullPaperProcessing", False)
        
        if debug:
            print(f"\n[DEBUG] ========== FULL PAPER PROCESSING CHECK ==========")
            print(f"[DEBUG] user_filters: {user_filters}")
            print(f"[DEBUG] full_paper_processing: {full_paper_processing}")
            print(f"[DEBUG] final_paper_ids: {final_paper_ids}")
            print(f"[DEBUG] len(final_paper_ids): {len(final_paper_ids)}")
            print(f"[DEBUG] =================================================")
        
        # If full paper processing is enabled, retrieve ALL chunks for the final papers
        if full_paper_processing and final_paper_ids:
            if debug:
                print("\n" + "="*60)
                print("FULL PAPER PROCESSING ENABLED")
                print("="*60)
                print(f"Retrieving ALL chunks for {len(final_paper_ids)} papers: {final_paper_ids}")
            
            # Get all chunks for these papers (only for Elasticsearch papers, not arXiv)
            # Filter out arXiv papers since they don't have chunks in Elasticsearch
            es_paper_ids = [pid for pid in final_paper_ids if pid not in [ap.get("paper_id") for ap in arxiv_papers]]
            
            if es_paper_ids:
                if debug:
                    print(f"[Full Paper] Calling _get_all_chunks_for_papers with {len(es_paper_ids)} papers: {es_paper_ids}")
                
                all_paper_chunks = _get_all_chunks_for_papers(es_paper_ids, debug=debug)
                
                if debug:
                    print(f"[Full Paper] _get_all_chunks_for_papers returned {len(all_paper_chunks)} chunks")
                    for pid in es_paper_ids:
                        pid_chunks = [k for k in all_paper_chunks.keys() if k[0] == pid]
                        print(f"[Full Paper]   Paper {pid}: {len(pid_chunks)} chunks from Elasticsearch query")
                
                # Replace/update all_chunks_dict with ALL chunks from these papers
                # This ensures we have all chunks, not just the ones from search results
                chunks_before = len(all_chunks_dict)
                for key, chunk_data in all_paper_chunks.items():
                    all_chunks_dict[key] = chunk_data  # Overwrite to ensure we have all chunks
                chunks_after = len(all_chunks_dict)
                
                if debug:
                    print(f"[Full Paper] all_chunks_dict: {chunks_before} -> {chunks_after} chunks")
                    for pid in es_paper_ids:
                        pid_chunks = [k for k in all_chunks_dict.keys() if k[0] == pid]
                        print(f"[Full Paper]   Paper {pid}: {len(pid_chunks)} chunks in all_chunks_dict after merge")
        
        # Fetch metadata from metadata index
        metadata_dict = _get_paper_metadata(paper_ids_for_metadata, debug=debug)
        
        # Process Elasticsearch chunks if we have any
        if len(all_chunks_dict) > 0:
            if full_paper_processing:
                # If full paper processing, write FULL papers to files and use only top 5 chunks for GPT
                if debug:
                    print(f"\n[Full Paper] ========== PROCESSING FULL PAPERS ==========")
                    print(f"[Full Paper] Writing full papers from {len(final_paper_ids)} papers: {final_paper_ids}")
                    print(f"[Full Paper] Total chunks in all_chunks_dict: {len(all_chunks_dict)}")
                
                # Write full papers (all chunks) to text files in /public/ directory
                public_dir = os.path.join(os.path.dirname(__file__), '..', 'public')
                os.makedirs(public_dir, exist_ok=True)
                
                yield f"data: {json.dumps({'event': 'status', 'message': 'Summarizing papers...'})}\n\n"
                
                for rank, paper_id in enumerate(final_paper_ids[:5], 1):  # Limit to top 5 papers
                    paper_text_parts = []
                    
                    # Collect all chunks for this paper (sorted by chunk_index)
                    paper_chunks = [(k, v) for k, v in all_chunks_dict.items() if k[0] == paper_id]
                    if paper_chunks:
                        paper_chunks.sort(key=lambda x: x[0][1])
                        
                        if debug:
                            print(f"[Full Paper] Paper {paper_id}: found {len(paper_chunks)} chunks")
                        
                        # Collect all chunk texts for this paper
                        for (pid, chunk_index), chunk_data in paper_chunks:
                            source_data = chunk_data["source_data"]
                            chunk_text = source_data.get("chunk_text", "")
                            if chunk_text:
                                paper_text_parts.append(chunk_text)
                        
                        # Remove overlapping text between consecutive chunks
                        if len(paper_text_parts) > 1:
                            original_count = len(paper_text_parts)
                            paper_text_parts = _remove_overlap_between_chunks(paper_text_parts, min_overlap_length=50)
                            if debug and len(paper_text_parts) != original_count:
                                print(f"[Full Paper] Removed overlaps from {original_count} chunks")
                    else:
                        # Check if this is an arXiv paper
                        for arxiv_paper in arxiv_papers:
                            if arxiv_paper.get("paper_id") == paper_id:
                                chunk_text = arxiv_paper.get("chunk_text", "")
                                if chunk_text:
                                    paper_text_parts.append(chunk_text)
                                if debug:
                                    print(f"[Full Paper] Paper {paper_id}: arXiv paper (single chunk)")
                                break
                    
                    # Write the full paper content to a file
                    if paper_text_parts:
                        # Join chunks with line breaks - deduplication function already handles spacing for smooth flow
                        # This preserves chunk separation while maintaining seamless text flow
                        full_paper_text = "\n\n".join(paper_text_parts)
                        
                        # Post-processing: remove any remaining repeated phrases that might have been missed
                        full_paper_text = _remove_repeated_phrases(full_paper_text, min_phrase_words=10)
                        
                        filename = os.path.join(public_dir, f"{rank}.txt")
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(full_paper_text)
                            if debug:
                                print(f"[Full Paper] Wrote paper {paper_id} ({len(paper_text_parts)} chunks) to {filename}")
                        except Exception as e:
                            if debug:
                                print(f"[Full Paper] Error writing {filename}: {e}")
                        
                        # Summarize the paper and write summarized version
                        if debug:
                            print(f"[Summarization] Summarizing paper {paper_id}...")
                        
                        summarized_text = _summarize_paper(full_paper_text, target_ratio=0.15, debug=debug)
                        
                        summary_filename = os.path.join(public_dir, f"{rank}_summ.txt")
                        try:
                            with open(summary_filename, 'w', encoding='utf-8') as f:
                                f.write(summarized_text)
                            if debug:
                                print(f"[Summarization] Wrote summarized paper {paper_id} to {summary_filename}")
                        except Exception as e:
                            if debug:
                                print(f"[Summarization] Error writing {summary_filename}: {e}")
                
                if debug:
                    print(f"[Full Paper] Wrote {min(5, len(final_paper_ids))} papers to text files in /public/ directory")
                
                # Read summaries from files and pass them to GPT instead of chunks
                context_chunk_count = 0
                for rank in range(1, min(6, len(final_paper_ids) + 1)):
                    summary_filename = os.path.join(public_dir, f"{rank}_summ.txt")
                    try:
                        if os.path.exists(summary_filename):
                            with open(summary_filename, 'r', encoding='utf-8') as f:
                                summary_text = f.read().strip()
                            if summary_text:
                                retrieved_texts.append(summary_text)
                                context_chunk_count += 1
                                if debug:
                                    print(f"[Full Paper] Added summary {rank}/5 to GPT context from {summary_filename}")
                            else:
                                if debug:
                                    print(f"[Full Paper] Summary file {summary_filename} is empty")
                        else:
                            if debug:
                                print(f"[Full Paper] Summary file {summary_filename} not found")
                    except Exception as e:
                        if debug:
                            print(f"[Full Paper] Error reading summary {summary_filename}: {e}")
                
                if debug:
                    print(f"[Full Paper] Total summaries sent to GPT: {context_chunk_count}")
                    print(f"[Full Paper] ============================================\n")
                
                # Build sources list from the first chunk of each paper (for display)
                seen_paper_ids = set()  # Track papers already added to avoid duplicates
                rank_counter = 1
                for paper_id in final_paper_ids:
                    if paper_id in seen_paper_ids:
                        continue  # Skip if this paper already added
                    seen_paper_ids.add(paper_id)
                    
                    # Find all chunks for this paper
                    paper_chunks = [(k, v) for k, v in all_chunks_dict.items() if k[0] == paper_id]
                    if paper_chunks:
                        paper_chunks.sort(key=lambda x: x[0][1])
                        (first_paper_id, first_chunk_index), first_chunk_data = paper_chunks[0]
                        source_data = first_chunk_data["source_data"]
                        
                        # Calculate best scores from all chunks of this paper
                        best_similarity_score = max((chunk_data["score"] for _, chunk_data in paper_chunks), default=0.0)
                        # Find best RRF score for this paper from sorted_chunks
                        best_rrf_score = 0.0
                        for (pid, cidx), rrf_score in sorted_chunks:
                            if pid == paper_id:
                                best_rrf_score = max(best_rrf_score, rrf_score)
                        
                        # Get title and authors from metadata or chunk data
                        title = ""
                        authors = ""
                        if paper_id in metadata_dict:
                            metadata = metadata_dict[paper_id]
                            title = metadata.get("title", "")
                            if title:
                                title = str(title).strip()
                            authors = _normalize_authors(metadata.get("authors", ""))
                        
                        if not title:
                            title = source_data.get("title", "")
                            if title:
                                title = str(title).strip()
                        if not authors:
                            authors = _normalize_authors(source_data.get("authors", ""))
                        
                        arxiv_url = None
                        if paper_id and not paper_id.startswith("http"):
                            if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                        
                        sources.append({
                            "paper_id": paper_id,
                            "title": title if title else f"Paper {paper_id}",
                            "authors": authors if authors else None,
                            "chunk_index": first_chunk_index,
                            "rank": rank_counter,
                            "rrf_score": best_rrf_score,
                            "similarity_score": best_similarity_score,
                            "url": arxiv_url
                        })
                        rank_counter += 1
            else:
                # Normal processing: use top chunks from sorted_chunks
                if len(sorted_chunks) > 0:
                    context_chunk_count = min(5, len(sorted_chunks))
                    seen_paper_ids = set()  # Track papers already added to avoid duplicates
                    rank_counter = 1
                    for ((paper_id, chunk_index), rrf_score) in sorted_chunks[:context_chunk_count]:
                        if paper_id in seen_paper_ids:
                            continue  # Skip if this paper already added
                        seen_paper_ids.add(paper_id)
                        chunk_data = all_chunks_dict.get((paper_id, chunk_index))
                        if not chunk_data:
                            continue
                        
                        source_data = chunk_data["source_data"]
                        chunk_text = source_data.get("chunk_text", "")
                        if chunk_text:
                            retrieved_texts.append(chunk_text)
                        
                        # Try to get title from metadata index first, fall back to chunk data
                        title = ""
                        authors = ""
                        if paper_id in metadata_dict:
                            metadata = metadata_dict[paper_id]
                            title = metadata.get("title", "")
                            if title:
                                title = str(title).strip()
                            authors = _normalize_authors(metadata.get("authors", ""))
                        
                        # Fall back to chunk data if metadata not available
                        if not title:
                            title = source_data.get("title", "")
                            if title:
                                title = str(title).strip()
                        if not authors:
                            authors = _normalize_authors(source_data.get("authors", ""))
                        
                        # Construct arXiv URL if paper_id looks like an arXiv ID
                        arxiv_url = None
                        if paper_id and not paper_id.startswith("http"):
                            if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                        
                        sources.append({
                            "paper_id": paper_id,
                            "title": title if title else f"Paper {paper_id}",
                            "authors": authors if authors else None,
                            "chunk_index": chunk_index,
                            "rank": rank_counter,
                            "rrf_score": rrf_score,
                            "similarity_score": chunk_data["score"],
                            "url": arxiv_url
                        })
                        rank_counter += 1
        
        # Add arXiv papers as additional chunks (avoid duplicates with existing sources)
        seen_paper_ids_from_sources = {s.get("paper_id") for s in sources}
        arxiv_start_rank = len(sources) + 1
        arxiv_paper_count = min(5, len(arxiv_papers))
        arxiv_added_count = 0
        if len(arxiv_papers) > 0:
            # Update context_chunk_count to include arXiv papers
            context_chunk_count = context_chunk_count + arxiv_paper_count
        for idx, arxiv_paper in enumerate(arxiv_papers[:arxiv_paper_count], start=arxiv_start_rank):
            paper_id = arxiv_paper.get("paper_id", "unknown")
            if paper_id in seen_paper_ids_from_sources:
                continue  # Skip if this arXiv paper already exists in sources
            seen_paper_ids_from_sources.add(paper_id)
            chunk_text = arxiv_paper.get("chunk_text", "")
            if chunk_text:
                retrieved_texts.append(chunk_text)
            
            # Try to get title from metadata index first, fall back to arXiv data
            title = ""
            authors = ""
            if paper_id in metadata_dict:
                metadata = metadata_dict[paper_id]
                title = metadata.get("title", "")
                if title:
                    title = str(title).strip()
                authors = _normalize_authors(metadata.get("authors", ""))
            
            # Fall back to arXiv data if metadata not available
            if not title:
                title = arxiv_paper.get("title", "")
                if title:
                    title = str(title).strip()
            if not authors:
                authors = _normalize_authors(arxiv_paper.get("authors", ""))
            
            sources.append({
                "paper_id": paper_id,
                "title": title if title else f"Paper {paper_id}",
                "authors": authors if authors else None,
                "chunk_index": 0,
                "rank": arxiv_start_rank + arxiv_added_count,
                "rrf_score": 0.0,  # arXiv papers don't have RRF scores
                "similarity_score": 0.0,  # arXiv papers don't have similarity scores
                "url": arxiv_paper.get("url"),
                "source": "arxiv_api"
            })
            arxiv_added_count += 1
        
        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."
        
        # Prepare messages for the generator
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a paper finder assistant. Your primary role is to provide a very concise, high-level summary of what the retrieved academic papers are about. "
                    "Keep the user's query in mind, but focus on giving a brief overview (one to two paragraphs maximum) of the key topics and contributions across the papers. "
                    "Do NOT cite papers inline (no [Paper Title] or similar citations). The papers are already listed separately as sources. "
                    "Use conversation history to provide context-aware responses. "
                    "When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) "
                    "and display math with \\[ ... \\]. Do not wrap LaTeX in code blocks and do not escape backslashes."
                )
            }
        ]
        
        if conversation_history:
            for msg in conversation_history[-4:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        
        messages.append({
            "role": "user",
            "content": (
                f"Best keyword query (captures main intent): {best_query}\n\n"
                f"Original user question: {query}\n\n"
                f"Retrieved context snippets (top {context_chunk_count} chunks after RRF):\n{context}\n\n"
                "Provide a very concise, high-level summary (one to two paragraphs maximum) of what these papers are about, keeping the user's query in mind. "
                "Focus on the key topics and contributions across the papers rather than detailed explanations. "
                "Do NOT cite papers inline - the papers are already listed separately as sources."
            )
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
        
        # Emit sources first (before the summary)
        if sources:
            yield f"data: {json.dumps({'event': 'sources', 'sources': sources})}\n\n"
        
        # Stream deltas (the summary)
        yield f"data: {json.dumps({'event': 'status', 'message': 'Generating response...'})}\n\n"
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
        
        # Generate individual summaries for each source (top 5)
        if sources and len(sources) > 0:
            yield f"data: {json.dumps({'event': 'status', 'message': 'Generating relevancy explanations...'})}\n\n"
            if debug:
                print("\n[Streaming] Generating individual source summaries...")
            
            # Get the chunk text for each source to generate summaries
            source_chunks_dict = {}
            for source in sources[:5]:
                paper_id = source.get("paper_id")
                chunk_index = source.get("chunk_index", 0)
                key = (paper_id, chunk_index)
                
                # Get chunk text from all_chunks_dict or arxiv_papers
                chunk_text = ""
                if key in all_chunks_dict:
                    chunk_text = all_chunks_dict[key]["source_data"].get("chunk_text", "")
                else:
                    # Check if it's from arXiv
                    for arxiv_paper in arxiv_papers:
                        if arxiv_paper.get("paper_id") == paper_id:
                            chunk_text = arxiv_paper.get("chunk_text", "")
                            break
                
                if chunk_text:
                    source_chunks_dict[paper_id] = chunk_text
            
            # Generate summaries for each source
            for source in sources[:5]:
                paper_id = source.get("paper_id")
                chunk_text = source_chunks_dict.get(paper_id, "")
                
                if chunk_text:
                    try:
                        summary_messages = [
                            {
                                "role": "system",
                                "content": "You are a paper finder assistant. Write a concise one-paragraph explanation that jumps directly into HOW and WHY the paper connects to the query. Do NOT start with phrases like 'The paper X is highly relevant to the user's query about Y as it explores...' or 'This paper is relevant because...' Instead, immediately begin describing the specific content, methods, findings, or concepts that connect to the query. Write as if continuing a conversation, not introducing a topic."
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"User's query: {query}\n\n"
                                    f"Paper title: {source.get('title', 'Unknown')}\n\n"
                                    f"Paper content snippet:\n{chunk_text[:2000]}\n\n"
                                    f"Write a concise one-paragraph explanation that immediately describes the specific ways this paper connects to the query. Start directly with the content, methods, or findings - no introductory phrases about relevance."
                                )
                            }
                        ]
                        
                        summary_response = _openai_client.chat.completions.create(
                            model=MODEL_ID,
                            messages=summary_messages,
                            max_tokens=150,
                            temperature=0.7
                        )
                        
                        source["relevance_summary"] = summary_response.choices[0].message.content.strip()
                        
                        if debug:
                            print(f"  Generated summary for {paper_id}")
                    except Exception as e:
                        if debug:
                            print(f"  Error generating summary for {paper_id}: {e}")
                        source["relevance_summary"] = None
                else:
                    source["relevance_summary"] = None
        
        # Emit final metadata
        yield f"data: {json.dumps({'event': 'done', 'sources': sources, 'reworded_queries': reworded_queries, 'unique_chunks': unique_chunks, 'duplicates_removed': duplicates_removed, 'analysis': query_analysis})}\n\n"
    except Exception as e:
        import traceback
        error_msg = str(e)
        if debug:
            print(f"[ERROR] Exception in stream_rag_response: {error_msg}")
            traceback.print_exc()
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg})}\n\n"
