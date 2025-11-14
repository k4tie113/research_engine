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
        if debug:
            print("\n" + "="*60)
            print("STREAMING: Generating reworded queries")
            print("="*60)
        
        reworded_queries = _generate_reworded_queries(query, conversation_history, num_queries=10, debug=debug)
        
        # Step 2: For each query, retrieve top 20 chunks from Elasticsearch
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
                # Merge authors, avoiding duplicates
                merged_authors = list(set(existing_authors + user_filters["authors"]))
                query_analysis["authors"] = merged_authors
            
            if user_filters.get("venues") and len(user_filters["venues"]) > 0:
                existing_venues = query_analysis.get("venues", [])
                # Merge venues, avoiding duplicates
                merged_venues = list(set(existing_venues + user_filters["venues"]))
                query_analysis["venues"] = merged_venues
            
            if user_filters.get("queryType"):
                query_analysis["query_type"] = user_filters["queryType"]
        
        # Filter chunks based on merged analysis
        all_chunks_dict = _filter_chunks_by_analysis(all_chunks_dict, query_analysis, debug=debug)
        
        # Also filter rank_lists to only include chunks that passed the filter
        filtered_rank_lists = []
        for rank_list in all_rank_lists:
            filtered_rank_list = [key for key in rank_list if key in all_chunks_dict]
            if filtered_rank_list:  # Only add non-empty rank lists
                filtered_rank_lists.append(filtered_rank_list)
        
        all_rank_lists = filtered_rank_lists
        
        if len(all_chunks_dict) == 0:
            no_data_msg = "No relevant chunks matched the specified criteria (year, author, etc.). Please try adjusting your filters."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return
        
        # Step 4: Apply RRF reranking on the deduplicated and filtered chunks
        if debug:
            print("\n" + "="*60)
            print("APPLYING RRF RERANKING")
            print("="*60)
        
        rrf_scores = _reciprocal_rank_fusion(all_rank_lists, k=60)
        sorted_chunks = sorted(
            [(key, score) for key, score in rrf_scores.items() if key in all_chunks_dict],
            key=lambda x: x[1],
            reverse=True
        )
        
        if not sorted_chunks:
            no_data_msg = "Unable to rank retrieved chunks. Please try again."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return
        
        if debug:
            print(f"RRF reranked {len(sorted_chunks)} unique chunks")
        
        # Step 5: Build final sources and context from top 5 results
        context_chunk_count = min(5, len(sorted_chunks))
        retrieved_texts = []
        sources = []
        best_query = reworded_queries[0] if reworded_queries else query
        
        for rank, ((paper_id, chunk_index), rrf_score) in enumerate(sorted_chunks[:context_chunk_count], 1):
            chunk_data = all_chunks_dict.get((paper_id, chunk_index))
            if not chunk_data:
                continue
            
            source_data = chunk_data["source_data"]
            chunk_text = source_data.get("chunk_text", "")
            if chunk_text:
                retrieved_texts.append(chunk_text)
            
            title = source_data.get("title", "").strip()
            authors = source_data.get("authors", "").strip()
            
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
                "rank": rank,
                "rrf_score": rrf_score,
                "similarity_score": chunk_data["score"],
                "url": arxiv_url
            })
        
        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."
        
        # Prepare messages for the generator
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Use the provided context from academic papers to answer questions clearly and concisely. "
                    "If the context doesn't contain enough information, say so. Always cite the relevant papers when possible. "
                    "Use conversation history to provide context-aware responses. When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) "
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
                "Using the conversation history and the context above, answer the user's question. "
                "Explicitly cite relevant papers inline (e.g., [Paper Title]). If information is insufficient, explain what is missing."
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
        
        # Emit final sources and analysis
        yield f"data: {json.dumps({'event': 'done', 'sources': sources, 'reworded_queries': reworded_queries, 'unique_chunks': unique_chunks, 'duplicates_removed': duplicates_removed, 'analysis': query_analysis})}\n\n"
    except Exception as e:
        import traceback
        error_msg = str(e)
        if debug:
            print(f"[ERROR] Exception in stream_rag_response: {error_msg}")
            traceback.print_exc()
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg})}\n\n"
