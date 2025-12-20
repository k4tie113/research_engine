#!/usr/bin/env python3
"""
rag_service.py
--------------
A service module for RAG operations that can be imported by app.py.
Uses Elasticsearch for retrieval instead of FAISS.

UPDATED (this version):
- Deduplicate so the same paper is not returned multiple times (keep best chunk per paper).
- Add ASTA-style retrieval relevance scoring + label (0..3) with thresholds.
- Add ASTA Layer 2: OPTIONAL LLM semantic relevance judgment (gated), applied ONLY to top-5 returned docs.
- Convert LLM labels -> scores (0..3) and CLIP final relevance = min(retrieval_score, semantic_score).
- Preserve backward-compatibility: source["relevance"]["label"/"score"/"normalized"] still exist (final),
  while also exposing nested source["relevance"]["retrieval"] and source["relevance"]["semantic"].
- Fix bugs: undefined chunk_index / chunk_data in get_rag_response sources construction.
- Add relevance display in format_sources().
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

# ASTA Layer 2 gate config
DEFAULT_APPLY_RELEVANCE_JUDGEMENT = True  # global default; still gated per-query
DEFAULT_JUDGE_TOP_N = 5                   # only for the top N returned docs

# === ELASTICSEARCH CONFIG ===
ES_URL = os.getenv("ES_URL", "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_INDEX = "chunks"  # Index name in Elasticsearch cluster
ES_METADATA_INDEX = None  # Will be discovered dynamically

# Global variables for loaded resources
_es_client = None
_openai_client = None
_metadata_index_cache = None  # Cache the discovered metadata index name


# =============================================================================
# ASTA-inspired relevance logic (same thresholds/labels pattern)
# =============================================================================
class RelevanceThresholds:
    NOT_RELEVANT = 0.25
    SOMEWHAT_RELEVANT = 0.67
    HIGHLY_RELEVANT = 0.99


class RelevanceLabels:
    PERFECTLY_RELEVANT = "Perfectly Relevant"
    HIGHLY_RELEVANT = "Highly Relevant"
    SOMEWHAT_RELEVANT = "Somewhat Relevant"
    NOT_RELEVANT = "Not Relevant"


RELEVANCE_LABEL_TO_SCORE = {
    RelevanceLabels.PERFECTLY_RELEVANT: 3,
    RelevanceLabels.HIGHLY_RELEVANT: 2,
    RelevanceLabels.SOMEWHAT_RELEVANT: 1,
    RelevanceLabels.NOT_RELEVANT: 0,
}

SCORE_TO_RELEVANCE_LABEL = {v: k for k, v in RELEVANCE_LABEL_TO_SCORE.items()}


def _label_relevance(norm: float) -> str:
    """
    Map a normalized [0,1] score to ASTA-style labels.

    >= 0.99  -> Perfectly Relevant (3)
    >= 0.67  -> Highly Relevant (2)
    >= 0.25  -> Somewhat Relevant (1)
    else     -> Not Relevant (0)
    """
    if norm >= RelevanceThresholds.HIGHLY_RELEVANT:
        return RelevanceLabels.PERFECTLY_RELEVANT
    if norm >= RelevanceThresholds.SOMEWHAT_RELEVANT:
        return RelevanceLabels.HIGHLY_RELEVANT
    if norm >= RelevanceThresholds.NOT_RELEVANT:
        return RelevanceLabels.SOMEWHAT_RELEVANT
    return RelevanceLabels.NOT_RELEVANT


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _attach_retrieval_relevance(source: Dict[str, Any], norm: float) -> None:
    """
    Attach ASTA-style retrieval relevance into a compatibility-friendly structure.

    Backward compatible:
      source["relevance"]["label"/"score"/"normalized"] exists.

    Extended:
      source["relevance"]["retrieval"] exists and contains retrieval-specific fields.
      source["relevance"]["semantic"] is added later (optional).
      source["relevance"]["final_score"] is set later (or equals retrieval score if no semantic judge).
    """
    norm = _clamp01(float(norm))
    label = _label_relevance(norm)
    score = RELEVANCE_LABEL_TO_SCORE[label]

    # Back-compat + extensibility
    source["relevance"] = {
        # Default "final" == retrieval until semantic judgement runs
        "label": label,
        "score": score,
        "normalized": round(norm, 4),

        "retrieval": {
            "label": label,
            "score": score,
            "normalized": round(norm, 4),
        },

        # semantic will be filled in later if enabled
        "semantic": None,

        # final_score will be updated after semantic (or stays retrieval)
        "final_score": score,
    }


def _finalize_relevance(source: Dict[str, Any]) -> None:
    """
    If semantic relevance exists, clip final_score = min(retrieval_score, semantic_score).
    Keep compatibility fields source["relevance"]["label"/"score"] aligned to final_score.
    """
    rel = source.get("relevance") or {}
    retrieval = rel.get("retrieval") or {}
    semantic = rel.get("semantic")

    retrieval_score = int(retrieval.get("score", rel.get("score", 0)) or 0)

    if semantic and isinstance(semantic, dict):
        semantic_score = int(semantic.get("score", 0) or 0)
        final_score = min(retrieval_score, semantic_score)
    else:
        final_score = retrieval_score

    final_label = SCORE_TO_RELEVANCE_LABEL.get(final_score, RelevanceLabels.NOT_RELEVANT)

    # Update compatibility fields
    rel["final_score"] = final_score
    rel["score"] = final_score
    rel["label"] = final_label

    # Keep existing normalized (retrieval normalized) untouched
    source["relevance"] = rel


# =============================================================================
# ASTA Layer 2: LLM semantic relevance judgment (OPTIONAL, gated)
# =============================================================================
def _should_apply_semantic_judgement(
    analysis: Optional[Dict[str, Any]],
    user_filters: Optional[Dict[str, Any]]
) -> bool:
    """
    Gate semantic relevance judgement (LLM) on/off.

    Priority:
    1) explicit user override (applyRelevanceJudgement OR apply_relevance_judgement)
    2) global default
    3) (optional) query_type heuristics if analysis is present
    """

    # 1) frontend override (support both naming styles)
    if user_filters:
        if "applyRelevanceJudgement" in user_filters:
            return bool(user_filters.get("applyRelevanceJudgement"))
        if "apply_relevance_judgement" in user_filters:
            return bool(user_filters.get("apply_relevance_judgement"))

    # 2) global default
    if not DEFAULT_APPLY_RELEVANCE_JUDGEMENT:
        return False

    # 3) if no analysis, still allow (don’t silently disable L2)
    if not isinstance(analysis, dict):
        return True

    qtype = analysis.get("query_type") or analysis.get("queryType")

    # If analyzer didn’t return a qtype, still allow.
    if not qtype:
        return True

    # Your intended allowlist
    return qtype in ("BROAD_BY_DESCRIPTION", "BROAD", "EXPLORATORY")



def _extract_criteria_names(analysis: Optional[Dict[str, Any]]) -> List[str]:
    if not analysis:
        return []
    crit = analysis.get("relevance_criteria") or analysis.get("relevanceCriteria") or []
    names = []
    for c in crit:
        if isinstance(c, dict):
            n = c.get("name")
            if n:
                names.append(str(n).strip())
        elif isinstance(c, str) and c.strip():
            names.append(c.strip())
    # Dedup while preserving order
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract the first {...} JSON object
    try:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None

    return None


def _normalize_llm_relevance_label(label: Any) -> str:
    if not label:
        return RelevanceLabels.NOT_RELEVANT
    label_str = str(label).strip()

    # Exact matches
    if label_str in RELEVANCE_LABEL_TO_SCORE:
        return label_str

    # Soft matches
    low = label_str.lower()
    if "perfect" in low:
        return RelevanceLabels.PERFECTLY_RELEVANT
    if "high" in low:
        return RelevanceLabels.HIGHLY_RELEVANT
    if "some" in low or "partial" in low or "moderate" in low:
        return RelevanceLabels.SOMEWHAT_RELEVANT
    return RelevanceLabels.NOT_RELEVANT


def _judge_relevance_llm(
    query: str,
    criteria: List[str],
    title: str,
    snippet: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    ASTA-style semantic entailment judgment for ONE document.

    Output schema:
    {
      "relevance": one of the 4 labels,
      "relevant_snippet": "...",
      "criteria_judgements": { criterion: true/false, ... }
    }
    """
    global _openai_client

    system_prompt = (
        "You are a scientific relevance judge.\n"
        "Determine whether a paper is semantically relevant to the user's query.\n\n"
        "This is NOT a similarity score. Judge semantic entailment.\n\n"
        "Return ONLY valid JSON with keys:\n"
        '  "relevance": "Perfectly Relevant" | "Highly Relevant" | "Somewhat Relevant" | "Not Relevant"\n'
        '  "relevant_snippet": string (a short excerpt/phrase from the provided snippet)\n'
        '  "criteria_judgements": object mapping each criterion string -> true/false\n'
        "If criteria are empty, still include criteria_judgements as an empty object.\n"
    )

    payload = {
        "query": query,
        "relevance_criteria": criteria,
        "paper_title": title,
        "paper_snippet": (snippet or "")[:2000],
    }

    if debug:
        print("\n" + "-" * 80)
        print("[Semantic Relevance] LLM judge payload:")
        print(json.dumps(payload, indent=2)[:4000])
        print("-" * 80)

    try:
        resp = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.0,
            max_tokens=280,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        if debug:
            print(f"[Semantic Relevance] LLM call failed: {e}")
        return {
            "relevance": RelevanceLabels.NOT_RELEVANT,
            "relevant_snippet": "",
            "criteria_judgements": {},
            "error": str(e),
        }

    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, dict):
        # Fall back: infer label from text
        label = _normalize_llm_relevance_label(raw)
        return {
            "relevance": label,
            "relevant_snippet": "",
            "criteria_judgements": {},
            "raw": raw[:800],
            "parse_error": True,
        }

    # Normalize and harden fields
    label = _normalize_llm_relevance_label(parsed.get("relevance"))
    snippet_out = parsed.get("relevant_snippet") or ""
    cj = parsed.get("criteria_judgements") or {}
    if not isinstance(cj, dict):
        cj = {}

    return {
        "relevance": label,
        "relevant_snippet": str(snippet_out)[:400],
        "criteria_judgements": cj,
    }


def _apply_semantic_relevance_to_top_sources(
    query: str,
    analysis: Optional[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    snippet_by_paper_id: Dict[str, str],
    debug: bool = False,
    top_n: int = DEFAULT_JUDGE_TOP_N,
) -> None:
    """
    Apply ASTA Layer 2 to top N sources only, in-place.
    """
    if not sources:
        return

    criteria = _extract_criteria_names(analysis)
    top_n = min(top_n, len(sources))

    for i in range(top_n):
        s = sources[i]
        pid = s.get("paper_id") or ""
        title = s.get("title") or ""
        snippet = snippet_by_paper_id.get(pid) or s.get("relevance_summary") or ""

        judgement = _judge_relevance_llm(
            query=query,
            criteria=criteria,
            title=title,
            snippet=snippet,
            debug=debug,
        )
        label = judgement.get("relevance", RelevanceLabels.NOT_RELEVANT)
        label = _normalize_llm_relevance_label(label)
        sem_score = RELEVANCE_LABEL_TO_SCORE.get(label, 0)

        rel = s.get("relevance") or {}
        rel["semantic"] = {
            "label": label,
            "score": sem_score,
            "relevant_snippet": judgement.get("relevant_snippet", ""),
            "criteria_judgements": judgement.get("criteria_judgements", {}),
        }
        s["relevance"] = rel

        # Clip final
        _finalize_relevance(s)


# =============================================================================
# Initialization / metadata discovery
# =============================================================================
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
    """
    global _es_client, _metadata_index_cache

    if not _es_client:
        return {}

    metadata_index = _discover_metadata_index()
    if not metadata_index:
        if debug:
            print("[Metadata] No metadata index available")
        return {}

    if not paper_ids:
        return {}

    unique_paper_ids = list(set(paper_ids))

    try:
        results = {}

        # Try mget by doc IDs first
        try:
            mget_response = _es_client.mget(
                index=metadata_index,
                body={"ids": unique_paper_ids},
                _source=["paper_id", "title", "authors", "year", "categories"]
            )

            for doc in mget_response.get("docs", []):
                if doc.get("found"):
                    source = doc.get("_source", {})
                    paper_id = source.get("paper_id") or doc.get("_id")
                    if paper_id:
                        results[paper_id] = {
                            "title": source.get("title", ""),
                            "authors": source.get("authors", ""),
                            "year": source.get("year"),
                            "categories": source.get("categories", [])
                        }

            if len(results) == len(unique_paper_ids):
                if debug:
                    print(f"[Metadata] Fetched metadata for {len(results)}/{len(unique_paper_ids)} papers via mget")
                return results
        except Exception as e:
            if debug:
                print(f"[Metadata] mget failed, trying search query: {e}")

        # Fall back to terms query on paper_id field
        query = {
            "query": {"terms": {"paper_id": unique_paper_ids}},
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
            response = _es_client.search(index=metadata_index, body=query, timeout="10s")

        hits = response.get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
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
    """Rewrite the current user question into a standalone, retrieval-optimized query using recent conversation history."""
    if not conversation_history:
        return query

    recent_history = conversation_history[-6:]

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into standalone search queries. "
                    "Preserve all specific entities, acronyms, methods, and constraints. "
                    "Resolve pronouns to explicit referents from the conversation. "
                    "Output only the rewritten query without commentary."
                ),
            }
        ]
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

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

        return rewritten or query

    except Exception as e:
        if debug:
            print(f"[Retrieval] Query rewrite failed, falling back. Error: {e}")
        return query


# =============================================================================
# Non-streaming RAG: dedup papers + relevance scoring + OPTIONAL semantic LLM judgement (top-5)
# =============================================================================
def get_rag_response(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    debug: bool = False,
    conversation_history: List[Dict] = None,
    user_filters: Dict = None
) -> Tuple[str, List[Dict]]:
    """
    Get a RAG response for a query with optional conversation history.

    Returns:
        Tuple of (answer_text, sources_list)
    """
    if not initialize_rag_system():
        return "RAG system not properly initialized. Please check the logs.", []

    if not all([_es_client, _openai_client]):
        return "RAG system not properly initialized. Please check the logs.", []

    try:
        # Analyze query once (for ASTA-style gating + criteria extraction)
        query_analysis = analyze_query(query)

        retrieval_query = _rewrite_query_with_history(query, conversation_history, debug=debug)

        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text^3", "title^2", "authors", "authors.keyword"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "multi_match": {
                                "query": retrieval_query,
                                "fields": ["chunk_text^3", "title^2", "authors", "authors.keyword"],
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

        filters = []

        # AUTHORS FILTER
        if user_filters and user_filters.get("authors"):
            filters.append({"terms": {"authors.keyword": user_filters["authors"]}})

        # YEAR FILTER
        year_filters = {}
        if user_filters and user_filters.get("yearStart") is not None:
            year_filters["gte"] = user_filters["yearStart"]
        if user_filters and user_filters.get("yearEnd") is not None:
            year_filters["lte"] = user_filters["yearEnd"]

        if year_filters:
            filters.append({"range": {"year": year_filters}})

        if filters:
            search_body["query"]["bool"]["filter"] = filters

        if debug:
            print(f"\n[Retrieval] Searching Elasticsearch with query: {retrieval_query}")
            print(f"[Retrieval] Index: {ES_INDEX}, Top K: {top_k}")

        # Search
        try:
            response = _es_client.search(
                index=ES_INDEX,
                query=search_body["query"],
                size=search_body["size"],
                _source=search_body["_source"],
                timeout="60s"
            )
        except (TypeError, KeyError):
            search_body_with_timeout = search_body.copy()
            search_body_with_timeout["timeout"] = "60s"
            response = _es_client.search(index=ES_INDEX, body=search_body_with_timeout)

        raw_hits = response.get("hits", {}).get("hits", [])

        if debug:
            print(f"[Retrieval] Elasticsearch returned {len(raw_hits)} results")

        if not raw_hits:
            if debug:
                print("[Retrieval] WARNING: No results found in Elasticsearch!")
            raw_hits = []

        # Deduplicate by paper_id (keep best scoring chunk per paper)
        best_hit_by_paper: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for hit in raw_hits:
            src = hit.get("_source", {}) or {}
            paper_id = src.get("paper_id", "unknown")
            score = float(hit.get("_score", 0.0))
            if paper_id not in best_hit_by_paper or score > best_hit_by_paper[paper_id][0]:
                best_hit_by_paper[paper_id] = (score, hit)

        deduped_hits = [h for (_, h) in best_hit_by_paper.values()]
        deduped_hits.sort(key=lambda h: float(h.get("_score", 0.0)), reverse=True)

        # top_k papers (not chunks)
        hits = deduped_hits[:top_k]

        max_score = max((float(h.get("_score", 0.0)) for h in hits), default=0.0)

        # metadata lookup
        paper_ids_for_metadata = []
        for hit in hits:
            source_data = hit.get("_source", {}) or {}
            paper_id = source_data.get("paper_id", "unknown")
            if paper_id and paper_id != "unknown":
                paper_ids_for_metadata.append(paper_id)

        metadata_dict = _get_paper_metadata(paper_ids_for_metadata, debug=debug)

        retrieved_texts: List[str] = []
        sources: List[Dict] = []
        snippet_by_pid: Dict[str, str] = {}

        for rank, hit in enumerate(hits, 1):
            source_data = hit.get("_source", {}) or {}
            chunk_text = source_data.get("chunk_text", "") or ""
            retrieved_texts.append(chunk_text)

            paper_id = source_data.get("paper_id", "unknown")
            chunk_index = int(source_data.get("chunk_index", 0))
            similarity_score = float(hit.get("_score", 0.0))

            # Save snippet for semantic judge
            if paper_id and paper_id != "unknown" and chunk_text:
                snippet_by_pid[paper_id] = chunk_text[:2000]

            # title/authors from metadata first
            title = ""
            authors = ""
            if paper_id in metadata_dict:
                metadata = metadata_dict[paper_id]
                title = str(metadata.get("title", "") or "").strip()
                authors = _normalize_authors(metadata.get("authors", ""))

            if not title:
                title = str(source_data.get("title", "") or "").strip()
            if not authors:
                authors = _normalize_authors(source_data.get("authors", ""))

            arxiv_url = None
            if paper_id and not paper_id.startswith("http"):
                if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                    arxiv_url = f"https://arxiv.org/abs/{paper_id}"

            src_obj = {
                "paper_id": paper_id,
                "title": title if title else f"Paper {paper_id}",
                "authors": authors if authors else None,
                "year": source_data.get("year"),
                "chunk_index": chunk_index,
                "rank": rank,
                "similarity_score": similarity_score,
                "url": arxiv_url
            }

            # Attach ASTA-style retrieval relevance (normalize by max_score)
            norm = (similarity_score / max_score) if max_score > 0 else 0.0
            _attach_retrieval_relevance(src_obj, norm)

            sources.append(src_obj)

        # OPTIONAL: Apply ASTA Layer 2 semantic judgement ONLY for top-5 docs
        if _should_apply_semantic_judgement(query_analysis, user_filters):
            if debug:
                print("[Semantic Relevance] Applying ASTA Layer 2 semantic judgement to top sources...")
            _apply_semantic_relevance_to_top_sources(
                query=query,
                analysis=query_analysis,
                sources=sources,
                snippet_by_paper_id=snippet_by_pid,
                debug=debug,
                top_n=DEFAULT_JUDGE_TOP_N,
            )
        else:
            # No semantic judge: finalize relevance = retrieval
            for s in sources:
                _finalize_relevance(s)

        if debug:
            print(f"[Retrieval] Retrieved {len(retrieved_texts)} chunks, total context length: {sum(len(t) for t in retrieved_texts)} chars")

        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

        messages = [
            {"role": "system", "content": "You are a paper finder assistant. Your primary role is to provide a very concise, high-level summary of what the retrieved academic papers are about. Keep the user's query in mind, but focus on giving a brief overview (one to two paragraphs maximum) of the key topics and contributions across the papers. Do NOT cite papers inline (no [Paper Title] or similar citations). The papers are already listed separately as sources. Use conversation history to provide context-aware responses. When including mathematics, write formulas in LaTeX and delimit inline math with \\( ... \\) and display math with \\[ ... \\]. Do not wrap LaTeX in code blocks and do not escape backslashes. Do NOT use headings, tables, or section numbering. Keep formatting simple: short paragraphs, bullet lists ( - item ), and optional **bold**/*italics* only."}
        ]

        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append(msg)

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

        response = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

        answer = response.choices[0].message.content

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
        paper_id = source.get("paper_id", "unknown")
        title = source.get("title", "No title")
        authors = source.get("authors")

        rel = source.get("relevance") or {}
        final_label = rel.get("label")
        final_score = rel.get("score")
        retrieval = rel.get("retrieval") or {}
        semantic = rel.get("semantic") or {}

        # compact display: final + (retrieval/semantic)
        source_line = f"{source.get('rank', '?')}. [{paper_id}] {title}"
        if final_label is not None and final_score is not None:
            source_line += f" — {final_label} ({final_score}/3)"

        r_label = retrieval.get("label")
        r_score = retrieval.get("score")
        s_label = semantic.get("label")
        s_score = semantic.get("score")

        if r_label is not None and r_score is not None:
            source_line += f" [retrieval: {r_label} {r_score}/3]"
        if s_label is not None and s_score is not None:
            source_line += f" [semantic: {s_label} {s_score}/3]"

        if authors:
            source_line += f" - {authors}"

        url = source.get("url")
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
            index_count = stats["count"]
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

    criteria = analysis.get("relevance_criteria", [])
    if criteria:
        names = [c.get("name") for c in criteria if isinstance(c, dict) and c.get("name")]
        if names:
            lines.append("Key topics: " + ", ".join(names))

    authors = analysis.get("authors", [])
    if authors:
        lines.append("Authors: " + ", ".join(authors))

    venues = analysis.get("venues", [])
    if venues:
        lines.append("Venues: " + ", ".join(venues))

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
                    f"Return exactly {num_queries} keyword queries, one per line, without numbering or bullets. "
                    "The FIRST query must be the single best representation of the user's core intent and should be the one most likely to retrieve the perfect answer. "
                    "Each query should be optimized for retrieval - use keywords and phrases, not complete sentences."
                ),
            }
        ]

        recent_history = []
        if conversation_history:
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

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
            print("\n" + "=" * 60)
            print("GENERATING REWORDED QUERIES")
            print("=" * 60)
            print(f"Original query: {query}")
            if conversation_history:
                print(f"Using {len(recent_history)} messages from conversation history")

        response = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content.strip()
        queries = [q.strip() for q in raw_output.split("\n") if q.strip()]

        if len(queries) < num_queries:
            queries.append(query)
            while len(queries) < num_queries:
                queries.append(f"{query} (alternative perspective)")

        return queries[:num_queries]

    except Exception as e:
        if debug:
            print(f"Error generating reworded queries: {e}")
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
                            "fields": ["chunk_text^3", "title^2", "authors", "authors.keyword"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_text^3", "title^2", "authors", "authors.keyword"],
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
        response = _es_client.search(
            index=ES_INDEX,
            query=search_body["query"],
            size=search_body["size"],
            _source=search_body["_source"],
            timeout="60s"
        )
    except (TypeError, KeyError):
        search_body_with_timeout = search_body.copy()
        search_body_with_timeout["timeout"] = "60s"
        response = _es_client.search(index=ES_INDEX, body=search_body_with_timeout)

    hits = response.get("hits", {}).get("hits", [])
    return hits


# =============================================================================
# Summarization + overlap removal (unchanged)
# =============================================================================
def _summarize_paper(full_text: str, target_ratio: float = 0.15, debug: bool = False) -> str:
    """
    Summarize a full paper to exactly 500 words as a single paragraph using a lightweight LLM.
    """
    global _openai_client

    if not _openai_client:
        if debug:
            print("[Summarization] OpenAI client not initialized")
        return full_text

    if not full_text or len(full_text) < 200:
        return full_text

    target_word_count = 500
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

        estimated_tokens = int(target_word_count * 1.5)
        max_tokens = min(4000, max(800, estimated_tokens))

        response = _openai_client.chat.completions.create(
            model=SUMMARIZATION_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3
        )

        choice = response.choices[0]
        content = choice.message.content
        summarized_text = content.strip() if isinstance(content, str) else str(content).strip()
        if not summarized_text:
            return full_text

        return summarized_text

    except Exception as e:
        if debug:
            print(f"[Summarization] Error summarizing paper: {e}")
            import traceback
            traceback.print_exc()
        return full_text


def _remove_repeated_phrases(text: str, min_phrase_words: int = 10) -> str:
    if not text or len(text) < 100:
        return text

    words = text.split()
    if len(words) < min_phrase_words * 2:
        return text

    max_phrase_length = min(len(words) // 2, 100)
    max_iterations = 5
    iteration = 0

    result_text = text

    while iteration < max_iterations:
        iteration += 1
        found_repeat = False

        for phrase_length in range(max_phrase_length, min_phrase_words - 1, -1):
            phrase_positions = {}

            for start_idx in range(len(words) - phrase_length + 1):
                phrase = words[start_idx:start_idx + phrase_length]
                phrase_text = " ".join(phrase)

                if len(phrase_text) < 50:
                    continue

                occurrences = []
                search_start = 0

                while True:
                    pos = result_text.find(phrase_text, search_start)
                    if pos == -1:
                        break
                    occurrences.append(pos)
                    search_start = pos + 1

                if len(occurrences) > 1:
                    phrase_positions[phrase_text] = sorted(occurrences)
                    found_repeat = True
                    break

        if found_repeat and phrase_positions:
            phrase_text = list(phrase_positions.keys())[0]
            occurrences = phrase_positions[phrase_text]

            for i in range(len(occurrences) - 1, 0, -1):
                occ_pos = occurrences[i]
                before = result_text[:occ_pos]
                after = result_text[occ_pos + len(phrase_text):]

                before = before.rstrip()
                after = after.lstrip()

                if before and after:
                    if before[-1] not in ".!?;:\n" and after[0] not in ".!?;:,":
                        if not before.endswith(" "):
                            before += " "

                result_text = before + after

            words = result_text.split()
            if len(words) < min_phrase_words * 2:
                break
        else:
            break

    return result_text


def _remove_overlap_between_chunks(chunks: List[str], min_overlap_length: int = 50) -> List[str]:
    # (unchanged from your version)
    if not chunks or len(chunks) <= 1:
        return chunks

    deduplicated = []

    for i, curr_chunk in enumerate(chunks):
        if i == 0:
            deduplicated.append(curr_chunk)
            continue

        prev_chunk = deduplicated[-1]
        prev_chunk_clean = prev_chunk.strip()
        curr_chunk_clean = curr_chunk.strip()

        if not prev_chunk_clean or not curr_chunk_clean:
            deduplicated.append(curr_chunk)
            continue

        prev_words = prev_chunk_clean.split()
        curr_words = curr_chunk_clean.split()

        if not prev_words or not curr_words:
            deduplicated.append(curr_chunk)
            continue

        overlap_found = False
        overlap_num_words = 0
        overlap_text = ""

        max_possible_words = min(len(prev_words), len(curr_words))
        for num_words in range(max_possible_words, 0, -1):
            prev_end_words = prev_words[-num_words:]
            curr_start_words = curr_words[:num_words]

            if prev_end_words == curr_start_words:
                overlap_text_words = " ".join(curr_start_words)
                if len(overlap_text_words) >= min_overlap_length:
                    overlap_found = True
                    overlap_num_words = num_words
                    overlap_text = overlap_text_words
                    break

        overlap_len_chars = 0
        if not overlap_found:
            prev_len = len(prev_chunk_clean)
            curr_len = len(curr_chunk_clean)
            max_possible_overlap = min(prev_len, curr_len, 2000)

            for test_len in range(max_possible_overlap, min_overlap_length - 1, -10):
                prev_suffix = prev_chunk_clean[-test_len:]
                curr_prefix = curr_chunk_clean[:test_len]

                if prev_suffix == curr_prefix:
                    overlap_found = True
                    overlap_len_chars = test_len
                    overlap_text = prev_suffix
                    break

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

        if overlap_found:
            overlap_pos = curr_chunk.find(overlap_text)
            cut_position = None

            if overlap_pos == -1:
                search_start = max(0, len(curr_chunk) - len(curr_chunk_clean) - 100)
                overlap_pos = curr_chunk.find(overlap_text, search_start)

            if overlap_pos == -1:
                overlap_text_normalized = re.sub(r"\s+", " ", overlap_text).strip()
                curr_chunk_normalized = re.sub(r"\s+", " ", curr_chunk).strip()
                normalized_pos = curr_chunk_normalized.find(overlap_text_normalized)
                if normalized_pos >= 0:
                    char_count = 0
                    for j, char in enumerate(curr_chunk):
                        if char not in " \n\t":
                            char_count += 1
                        if char_count > normalized_pos:
                            overlap_pos = j
                            break

            if overlap_pos >= 0:
                if cut_position is None:
                    cut_position = overlap_pos + len(overlap_text)

                while cut_position < len(curr_chunk) and curr_chunk[cut_position] in " \n\t":
                    cut_position += 1

                remaining_text = curr_chunk[cut_position:].strip()
                if remaining_text:
                    prev_end = prev_chunk.rstrip()
                    if prev_end and not prev_end[-1] in ".!?;:\n":
                        if not remaining_text[0] in ".!?;:," and not prev_end.endswith(" "):
                            remaining_text = " " + remaining_text

                    deduplicated.append(remaining_text)
            else:
                deduplicated.append(curr_chunk)
        else:
            prev_end = prev_chunk.rstrip()
            if prev_end and curr_chunk_clean:
                if (prev_end[-1] not in ".!?;:\n" and curr_chunk_clean[0] not in ".!?;:," and not prev_end.endswith(" ")):
                    deduplicated.append(" " + curr_chunk)
                else:
                    deduplicated.append(curr_chunk)
            else:
                deduplicated.append(curr_chunk)

    return deduplicated


# =============================================================================
# Full-paper support + filtering utilities (unchanged from your file)
# =============================================================================
def _get_all_chunks_for_papers(paper_ids: List[str], debug: bool = False) -> Dict[Tuple[str, int], Dict]:
    global _es_client

    if not _es_client or not paper_ids:
        return {}

    unique_paper_ids = list(set(paper_ids))
    all_chunks = {}

    try:
        query = {
            "query": {
                "terms": {
                    "paper_id": unique_paper_ids
                }
            },
            "size": 10000,
            "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count", "year"]
        }

        if debug:
            print(f"[Full Paper] Querying Elasticsearch for {len(unique_paper_ids)} papers: {unique_paper_ids}")

        try:
            response = _es_client.search(
                index=ES_INDEX,
                query=query["query"],
                size=query["size"],
                _source=query["_source"],
                timeout="60s"
            )
        except (TypeError, KeyError):
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

        sorted_chunk_items = sorted(all_chunks.items(), key=lambda x: (x[0][0], x[0][1]))
        all_chunks = dict(sorted_chunk_items)

        return all_chunks

    except Exception as e:
        if debug:
            print(f"[Full Paper] Error retrieving all chunks: {e}")
        return {}


def _extract_year_from_paper_id(paper_id: str) -> Optional[int]:
    if not paper_id:
        return None

    try:
        paper_id_clean = paper_id.split("v")[0]

        if "." in paper_id_clean:
            parts = paper_id_clean.split(".")
            if len(parts) >= 1 and len(parts[0]) == 4 and parts[0].isdigit():
                year = int(parts[0])
                if 1990 <= year <= 2100:
                    return year

        match = re.search(r"(\d{2})(\d{2})", paper_id_clean)
        if match:
            yy = int(match.group(1))
            mm = int(match.group(2))
            if 1 <= mm <= 12:
                return 2000 + yy if yy < 50 else 1900 + yy
    except (ValueError, IndexError, AttributeError):
        pass

    return None


def _get_chunk_year(chunk_data: Dict) -> Optional[int]:
    source_data = chunk_data.get("source_data", {})
    year = source_data.get("year")
    if year is not None:
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2100:
                return year_int
        except (ValueError, TypeError):
            pass

    paper_id = source_data.get("paper_id", "")
    return _extract_year_from_paper_id(paper_id)


def _filter_chunks_by_analysis(all_chunks_dict: Dict, analysis: Dict[str, Any], debug: bool = False) -> Dict:
    if not analysis or analysis.get("status") != "success":
        if debug:
            print("[Filtering] Analysis not available or failed, skipping filtering")
        return all_chunks_dict

    time_range = analysis.get("time_range", {})
    requested_authors = analysis.get("authors", [])
    requested_venues = analysis.get("venues", [])

    has_year_filter = time_range.get("start") is not None or time_range.get("end") is not None
    has_author_filter = len(requested_authors) > 0
    has_venue_filter = len(requested_venues) > 0

    if not (has_year_filter or has_author_filter or has_venue_filter):
        if debug:
            print("[Filtering] No filtering criteria found in analysis, keeping all chunks")
        return all_chunks_dict

    filtered_chunks = {}
    year_start = time_range.get("start")
    year_end = time_range.get("end")

    for key, chunk_data in all_chunks_dict.items():
        source_data = chunk_data.get("source_data", {})
        should_keep = True

        if has_year_filter:
            chunk_year = _get_chunk_year(chunk_data)
            if chunk_year is not None:
                if year_start and chunk_year < year_start:
                    should_keep = False
                if year_end and chunk_year > year_end:
                    should_keep = False

        if should_keep and has_author_filter:
            chunk_authors = source_data.get("authors", []) or []
            chunk_authors_lower = [str(a).lower() for a in chunk_authors]
            requested_authors_lower = [str(a).lower() for a in requested_authors]

            author_match = any(
                any(req in chunk_author for chunk_author in chunk_authors_lower)
                for req in requested_authors_lower
            )
            if not author_match:
                should_keep = False

        if should_keep and has_venue_filter:
            # Venue filtering not implemented (left as in your version)
            pass

        if should_keep:
            filtered_chunks[key] = chunk_data

    return filtered_chunks


def _fetch_arxiv_papers_by_author(author_names: List[str], max_results_per_author: int = 5, debug: bool = False) -> List[Dict]:
    if not author_names:
        return []

    all_papers = []
    arxiv_ns = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

    for author_name in author_names:
        if not author_name or not author_name.strip():
            continue

        try:
            search_query = f'au:"{author_name.strip()}"'
            url = "http://export.arxiv.org/api/query"
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

            root = ET.fromstring(response.text)
            entries = root.findall("atom:entry", arxiv_ns)

            for entry in entries:
                paper_id_elem = entry.find("atom:id", arxiv_ns)
                if paper_id_elem is None:
                    continue

                paper_id = paper_id_elem.text.split("/")[-1] if paper_id_elem.text else None
                if not paper_id:
                    continue

                title_elem = entry.find("atom:title", arxiv_ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

                author_elems = entry.findall("atom:author", arxiv_ns)
                authors_list = []
                for author_elem in author_elems:
                    name_elem = author_elem.find("atom:name", arxiv_ns)
                    if name_elem is not None and name_elem.text:
                        authors_list.append(name_elem.text.strip())
                authors = ", ".join(authors_list) if authors_list else ""

                summary_elem = entry.find("atom:summary", arxiv_ns)
                abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""

                published_elem = entry.find("atom:published", arxiv_ns)
                published_date = published_elem.text.split("T")[0] if published_elem is not None and published_elem.text else None
                year = None
                if published_date and len(published_date) >= 4:
                    try:
                        year = int(published_date.split("-")[0])
                    except (ValueError, AttributeError):
                        year = None

                chunk_text = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
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
                    "chunk_index": 0,
                    "source": "arxiv_api"
                }

                all_papers.append(paper_data)

        except Exception as e:
            if debug:
                print(f"[arXiv] Error fetching papers for author '{author_name}': {e}")
            continue

    return all_papers


def _reciprocal_rank_fusion(rank_lists: List[List[Tuple[str, int]]], k: int = 60) -> Dict[Tuple[str, int], float]:
    scores: Dict[Tuple[str, int], float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, (paper_id, chunk_index) in enumerate(ranks, start=1):
            key = (paper_id, chunk_index)
            scores[key] += 1.0 / (k + rank)
    return scores


# =============================================================================
# Streaming RAG: add dedup-by-paper + retrieval relevance + OPTIONAL semantic LLM judgement (top-5)
# =============================================================================
def stream_rag_response(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    debug: bool = False,
    conversation_history: List[Dict] = None,
    user_filters: Dict = None
):
    """Generator that streams the LLM answer tokens and finally emits sources."""
    if not initialize_rag_system() or not all([_es_client, _openai_client]):
        yield f"data: {json.dumps({'event': 'error', 'message': 'RAG not initialized'})}\n\n"
        return

    try:
        yield f"data: {json.dumps({'event': 'status', 'message': 'Generating reformulated queries...'})}\n\n"
        reworded_queries = _generate_reworded_queries(query, conversation_history, num_queries=10, debug=debug)

        yield f"data: {json.dumps({'event': 'status', 'message': 'Retrieving best chunks...'})}\n\n"

        all_rank_lists = []
        all_chunks_dict = {}

        for reworded_query in reworded_queries:
            hits = _search_elasticsearch(reworded_query, top_k=20, debug=debug)

            rank_list = []
            for hit in hits:
                source_data = hit.get("_source", {}) or {}
                paper_id = source_data.get("paper_id", "unknown")
                chunk_index = int(source_data.get("chunk_index", 0))
                key = (paper_id, chunk_index)

                rank_list.append(key)

                if key not in all_chunks_dict:
                    all_chunks_dict[key] = {
                        "source_data": source_data,
                        "score": float(hit.get("_score", 0.0)),
                        "hit": hit
                    }

            all_rank_lists.append(rank_list)

        total_before_dedup = sum(len(rank_list) for rank_list in all_rank_lists)
        unique_chunks = len(all_chunks_dict)
        duplicates_removed = total_before_dedup - unique_chunks

        if unique_chunks == 0:
            no_data_msg = "No relevant chunks could be retrieved for your request. Please try rephrasing the question."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return

        # Query analysis (used for filtering + semantic judgement gate)
        query_analysis = analyze_query(query)

        # Merge filters into analysis (as in your original)
        if user_filters:
            if user_filters.get("yearStart") is not None or user_filters.get("yearEnd") is not None:
                if not query_analysis.get("time_range"):
                    query_analysis["time_range"] = {"start": None, "end": None}
                if user_filters.get("yearStart") is not None:
                    query_analysis["time_range"]["start"] = user_filters["yearStart"]
                if user_filters.get("yearEnd") is not None:
                    query_analysis["time_range"]["end"] = user_filters["yearEnd"]

            if user_filters.get("authors") and len(user_filters["authors"]) > 0:
                existing_authors = query_analysis.get("authors", [])
                merged_authors = list(set(existing_authors + user_filters["authors"]))
                query_analysis["authors"] = merged_authors

            if user_filters.get("venues") and len(user_filters["venues"]) > 0:
                existing_venues = query_analysis.get("venues", [])
                merged_venues = list(set(existing_venues + user_filters["venues"]))
                query_analysis["venues"] = merged_venues

            if user_filters.get("queryType"):
                query_analysis["query_type"] = user_filters["queryType"]

        # Optional arXiv author fetch (unchanged)
        arxiv_papers = []
        authors_list = query_analysis.get("authors", []) or []
        if authors_list:
            arxiv_papers = _fetch_arxiv_papers_by_author(authors_list, max_results_per_author=5, debug=debug)

        # Filter chunks by analysis (unchanged)
        all_chunks_dict = _filter_chunks_by_analysis(all_chunks_dict, query_analysis, debug=debug)

        filtered_rank_lists = []
        for rank_list in all_rank_lists:
            filtered_rank_list = [key for key in rank_list if key in all_chunks_dict]
            if filtered_rank_list:
                filtered_rank_lists.append(filtered_rank_list)
        all_rank_lists = filtered_rank_lists

        if len(all_chunks_dict) == 0 and len(arxiv_papers) == 0:
            no_data_msg = "No relevant chunks matched the specified criteria (year, author, etc.). Please try adjusting your filters."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return

        sorted_chunks = []
        if len(all_chunks_dict) > 0 and len(all_rank_lists) > 0:
            rrf_scores = _reciprocal_rank_fusion(all_rank_lists, k=60)
            sorted_chunks = sorted(
                [(key, score) for key, score in rrf_scores.items() if key in all_chunks_dict],
                key=lambda x: x[1],
                reverse=True
            )

        if len(sorted_chunks) == 0 and len(arxiv_papers) == 0:
            no_data_msg = "No relevant chunks matched the specified criteria and no papers found from authors. Please try adjusting your filters."
            yield f"data: {json.dumps({'event': 'delta', 'text': no_data_msg})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []})}\n\n"
            return

        retrieved_texts = []
        sources = []
        best_query = reworded_queries[0] if reworded_queries else query
        context_chunk_count = 0

        paper_ids_for_metadata = []
        final_paper_ids = []

        # Collect final_paper_ids up to 5
        if len(all_chunks_dict) > 0 and len(sorted_chunks) > 0:
            for (paper_id, chunk_index), _ in sorted_chunks[:25]:
                if paper_id and paper_id != "unknown":
                    if paper_id not in final_paper_ids:
                        final_paper_ids.append(paper_id)
                        paper_ids_for_metadata.append(paper_id)
                if len(final_paper_ids) >= 5:
                    break

        for arxiv_paper in arxiv_papers:
            paper_id = arxiv_paper.get("paper_id")
            if paper_id and paper_id != "unknown" and paper_id not in final_paper_ids and len(final_paper_ids) < 5:
                final_paper_ids.append(paper_id)
                paper_ids_for_metadata.append(paper_id)

        final_paper_ids = final_paper_ids[:5]
        full_paper_processing = bool(user_filters and user_filters.get("fullPaperProcessing", False))

        # Full paper processing logic (your original) — unchanged structure, we just ensure relevance attached later
        max_rrf_in_sources = 0.0
        snippet_by_pid: Dict[str, str] = {}

        metadata_dict = _get_paper_metadata(paper_ids_for_metadata, debug=debug)

        if len(all_chunks_dict) > 0:
            if full_paper_processing and final_paper_ids:
                public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
                os.makedirs(public_dir, exist_ok=True)

                yield f"data: {json.dumps({'event': 'status', 'message': 'Summarizing papers...'})}\n\n"

                for rank, paper_id in enumerate(final_paper_ids[:5], 1):
                    paper_text_parts = []

                    paper_chunks = [(k, v) for k, v in all_chunks_dict.items() if k[0] == paper_id]
                    if paper_chunks:
                        paper_chunks.sort(key=lambda x: x[0][1])
                        for (pid, chunk_index), chunk_data in paper_chunks:
                            chunk_text = chunk_data["source_data"].get("chunk_text", "") or ""
                            if chunk_text:
                                paper_text_parts.append(chunk_text)

                        if len(paper_text_parts) > 1:
                            paper_text_parts = _remove_overlap_between_chunks(paper_text_parts, min_overlap_length=50)
                    else:
                        for arxiv_paper in arxiv_papers:
                            if arxiv_paper.get("paper_id") == paper_id:
                                chunk_text = arxiv_paper.get("chunk_text", "") or ""
                                if chunk_text:
                                    paper_text_parts.append(chunk_text)
                                break

                    if paper_text_parts:
                        full_paper_text = "\n\n".join(paper_text_parts)
                        full_paper_text = _remove_repeated_phrases(full_paper_text, min_phrase_words=10)

                        filename = os.path.join(public_dir, f"{rank}.txt")
                        try:
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(full_paper_text)
                        except Exception:
                            pass

                        summarized_text = _summarize_paper(full_paper_text, target_ratio=0.15, debug=debug)

                        summary_filename = os.path.join(public_dir, f"{rank}_summ.txt")
                        try:
                            with open(summary_filename, "w", encoding="utf-8") as f:
                                f.write(summarized_text)
                        except Exception:
                            pass

                for rank in range(1, min(6, len(final_paper_ids) + 1)):
                    summary_filename = os.path.join(public_dir, f"{rank}_summ.txt")
                    try:
                        if os.path.exists(summary_filename):
                            with open(summary_filename, "r", encoding="utf-8") as f:
                                summary_text = f.read().strip()
                            if summary_text:
                                retrieved_texts.append(summary_text)
                                context_chunk_count += 1
                    except Exception:
                        pass

                # Sources: one per paper, best RRF per paper
                seen_paper_ids = set()
                rank_counter = 1
                for paper_id in final_paper_ids:
                    if paper_id in seen_paper_ids:
                        continue
                    seen_paper_ids.add(paper_id)

                    paper_chunks = [(k, v) for k, v in all_chunks_dict.items() if k[0] == paper_id]
                    if paper_chunks:
                        paper_chunks.sort(key=lambda x: x[0][1])
                        (first_paper_id, first_chunk_index), first_chunk_data = paper_chunks[0]
                        source_data = first_chunk_data["source_data"]

                        best_similarity_score = max((chunk_data["score"] for _, chunk_data in paper_chunks), default=0.0)
                        best_rrf_score = 0.0
                        for (pid, cidx), rrf_score in sorted_chunks:
                            if pid == paper_id:
                                best_rrf_score = max(best_rrf_score, rrf_score)

                        title = ""
                        authors = ""
                        if paper_id in metadata_dict:
                            md = metadata_dict[paper_id]
                            title = str(md.get("title", "") or "").strip()
                            authors = _normalize_authors(md.get("authors", ""))

                        if not title:
                            title = str(source_data.get("title", "") or "").strip()
                        if not authors:
                            authors = _normalize_authors(source_data.get("authors", ""))

                        arxiv_url = None
                        if paper_id and not paper_id.startswith("http"):
                            if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                                arxiv_url = f"https://arxiv.org/abs/{paper_id}"

                        src_obj = {
                            "paper_id": paper_id,
                            "title": title if title else f"Paper {paper_id}",
                            "authors": authors if authors else None,
                            "year": source_data.get("year"),
                            "chunk_index": first_chunk_index,
                            "rank": rank_counter,
                            "rrf_score": best_rrf_score,
                            "similarity_score": best_similarity_score,
                            "url": arxiv_url
                        }
                        sources.append(src_obj)
                        max_rrf_in_sources = max(max_rrf_in_sources, float(best_rrf_score))
                        # snippet for semantic judge: use first chunk if available
                        snippet_by_pid[paper_id] = (source_data.get("chunk_text", "") or "")[:2000]
                        rank_counter += 1
            else:
                # Normal processing: top 5 unique papers from sorted_chunks
                if sorted_chunks:
                    seen_paper_ids = set()
                    rank_counter = 1
                    for ((paper_id, chunk_index), rrf_score) in sorted_chunks:
                        if len(sources) >= 5:
                            break
                        if paper_id in seen_paper_ids:
                            continue
                        seen_paper_ids.add(paper_id)

                        chunk_data = all_chunks_dict.get((paper_id, chunk_index))
                        if not chunk_data:
                            continue

                        source_data = chunk_data["source_data"]
                        chunk_text = source_data.get("chunk_text", "") or ""
                        if chunk_text:
                            retrieved_texts.append(chunk_text)
                            context_chunk_count += 1
                            snippet_by_pid[paper_id] = chunk_text[:2000]

                        title = ""
                        authors = ""
                        if paper_id in metadata_dict:
                            md = metadata_dict[paper_id]
                            title = str(md.get("title", "") or "").strip()
                            authors = _normalize_authors(md.get("authors", ""))

                        if not title:
                            title = str(source_data.get("title", "") or "").strip()
                        if not authors:
                            authors = _normalize_authors(source_data.get("authors", ""))

                        arxiv_url = None
                        if paper_id and not paper_id.startswith("http"):
                            if len(paper_id) >= 4 and paper_id.replace(".", "").replace("v", "").replace("/", "").isdigit():
                                arxiv_url = f"https://arxiv.org/abs/{paper_id}"

                        src_obj = {
                            "paper_id": paper_id,
                            "title": title if title else f"Paper {paper_id}",
                            "authors": authors if authors else None,
                            "year": source_data.get("year"),
                            "chunk_index": chunk_index,
                            "rank": rank_counter,
                            "rrf_score": rrf_score,
                            "similarity_score": chunk_data["score"],
                            "url": arxiv_url
                        }
                        sources.append(src_obj)
                        max_rrf_in_sources = max(max_rrf_in_sources, float(rrf_score))
                        rank_counter += 1

        # Add arXiv papers to fill remaining slots up to 5 (unchanged)
        seen_paper_ids_from_sources = {s.get("paper_id") for s in sources}
        remaining_slots = max(0, 5 - len(sources))
        arxiv_paper_count = min(remaining_slots, len(arxiv_papers))

        arxiv_added_count = 0
        arxiv_start_rank = len(sources) + 1
        for idx, arxiv_paper in enumerate(arxiv_papers[:arxiv_paper_count], start=arxiv_start_rank):
            paper_id = arxiv_paper.get("paper_id", "unknown")
            if paper_id in seen_paper_ids_from_sources:
                continue
            seen_paper_ids_from_sources.add(paper_id)

            chunk_text = arxiv_paper.get("chunk_text", "") or ""
            if chunk_text:
                retrieved_texts.append(chunk_text)
                snippet_by_pid[paper_id] = chunk_text[:2000]
                context_chunk_count += 1

            title = str(arxiv_paper.get("title", "") or "").strip()
            authors = _normalize_authors(arxiv_paper.get("authors", ""))

            src_obj = {
                "paper_id": paper_id,
                "title": title if title else f"Paper {paper_id}",
                "authors": authors if authors else None,
                "year": arxiv_paper.get("year"),
                "chunk_index": 0,
                "rank": arxiv_start_rank + arxiv_added_count,
                "rrf_score": 0.0,
                "similarity_score": 0.0,
                "url": arxiv_paper.get("url"),
                "source": "arxiv_api"
            }
            sources.append(src_obj)
            arxiv_added_count += 1
            if len(sources) >= 5:
                break

        # Attach ASTA-style RETRIEVAL relevance for streaming sources (normalize by max_rrf_in_sources)
        for s in sources:
            rrf = float(s.get("rrf_score", 0.0) or 0.0)
            norm = (rrf / max_rrf_in_sources) if max_rrf_in_sources > 0 else 0.0
            _attach_retrieval_relevance(s, norm)

        # OPTIONAL: Apply ASTA Layer 2 semantic judgement only to top-5 returned docs
        if _should_apply_semantic_judgement(query_analysis, user_filters):
            yield f"data: {json.dumps({'event': 'status', 'message': 'Running semantic relevance judgement (top papers)...'})}\n\n"
            _apply_semantic_relevance_to_top_sources(
                query=query,
                analysis=query_analysis,
                sources=sources,
                snippet_by_paper_id=snippet_by_pid,
                debug=debug,
                top_n=DEFAULT_JUDGE_TOP_N,
            )
        else:
            for s in sources:
                _finalize_relevance(s)

        context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

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

        stream = _openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )

        # Emit sources (now includes retrieval + semantic + final)
        if sources:
            yield f"data: {json.dumps({'event': 'sources', 'sources': sources})}\n\n"

        yield f"data: {json.dumps({'event': 'status', 'message': 'Generating response...'})}\n\n"

        for event in stream:
            try:
                delta = event.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'event': 'delta', 'text': delta})}\n\n"
            except Exception as e:
                if debug:
                    print(f"[Streaming] Error processing delta: {e}")

        # Keep your relevance_summary generation unchanged
        if sources and len(sources) > 0:
            yield f"data: {json.dumps({'event': 'status', 'message': 'Generating relevancy explanations...'})}\n\n"

            source_chunks_dict = {}
            for source in sources[:5]:
                paper_id = source.get("paper_id")
                chunk_index = source.get("chunk_index", 0)
                key = (paper_id, chunk_index)

                chunk_text = ""
                if key in all_chunks_dict:
                    chunk_text = all_chunks_dict[key]["source_data"].get("chunk_text", "")
                else:
                    for arxiv_paper in arxiv_papers:
                        if arxiv_paper.get("paper_id") == paper_id:
                            chunk_text = arxiv_paper.get("chunk_text", "")
                            break

                if chunk_text:
                    source_chunks_dict[paper_id] = chunk_text

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
                    except Exception as e:
                        if debug:
                            print(f"  Error generating summary for {paper_id}: {e}")
                        source["relevance_summary"] = None
                else:
                    source["relevance_summary"] = None

        yield f"data: {json.dumps({'event': 'done', 'sources': sources, 'reworded_queries': reworded_queries, 'unique_chunks': unique_chunks, 'duplicates_removed': duplicates_removed, 'analysis': query_analysis})}\n\n"

    except Exception as e:
        import traceback
        error_msg = str(e)
        if debug:
            print(f"[ERROR] Exception in stream_rag_response: {error_msg}")
            traceback.print_exc()
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg})}\n\n"