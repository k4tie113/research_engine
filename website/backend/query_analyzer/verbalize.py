from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def _format_time_range(tr: Dict[str, Optional[int]] | None) -> Optional[str]:
    """
    Convert time range dict to human-readable phrase.
    
    Args:
        tr: Dict with 'start' and 'end' keys (both Optional[int])
    
    Returns:
        Formatted string like "published in 2020" or None if no range
    
    Examples:
        {"start": 2020, "end": 2020} -> "published in 2020"
        {"start": 2018, "end": 2021} -> "published between 2018 and 2021"
        {"start": 2015, "end": None} -> "published after 2015"
        {"start": None, "end": 2010} -> "published before 2010"
    """
    if not isinstance(tr, dict):
        return None

    start = tr.get("start")
    end = tr.get("end")

    # No time constraints
    if not start and not end:
        return None

    # Single year
    if start and end and start == end:
        return f"published in {start}"
    
    # Year range
    if start and end:
        return f"published between {start} and {end}"
    
    # After year
    if start and not end:
        return f"published after {start}"
    
    # Before year
    if end and not start:
        return f"published before {end}"
    
    return None


def _format_authors(authors: List[str] | None) -> Optional[str]:
    """
    Format list of authors into human-readable phrase.
    
    Args:
        authors: List of author names
    
    Returns:
        Formatted string or None if no authors
    
    Examples:
        ["Alice Smith"] -> "authored by Alice Smith"
        ["Alice Smith", "Bob Jones"] -> "authored by Alice Smith and Bob Jones"
        ["A", "B", "C"] -> "authored by A, B, and C"
    """
    if not authors:
        return None
    
    if len(authors) == 1:
        return f"authored by {authors[0]}"
    
    if len(authors) == 2:
        return f"authored by {authors[0]} and {authors[1]}"
    
    # Oxford comma for 3+ authors
    return f"authored by {', '.join(authors[:-1])}, and {authors[-1]}"


def _format_venues(venues: List[str] | None) -> Optional[str]:
    """
    Format list of venues into human-readable phrase.
    
    Args:
        venues: List of venue names
    
    Returns:
        Formatted string or None if no venues
    
    Examples:
        ["NeurIPS"] -> "in venue: NeurIPS"
        ["ICML", "NeurIPS"] -> "in venues: ICML, NeurIPS"
    """
    if not venues:
        return None
    
    if len(venues) == 1:
        return f"in venue: {venues[0]}"
    
    return "in venues: " + ", ".join(venues)


def _format_relevance_criteria(criteria_list: List[Dict[str, Any]] | None) -> Optional[str]:
    """
    Format relevance criteria into human-readable list.
    
    Args:
        criteria_list: List of criteria dicts with name, description, weight
    
    Returns:
        Multi-line formatted string or None if no criteria
    
    Examples:
        [{"name": "topic", "description": "About RAG", "weight": 0.8},
         {"name": "year", "description": "Recent", "weight": 0.2}]
        ->
        "Content must satisfy:
        - topic (weight ≈ 0.80): About RAG
        - year (weight ≈ 0.20): Recent"
    """
    if not criteria_list:
        return None

    lines: List[str] = ["Content must satisfy:"]
    
    for c in criteria_list:
        name = c.get("name", "criterion")
        desc = (c.get("description") or "").strip()
        weight = c.get("weight", None)
        
        if weight is not None:
            try:
                weight_val = float(weight)
                lines.append(f"- {name} (weight ≈ {weight_val:.2f}): {desc}")
            except (ValueError, TypeError):
                lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}: {desc}")
    
    return "\n".join(lines)


def _describe_query_type(query_type: str) -> str:
    """
    Map query type codes to human-readable descriptions.
    
    Args:
        query_type: One of the four query type codes
    
    Returns:
        Human-readable description
    """
    descriptions = {
        "BROAD_BY_DESCRIPTION": "Query interpreted as a broad topic search for a set of papers.",
        "SPECIFIC_BY_TITLE": "Query interpreted as a search for a specific paper by (near-)exact title.",
        "SPECIFIC_BY_NAME": "Query interpreted as a search for a specific paper by name or nickname (e.g., "the BERT paper").",
        "BY_AUTHOR": "Query interpreted as a search for papers by specific author(s).",
    }
    
    return descriptions.get(query_type, f"Query type: {query_type}.")


def verbalize_analyzed_query(query_analysis_result: dict) -> str | None:
    """
    Convert query analysis result dict into human-readable summary.
    
    This function takes the structured output from QueryAnalyzer and
    renders it as natural language text suitable for display to users.
    
    Args:
        query_analysis_result: Dict from QueryAnalyzer.analyze_query()
    
    Returns:
        Multi-line human-readable summary, or None if not successful
    
    Expected input structure:
        {
            "status": "success",
            "content": str,
            "authors": List[str],
            "venues": List[str],
            "time_range": {"start": int|None, "end": int|None},
            "query_type": str,
            "broad_or_specific": "broad" | "specific",
            "relevance_criteria": List[Dict],
            "original_query": str,
        }
    
    Example output:
        "Query interpreted as a broad topic search for a set of papers.
        Mode: Broad
        Metadata filters: authored by Patrick Lewis; in venue: ACL; published after 2020.
        Content to search for: RAG hallucination mitigation.
        Content must satisfy:
        - topic_relevance (weight ≈ 0.70): Papers about RAG and hallucinations
        - methodology (weight ≈ 0.30): Focus on mitigation techniques"
    """
    # Validate input
    if not isinstance(query_analysis_result, dict):
        logger.warning("verbalize_analyzed_query called with non-dict input")
        return None
    
    # Only verbalize successful analyses
    if query_analysis_result.get("status") != "success":
        logger.debug("Query analysis status is not 'success', skipping verbalization")
        return None

    parts: List[str] = []

    # 1) Query type description and mode
    qtype = query_analysis_result.get("query_type", "BROAD_BY_DESCRIPTION")
    qtype_str = str(qtype)

    # Determine broad/specific mode
    broad_or_specific = query_analysis_result.get("broad_or_specific")
    if broad_or_specific not in {"broad", "specific"}:
        # Derive from query_type if missing
        broad_or_specific = "specific" if qtype_str.startswith("SPECIFIC_") else "broad"

    parts.append(_describe_query_type(qtype_str))
    parts.append(f"Mode: {broad_or_specific.capitalize()}")

    # 2) Extract all fields from result
    authors = query_analysis_result.get("authors") or []
    venues = query_analysis_result.get("venues") or []
    tr = query_analysis_result.get("time_range") or {"start": None, "end": None}
    content = (query_analysis_result.get("content") or "").strip()
    rc = query_analysis_result.get("relevance_criteria") or []

    # 3) Build metadata filters section
    meta_bits: List[str] = []

    authors_str = _format_authors(authors)
    if authors_str:
        meta_bits.append(authors_str)

    venues_str = _format_venues(venues)
    if venues_str:
        meta_bits.append(venues_str)

    time_str = _format_time_range(tr)
    if time_str:
        meta_bits.append(time_str)

    if meta_bits:
        parts.append("Metadata filters: " + "; ".join(meta_bits) + ".")

    # 4) Content description (mainly for broad queries)
    if content and broad_or_specific == "broad":
        parts.append(f"Content to search for: {content}.")

    # 5) Relevance criteria
    rc_text = _format_relevance_criteria(rc)
    if rc_text:
        parts.append(rc_text)

    # 6) Optionally include original query (commented out by default)
    # original_query = (query_analysis_result.get("original_query") or "").strip()
    # if original_query:
    #     parts.append(f'Original query: "{original_query}"')

    # Join all parts with newlines
    result = "\n".join(parts) if parts else None
    
    if not result:
        logger.warning("Verbalization produced empty result")
    
    return result


def verbalize_analyzed_query_compact(query_analysis_result: dict) -> str | None:
    """
    Compact single-line version of verbalize_analyzed_query.
    
    Args:
        query_analysis_result: Dict from QueryAnalyzer.analyze_query()
    
    Returns:
        Single-line summary or None if not successful
    
    Example:
        "Broad search for 'RAG techniques' by Patrick Lewis in ACL (2020+)"
    """
    if not isinstance(query_analysis_result, dict):
        return None
    
    if query_analysis_result.get("status") != "success":
        return None

    parts: List[str] = []
    
    # Mode
    broad_or_specific = query_analysis_result.get("broad_or_specific", "broad")
    parts.append(f"{broad_or_specific.capitalize()} search")
    
    # Content
    content = (query_analysis_result.get("content") or "").strip()
    if content:
        parts.append(f"for '{content}'")
    
    # Authors
    authors = query_analysis_result.get("authors") or []
    if authors:
        if len(authors) == 1:
            parts.append(f"by {authors[0]}")
        else:
            parts.append(f"by {len(authors)} authors")
    
    # Venues
    venues = query_analysis_result.get("venues") or []
    if venues:
        parts.append(f"in {', '.join(venues)}")
    
    # Time range
    tr = query_analysis_result.get("time_range") or {}
    start, end = tr.get("start"), tr.get("end")
    if start and end:
        if start == end:
            parts.append(f"({start})")
        else:
            parts.append(f"({start}-{end})")
    elif start:
        parts.append(f"({start}+)")
    elif end:
        parts.append(f"(up to {end})")
    
    return " ".join(parts) if parts else None