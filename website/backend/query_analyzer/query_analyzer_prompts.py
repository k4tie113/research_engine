from __future__ import annotations
from typing import Literal, Optional, TypedDict, Any
from pydantic import BaseModel

# --- Minimal local schemas to avoid external deps ---

class ExtractedContent(BaseModel):
    content: Optional[str] = None

class ExtractedAuthors(BaseModel):
    authors: list[str] = []

class ExtractedVenues(BaseModel):
    venues: list[str] = []

class ExtractedRecency(BaseModel):
    recency: Optional[str] = None  # "first" | "last" | None

class ExtractedCentrality(BaseModel):
    centrality: Optional[str] = None  # "first" | "last" | None

class ExtractedYearlyTimeRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None

class BroadOrSpecificType(BaseModel):
    type: Literal["broad", "specific"]

class ByNameOrTitleType(BaseModel):
    type: Literal["name", "title"]

class RelevanceCriterion(BaseModel):
    name: str
    description: str
    weight: float

class RelevanceCriteria(BaseModel):
    query: str
    required_relevance_critieria: Optional[list[RelevanceCriterion]] = None
    nice_to_have_relevance_criteria: Optional[list[RelevanceCriterion]] = None
    clarification_questions: Optional[list[str]] = None

class DomainsIdentified(BaseModel):
    main: str = "Unknown"
    others: list[str] = []

class PossibleRefusal(BaseModel):
    type: Optional[Literal["not paper finding", "similar to", "web access", "affiliation", "author ID"]] = None

InputQueryJson = dict[str, Any]

# --- Heuristic implementations (no LLM) ---

def content_extraction(query_json: InputQueryJson) -> ExtractedContent:
    q = (query_json.get("query") or "").strip()
    return ExtractedContent(content=q if q else None)

def author_extraction(query_json: InputQueryJson) -> ExtractedAuthors:
    # Extremely naive: no authors unless a capitalized full name appears
    return ExtractedAuthors(authors=[])

def venue_extraction(query_json: InputQueryJson) -> ExtractedVenues:
    vs = []
    for v in ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "CVPR", "ECCV", "KDD", "SIGGRAPH"]:
        if v.lower() in (query_json.get("query") or "").lower():
            vs.append(v)
    return ExtractedVenues(venues=vs)

def recency_extraction(query_json: InputQueryJson) -> ExtractedRecency:
    q = (query_json.get("query") or "").lower()
    if "recent" in q or "latest" in q or "since" in q:
        return ExtractedRecency(recency="first")
    if "early" in q or "earliest" in q or "foundational" in q:
        return ExtractedRecency(recency="last")
    return ExtractedRecency(recency=None)

def centrality_extraction(query_json: InputQueryJson) -> ExtractedCentrality:
    q = (query_json.get("query") or "").lower()
    if "seminal" in q or "highly cited" in q or "influential" in q:
        return ExtractedCentrality(centrality="first")
    if "less cited" in q or "obscure" in q:
        return ExtractedCentrality(centrality="last")
    return ExtractedCentrality(centrality=None)

def time_range_extraction(query_json: InputQueryJson) -> ExtractedYearlyTimeRange:
    return ExtractedYearlyTimeRange(start=None, end=None)

def broad_or_specific_query_type(query_json: InputQueryJson) -> BroadOrSpecificType:
    q = (query_json.get("query") or "")
    if q and (q.istitle() or q.count('"') >= 2):
        return BroadOrSpecificType(type="specific")
    return BroadOrSpecificType(type="broad")

def by_title_or_name_query_type(query_json: InputQueryJson) -> ByNameOrTitleType:
    q = (query_json.get("query") or "").lower()
    if "paper" in q and '"' not in q:
        return ByNameOrTitleType(type="name")
    return ByNameOrTitleType(type="title")

def identify_relevance_criteria(query_json: InputQueryJson) -> RelevanceCriteria:
    q = (query_json.get("query") or "")
    crit = [RelevanceCriterion(name="topic_match", description=f"About: {q}", weight=1.0)] if q else []
    return RelevanceCriteria(query=q, required_relevance_critieria=crit)

def domain_identification(query_json: InputQueryJson) -> DomainsIdentified:
    return DomainsIdentified(main="Unknown", others=[])

def check_refusal(query_json: InputQueryJson) -> PossibleRefusal:
    return PossibleRefusal(type=None)

def decompose_query(query: str) -> dict[str, Any]:
    qj: InputQueryJson = {"query": query}
    return {
        "content": content_extraction(qj),
        "authors": author_extraction(qj),
        "venues": venue_extraction(qj),
        "recency": recency_extraction(qj),
        "centrality": centrality_extraction(qj),
        "time_range": time_range_extraction(qj),
        "broad_or_specific": broad_or_specific_query_type(qj),
        "by_name_or_title": by_title_or_name_query_type(qj),
        "relevance_criteria": identify_relevance_criteria(qj),
        "domains": domain_identification(qj),
        "possible_refusal": check_refusal(qj),
    }
