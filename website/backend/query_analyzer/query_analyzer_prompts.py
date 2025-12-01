from __future__ import annotations
from typing import Literal, Optional, TypedDict, Any
from pydantic import BaseModel
import re
from datetime import datetime

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
    """
    Very simple heuristic:
    - Look for patterns like "papers by Andrej Karpathy" or "by Mayer Godberg and Yossi Matias".
    - Only considers capitalized names after the word 'by'.
    - Returns a list of names, or [] if nothing obvious is found.
    """
    q = (query_json.get("query") or "")
    authors: list[str] = []

    # Look for "by <Name [and Name]...>"
    # Example matched group: "Mayer Godberg and Yossi Matias"
    match = re.search(
        r"\bby ([A-Z][a-z]+(?: [A-Z][a-z]+)*(?: and [A-Z][a-z]+(?: [A-Z][a-z]+)*)*)",
        q,
    )
    if not match:
        return ExtractedAuthors(authors=[])

    names_blob = match.group(1)  # everything after "by "
    # Split multiple authors joined by "and"
    candidate_parts = [p.strip() for p in names_blob.split(" and ") if p.strip()]

    for part in candidate_parts:
        # Basic sanity check: at least "First Last"
        tokens = part.split()
        if len(tokens) < 2:
            continue
        # Require that most tokens start uppercase (to avoid phrases like "any coauthor")
        if sum(t[0].isupper() for t in tokens) >= len(tokens) - 1:
            authors.append(part)

    # Deduplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for a in authors:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    return ExtractedAuthors(authors=deduped)

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
    """
    Heuristic year extractor:
    - Handles phrases like:
        * "before 1980"  -> end = 1979
        * "after 2010"   -> start = 2011
        * "since 2010"   -> start = 2010
        * "from 2010 to 2015", "2018 to 2020" -> start/end range
        * "ICLR 2024"    -> start = end = 2024
        * "last 3 years" -> start = current_year-2, end = current_year
    - Otherwise, if there are multiple years, uses [min, max].
    """
    q = (query_json.get("query") or "")
    ql = q.lower()

    # 1) Handle "last N years"
    m = re.search(r"last\s+(\d+)\s+years", ql)
    if m:
        try:
            n = int(m.group(1))
            current_year = datetime.now().year
            if n > 0:
                start = current_year - n + 1
                end = current_year
                return ExtractedYearlyTimeRange(start=start, end=end)
        except Exception:
            # fall through to generic parsing if something goes wrong
            pass

    # 2) Extract all 4-digit year-like numbers
    raw_years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", q)
    years = []
    for y in raw_years:
        yi = int(y)
        if 1800 <= yi <= 2100:
            years.append(yi)

    if not years:
        return ExtractedYearlyTimeRange(start=None, end=None)

    years.sort()
    first, last = years[0], years[-1]

    # 3) Directional phrases
    if "before" in ql or "earlier than" in ql or "prior to" in ql:
        # "before 1980" -> end = 1979
        return ExtractedYearlyTimeRange(start=None, end=first - 1)

    if "after" in ql or "later than" in ql:
        # "after 2010" -> start = 2011
        return ExtractedYearlyTimeRange(start=first + 1, end=None)

    if "since" in ql:
        # "since 2010" -> start = 2010
        return ExtractedYearlyTimeRange(start=first, end=None)

    # 4) Explicit ranges: "between ...", "from ... to ..."
    if "between" in ql or ("from" in ql and "to" in ql):
        return ExtractedYearlyTimeRange(start=first, end=last)

    # 5) Single year: treat as that exact year
    if len(years) == 1:
        return ExtractedYearlyTimeRange(start=first, end=first)

    # 6) Multiple years but no range words: just span min..max
    return ExtractedYearlyTimeRange(start=first, end=last)

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
