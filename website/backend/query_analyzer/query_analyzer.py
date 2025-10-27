#!/usr/bin/env python3
"""
Simplified query analyzer that extracts key information from research queries.
Only dependency: openai. Set OPENAI_API_KEY in the environment.
"""

import os
import json as _json
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Small helpers / normalizers
# -----------------------------
def _safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if isinstance(i, (str, int, float))]
    if x is None:
        return []
    if isinstance(x, str) and x.strip():
        return [s.strip() for s in x.split(",")]
    return []

def _safe_year(y: Any) -> Optional[int]:
    try:
        if y is None:
            return None
        yi = int(y)
        if 1800 <= yi <= 2100:
            return yi
    except Exception:
        pass
    return None

def _normalize_time_range(tr: Any) -> Dict[str, Optional[int]]:
    if not isinstance(tr, dict):
        return {"start": None, "end": None}
    start = _safe_year(tr.get("start"))
    end = _safe_year(tr.get("end"))
    if start and end and end < start:
        start, end = end, start
    return {"start": start, "end": end}

def _normalize_relevance_criteria(rc: Any) -> List[Dict[str, Any]]:
    if not isinstance(rc, list) or not rc:
        return []
    out: List[Dict[str, Any]] = []
    for item in rc:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip() or "criterion"
        desc = str(item.get("description") or "").strip()
        weight = item.get("weight")
        try:
            weight = float(weight)
        except Exception:
            weight = None
        out.append({"name": name, "description": desc, "weight": weight})
    if any(i["weight"] is None for i in out):
        n = len(out)
        for i in out:
            i["weight"] = 1.0 / n
        return out
    total = sum(max(0.0, float(i["weight"])) for i in out) or 1.0
    for i in out:
        i["weight"] = max(0.0, float(i["weight"])) / total
    return out

def _infer_query_type(analysis: Dict[str, Any], original_query: str) -> str:
    """
    Allowed:
      - BROAD_BY_DESCRIPTION
      - SPECIFIC_BY_TITLE
      - SPECIFIC_BY_NAME
      - BY_AUTHOR
    """
    qtype = (analysis.get("query_type") or "").strip()
    if qtype:
        return qtype

    content = (analysis.get("content") or "").strip()
    authors = _safe_list(analysis.get("authors"))

    if authors:
        return "BY_AUTHOR"

    # heuristic for exact title
    if content and len(content.split()) <= 12 and content[:1].isupper():
        if content == original_query.strip() or any(k in original_query.lower() for k in ["paper", "title"]):
            return "SPECIFIC_BY_TITLE"

    if "paper" in original_query.lower() and not authors:
        return "SPECIFIC_BY_NAME"

    return "BROAD_BY_DESCRIPTION"


class QueryAnalyzer:
    def __init__(self) -> None:
        self._model = "gpt-4o-mini"
        self._cached_client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._cached_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._cached_client = OpenAI(api_key=api_key)
        return self._cached_client

    def _system_prompt(self) -> str:
        return (
            "You are a research query analyzer. Extract structured information from academic paper search queries.\n\n"
            "Return ONLY a single JSON object with these fields:\n"
            '1. "content": string — The main research topic (strip metadata like authors/venues/years)\n'
            '2. "authors": string[] — List of author names mentioned (empty if none)\n'
            '3. "venues": string[] — List of conference/journal venues mentioned (empty if none)\n'
            '4. "time_range": {"start": year|null, "end": year|null} — Only if the query explicitly constrains years\n'
            '5. "query_type": one of ["BROAD_BY_DESCRIPTION","SPECIFIC_BY_TITLE","SPECIFIC_BY_NAME","BY_AUTHOR"]\n'
            '6. "relevance_criteria": [{"name": str, "description": str, "weight": number}] — content-based; weights sum to 1.0\n\n'
            "Rules:\n"
            "- Do not infer years unless explicitly requested; if unsure, use nulls.\n"
            "- If the user asks for a specific paper by exact title, use SPECIFIC_BY_TITLE.\n"
            "- If they say things like “the BERT paper” without exact title, use SPECIFIC_BY_NAME.\n"
            "- If authors are provided as filters, you may set BY_AUTHOR.\n"
            "- Keep the response compact and valid JSON only."
        )

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_query},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            if isinstance(raw, list):  # very defensive
                raw = "".join(str(p) for p in raw)
            parsed: Dict[str, Any] = _json.loads(raw)

            content = str(parsed.get("content") or "").strip()
            authors = _safe_list(parsed.get("authors"))
            venues = _safe_list(parsed.get("venues"))
            time_range = _normalize_time_range(parsed.get("time_range"))
            query_type = _infer_query_type(parsed, user_query)
            relevance = _normalize_relevance_criteria(parsed.get("relevance_criteria"))

            return {
                "status": "success",
                "content": content,
                "authors": authors,
                "venues": venues,
                "time_range": time_range,
                "query_type": query_type,
                "relevance_criteria": relevance,
                "specifications": parsed.get("specifications", []),  # forward-compat
                "original_query": user_query,
            }

        except Exception as e:
            return {
                "status": "failure",
                "error": str(e),
                "content": user_query,
                "authors": [],
                "venues": [],
                "time_range": {"start": None, "end": None},
                "query_type": "BROAD_BY_DESCRIPTION",
                "relevance_criteria": [],
                "specifications": [],
                "original_query": user_query,
            }


# Singleton instance and sync wrapper
_analyzer = QueryAnalyzer()

def analyze_query(text: str) -> dict:
    return _analyzer.analyze_query(text)

__all__ = ["analyze_query", "QueryAnalyzer"]
