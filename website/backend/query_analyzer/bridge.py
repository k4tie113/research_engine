import asyncio
from typing import Dict, Any
from .query_analyzer import QueryAnalyzer

def _broad_or_specific(query_type: str) -> str:
    # Treat SPECIFIC_* as "specific", everything else as "broad"
    return "specific" if query_type.startswith("SPECIFIC_") else "broad"

class QueryAnalyzerBridge:
    """Bridge between the app and the OpenAI-based query analyzer."""

    def __init__(self) -> None:
        self._qa = QueryAnalyzer()

    async def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze a user query and return structured results.
        """
        try:
            res = self._qa.analyze_query(user_query)

            if res.get("status") == "success":
                qtype = res.get("query_type", "BROAD_BY_DESCRIPTION")
                return {
                    "status": "success",
                    "content": res.get("content", ""),
                    "authors": res.get("authors", []),
                    "venues": res.get("venues", []),
                    "time_range": res.get("time_range", {"start": None, "end": None}),
                    "query_type": qtype,
                    "broad_or_specific": _broad_or_specific(qtype),
                    "specifications": res.get("specifications", []),  # keep key for forward-compat
                    "relevance_criteria": res.get("relevance_criteria", []),
                    "original_query": res.get("original_query", user_query),
                }

            # failure path
            return {
                "status": "failure",
                "error": res.get("error", "Unknown error"),
                "original_query": user_query,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "original_query": user_query,
            }

    def analyze_query_sync(self, user_query: str) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_query."""
        return asyncio.run(self.analyze_query(user_query))
