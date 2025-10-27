from .bridge import QueryAnalyzerBridge

_bridge = QueryAnalyzerBridge()

def analyze_query(text: str) -> dict:
    # synchronous wrapper
    return _bridge.analyze_query_sync(text)

__all__ = ["analyze_query", "QueryAnalyzerBridge"]
