def verbalize_analyzed_query(query_analysis_result: dict) -> str | None:
    """
    Very simple renderer for the dict returned by QueryAnalyzer/analyze_query.
    """
    if not isinstance(query_analysis_result, dict):
        return None
    if query_analysis_result.get("status") != "success":
        return None

    parts = []
    qtype = query_analysis_result.get("query_type", "BROAD_BY_DESCRIPTION")
    if qtype.startswith("SPECIFIC_"):
        parts.append("Looking for a specific paper.")
    else:
        parts.append("Looking for a set of papers.")

    authors = query_analysis_result.get("authors") or []
    venues = query_analysis_result.get("venues") or []
    tr = query_analysis_result.get("time_range") or {"start": None, "end": None}
    content = (query_analysis_result.get("content") or "").strip()

    meta = []
    if authors:
        meta.append(f"authored by {', '.join(authors)}")
    if venues:
        meta.append(f"in venues: {', '.join(venues)}")
    if tr.get("start") or tr.get("end"):
        s = tr.get("start")
        e = tr.get("end")
        if s and e and s == e:
            meta.append(f"published in {s}")
        elif s and e:
            meta.append(f"published between {s} and {e}")
        elif s:
            meta.append(f"published after {s}")
        elif e:
            meta.append(f"published before {e}")
    if meta:
        parts.append("Metadata filters: " + "; ".join(meta) + ".")

    if content and not qtype.startswith("SPECIFIC_"):
        parts.append(f"Content to search for: {content}.")

    rc = query_analysis_result.get("relevance_criteria") or []
    if rc:
        crit_lines = [f"- {c.get('name', 'criterion')}: {c.get('description','')}" for c in rc]
        parts.append("Content must satisfy:\n" + "\n".join(crit_lines))

    return "\n".join(parts) or None
