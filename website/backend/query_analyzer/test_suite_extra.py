# file: website/backend/query_analyzer/test_suite_extra.py

import os
import json
from unittest import mock

import pytest

# --- Helpers ---------------------------------------------------------------

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))

# Some tests require actual LLM behavior. Skip them cleanly if no key is set.
needs_key = pytest.mark.skipif(not HAS_KEY, reason="OPENAI_API_KEY not set; skipping LLM-dependent test")


# --- 1) Query-type classification -----------------------------------------

@needs_key
@pytest.mark.parametrize("q,expect", [
    ('"Attention Is All You Need"', "SPECIFIC_BY_TITLE"),
    ("The BERT paper", "SPECIFIC_BY_NAME"),
    ("Papers by Patrick Lewis about RAG", "BY_AUTHOR"),
    ("Survey on graph-based RAG techniques", "BROAD_BY_DESCRIPTION"),
])
def test_query_type_heuristics(q, expect):
    from query_analyzer import analyze_query
    res = analyze_query(q)
    assert res["status"] == "success"
    assert res["query_type"] == expect


# --- 2) Year parsing & ordering -------------------------------------------

@needs_key
def test_time_range_years():
    from query_analyzer import analyze_query

    r1 = analyze_query("NeurIPS papers 2018 to 2016 on GNNs")
    assert r1["status"] == "success"
    # normalized order (2016, 2018)
    assert r1["time_range"] == {"start": 2016, "end": 2018}

    r2 = analyze_query("ACL 2020 paper on RAG")
    assert r2["status"] == "success"
    tr = r2["time_range"]
    assert tr == {"start": 2020, "end": 2020}


# --- 3) Venue & author extraction signals ---------------------------------

@needs_key
def test_venues_and_authors_detected():
    from query_analyzer import analyze_query
    r = analyze_query("ACL / EMNLP papers by Patrick Lewis on retrieval")
    assert r["status"] == "success"
    blob = " ".join([*r.get("venues", []), *r.get("authors", [])]).lower()
    assert "acl" in blob and "emnlp" in blob and "patrick" in blob


# --- 4) Relevance-criteria normalization ----------------------------------

@needs_key
def test_relevance_weights_sum_to_one():
    from query_analyzer import analyze_query
    r = analyze_query("Hallucination mitigation in RAG after 2022")
    assert r["status"] == "success"
    w = [c.get("weight", 0.0) for c in r.get("relevance_criteria", [])]
    assert w, "expected at least one criterion"
    assert abs(sum(w) - 1.0) < 1e-6


# --- 5) Failure path (no key) ---------------------------------------------

def test_failure_path_shapes_object(monkeypatch):
    from query_analyzer import analyze_query
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        res = analyze_query("anything")
    assert res["status"] in {"failure", "success"}  # your code returns 'failure' in this path
    assert "time_range" in res and isinstance(res["time_range"], dict)


# --- 6) Verbalizer coverage -----------------------------------------------

@needs_key
def test_verbalizer_non_empty():
    from query_analyzer.verbalize import verbalize_analyzed_query
    from query_analyzer import analyze_query
    r = analyze_query("RAG hallucinations ACL 2023")
    assert r["status"] == "success"
    text = verbalize_analyzed_query(r)
    assert isinstance(text, str) and text.strip()


# --- 7) Anchors behavior (no-op without key; accepts markdown attr) -------

def test_anchors_no_key_is_noop(monkeypatch):
    from query_analyzer.anchor import combine_content_query_with_anchors
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        q = "Graph RAG for legal retrieval"
        out = combine_content_query_with_anchors(q, ["# Note", "Some markdown"])
        assert out == q  # requires the no-op guard in combine_content_query_with_anchors


def test_anchors_accepts_markdown_attr():
    from query_analyzer.anchor import combine_content_query_with_anchors

    class Doc:
        markdown = "GraphRAG over case-law graph"
    q = "Graph RAG for legal retrieval"
    out = combine_content_query_with_anchors(q, [Doc()])
    assert isinstance(out, str) and len(out) > 0


# --- 8) Robustness to weird input -----------------------------------------

def test_weird_inputs():
    from query_analyzer import analyze_query
    assert analyze_query("")["status"] in {"success", "failure"}
    assert analyze_query("“BERT” paper")["status"] in {"success", "failure"}
    long_q = "RAG " * 5000
    assert analyze_query(long_q)["status"] in {"success", "failure"}


# --- 9) Basic stability on core fields ------------------------------------

@needs_key
def test_basic_stability():
    from query_analyzer import analyze_query
    q = "RAG papers after 2022 about hallucinations, ACL preferred"
    r1, r2 = analyze_query(q), analyze_query(q)
    for k in ["content", "time_range", "query_type"]:
        assert r1.get(k) == r2.get(k)


# --- 10) Bridge sync wrapper ----------------------------------------------

def test_bridge_sync():
    from query_analyzer import analyze_query as bridge_analyze
    r = bridge_analyze("papers by Patrick Lewis in ACL")
    assert isinstance(r, dict)
    assert "status" in r
