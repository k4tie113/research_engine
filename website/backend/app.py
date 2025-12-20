from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import sys
from pathlib import Path
import json

# Add the retrieval directory to the path
sys.path.append(str(Path(__file__).resolve().parents[2] / "retrieval"))

# Import RAG service from retrieval module
from rag_service import (
    get_rag_response,
    get_system_status,
    stream_rag_response,
)
from query_analyzer import analyze_query

app = Flask(__name__)
CORS(app)


@app.get("/api/status")
def status():
    """Check if the RAG system is initialized."""
    return jsonify(get_system_status())


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

    if not lines:
        return base

    return base + "\n\nFilters:\n" + "\n".join(lines)


@app.post("/api/analyze")
def api_analyze():
    """
    Endpoint to analyze a query without executing the search.
    Useful for debugging or understanding how the query is interpreted.
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("message") or data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        analysis = analyze_query(query)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/chat")
def api_chat():
    """
    Main chat endpoint.
    1) Analyze user's query
    2) Enhance it with extracted metadata
    3) Retrieve + generate answer (RAG)
    4) Return answer + structured sources (incl relevance scoring)
    """
    data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()
    conversation_history = data.get("conversation_history", [])
    user_filters = data.get("filters", {}) or {}

    if not message:
        return jsonify({"reply": "Please enter a message.", "sources": []}), 400

    try:
        # Step 1: Analyze the query
        print("\n" + "=" * 60)
        print("NEW QUERY")
        print("=" * 60)

        analysis = analyze_query(message)

        print(f"Original Query: {message}")
        print(f"Extracted Content: {analysis.get('content')}")
        print(f"Query Type: {analysis.get('query_type')}")
        print(f"Authors: {analysis.get('authors')}")
        print(f"Venues: {analysis.get('venues')}")
        print(f"Time Range: {analysis.get('time_range')}")
        print("=" * 60 + "\n")

        # Step 2: Build enhanced query
        base_query = analysis.get("content") or message
        enhanced_query = _augment_query_with_analysis(base_query, analysis)

        print(f"Enhanced Query for RAG:\n{enhanced_query}\n")

        # Step 3: Retrieve relevant papers and generate answer
        answer, sources = get_rag_response(
            enhanced_query,
            top_k=10,
            debug=True,
            conversation_history=conversation_history,
            user_filters=user_filters,
            query_analysis=analysis,
        )

        return jsonify(
            {
                "reply": answer,
                "analysis": analysis,
                "enhanced_query": enhanced_query,
                "sources": sources,  # should already include relevance_* fields
            }
        )

    except Exception as e:
        print(f"ERROR in /api/chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"reply": f"Sorry, an error occurred: {str(e)}", "sources": []}), 500


@app.post("/api/chat_stream")
def chat_stream():
    """
    Streaming chat endpoint (Server-Sent Events).
    IMPORTANT: We run analyze_query + augment query here too, so streaming and non-streaming match.
    """
    data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()
    conversation_history = data.get("conversation_history", [])
    user_filters = data.get("filters", {}) or {}

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    def generate():
        try:
            # Analyze + enhance query (match /api/chat behavior)
            analysis = analyze_query(message)
            base_query = analysis.get("content") or message
            enhanced_query = _augment_query_with_analysis(base_query, analysis)

            # stream_rag_response emits SSE lines: `data: {json}\n\n`
            # It also emits sources with relevance fields.
            yield from stream_rag_response(
                enhanced_query,
                top_k=10,
                max_tokens=600,
                debug=True,
                conversation_history=conversation_history,
                user_filters=user_filters,
            )
        except Exception as e:
            payload = {"event": "error", "message": str(e)}
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Paper Finder Backend")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /api/status      - Check system status")
    print("  POST /api/analyze     - Analyze a query")
    print("  POST /api/chat        - Full chat with RAG")
    print("  POST /api/chat_stream - Streaming chat with RAG")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000, host="0.0.0.0")
