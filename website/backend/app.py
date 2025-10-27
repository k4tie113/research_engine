from flask import Flask, jsonify, request
from flask_cors import CORS

from rag_service import get_rag_response, format_sources, get_system_status
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

    # Combine base query with hints
    if not lines:
        return base
    
    return base + "\n\nFilters:\n" + "\n".join(lines)

@app.post("/api/analyze")
def api_analyze():
    """
    Endpoint to analyze a query without executing the search.
    Useful for debugging or understanding how the query is interpreted.
    """
    data = request.get_json() or {}
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
    1. Analyzes the user's query
    2. Enhances it with extracted metadata
    3. Retrieves relevant papers using RAG
    4. Returns answer with sources
    """
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    conversation_history = data.get("conversation_history", [])

    if not message:
        return jsonify({"reply": "Please enter a message."})

    try:
        # Step 1: Analyze the query
        print("\n" + "="*60)
        print("NEW QUERY")
        print("="*60)
        analysis = analyze_query(message)
        
        print(f"Original Query: {message}")
        print(f"Extracted Content: {analysis.get('content')}")
        print(f"Query Type: {analysis.get('query_type')}")
        print(f"Authors: {analysis.get('authors')}")
        print(f"Venues: {analysis.get('venues')}")
        print(f"Time Range: {analysis.get('time_range')}")
        print("="*60 + "\n")

        # Step 2: Build enhanced query
        # Use extracted content as base, fall back to original message
        base_query = analysis.get("content") or message
        enhanced_query = _augment_query_with_analysis(base_query, analysis)
        
        print(f"Enhanced Query for RAG:\n{enhanced_query}\n")

        # Step 3: Retrieve relevant papers and generate answer
        answer, sources = get_rag_response(
            enhanced_query, 
            top_k=5, 
            debug=True,  # Shows the full prompt sent to GPT
            conversation_history=conversation_history
        )

        # Step 4: Format and return response
        reply = answer + format_sources(sources, max_sources=5)
        
        return jsonify({
            "reply": reply, 
            "analysis": analysis,
            "enhanced_query": enhanced_query
        })
        
    except Exception as e:
        print(f"ERROR in /api/chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "reply": f"Sorry, an error occurred: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Paper Finder Backend")
    print("="*60)
    print("Endpoints:")
    print("  GET  /api/status  - Check system status")
    print("  POST /api/analyze - Analyze a query")
    print("  POST /api/chat    - Full chat with RAG")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')