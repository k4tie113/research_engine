from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import sys

# Add the retrieval directory to the path
sys.path.append(str(Path(__file__).resolve().parents[2] / "retrieval"))

# Import RAG service from retrieval module
from rag_service import get_rag_response, format_sources, get_system_status

app = Flask(__name__)
CORS(app)

# === RAG SYSTEM INITIALIZATION ===
print("Initializing RAG system from retrieval module...")
# The RAG system will be initialized lazily on first use
# via the rag_service module
status = get_system_status()
print(f"System status: {status}")
print("RAG system ready!")

def handle_rag_query(query, top_k=5, debug=True, conversation_history=None):
    """Handle RAG query using the imported service."""
    answer, sources = get_rag_response(query, top_k=top_k, debug=debug, conversation_history=conversation_history)
    
    # Format the response with sources
    source_text = format_sources(sources, max_sources=5)
    
    return answer + source_text

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    conversation_history = data.get("conversation_history", [])

    if not message:
        return jsonify({"reply": "Please enter a message."})

    # Use RAG system from retrieval module
    reply = handle_rag_query(message, top_k=5, debug=True, conversation_history=conversation_history)
    
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=5000)