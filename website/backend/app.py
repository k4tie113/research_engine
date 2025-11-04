from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pathlib import Path
import sys

# Add the retrieval directory to the path
sys.path.append(str(Path(__file__).resolve().parents[2] / "retrieval"))

# Import RAG service from retrieval module
from rag_service import get_rag_response, format_sources, get_system_status, stream_rag_response

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
    answer, result = get_rag_response(query, top_k=top_k, debug=debug, conversation_history=conversation_history)
    sources = result["sources"]
    sentence_alignments = result["sentence_alignments"]

    # Optional: still include bottom citations
    source_text = format_sources(sources, max_sources=5)

    return {
        "reply": answer + source_text,
        "sources": sources,
        "sentence_alignments": sentence_alignments
    }
""" COMMENTED OUT CODE
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
    """
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    conversation_history = data.get("conversation_history", [])

    if not message:
        return jsonify({"reply": "Please enter a message."})

    result = handle_rag_query(message, top_k=5, debug=True, conversation_history=conversation_history)
    return jsonify(result)


@app.route("/api/chat_stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    message = data.get("message", "").strip()
    conversation_history = data.get("conversation_history", [])

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    def generate():
        yield from stream_rag_response(message, top_k=5, max_tokens=600, debug=True, conversation_history=conversation_history)

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, port=5000)