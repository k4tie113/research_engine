from flask import Flask, jsonify, request
from flask_cors import CORS
from services.semantic_api import search_semantic
from services.faiss_search import search_faiss

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please enter a message."})

    # Use Semantic Scholar API
    results = search_semantic(message)

    if not results:
        return jsonify({"reply": "I couldn’t find any papers matching that topic."})

    # Only take the top (best) paper
    paper = results[0]

    title = paper.get("title", "Untitled")
    year = paper.get("year", "n/a")
    abstract = paper.get("abstract", "No abstract available.")
    url = paper.get("url", "")

    # Bot’s formatted reply
    reply = f"**{title}** ({year})\n\n{abstract}\n\n🔗 [View Paper]({url})"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=5000)