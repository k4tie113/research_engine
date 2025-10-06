from flask import Flask, jsonify, request
from flask_cors import CORS
from services.semantic_api import search_semantic
from services.faiss_search import search_faiss

app = Flask(__name__)
CORS(app)

@app.route("/api/search")
def search():
    q = request.args.get("q", "")
    source = request.args.get("source", "semantic")  # default to semantic

    if not q:
        return jsonify({"results": []})

    if source == "semantic":
        results = search_semantic(q)
    elif source == "local":
        results = search_faiss(q)
    else:
        results = []

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
