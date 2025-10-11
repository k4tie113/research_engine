# retrieval/es_index_chunks.py
import json
from pathlib import Path
from elasticsearch import Elasticsearch, helpers

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks"

ROOT = Path(__file__).resolve().parents[1]
# prefer OAI chunks if present, else fallback
CANDIDATES = [
    ROOT / "data" / "chunks_oai.jsonl",
    ROOT / "data" / "chunks.jsonl",
]
CHUNKS = next((p for p in CANDIDATES if p.exists()), None)

def actions():
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            _id = f"{obj.get('paper_id','na')}::{obj.get('chunk_index',0)}"
            yield {"_index": INDEX, "_id": _id, "_source": obj}

def main():
    assert CHUNKS, "No chunks file found (looked for data/chunks_oai.jsonl then data/chunks.jsonl)"
    es = Elasticsearch(ES_URL)
    helpers.bulk(es, actions(), chunk_size=2000, request_timeout=120)
    es.indices.refresh(index=INDEX)
    print("Indexed:", es.count(index=INDEX)["count"])

if __name__ == "__main__":
    main()
