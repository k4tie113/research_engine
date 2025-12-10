#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = BASE_DIR / "elasticsearch" / "embedded_chunks.jsonl"
METADATA_PATH = BASE_DIR / "elasticsearch" / "papers_oai_combined.csv"

ES_HOST = "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud:443"
ES_API_KEY = "api key"

NEW_INDEX = "chunks"

# ----------------------------------------------------
# CONNECT
# ----------------------------------------------------

es = Elasticsearch(
    ES_HOST,
    api_key=ES_API_KEY,
    request_timeout=200,
    verify_certs=True,
)

print("Connected to Elastic:", es.info())

# ----------------------------------------------------
# LOAD METADATA
# ----------------------------------------------------

print("Loading metadata CSV...")
metadata_map = {}

with open(METADATA_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["id"]
        year = int(row["created"][:4]) if row.get("created") else None
        authors = [a.strip() for a in row.get("authors", "").split(";") if a.strip()]
        categories = row.get("categories", "").split()

        metadata_map[pid] = {
            "year": year,
            "authors": authors,
            "categories": categories,
        }

print(f"Loaded metadata for {len(metadata_map)} papers.")

# ----------------------------------------------------
# DELETE OLD INDEX IF EXISTS
# ----------------------------------------------------

if es.indices.exists(index=NEW_INDEX):
    print(f"Deleting existing index '{NEW_INDEX}'...")
    es.indices.delete(index=NEW_INDEX)

# ----------------------------------------------------
# CREATE INDEX
# ----------------------------------------------------

index_body = {
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},

            # TITLE: BM25 + exact match
            "title": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"}  # filtering + sorting + exact equality
                }
            },

            # AUTHORS: array of strings with BOTH text search + exact match
            "authors": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            # YEAR: numeric filter
            "year": {"type": "integer"},

            # CATEGORIES: filterable array of strings
            "categories": {"type": "keyword"},

            # CHUNK TEXT: BM25 search
            "chunk_text": {"type": "text"},

            "token_count": {"type": "integer"},

            # VECTOR EMBEDDING (RAG top-k)
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}



print(f"Creating index '{NEW_INDEX}'...")
es.indices.create(index=NEW_INDEX, body=index_body)
print("Index created.")

# ----------------------------------------------------
# FAST LINE COUNT FOR PROGRESS BAR
# ----------------------------------------------------

total_chunks = 3927680 

# ----------------------------------------------------
# DOCUMENT GENERATOR
# ----------------------------------------------------

def generate_docs():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            pid = chunk["paper_id"]
            meta = metadata_map.get(pid, {})

            yield {
                "_index": NEW_INDEX,
                "_id": f"{pid}_{chunk['chunk_index']}",
                "_source": {
                    "paper_id": pid,
                    "chunk_index": chunk["chunk_index"],
                    "title": chunk.get("title", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "token_count": chunk.get("token_count", 0),
                    "embedding": chunk["embedding"],
                    "year": meta.get("year"),
                    "authors": meta.get("authors", []),
                    "categories": meta.get("categories", []),
                },
            }

# ----------------------------------------------------
# STREAMING BULK WITH REAL PROGRESS BAR
# ----------------------------------------------------

print("Indexing enriched chunks...")

success = 0

progress = tqdm(total=total_chunks, desc="Uploading", smoothing=0.02)

for ok, result in helpers.streaming_bulk(
    es,
    generate_docs(),
    chunk_size=2000,     # FASTEST for Elastic Cloud
    max_retries=3,
    request_timeout=200,
):
    success += ok
    progress.update(1)

progress.close()

print(f"\nSuccessfully indexed {success:,} chunks.")
print("DONE.")
