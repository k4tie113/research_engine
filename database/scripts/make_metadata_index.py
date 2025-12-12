#!/usr/bin/env python3
"""
make_metadata_index.py
----------------------
Creates an Elasticsearch metadata index (paper_metadata)
storing one document per paper, including title, authors, abstract, year, and categories.
Uses papers_oai_combined.csv as the source.
"""

import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

# === CONFIGURATION ===
# Path relative to this script: ../data/metadata/papers_oai_combined.csv
BASE_DIR = Path(__file__).resolve().parents[1]  # /research_engine/database
CSV_PATH = BASE_DIR / "data" / "metadata" / "papers_oai_combined.csv"

ES_HOST = "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud:443"
ES_API_KEY = "SHR3SFY1b0J5WWFqM1RHdHBMQ2g6U2JsaEZnNDFadEZ2RFNSUzdQY3VYZw=="
INDEX_NAME = "paper_metadata"

# === CONNECT TO ELASTICSEARCH ===
print(f"Connecting to Elasticsearch at {ES_HOST} ...")
es = Elasticsearch(ES_HOST, api_key=ES_API_KEY, verify_certs=True, request_timeout=120)

# === CREATE INDEX IF NOT EXISTS ===
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "paper_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "authors": {"type": "text"},
                    "abstract": {"type": "text"},
                    "categories": {"type": "keyword"},
                    "year": {"type": "integer"}
                }
            }
        }
    )
    print(f"✅ Created index '{INDEX_NAME}'")
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists — new documents will be appended.")

# === HELPER FUNCTION ===
def parse_year(date_str):
    """Extract year from 'created' column (e.g. '2024-07-15T00:00:00Z')."""
    try:
        return int(date_str[:4])
    except Exception:
        return None

# === READ CSV AND UPLOAD ===
def upload_metadata():
    docs = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Preparing metadata docs"):
            doc = {
                "_index": INDEX_NAME,
                "_id": row["id"],
                "_source": {
                    "paper_id": row["id"],
                    "title": row.get("title", "").strip(),
                    "authors": [a.strip() for a in row.get("authors", "").split(";") if a.strip()],
                    "abstract": row.get("abstract", "").strip(),
                    "categories": row.get("categories", "").split(),
                    "year": parse_year(row.get("created", "")),
                },
            }
            docs.append(doc)

            # Bulk upload in batches
            if len(docs) >= 1000:
                helpers.bulk(es, docs)
                docs.clear()

        # Upload leftovers
        if docs:
            helpers.bulk(es, docs)

    print("All metadata uploaded successfully!")

# === MAIN ===
if __name__ == "__main__":
    upload_metadata()
