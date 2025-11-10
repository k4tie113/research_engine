#!/usr/bin/env python3
"""
check_es_data.py
----------------
Check what data is actually in Elasticsearch for a sample document.
"""

from elasticsearch import Elasticsearch

# === ELASTICSEARCH CONFIG ===
ES_URL = "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud"
ES_API_KEY = "SHR3SFY1b0J5WWFqM1RHdHBMQ2g6U2JsaEZnNDFadEZ2RFNSUzdQY3VYZw=="
ES_INDEX = "paper_chunks"

def check_sample_data():
    """Check a sample document from Elasticsearch to see what fields are available."""
    try:
        es = Elasticsearch(
            [ES_URL],
            api_key=ES_API_KEY,
            request_timeout=60
        )
        
        if not es.ping():
            print("[ERROR] Failed to connect to Elasticsearch")
            return
        
        print("[OK] Connected to Elasticsearch\n")
        
        # Get a sample document
        search_body = {
            "size": 3,
            "query": {"match_all": {}},
            "_source": True
        }
        
        try:
            response = es.search(index=ES_INDEX, body=search_body)
        except TypeError:
            response = es.search(index=ES_INDEX, **search_body)
        
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            print("No documents found in index")
            return
        
        print(f"Found {len(hits)} sample documents:\n")
        print("=" * 80)
        
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            print(f"\n--- Document {i} ---")
            print(f"ID: {hit.get('_id', 'N/A')}")
            print(f"Score: {hit.get('_score', 'N/A')}")
            print(f"\nSource fields:")
            for key, value in source.items():
                if key == "chunk_text":
                    # Truncate long text
                    text_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {text_preview}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_sample_data()

