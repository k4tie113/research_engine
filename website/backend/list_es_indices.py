#!/usr/bin/env python3
"""
list_es_indices.py
------------------
Simple script to list all indices in your Elasticsearch cluster.
"""

from elasticsearch import Elasticsearch

# === ELASTICSEARCH CONFIG ===
ES_URL = "https://my-elasticsearch-project-fb6996.es.us-central1.gcp.elastic.cloud"
ES_API_KEY = "SHR3SFY1b0J5WWFqM1RHdHBMQ2g6U2JsaEZnNDFadEZ2RFNSUzdQY3VYZw=="

def list_indices():
    """List all indices in the Elasticsearch cluster."""
    try:
        # Initialize Elasticsearch client
        es = Elasticsearch(
            [ES_URL],
            api_key=ES_API_KEY
        )
        
        # Test connection
        if not es.ping():
            print("[ERROR] Failed to connect to Elasticsearch")
            return
        
        print("[OK] Connected to Elasticsearch successfully!\n")
        
        # Get all indices
        indices = es.indices.get_alias(index="*")
        
        if not indices:
            print("No indices found in the cluster.")
            return
        
        print(f"Found {len(indices)} index(es):\n")
        print("-" * 80)
        
        for index_name in sorted(indices.keys()):
            # Skip system indices (optional - you can remove this filter if you want to see them)
            if index_name.startswith('.'):
                continue
                
            try:
                # Get index stats
                stats = es.count(index=index_name)
                doc_count = stats['count']
                
                # Get index settings to see creation date, etc.
                settings = es.indices.get_settings(index=index_name)
                index_settings = settings[index_name].get('settings', {})
                
                print(f"Index: {index_name}")
                print(f"   Documents: {doc_count:,}")
                print(f"   Status: {index_settings.get('index', {}).get('verified_before_close', 'N/A')}")
                print()
            except Exception as e:
                print(f"Index: {index_name}")
                print(f"   Error getting stats: {e}")
                print()
        
        print("-" * 80)
        print("\n[TIP] Copy the index name you want to use and update ES_INDEX in rag_service.py")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_indices()

