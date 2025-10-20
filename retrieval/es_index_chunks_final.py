# retrieval/es_index_chunks_final.py
import json
from pathlib import Path
from elasticsearch import Elasticsearch, helpers

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks_final"

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "database" / "data" / "chunks_final.jsonl"

def actions():
    """Generate Elasticsearch actions for bulk indexing"""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                # Create unique ID from paper_id and chunk_index
                _id = f"{obj.get('paper_id','na')}::{obj.get('chunk_index',0)}"
                yield {"_index": INDEX, "_id": _id, "_source": obj}
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

def main():
    assert CHUNKS_FILE.exists(), f"Chunks file not found: {CHUNKS_FILE}"
    
    print(f"Indexing chunks from: {CHUNKS_FILE}")
    print(f"Target index: {INDEX}")
    
    es = Elasticsearch(ES_URL)
    
    # Check if index exists
    if not es.indices.exists(index=INDEX):
        print(f"Index '{INDEX}' does not exist. Please run es_create_index_final.py first.")
        return
    
    # Count chunks in file
    chunk_count = 0
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunk_count = sum(1 for line in f)
    
    print(f"Found {chunk_count} chunks to index...")
    
    # Bulk index the chunks
    try:
        success_count, failed_items = helpers.bulk(
            es, 
            actions(), 
            chunk_size=1000, 
            request_timeout=120,
            max_retries=3
        )
        
        # Refresh index to make changes searchable
        es.indices.refresh(index=INDEX)
        
        # Get final count
        final_count = es.count(index=INDEX)["count"]
        
        print(f"Successfully indexed {success_count} chunks")
        print(f"Final index count: {final_count}")
        
        if failed_items:
            print(f"Failed items: {len(failed_items)}")
            for item in failed_items[:5]:  # Show first 5 failures
                print(f"  - {item}")
    
    except Exception as e:
        print(f"Error during indexing: {e}")

if __name__ == "__main__":
    main()
