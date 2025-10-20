# retrieval/es_create_index_final.py
from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks_final"

def main():
    es = Elasticsearch(ES_URL)
    
    # Delete existing index if it exists
    if es.indices.exists(index=INDEX):
        print(f"Deleting existing index '{INDEX}'...")
        es.indices.delete(index=INDEX)
    
    print(f"Creating new index '{INDEX}'...")
    
    mapping = {
        "settings": {
            "index": {
                "analysis": {
                    "analyzer": {
                        "english_folded": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "porter_stem"]
                        }
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "paper_id":   {"type": "keyword"},
                "chunk_index":{"type": "integer"},
                "title":      {"type": "text", "analyzer": "english_folded"},
                "authors":    {"type": "text", "analyzer": "english_folded"},
                "chunk_text": {"type": "text", "analyzer": "english_folded"},
                "token_count":{"type": "integer"}
                # Note: year and categories removed since chunks_final.jsonl doesn't have them
            }
        }
    }
    
    es.indices.create(index=INDEX, body=mapping)
    print(f"Successfully created index '{INDEX}' with improved mapping")

if __name__ == "__main__":
    main()
