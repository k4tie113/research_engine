# retrieval/es_create_index.py
from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks"

def main():
    es = Elasticsearch(ES_URL)
    if es.indices.exists(index=INDEX):
        print(f"Index '{INDEX}' already exists"); return

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
                "token_count":{"type": "integer"},
                "year":       {"type": "integer"},
                "categories": {"type": "keyword"}
                # later (hybrid): "embedding": {"type":"dense_vector","dims":1024,"index":True,"similarity":"cosine"}
            }
        }
    }
    es.indices.create(index=INDEX, body=mapping)
    print(f"Created '{INDEX}'")

if __name__ == "__main__":
    main()
