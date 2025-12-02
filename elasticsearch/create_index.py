from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "jHBuBAEd")
)

index_name = "paper_chunks"

mapping = {
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "title": {"type": "text"},
            "authors": {"type": "text"},
            "token_count": {"type": "integer"},
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

# Create fresh index (delete if exists)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

es.indices.create(index=index_name, body=mapping)
print("Index created!")
