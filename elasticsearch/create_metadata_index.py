from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
USERNAME = "elastic"
PASSWORD = "jHBuBAEd"
INDEX_NAME = "paper_metadata"

es = Elasticsearch(
    ES_URL,
    basic_auth=(USERNAME, PASSWORD)
)

mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "authors": {"type": "text"},
            "abstract": {"type": "text"},
            "categories": {"type": "keyword"},
            "created": {"type": "date"}  # YYYY-MM-DD
        }
    }
}

# recreate clean index
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

es.indices.create(index=INDEX_NAME, body=mapping)
print("Metadata index created!")
