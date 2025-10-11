# retrieval/es_search.py
from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks"

def search(query: str, k: int = 10):
    es = Elasticsearch(ES_URL)
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["chunk_text^4", "title^2", "authors"]
            }
        },
        "_source": ["paper_id","chunk_index","title","authors","chunk_text"]
    }
    resp = es.search(index=INDEX, body=body)
    return [{"score":h["_score"], **h["_source"]} for h in resp["hits"]["hits"]]

if __name__ == "__main__":
    for h in search("contrastive pretraining retrieval augmentation", 5):
        print(h["score"], h["paper_id"], h["chunk_index"], h["title"][:80])
