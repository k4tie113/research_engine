# retrieval/es_search_final.py
from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
INDEX = "papers_chunks_final"

def search_chunks(query: str, size: int = 10):
    """Search the final chunks index with improved query"""
    es = Elasticsearch(ES_URL)
    
    # Enhanced search query with better scoring
    search_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_text^3", "title^2", "authors"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_text", "title", "authors"],
                            "type": "phrase",
                            "boost": 2.0
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "size": size,
        "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "token_count"]
    }
    
    response = es.search(index=INDEX, body=search_body)
    return response["hits"]["hits"]

def main():
    es = Elasticsearch(ES_URL)
    
    # Check if index exists
    if not es.indices.exists(index=INDEX):
        print(f"Index '{INDEX}' does not exist. Please run es_create_index_final.py and es_index_chunks_final.py first.")
        return
    
    # Get index stats
    stats = es.count(index=INDEX)
    print(f"Index '{INDEX}' contains {stats['count']} chunks")
    
    # Test search
    test_queries = [
        "neural networks",
        "transformer architecture", 
        "language models",
        "machine learning",
        "natural language processing"
    ]
    
    for query in test_queries:
        print(f"\n--- Searching for: '{query}' ---")
        results = search_chunks(query, size=3)
        
        for i, hit in enumerate(results, 1):
            source = hit["_source"]
            score = hit["_score"]
            print(f"{i}. Score: {score:.2f}")
            print(f"   Paper: {source.get('paper_id', 'N/A')}")
            print(f"   Title: {source.get('title', 'N/A')}")
            print(f"   Authors: {source.get('authors', 'N/A')}")
            text_preview = source.get('chunk_text', '')[:200]
            # Handle Unicode characters for display
            text_preview = text_preview.encode('ascii', 'ignore').decode('ascii')
            print(f"   Text preview: {text_preview}...")
            print()

if __name__ == "__main__":
    main()
