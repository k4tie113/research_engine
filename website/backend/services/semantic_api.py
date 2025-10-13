import requests

def search_semantic(query):
    """Query Semantic Scholar API and return normalized results."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,year,authors,url,abstract"
    res = requests.get(url)

    if res.status_code != 200:
        return []

    data = res.json()
    results = []
    for paper in data.get("data", []):
        results.append({
            "title": paper.get("title", "Untitled"),
            "authors": [a["name"] for a in paper.get("authors", [])],
            "year": paper.get("year"),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", "")
        })
    return results
