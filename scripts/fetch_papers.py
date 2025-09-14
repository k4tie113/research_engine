"""
fetch_papers.py
---------------
Fetches papers from arXiv API (NLP-related).
Downloads PDFs into data/ and saves metadata into papers.csv.
"""

import arxiv
import pandas as pd
import requests, os, time

os.makedirs("data", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# Number of papers to fetch (max ~2000 per query)
MAX_RESULTS = 200

search = arxiv.Search(
    query="natural language processing",
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

records = []

for result in search.results():
    paper_id = result.get_short_id()
    title = result.title
    abstract = result.summary
    pdf_url = result.pdf_url

    records.append({"id": paper_id, "title": title, "abstract": abstract, "pdf_url": pdf_url})

    pdf_path = f"data/{paper_id}.pdf"
    if not os.path.exists(pdf_path):
        try:
            r = requests.get(pdf_url, timeout=15)
            if r.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
                print(f"📄 Downloaded {pdf_path}")
                time.sleep(1)  # avoid hitting rate limits
            else:
                print(f"⚠️ Failed to download {pdf_url}")
        except Exception as e:
            print(f"Error for {paper_id}: {e}")

df = pd.DataFrame(records)
df.to_csv("metadata/papers.csv", index=False)
print(f"Saved {len(records)} papers to metadata/papers.csv")
