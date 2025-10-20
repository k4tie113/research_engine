#!/usr/bin/env python3
import random
import jsonlines
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "metadata" / "papers_oai.csv"  # fallback if JSONL not used
CHUNKS_PATH = ROOT / "data" / "chunks_oai.jsonl"

# === HELPER: Get arXiv date ===
def get_arxiv_date(paper_id: str) -> str:
    """Fetch publication date using arXiv API."""
    arxiv_id = paper_id.replace("_", "/")
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return "Unknown"
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return "Unknown"
        published = entry.find("atom:published", ns)
        return published.text.split("T")[0] if published is not None else "Unknown"
    except Exception as e:
        return f"Unknown ({e.__class__.__name__})"

# === LOAD UNIQUE PAPER IDS ===
paper_ids = set()
with jsonlines.open(CHUNKS_PATH, "r") as reader:
    for rec in reader:
        paper_ids.add(rec["paper_id"])

paper_ids = list(paper_ids)
sample = random.sample(paper_ids, min(100, len(paper_ids)))

# === FETCH DATES ===
print(f"📚 Checking publication dates for {len(sample)} random papers…\n")
for pid in sample:
    date = get_arxiv_date(pid)
    print(f"{pid:20}  📅  {date}")

print("\n✅ Done! You can use this to see how old your corpus is.")
