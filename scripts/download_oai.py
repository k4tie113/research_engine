#!/usr/bin/env python3
"""
download_oai.py
----------------
Read data/metadata/papers_oai.csv and download each paper's PDF
into data/pdfs/<sanitized_id>.pdf.

If the arXiv id contains a slash (old-style IDs), we replace it
with an underscore for the local file name.
"""

import csv
import requests
from pathlib import Path

CSV_PATH = Path("data/metadata/papers_oai.csv")
PDF_DIR  = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_id(arxiv_id: str) -> str:
    """Replace '/' with '_' so it can be used as a file name."""
    return arxiv_id.replace("/", "_")

def download_pdf(arxiv_id: str):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = PDF_DIR / f"{sanitize_id(arxiv_id)}.pdf"

    if out_path.exists():
        print(f"↩️  Skipping {arxiv_id} (already downloaded)")
        return

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {arxiv_id} → {out_path.name}")

def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                download_pdf(row["id"])
            except Exception as e:
                print(f"⚠️  Failed {row['id']}: {e}")

if __name__ == "__main__":
    main()
