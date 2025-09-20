#!/usr/bin/env python3
"""
download_pdfs.py
----------------
Read data/metadata/papers.csv and download each paper's PDF
into data/pdfs/<arxiv_id>.pdf.

We keep PDFs locally so that chunk_pdfs.py can extract text.
"""

import csv
import requests
from pathlib import Path

METADATA_CSV = Path("../data/metadata/papers.csv")
PDF_DIR = Path("../data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

def download_pdf(arxiv_id: str, pdf_url: str):
    """
    Stream the PDF to disk to avoid large memory usage.
    """
    out_path = PDF_DIR / f"{arxiv_id}.pdf"
    if out_path.exists():
        print(f"Skipping {arxiv_id} (already downloaded)")
        return
    r = requests.get(pdf_url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {arxiv_id}")

def main():
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            download_pdf(row["id"], row["pdf_url"])

if __name__ == "__main__":
    main()
