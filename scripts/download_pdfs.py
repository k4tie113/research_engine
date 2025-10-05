#!/usr/bin/env python3
"""
download_pdfs.py
----------------
Read data/metadata/papers.csv and download each paper's PDF
into data/pdfs/<arxiv_id>.pdf.
"""

from __future__ import annotations
import csv
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = ROOT / "data" / "metadata" / "papers.csv"
PDF_DIR = ROOT / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "research_engine/0.1 (+https://github.com/k4tie113/research_engine)"
})
TIMEOUT = 60
MAX_RETRIES = 3

def download_pdf(arxiv_id: str, pdf_url: str) -> None:
    out_path = PDF_DIR / f"{arxiv_id}.pdf"
    if out_path.exists():
        print(f"Skipping {arxiv_id} (already downloaded)")
        return
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with SESSION.get(pdf_url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded {arxiv_id}")
            return
        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                print(f"FAILED {arxiv_id}: {e}")
            else:
                sleep_s = 1.5 * attempt
                print(f"Retry {arxiv_id} in {sleep_s:.1f}s … ({e})")
                time.sleep(sleep_s)

def main():
    assert METADATA_CSV.exists(), f"Missing {METADATA_CSV}. Run fetch_metadata.py first."
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("pdf_url"):
                print(f"Missing pdf_url for {row.get('id')}")
                continue
            download_pdf(row["id"], row["pdf_url"])

if __name__ == "__main__":
    main()
