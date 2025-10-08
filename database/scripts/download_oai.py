#!/usr/bin/env python3
"""
download_oai_batched.py
------------------------
Downloads arXiv PDFs in batches of 3,000 per run.

Each time you run this script, it picks up where it left off
(using a checkpoint file) and downloads the next 3,000 PDFs.

Example:
  $ python3 scripts/download_oai_batched.py
"""

import csv
import requests
from pathlib import Path

# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "papers.csv"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "download_checkpoint.txt"
BATCH_SIZE = 3000

PDF_DIR.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
def sanitize_id(arxiv_id: str) -> str:
    """Replace '/' with '_' for safe file names."""
    return arxiv_id.replace("/", "_")

def get_last_index() -> int:
    """Read checkpoint file; return 0 if missing."""
    if CHECKPOINT_PATH.exists():
        try:
            return int(CHECKPOINT_PATH.read_text().strip())
        except ValueError:
            return 0
    return 0

def save_checkpoint(idx: int):
    """Write current index to checkpoint."""
    CHECKPOINT_PATH.write_text(str(idx))

def download_pdf(arxiv_id: str):
    """Download one PDF if not already present."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = PDF_DIR / f"{sanitize_id(arxiv_id)}.pdf"

    if out_path.exists():
        print(f"↩️  Skipping {arxiv_id} (already exists)")
        return

    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded {arxiv_id}")
    except Exception as e:
        print(f"⚠️  Failed {arxiv_id}: {e}")

# === MAIN ===
def main():
    start_index = get_last_index()
    print(f"🚀 Starting new batch download (index {start_index}, batch size {BATCH_SIZE})")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        total = len(reader)

        end_index = min(start_index + BATCH_SIZE, total)
        print(f"📦 Downloading rows {start_index}–{end_index - 1} of {total}")

        for i in range(start_index, end_index):
            arxiv_id = reader[i]["id"]
            download_pdf(arxiv_id)

        save_checkpoint(end_index)
        print(f"\n✅ Batch complete. Downloaded {end_index - start_index} papers.")
        print(f"💾 Checkpoint saved at index {end_index}/{total}.")
        print("👉 Run again to download the next batch.")

if __name__ == "__main__":
    main()
