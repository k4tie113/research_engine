#!/usr/bin/env python3
"""
download_oai_fixed_parallel_sampled_batches.py
----------------------------------------------
Downloads ~3,000 evenly spaced PDFs per run from papers_oai_combined.csv
into database/data/pdfs/batch_[offset]/, continuing from last offset.

Each run samples every STEP_SIZE-th paper (e.g., every 120th) to maintain coverage.
"""

import csv
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "papers_oai_combined.csv"
PDF_ROOT = PROJECT_ROOT / "data" / "pdfs"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "metadata" / "offset.txt"

BATCH_SIZE = 3000
STEP_SIZE = 120     # how spaced out papers are
MAX_WORKERS = 10    # number of concurrent downloads

PDF_ROOT.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
def sanitize_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(":", "_").strip()


def get_last_offset() -> int:
    if CHECKPOINT_PATH.exists():
        try:
            return int(CHECKPOINT_PATH.read_text().strip())
        except ValueError:
            return 0
    return 0


def save_offset(offset: int):
    CHECKPOINT_PATH.write_text(str(offset))


def download_pdf(arxiv_id: str, batch_dir: Path):
    """Try downloading a single arXiv paper."""
    out_path = batch_dir / f"{sanitize_id(arxiv_id)}.pdf"
    if out_path.exists():
        return f"↩️  Skipped {arxiv_id}"

    base_ids = [arxiv_id, f"{arxiv_id}v1", f"{arxiv_id}v2"]
    base_urls = [f"https://export.arxiv.org/pdf/{bid}.pdf" for bid in base_ids]

    for url in base_urls:
        try:
            with requests.get(url, stream=True, timeout=25) as r:
                if r.status_code != 200:
                    continue
                if "application/pdf" in r.headers.get("Content-Type", "") or r.content[:4] == b"%PDF":
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return f"✅ {arxiv_id}"
        except Exception:
            continue

    return f"⚠️  Failed {arxiv_id}"


def main():
    offset = get_last_offset()
    batch_dir = PDF_ROOT / f"batch_{offset}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting sampled batch (offset={offset}, batch={BATCH_SIZE}, step={STEP_SIZE}, workers={MAX_WORKERS})")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        total = len(reader)

        # sample every STEP_SIZE entries
        indices = list(range(offset, total, STEP_SIZE))[:BATCH_SIZE]
        print(f"📦 Downloading {len(indices)} papers out of {total:,} total → into {batch_dir}")

        arxiv_ids = [
            reader[i]["id"].strip().replace("https://arxiv.org/abs/", "").replace("arXiv:", "")
            for i in indices
        ]

        success = fail = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_pdf, aid, batch_dir): aid for aid in arxiv_ids}
            for future in as_completed(futures):
                msg = future.result()
                print(msg)
                if msg.startswith("✅"):
                    success += 1
                elif msg.startswith("⚠️"):
                    fail += 1

        save_offset(offset + 1)
        print(f"\n✅ Batch complete: {success} ok, {fail} failed.")
        print(f"💾 Offset updated to {offset + 1}.")
        print("👉 Run again to continue.\n")


if __name__ == "__main__":
    main()
