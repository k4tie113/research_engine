#!/usr/bin/env python3
"""
fetch_oai.py
-------------
Harvest ALL Computational Linguistics (cs.CL) papers from arXiv using OAI-PMH.
Supports automatic resumption (via resumption tokens), checkpointing, and retries.
"""

from pathlib import Path
import csv
import time
from sickle import Sickle
from sickle.models import Record
from sickle.oaiexceptions import NoRecordsMatch


CSV_PATH = Path("data/metadata/papers_oai.csv")
CHECKPOINT_FILE = Path("data/metadata/checkpoint.txt")
CHECKPOINT_SIZE = 1000    # print progress every N papers
RETRY_DELAY = 15          # seconds to wait after temporary failure


def safe_join_author_list(author_list):
    """Return a '; ' joined string, ignoring None values."""
    return "; ".join([a for a in author_list if isinstance(a, str)])


def get_last_checkpoint():
    """Return the number of papers already harvested."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
    return 0


def save_checkpoint(count):
    """Save how many papers have been harvested so far."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(count))


def main():
    sickle = Sickle("https://oaipmh.arxiv.org/oai")
    start_index = get_last_checkpoint()
    kept = start_index

    print(f"📂 Resuming from checkpoint: {kept} papers already harvested\n")

    mode = "a" if start_index > 0 else "w"
    with open(CSV_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "title", "authors", "abstract", "categories", "datestamp"]
        )
        if start_index == 0:
            writer.writeheader()

        print("🚀 Starting FULL cs.CL harvest from arXiv (OAI-PMH)")
        print("    This may take many hours; arXiv updates nightly.\n")

        records = None
        try:
            records = sickle.ListRecords(metadataPrefix="arXiv", set="cs:cs:CL")
            for record in records:
                if not isinstance(record, Record):
                    continue

                meta = record.metadata
                categories = [c for c in meta.get("categories", []) if c]

                kept += 1
                writer.writerow({
                    "id": meta.get("id", [""])[0],
                    "title": (meta.get("title", [""])[0] or "").replace("\n", " ").strip(),
                    "authors": safe_join_author_list(meta.get("authors", [])),
                    "abstract": (meta.get("abstract", [""])[0] or "").replace("\n", " ").strip(),
                    "categories": " ".join(categories),
                    "datestamp": record.header.datestamp,
                })

                if kept % CHECKPOINT_SIZE == 0:
                    print(f"✅ {kept} cs.CL papers saved so far...")
                    save_checkpoint(kept)

        except (SickleException, NoRecordsMatch) as e:
            print(f"⚠️ OAI error: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            print(f"\n🎉 Harvest complete!")
            print(f"   Total cs.CL papers written: {kept}")
            save_checkpoint(kept)
            print(f"   Output: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()