#!/usr/bin/env python3
"""
fetch_oai.py
-------------
Harvest up to MAX_PAPERS Computational Linguistics (cs.CL) papers
from arXiv using the OAI-PMH interface.
"""

from pathlib import Path
import csv
from sickle import Sickle

CSV_PATH = Path("data/metadata/papers_oai.csv")
CHECKPOINT_SIZE = 1000   # progress print interval
MAX_PAPERS = 500         # ALERT THIS IS FOR MY TESTING ONLY. change this

def safe_join_author_list(author_list):
    """Return a '; ' joined string, ignoring None values."""
    return "; ".join([a for a in author_list if isinstance(a, str)])

def main():
    sickle = Sickle("https://oaipmh.arxiv.org/oai")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "title", "authors", "abstract", "categories", "datestamp"]
        )
        writer.writeheader()

        print(f"🚀 Harvesting up to {MAX_PAPERS} cs.CL papers from arXiv…")

        records = sickle.ListRecords(metadataPrefix="arXiv", set="cs:cs:CL")

        kept = 0
        for record in records:
            meta = record.metadata
            categories = [c for c in meta.get("categories", []) if c]

            kept += 1
            writer.writerow({
                "id":        meta.get("id", [""])[0],
                "title":     (meta.get("title", [""])[0] or "").replace("\n", " ").strip(),
                "authors":   safe_join_author_list(meta.get("authors", [])),
                "abstract":  (meta.get("abstract", [""])[0] or "").replace("\n", " ").strip(),
                "categories": " ".join(categories),
                "datestamp": record.header.datestamp
            })

            if kept % CHECKPOINT_SIZE == 0:
                print(f"{kept} cs.CL papers saved so far")

            # ---- stop early when we reach the max ----
            if kept >= MAX_PAPERS:
                print(f"\n⏹️  Reached MAX_PAPERS limit ({MAX_PAPERS}). Stopping early.")
                break

        print(f"\n🎉 Harvest complete!")
        print(f"   cs.CL papers written: {kept}")
        print(f"   Output file: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
