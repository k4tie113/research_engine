#!/usr/bin/env python3
"""
fetch_metadata.py
------------------
Query the arXiv API for the latest NLP (cs.CL) papers and
save their metadata to research_engine/data/metadata/papers.csv.

This version fixes the path issue by resolving the project root
from the script's own location, so the output always ends up
inside the repository regardless of where you run the script.

It does NOT download the PDF files themselves—only the basic
information needed to fetch them later.
"""

import csv
import feedparser      # pip install feedparser
from pathlib import Path

# --------------------------------------------------------------------
# Determine the absolute path to the project root.
# __file__ is this script's path; parent = scripts/, parent.parent = project root.
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------
# Define the metadata directory INSIDE the project (not relative to CWD).
# This guarantees the CSV is created in research_engine/data/metadata/.
# --------------------------------------------------------------------
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# The CSV file that will hold the paper metadata.
CSV_PATH = METADATA_DIR / "papers.csv"

# --------------------------------------------------------------------
# Number of papers to request. 300 is a safe batch size for arXiv.
# --------------------------------------------------------------------
BATCH_SIZE = 300

# --------------------------------------------------------------------
# arXiv category for NLP research: 'cs.CL' (Computational Linguistics).
# --------------------------------------------------------------------
CATEGORY = "cs.CL"


def fetch_arxiv_metadata(category: str, max_results: int):
    """
    Query the arXiv API and yield metadata for each paper.

    Parameters
    ----------
    category : str
        arXiv category code (e.g. 'cs.CL').
    max_results : int
        Number of papers to fetch.

    Yields
    ------
    dict
        Dictionary with keys:
        'id', 'title', 'authors', and 'pdf_url'.
    """
    # ----------------------------------------------------------------
    # Build the URL for the arXiv API query.
    # Sort by submission date (newest first).
    # ----------------------------------------------------------------
    base_url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=cat:{category}&"
        f"start=0&max_results={max_results}&"
        "sortBy=submittedDate&sortOrder=descending"
    )

    # ----------------------------------------------------------------
    # Parse the Atom feed returned by arXiv.
    # ----------------------------------------------------------------
    feed = feedparser.parse(base_url)

    # ----------------------------------------------------------------
    # Each entry corresponds to a paper; collect the fields we need.
    # ----------------------------------------------------------------
    for entry in feed.entries:
        # Find the PDF link
        pdf_url = next((l.href for l in entry.links if l.type == "application/pdf"), "")
        # Join all author names into a single string separated by semicolons
        authors = "; ".join(a.name for a in entry.authors)
        yield {
            "id": entry.id.split("/")[-1],                 # arXiv identifier only
            "title": entry.title.replace("\n", " ").strip(),  # remove newlines and trim
            "authors": authors,
            "pdf_url": pdf_url
        }


def main():
    """
    Fetch metadata and write it to research_engine/data/metadata/papers.csv.
    """
    rows = list(fetch_arxiv_metadata(CATEGORY, BATCH_SIZE))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "authors", "pdf_url"])
        writer.writeheader()
        writer.writerows(rows)
    # Print the absolute path so you know exactly where the file was written.
    print(f"Wrote metadata for {len(rows)} papers to {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
