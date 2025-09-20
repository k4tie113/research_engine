#!/usr/bin/env python3
"""
chunk_pdfs.py
-------------
This script reads every PDF in data/pdfs/,
extracts the text, and splits it into overlapping chunks
of about 800 tokens each.

Each chunk is written as a single line of JSON in
data/chunks.jsonl, along with metadata identifying
the paper it came from.

No embeddings or model calls are made here—we are
only preparing the data for later retrieval-augmented
generation (RAG) steps.
"""

import csv
import json
from pathlib import Path
from pypdf import PdfReader           # pure-Python PDF text extractor
import tiktoken                        # tokenizer for measuring tokens

# --------------------------------------------------------------------
# Base directory = parent of this scripts folder (the project root).
# This ensures we always read/write inside the repository no matter
# where we launch the script from.
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute paths inside the project
METADATA_CSV = BASE_DIR / "data" / "metadata" / "papers.csv"
PDF_DIR      = BASE_DIR / "data" / "pdfs"
CHUNKS_PATH  = BASE_DIR / "data" / "chunks.jsonl"

# --------------------------------------------------------------------
# Desired chunk size and overlap (both measured in tokens).
# 800 tokens per chunk with 200-token overlap is typical for RAG.
# --------------------------------------------------------------------
MAX_TOKENS   = 800
OVERLAP      = 200

# Initialize tokenizer compatible with common OpenAI embeddings.
enc = tiktoken.get_encoding("cl100k_base")

def pdf_to_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF, page by page.
    """
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_by_tokens(text: str, max_tokens: int, overlap: int):
    """
    Split text into overlapping chunks of approximately max_tokens tokens.
    """
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])   # decode back to text
        chunks.append(chunk)
        start += max_tokens - overlap           # slide window forward
    return chunks

def main():
    """
    1. Read paper metadata from the CSV.
    2. For each paper, load its PDF and extract text.
    3. Break the text into overlapping chunks.
    4. Write one JSON object per chunk to chunks.jsonl.
    """
    with open(METADATA_CSV, newline="", encoding="utf-8") as fmeta, \
         open(CHUNKS_PATH, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fmeta)
        for row in reader:
            pdf_file = PDF_DIR / f"{row['id']}.pdf"

            if not pdf_file.exists():
                print(f"PDF missing: {row['id']}")
                continue

            text = pdf_to_text(pdf_file)

            for i, chunk in enumerate(chunk_by_tokens(text, MAX_TOKENS, OVERLAP)):
                record = {
                    "paper_id": row["id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "chunk_index": i,
                    "chunk_text": chunk
                }
                fout.write(json.dumps(record) + "\n")
            print(f"Chunked {row['id']}")

    print(f"\nChunks written to: {CHUNKS_PATH.resolve()}")

if __name__ == "__main__":
    main()
