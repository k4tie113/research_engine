#!/usr/bin/env python3
"""
chunk_pdfs_oai.py
-----------------
Extract text from PDFs downloaded via fetch_oai.py and split them into
~800-token chunks with 200-token overlap. Output to data/chunks_oai.jsonl.

This prepares data for embedding with GritLM or other models.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

from tqdm import tqdm
import tiktoken
import fitz          # PyMuPDF
import pdfplumber
from pypdf import PdfReader

from text_clean import basic_clean

# === PATHS ===
ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = ROOT / "data" / "metadata" / "papers_oai.csv"
PDF_DIR = ROOT / "data" / "pdfs"
OUT_JSONL = ROOT / "data" / "chunks_oai.jsonl"

# === CONFIG ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_DOC_TOKENS = 50  # skip too-short extracts
ENC = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(ENC.encode(text))

# === MULTI-BACKEND PDF TEXT EXTRACTION ===
def extract_text_pymupdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        return ""

def extract_text_pdfplumber(pdf_path: Path) -> str:
    try:
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_pypdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""

def extract_text(pdf_path: Path) -> str:
    for fn in (extract_text_pymupdf, extract_text_pdfplumber, extract_text_pypdf):
        txt = fn(pdf_path)
        if txt and txt.strip():
            return txt
    return ""

# === CHUNKING HELPERS ===
def sliding_windows(tokens, size: int, overlap: int):
    step = max(1, size - overlap)
    n = len(tokens)
    i = 0
    while i < n:
        j = min(n, i + size)
        yield i, j
        if j >= n:
            break
        i += step

def chunk_tokens_to_text(tokens, size: int, overlap: int):
    for idx, (s, e) in enumerate(sliding_windows(tokens, size, overlap)):
        subtoks = tokens[s:e]
        yield idx, ENC.decode(subtoks), len(subtoks)

def load_meta(csv_path: Path):
    meta = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["id"]] = {
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
            }
    return meta

# === MAIN ===
def main():
    assert METADATA_CSV.exists(), f"Missing {METADATA_CSV}. Run fetch_oai.py first."
    assert PDF_DIR.exists(), f"Missing {PDF_DIR}. Run download_oai_batched.py first."

    meta = load_meta(METADATA_CSV)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for pdf_path in tqdm(pdfs, desc="Chunking OAI PDFs"):
            pid = pdf_path.stem
            m = meta.get(pid, {"title": "", "authors": ""})

            raw = extract_text(pdf_path)
            if not raw.strip():
                print(f"⚠️  Empty extract: {pdf_path.name}")
                continue

            cleaned = basic_clean(raw, drop_refs=True)
            if token_len(cleaned) < MIN_DOC_TOKENS:
                continue

            toks = ENC.encode(cleaned)
            for idx, txt, tok_count in chunk_tokens_to_text(toks, CHUNK_SIZE, CHUNK_OVERLAP):
                out_f.write(json.dumps({
                    "paper_id": pid,
                    "chunk_index": idx,
                    "title": m["title"],
                    "authors": m["authors"],
                    "token_count": tok_count,
                    "chunk_text": txt
                }, ensure_ascii=False) + "\n")
                written += 1

    print(f"\n✅ Wrote {written} total chunks → {OUT_JSONL.resolve()}")

if __name__ == "__main__":
    main()
