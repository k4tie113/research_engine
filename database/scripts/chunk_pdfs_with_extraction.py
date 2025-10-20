#!/usr/bin/env python3
"""
chunk_pdfs_with_extraction.py
-----------------------------
Improved PDF chunking script that extracts title and author information directly from PDFs
instead of relying on the metadata CSV (which has empty authors field).

This script:
1. Extracts title and authors directly from PDF content
2. Filters out metadata, citations, URLs, and other irrelevant content
3. Focuses on actual paper content
4. Provides better text cleaning and quality filtering
"""

from __future__ import annotations
import csv
import json
import re
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
OUT_JSONL = ROOT / "data" / "chunks_with_extraction.jsonl"

# === CONFIG ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_DOC_TOKENS = 50
MIN_CHUNK_TOKENS = 100
ENC = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))

# === TITLE AND AUTHOR EXTRACTION ===
def extract_title_and_authors(text: str) -> tuple[str, str]:
    """Extract title and authors from PDF text"""
    lines = text.split('\n')
    
    title = ""
    authors = ""
    
    # Look for title - it's usually the first substantial line that's not metadata
    for i, line in enumerate(lines[:15]):  # Check first 15 lines
        line = line.strip()
        if not line:
            continue
            
        # Skip common header/metadata patterns
        if any(pattern in line.lower() for pattern in [
            'arxiv:', 'proceedings', 'conference', 'journal', 'volume', 'page', 'doi:',
            'abstract', 'introduction', 'keywords', 'submitted', 'received', 'accepted'
        ]):
            continue
            
        # Skip very short lines or lines that look like metadata
        if len(line) < 15 or len(line) > 300:
            continue
            
        # Skip lines with too many special characters or numbers
        if len(re.findall(r'[^\w\s]', line)) > len(line) * 0.3:
            continue
            
        # Skip lines that look like author names or affiliations
        if (re.search(r'\b(et al|university|department|institute|laboratory|center|@|email|\.edu|\.org)\b', line.lower()) or
            re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line) or  # Just two names
            re.search(r'^[A-Z]\. [A-Z][a-z]+$', line)):  # Initial. Lastname
            continue
            
        # This looks like a title
        title = line
        break
    
    # Look for authors - usually after title, before abstract
    author_lines = []
    title_found = False
    
    for i, line in enumerate(lines[:25]):  # Check first 25 lines
        line = line.strip()
        if not line:
            continue
            
        # Start looking after we find the title
        if title and line == title:
            title_found = True
            continue
            
        if not title_found:
            continue
            
        # Stop if we hit abstract or other sections
        if any(pattern in line.lower() for pattern in ['abstract', 'introduction', 'keywords', '1.']):
            break
            
        # Look for author patterns
        if (re.search(r'\b(et al|and|university|department|institute|laboratory|center|@|email)\b', line.lower()) or
            re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+', line) or  # Name pattern
            re.search(r'^[A-Z]\. [A-Z][a-z]+', line) or  # Initial. Lastname pattern
            re.search(r'\([0-9,\s]+\)', line)):  # Affiliation numbers
            author_lines.append(line)
    
    # Combine author lines and clean them up
    if author_lines:
        # Clean up author lines
        cleaned_authors = []
        for line in author_lines:
            # Remove email addresses and URLs
            line = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', line)
            line = re.sub(r'https?://[^\s]+', '', line)
            line = line.strip()
            if line and len(line) > 5:
                cleaned_authors.append(line)
        
        if cleaned_authors:
            authors = '; '.join(cleaned_authors[:3])  # Take first 3 author lines
    
    return title, authors

def clean_metadata(text: str) -> str:
    """Remove common metadata patterns from academic papers"""
    
    # Remove page numbers (standalone numbers at start/end of lines)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove headers/footers with common patterns
    text = re.sub(r'^.*\b(?:Proceedings|Conference|Journal|Volume|Issue|Page|DOI|arXiv|arXiv:)\b.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove arXiv IDs
    text = re.sub(r'\b\d{4}\.\d{4,5}(v\d+)?\b', '', text)
    
    # Remove common citation patterns
    text = re.sub(r'\[[\d,\s-]+\]', '', text)  # [1, 2, 3] style citations
    text = re.sub(r'\([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4}\)', '', text)  # (Author et al., 2024)
    text = re.sub(r'\([A-Za-z]+\s+\d{4}\)', '', text)  # (Author 2024)
    
    # Remove standalone dates
    text = re.sub(r'^\s*\d{4}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove lines that are mostly punctuation or very short
    text = re.sub(r'^[^\w\s]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.{1,3}$', '', text, flags=re.MULTILINE)
    
    return text

def identify_sections(text: str) -> dict:
    """Identify different sections of the paper"""
    sections = {
        'abstract': '',
        'introduction': '',
        'main_content': '',
        'references': '',
        'acknowledgments': ''
    }
    
    lines = text.split('\n')
    current_section = 'main_content'
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identify section headers
        if re.match(r'^\d+\.?\s*(?:Abstract|ABSTRACT)', line, re.IGNORECASE):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'abstract'
            current_content = []
        elif re.match(r'^\d+\.?\s*(?:Introduction|INTRODUCTION)', line, re.IGNORECASE):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'introduction'
            current_content = []
        elif re.match(r'^\d+\.?\s*(?:References?|Bibliography|REFERENCES?|BIBLIOGRAPHY)', line, re.IGNORECASE):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'references'
            current_content = []
        elif re.match(r'^\d+\.?\s*(?:Acknowledgments?|ACKNOWLEDGMENTS?)', line, re.IGNORECASE):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'acknowledgments'
            current_content = []
        else:
            current_content.append(line)
    
    # Add remaining content
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def is_quality_content(text: str) -> bool:
    """Check if text contains quality academic content"""
    if not text or len(text.strip()) < 50:
        return False
    
    # Check for too many citations (likely reference section)
    citation_count = len(re.findall(r'\[[\d,\s-]+\]|\([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4}\)', text))
    if citation_count > len(text.split()) * 0.1:  # More than 10% citations
        return False
    
    # Check for too many URLs
    url_count = len(re.findall(r'https?://|www\.', text))
    if url_count > 3:
        return False
    
    # Check for meaningful content (has sentences, not just metadata)
    sentences = re.split(r'[.!?]+', text)
    meaningful_sentences = [s for s in sentences if len(s.strip()) > 20 and re.search(r'[a-zA-Z]', s)]
    
    if len(meaningful_sentences) < 2:
        return False
    
    # Check for academic language patterns
    academic_words = ['analysis', 'method', 'approach', 'model', 'algorithm', 'experiment', 'result', 'conclusion', 'study', 'research', 'data', 'performance', 'evaluation', 'comparison']
    text_lower = text.lower()
    academic_word_count = sum(1 for word in academic_words if word in text_lower)
    
    return academic_word_count >= 2

def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF with better formatting"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Skip pages that are mostly metadata
            if len(text.strip()) < 100:
                continue
                
            text_parts.append(text)
        
        return "\n".join(text_parts)
    except Exception:
        return ""

def extract_text_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber with better formatting"""
    try:
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 100:  # Skip mostly empty pages
                    out.append(text)
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_pypdf(pdf_path: Path) -> str:
    """Extract text using pypdf with better formatting"""
    try:
        reader = PdfReader(str(pdf_path))
        out = []
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 100:  # Skip mostly empty pages
                out.append(text)
        return "\n".join(out)
    except Exception:
        return ""

def extract_text(pdf_path: Path) -> str:
    """Extract text using multiple backends, preferring better results"""
    for fn in (extract_text_pymupdf, extract_text_pdfplumber, extract_text_pypdf):
        txt = fn(pdf_path)
        if txt and txt.strip() and len(txt.strip()) > 500:  # Ensure substantial content
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
    """Load metadata from CSV as fallback"""
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
    assert PDF_DIR.exists(), f"Missing {PDF_DIR}. Run download_oai.py first."

    meta = load_meta(METADATA_CSV)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # For testing, only process first 3 PDFs
    test_pdfs = pdfs[:3]
    print(f"Testing with {len(test_pdfs)} PDFs: {[p.name for p in test_pdfs]}")

    written = 0
    skipped = 0
    
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for pdf_path in tqdm(test_pdfs, desc="Chunking PDFs with extraction"):
            pid = pdf_path.stem
            
            print(f"\nProcessing: {pdf_path.name}")
            
            # Extract raw text
            raw = extract_text(pdf_path)
            if not raw.strip():
                print(f"  Empty extract: {pdf_path.name}")
                skipped += 1
                continue

            print(f"  Raw text length: {len(raw)} characters")
            
            # Extract title and authors from PDF content
            extracted_title, extracted_authors = extract_title_and_authors(raw)
            
            # Fallback to CSV metadata if extraction failed
            csv_meta = meta.get(pid, {"title": "", "authors": ""})
            title = extracted_title if extracted_title else csv_meta["title"]
            authors = extracted_authors if extracted_authors else csv_meta["authors"]
            
            print(f"  Extracted title: {title[:100]}..." if title else "  No title extracted")
            print(f"  Extracted authors: {authors[:100]}..." if authors else "  No authors extracted")
            
            # Clean metadata
            cleaned_metadata = clean_metadata(raw)
            print(f"  After metadata cleaning: {len(cleaned_metadata)} characters")
            
            # Identify sections
            sections = identify_sections(cleaned_metadata)
            print(f"  Sections found: {list(sections.keys())}")
            
            # Focus on main content, abstract, and introduction
            main_content = sections['main_content'] + " " + sections['abstract'] + " " + sections['introduction']
            
            # Apply basic cleaning
            cleaned = basic_clean(main_content, drop_refs=True)
            print(f"  After basic cleaning: {len(cleaned)} characters")
            
            if token_len(cleaned) < MIN_DOC_TOKENS:
                print(f"  Too short after cleaning: {token_len(cleaned)} tokens")
                skipped += 1
                continue

            # Chunk the text
            toks = ENC.encode(cleaned, disallowed_special=())
            chunk_count = 0
            
            for idx, txt, tok_count in chunk_tokens_to_text(toks, CHUNK_SIZE, CHUNK_OVERLAP):
                # Quality check for each chunk
                if not is_quality_content(txt):
                    print(f"    Chunk {idx}: Skipped (low quality)")
                    continue
                
                if tok_count < MIN_CHUNK_TOKENS:
                    print(f"    Chunk {idx}: Skipped (too short: {tok_count} tokens)")
                    continue
                
                out_f.write(json.dumps({
                    "paper_id": pid,
                    "chunk_index": idx,
                    "title": title,
                    "authors": authors,
                    "token_count": tok_count,
                    "chunk_text": txt
                }, ensure_ascii=False) + "\n")
                written += 1
                chunk_count += 1
                print(f"    Chunk {idx}: Added ({tok_count} tokens)")
            
            print(f"  Total chunks added: {chunk_count}")

    print(f"\nWrote {written} total chunks to {OUT_JSONL.resolve()}")
    print(f"Skipped {skipped} PDFs")

if __name__ == "__main__":
    main()
