# Database Scripts - Chunking Pipeline

This directory contains scripts for harvesting, downloading, and chunking NLP research papers from arXiv.

## Overview

The pipeline consists of the following steps:

1. **Snowballing** - Discover related papers using Semantic Scholar API
2. **Metadata Harvesting** - Fetch paper metadata from arXiv OAI-PMH
3. **PDF Download** - Download PDFs in batches with checkpointing
4. **Text Extraction & Chunking** - Extract text and create searchable chunks
5. **Elasticsearch Indexing** - Index chunks for search and retrieval

## Prerequisites

- Python 3.8+
- Virtual environment activated
- Required packages installed (see requirements.txt)
- Docker running (for Elasticsearch)

## Pipeline Steps

### 1. Snowballing (Discover Related Papers)

```bash
cd database/scripts
python snowballer.py
```

This script:
- Uses Semantic Scholar API to find papers related to your seed papers
- Discovers papers through references and citations
- Saves discovered paper IDs to `data/metadata/snowball_queue.csv`
- Supports resumption via checkpoint files

### 2. Fetch Metadata from arXiv

```bash
python fetch_oai.py
```

This script:
- Harvests ALL cs.CL (Computational Linguistics) papers from arXiv
- Uses OAI-PMH protocol for comprehensive coverage
- Saves metadata to `data/metadata/papers_oai.csv`
- Supports checkpointing and resumption
- May take several hours for complete harvest

### 3. Add Snowballed Papers to Metadata

```bash
python fetch_metadata_from_queue.py
```

This script:
- Adds discovered papers from snowballing to the main metadata file
- Avoids duplicates
- Extends your paper collection with related work

### 4. Download PDFs

```bash
python download_oai.py
```

This script:
- Downloads PDFs in batches of 3,000 papers
- Uses checkpointing to resume interrupted downloads
- Saves PDFs to `data/pdfs/`
- Handles rate limiting and errors gracefully
- Run multiple times to download all papers

### 5. Extract Text and Create Chunks

```bash
python chunk_pdfs_oai.py
```

This script:
- Extracts text from PDFs using multiple backends (PyMuPDF, pdfplumber, pypdf)
- Creates ~800-token chunks with 200-token overlap
- Cleans text and removes reference sections
- Saves chunks to `data/chunks_oai.jsonl`
- Handles special tokens and encoding issues

### 6. Index into Elasticsearch

```bash
cd ../../retrieval
python es_index_chunks.py
```

This script:
- Indexes chunks into Elasticsearch for search
- Creates searchable index with proper mappings
- Supports both old and new chunk files

## File Structure

```
database/
├── scripts/
│   ├── snowballer.py              # Discover related papers
│   ├── fetch_oai.py               # Harvest arXiv metadata
│   ├── fetch_metadata_from_queue.py # Add snowballed papers
│   ├── download_oai.py            # Download PDFs
│   ├── chunk_pdfs_oai.py          # Extract text and chunk
│   ├── text_clean.py              # Text cleaning utilities
│   └── run_all.py                 # Master script (runs steps 1-3)
├── data/
│   ├── metadata/
│   │   ├── papers_oai.csv         # Main metadata file
│   │   ├── snowball_queue.csv     # Discovered paper IDs
│   │   └── snowball_state.json    # Snowballing state
│   ├── pdfs/                      # Downloaded PDFs
│   └── chunks_oai.jsonl           # Extracted text chunks
└── README.md                      # This file
```

## Configuration

### Chunking Parameters

In `chunk_pdfs_oai.py`:
- `CHUNK_SIZE = 800` - Target tokens per chunk
- `CHUNK_OVERLAP = 200` - Overlap between chunks
- `MIN_DOC_TOKENS = 50` - Minimum tokens to keep a document

### Download Parameters

In `download_oai.py`:
- `BATCH_SIZE = 3000` - Papers per batch
- Rate limiting and retry logic built-in

### Snowballing Parameters

In `snowballer.py`:
- `max_new=1500` - Maximum new papers to discover
- `max_depth=2` - Maximum citation/reference depth

## Running the Complete Pipeline

### Option 1: Run All Steps Manually

```bash
# 1. Start with snowballing
python snowballer.py

# 2. Fetch metadata from arXiv
python fetch_oai.py

# 3. Add snowballed papers
python fetch_metadata_from_queue.py

# 4. Download PDFs (run multiple times as needed)
python download_oai.py

# 5. Create chunks
python chunk_pdfs_oai.py

# 6. Index into Elasticsearch
cd ../../retrieval
python es_index_chunks.py
```

### Option 2: Use Master Script

```bash
python run_all.py
```

This runs steps 1-3 automatically. You'll still need to run steps 4-6 manually.

## Troubleshooting

### Common Issues

1. **Unicode Errors**: Fixed in scripts, but may occur with special characters
2. **PDF Extraction Failures**: Scripts use multiple backends for robustness
3. **Rate Limiting**: Built-in delays and retry logic
4. **Memory Issues**: Chunking processes files one at a time

### Checkpoint Files

- `data/metadata/checkpoint.txt` - OAI harvest progress
- `data/download_checkpoint.txt` - PDF download progress
- `data/metadata/snowball_state.json` - Snowballing state

### Logs and Output

- Progress bars show real-time status
- Error messages indicate specific failures
- Checkpoint files allow resumption

## Output Files

- **papers_oai.csv**: Complete metadata for all papers
- **chunks_oai.jsonl**: Searchable text chunks (28,000+ chunks)
- **PDFs**: Raw PDF files for reference
- **Elasticsearch Index**: Searchable chunk store (40,000+ chunks)

## Performance Notes

- Full pipeline may take 6-12 hours depending on paper count
- PDF download is the most time-consuming step
- Chunking processes ~4-8 papers per second
- Elasticsearch indexing is very fast (~2,000 chunks/second)

## Next Steps

After completing the pipeline:

1. **Test Search**: Use `retrieval/es_search.py` to test search functionality
2. **RAG System**: Use `retrieval/rag_generator.py` for question-answering
3. **Web Interface**: Use the website backend for interactive search

## Dependencies

Key Python packages:
- `sickle` - OAI-PMH client
- `requests` - HTTP requests
- `tiktoken` - Tokenization
- `PyMuPDF` - PDF text extraction
- `pdfplumber` - Alternative PDF extraction
- `pypdf` - Another PDF extraction option
- `elasticsearch` - Search indexing
- `tqdm` - Progress bars
