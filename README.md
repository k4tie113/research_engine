# Research Engine – Local RAG Pre-Processing Pipeline

This repo provides a **local, end-to-end pipeline** for preparing arXiv NLP papers for Retrieval-Augmented Generation (RAG). It downloads papers, extracts text, and produces **overlapping ~800-token chunks** in JSON-Lines format, ready to be embedded (e.g., with **GritLM**).

---

## How the Pipeline Works

### UPDATE 9/28
### `fetch_oai.py`
Uses the **arXiv OAI-PMH interface (https://oaipmh.arxiv.org/oai)** to harvest all Computational Linguistics (cs.CL) papers.
Requests the official set spec **cs:cs:CL**, automatically follows resumption tokens so there is no 5 000-record limit, and writes
data/metadata/papers_oai.csv with columns: id, title, authors, abstract, categories, datestamp.

### `download_oai.py`
Reads papers_oai.csv and downloads each paper’s PDF to **data/pdfs/<sanitized_id>.pdf**.
Builds the correct arXiv PDF URL (https://arxiv.org/pdf/<id>.pdf) and replaces any “/” in old-style IDs with “_” for the local file name to avoid folder-path errors.
Streams each PDF in chunks to avoid high memory usage.

### `fetch_metadata.py`
Calls the arXiv API for category `cs.CL` and writes `data/metadata/papers.csv` with columns: `id`, `title`, `authors`, `pdf_url`.  
**Note:** Intentionally limited to **300 papers per run** to respect arXiv rate limits.

### `download_pdfs.py`
Reads the CSV and streams each PDF to `data/pdfs/<id>.pdf` to avoid high memory usage.

### `chunk_pdfs.py`
Extracts text from every PDF and splits it into **~800-token chunks with 200-token overlap** (tokenized via `tiktoken`).  
Writes one JSON object per chunk to `data/chunks.jsonl`:
```json
{"paper_id":"2301.01234","chunk_index":0,"title":"...","authors":"...","chunk_text":"..."}
```

### `run_all.py`
Runs the three steps above in order.  

Paths are resolved relative to the repository root (via __file__) so outputs always land in research_engine/data/... regardless of where you run the command.

---

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Creating Vector Store

### From Drive (recommended)
- Place `embeddings_minilm.npy`, `faiss_index_minilm.bin`, and `metadata_minilm.jsonl` in `database\data\embeddings`
- Place `chunks_oai.jsonl` in `database\data`

### End-to-end
```bash
python scripts/run_all.py
```
### Or step-by-step
```bash
python scripts/fetch_metadata.py
python scripts/download_pdfs.py
python scripts/chunk_pdfs.py
```
### Outputs
- `data/metadata/papers.csv`  
- `data/pdfs/*.pdf`  
- `data/chunks.jsonl`  

You may delete `data/pdfs/` after chunking if disk space is tight.

## Running Website

### Backend
```bash
python .\website\backend\app.py
```

### Frontend
```bash
cd .\website\paperfinder
npm start
```