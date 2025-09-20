# Research Engine – Local RAG Pre-Processing Pipeline

This repo provides a **local, end-to-end pipeline** for preparing arXiv NLP papers for Retrieval-Augmented Generation (RAG). It downloads papers, extracts text, and produces **overlapping ~800-token chunks** in JSON-Lines format, ready to be embedded (e.g., with **GritLM**).

---

## How the Pipeline Works

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

## How to Run

### 1 Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` should include:
```
feedparser
requests
pypdf
tiktoken
```

### 2 End-to-end (recommended)
```bash
python scripts/run_all.py
```
### 3 Or step-by-step
```bash
python scripts/fetch_metadata.py
python scripts/download_pdfs.py
python scripts/chunk_pdfs.py
```

---

## Outputs
- `data/metadata/papers.csv`  
- `data/pdfs/*.pdf`  
- `data/chunks.jsonl`  

You may delete `data/pdfs/` after chunking if disk space is tight.

---

## Next Steps – Integrating GritLM Embeddings

We now have a clean dataset of recent arXiv NLP papers ( i will find a way to expand this dataset ): their PDFs have been downloaded, converted to text, and split into overlapping ~800-token chunks saved in data/chunks.jsonl.

Next step is to embed each chunk (for example with GritLM) and store the vectors with metadata in a vector database such as FAISS or pgvector.

From there we can build a retrieval-augmented generation layer that embeds a user query and retrieves the most relevant chunks.


---

## Notes and Caveats
- **Limited download size:** Default is the latest 300 papers. Increase `BATCH_SIZE` in `fetch_metadata.py` if needed (be considerate of arXiv limits).  
- **PDF text quality:** Equations, tables, and multi-column layouts may extract imperfectly; consider `pdfplumber` or `PyMuPDF` if accuracy is critical.  
- **Token size:** Default 800 tokens per chunk with 200 overlap using `cl100k_base`. Adjust if your embedding model uses a different tokenizer.  
- **Disk usage:** PDFs can be deleted after chunking; the JSONL file contains all text needed for embedding.  
- **Reproducibility:** Because chunking depends on tokenization, changing the tokenizer or parameters requires re-chunking for consistent retrieval.  
