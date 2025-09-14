# 📚 Research Paper Search Engine (Phase 1 – Data & Storage)

This project ingests NLP research papers from **arXiv**, organizes them, and exposes simple functions for retrieval and QA experiments.

**NOTE**
The search function in storage_api.py is a basic SQL LIKE search: it just looks for the exact keyword string inside abstracts and returns any matching papers. This means it only finds literal matches (e.g., "transformer") and does not understand synonyms, context, or semantic similarity like Asta does.


This repo contains:
- Scripts to **fetch papers** from arXiv
- Conversion of PDFs → text
- Metadata database (`papers.db`)
- A **Storage API** (`storage_api.py`) for teammates to query papers

---

## 🔹 Setup

### 1. Clone Repo & Create venv
```bash
git clone https://github.com/k4tie113/research_engine.git
cd research_engine
python3 -m venv venv
source venv/bin/activate   # (Mac/Linux), to activate the Python virtual environment
.\venv\Scripts\activate    # (Windows PowerShell)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔹 Pipeline

Run these steps in order (SIMPLER: just use `python scripts/run_all.py` to run it all at once):

### 1. Fetch Papers from arXiv
Downloads PDFs + metadata.
```bash
python scripts/fetch_papers.py
```
- PDFs → `data/`  
- Metadata → `metadata/papers.csv`

### 2. Convert PDFs to Text
```bash
python scripts/pdf_to_text.py
```
- Creates `.txt` files in `data/`  

### 3. Build Database
```bash
python scripts/make_db.py
```
- Creates `metadata/papers.db`

---

## 🔹 Storage API

The **Storage API** exposes helper functions for teammates.

### Example Usage
```python
from scripts.storage_api import list_candidates, get_paper_text, get_metadata, get_chunks

# Search abstracts
print(list_candidates("transformer"))

# Get metadata for a paper
print(get_metadata("2501.12345"))

# Get full text
print(get_paper_text("2501.12345")[:500])

# Get chunks (for retrieval/QA)
print(get_chunks("2501.12345", 300)[0])
```

### Quick Test
```bash
python scripts/storage_api.py
```
- Searches for `"transformer"`  
- Prints first paper’s metadata  
- Prints first 200 chars of text  
- Prints first chunk  

---

## 🔹 Project Structure
```
research_engine/
│   requirements.txt
│   .gitignore
│   README.md
│
├── data/                 # PDFs + text (ignored in git)
├── metadata/             # CSV + DB
└── scripts/
    ├── fetch_papers.py
    ├── pdf_to_text.py
    ├── make_db.py
    ├── storage_api.py
    └── run_all.py
```

---

## 🔹 Notes
- GitHub repo **does not include PDFs/text** (`data/` is `.gitignore`d).  
- To expand dataset, edit `MAX_RESULTS` in `fetch_papers.py` (up to ~2000).  
- Full-text downloads take time — you can also start with abstracts only.  
