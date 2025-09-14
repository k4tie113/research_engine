"""
storage_api.py
--------------
Functions for teammates:
- get_paper_text(id): return full text
- get_metadata(id): return metadata
- list_candidates(keyword): search abstracts
- get_chunks(id, chunk_size): split text into chunks
"""

import sqlite3, os

def get_paper_text(paper_id):
    txt_path = f"data/{paper_id}.txt"
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file for {paper_id} not found.")
    with open(txt_path, "r") as f:
        return f.read()

def get_metadata(paper_id):
    conn = sqlite3.connect("metadata/papers.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers WHERE id=?", (paper_id,))
    row = cur.fetchone()
    conn.close()
    return row

def list_candidates(keyword):
    conn = sqlite3.connect("metadata/papers.db")
    cur = conn.cursor()
    cur.execute("SELECT id, title, abstract FROM papers WHERE abstract LIKE ?", (f"%{keyword}%",))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_chunks(paper_id, chunk_size=500):
    text = get_paper_text(paper_id)
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append((f"{paper_id}_chunk{i//chunk_size}", chunk_text))
    return chunks

if __name__ == "__main__":
    print("🔎 Searching for 'transformer' in abstracts...")
    results = list_candidates("transformer")
    print(f"Found {len(results)} matches.")
    if results:
        pid = results[0][0]
        print("\nFirst paper metadata:", get_metadata(pid))
        print("\nFirst 200 chars of text:", get_paper_text(pid)[:200])
        print("\nFirst chunk:", get_chunks(pid, 300)[0])
