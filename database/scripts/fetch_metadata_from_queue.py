# database/fetch_metadata_from_queue.py
import csv, time, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "metadata" / "snowball_queue.csv"
OUT = ROOT / "data" / "metadata" / "papers_oai.csv"  # keep a single master

def main():
    assert INP.exists(), "Run snowballer first."
    existing = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing.add(r["id"])

    new_rows = []
    with open(INP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            aid = r["id"]
            if aid in existing: continue
            # arXiv PDF URL is deterministic; full rich meta will be filled by your OAI run later
            pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
            new_rows.append({
                "id": aid, "title": "", "authors": "",
                "abstract": "", "categories": "", "datestamp": ""
            })
            time.sleep(0.1)

    mode = "a" if OUT.exists() else "w"
    with open(OUT, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","title","authors","abstract","categories","datestamp"])
        if mode == "w": w.writeheader()
        w.writerows(new_rows)
    print(f"Appended {len(new_rows)} new IDs to {OUT}")

if __name__ == "__main__":
    main()
