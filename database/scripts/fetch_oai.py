import sys, csv, xml.etree.ElementTree as ET, requests
from pathlib import Path
from time import sleep

BASE_URL = "http://export.arxiv.org/oai2"
TARGET_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI", "cs.CV"}

year = int(sys.argv[1])
params = {
    "verb": "ListRecords",
    "metadataPrefix": "arXiv",
    "set": "cs",
    "from": f"{year}-01-01",
    "until": f"{year}-12-31",
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"papers_{year}.csv"

total = 0
token = None

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "title", "authors", "abstract", "categories", "created"])

    while True:
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}

        try:
            r = requests.get(BASE_URL, params=params, timeout=90)
            r.raise_for_status()
        except Exception as e:
            print(f"[{year}] {e}, retrying in 10s...")
            sleep(10)
            continue

        root = ET.fromstring(r.text)
        for rec in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
            meta = rec.find(".//{http://arxiv.org/OAI/arXiv/}arXiv")
            if meta is None: 
                continue
            cats = meta.findtext("{http://arxiv.org/OAI/arXiv/}categories", "")
            if not any(c in cats.split() for c in TARGET_CATEGORIES):
                continue

            title = meta.findtext("{http://arxiv.org/OAI/arXiv/}title", "").replace("\n"," ").strip()
            abstr = meta.findtext("{http://arxiv.org/OAI/arXiv/}abstract", "").replace("\n"," ").strip()
            aid   = meta.findtext("{http://arxiv.org/OAI/arXiv/}id", "")
            date  = meta.findtext("{http://arxiv.org/OAI/arXiv/}created", "")
            authors = [
                (a.findtext("{http://arxiv.org/OAI/arXiv/}forenames","") + " " +
                 a.findtext("{http://arxiv.org/OAI/arXiv/}keyname","")).strip()
                for a in meta.findall(".//{http://arxiv.org/OAI/arXiv/}author")
            ]
            writer.writerow([aid, title, "; ".join(authors), abstr, cats, date])
            total += 1
            if total % 5000 == 0:
                print(f"[{year}] Downloaded {total:,}")

        tok_elem = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
        if tok_elem is not None and tok_elem.text:
            token = tok_elem.text
            sleep(1)
        else:
            break

print(f"[{year}] Done, saved {total:,} records to {OUTPUT_FILE}")
