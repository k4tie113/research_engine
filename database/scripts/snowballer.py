# database/snowballer.py
import csv, json, re, time
from collections import deque
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "metadata" / "papers_oai.csv"   # your OAI harvest
QUEUE = ROOT / "data" / "metadata" / "snowball_queue.csv"
STATE = ROOT / "data" / "metadata" / "snowball_state.json"

API = "https://api.semanticscholar.org/graph/v1/paper/"
FIELDS = "title,year,externalIds,references.externalIds,citations.externalIds"

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.I)

def load_seed_ids():
    ids = set()
    if not SEED.exists(): return ids
    with open(SEED, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            aid = (r.get("id") or "").replace("arXiv:", "").strip()
            if aid: ids.add(aid)
    return ids

def arxiv_from_external(eids: dict | None):
    if not eids: return None
    aid = eids.get("ArXiv") or eids.get("ARXIV")
    if not aid: return None
    aid = aid.replace("arXiv:", "").strip()
    return aid if ARXIV_ID_RE.match(aid) else None

def fetch_paper(key: str):
    url = f"{API}{key}"
    r = requests.get(url, params={"fields": FIELDS}, timeout=20)
    if r.status_code == 429:
        time.sleep(0.5); r = requests.get(url, params={"fields": FIELDS}, timeout=20)
    r.raise_for_status()
    return r.json()

def main(max_new=1500, max_depth=2):
    visited = set()
    frontier = deque()
    if STATE.exists():
        s = json.loads(STATE.read_text())
        visited = set(s.get("visited", []))
        frontier = deque(s.get("frontier", []))

    for aid in load_seed_ids():
        if aid not in visited:
            frontier.append({"type":"arxiv","id":aid,"depth":0})

    discovered = []
    while frontier and len(discovered) < max_new:
        node = frontier.popleft()
        if node["depth"] > max_depth: continue

        key = f"arXiv:{node['id']}" if node["type"]=="arxiv" else node["id"]
        try:
            data = fetch_paper(key)
        except Exception as e:
            print("warn:", e); continue

        visited.add(node["id"])

        def enqueue(edges):
            n = 0
            for it in edges or []:
                aid = arxiv_from_external(it.get("externalIds", {}))
                if not aid or aid in visited: continue
                visited.add(aid)  # avoid dup enqueues
                discovered.append(aid)
                frontier.append({"type":"arxiv","id":aid,"depth":node["depth"]+1})
                n += 1
            return n

        r = enqueue(data.get("references"))
        c = enqueue(data.get("citations"))
        print(f"depth={node['depth']} queued refs={r} cits={c} total_new={len(discovered)}")
        time.sleep(0.12)

        if len(discovered) % 200 == 0:
            STATE.write_text(json.dumps({"visited": list(visited), "frontier": list(frontier)}))

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id"])
        w.writeheader()
        for aid in discovered: w.writerow({"id": aid})

    STATE.write_text(json.dumps({"visited": list(visited), "frontier": list(frontier)}))
    print(f"\nSnowballing complete → queued {len(discovered)} new arXiv IDs at {QUEUE}")

if __name__ == "__main__":
    main()
