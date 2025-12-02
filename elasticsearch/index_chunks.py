import json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import time

ES_URL = "http://localhost:9200"
USERNAME = "elastic"
PASSWORD = "jHBuBAEd"   # <-- update with your password
INDEX_NAME = "paper_chunks"
INPUT_FILE = "embedded_chunks.jsonl"
BATCH_SIZE = 2000

# Connect to ES
es = Elasticsearch(
    ES_URL,
    basic_auth=(USERNAME, PASSWORD),
    request_timeout=60
)


def generate_batches():
    """Stream batches of documents from the JSONL file."""
    batch = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            doc = json.loads(line)

            action = {
                "_index": INDEX_NAME,
                "_id": f"{doc['paper_id']}_{doc['chunk_index']}",
                "_source": doc
            }

            batch.append(action)

            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

        # leftover docs
        if batch:
            yield batch


def bulk_index():
    """Bulk index with error handling + retries."""
    print("Counting total lines...")
    total_lines = 3927680

    print(f"Total documents to index: {total_lines}")

    progress = tqdm(total=total_lines, desc="Indexing", unit="docs")

    for batch in generate_batches():
        success = False
        retries = 0

        while not success and retries < 5:
            try:
                helpers.bulk(es, batch)
                success = True
            except Exception as e:
                retries += 1
                wait = retries * 2
                print(f"\n⚠️ Bulk error — retry {retries}/5 in {wait}s\n{e}")
                time.sleep(wait)

        if not success:
            print("Failed permanently on this batch. Skipping.")
        
        progress.update(len(batch))

    progress.close()
    print("\nAll chunks indexed.")


if __name__ == "__main__":
    print("Starting bulk indexing...\n")
    bulk_index()
