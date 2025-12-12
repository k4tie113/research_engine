import csv
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

ES_URL = "http://localhost:9200"
USERNAME = "elastic"
PASSWORD = "jHBuBAEd"
INDEX_NAME = "paper_metadata"

CSV_FILE = "papers_oai_combined.csv"
BATCH_SIZE = 2000

es = Elasticsearch(
    ES_URL,
    basic_auth=(USERNAME, PASSWORD),
    request_timeout=60
)


def generate_actions():
    with open(CSV_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # IMPORTANT: convert empty strings to None
            if row["created"] == "":
                row["created"] = None

            yield {
                "_index": INDEX_NAME,
                "_id": row["id"],
                "_source": row
            }


def bulk_index():
    total = sum(1 for _ in open(CSV_FILE)) - 1  # minus header
    print(f"Total metadata rows: {total}")

    batch = []
    progress = tqdm(total=total, desc="Indexing metadata")

    for action in generate_actions():
        batch.append(action)

        if len(batch) == BATCH_SIZE:
            helpers.bulk(es, batch)
            progress.update(len(batch))
            batch = []

    # leftover
    if batch:
        helpers.bulk(es, batch)
        progress.update(len(batch))

    progress.close()
    print("Metadata indexing done!")


if __name__ == "__main__":
    bulk_index()
