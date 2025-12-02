import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

input_path = "chunks.jsonl"
output_path = "embedded_chunks.jsonl"

BATCH_SIZE = 64  # Increase to 128 if memory allows

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

TOTAL_LINES = 3927680   # your known count

def embed_chunks():
    print(f"Total chunks: {TOTAL_LINES}")

    buffer_docs = []
    buffer_texts = []

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:

        for line in tqdm(f_in, total=TOTAL_LINES, desc="Embedding"):
            doc = json.loads(line)
            buffer_docs.append(doc)
            buffer_texts.append(doc["chunk_text"])

            # When batch is full → embed it
            if len(buffer_texts) == BATCH_SIZE:
                embeddings = model.encode(buffer_texts, batch_size=BATCH_SIZE)

                for d, emb in zip(buffer_docs, embeddings):
                    d["embedding"] = emb.tolist()
                    f_out.write(json.dumps(d) + "\n")

                buffer_docs = []
                buffer_texts = []

        # Process remainder
        if buffer_texts:
            embeddings = model.encode(buffer_texts, batch_size=BATCH_SIZE)
            for d, emb in zip(buffer_docs, embeddings):
                d["embedding"] = emb.tolist()
                f_out.write(json.dumps(d) + "\n")

    print("\nDone! Saved →", output_path)


if __name__ == "__main__":
    embed_chunks()
