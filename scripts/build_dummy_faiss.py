import argparse
import os
import random
import string
from pathlib import Path

from dotenv import load_dotenv


def generate_dummy_text(index: int, min_len: int = 64, max_len: int = 256) -> str:
    topics = [
        "natural language processing",
        "transformer architectures",
        "retrieval augmented generation",
        "vector databases and FAISS",
        "tokenization and tiktoken",
        "semantic search and embeddings",
        "arXiv paper indexing",
        "LangChain retrievers",
        "contrastive learning",
        "GritLM-7B usage",
    ]
    topic = random.choice(topics)
    words = [
        topic,
        "pipeline",
        "evaluation",
        "benchmark",
        "encoding",
        "query",
        "document",
        "similarity",
        "cosine",
        "faiss",
        "hnsw",
        "metadata",
    ]
    target_len = random.randint(min_len, max_len)
    blob = []
    while len(" ".join(blob)) < target_len:
        token = random.choice(words)
        if random.random() < 0.15:
            token += "-" + random.choice(words)
        blob.append(token)
    suffix = " ".join(random.choices(words, k=8))
    return f"Dummy doc {index}: {topic}. {' '.join(blob)} {suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dummy FAISS vector store (local or HF embeddings)")
    parser.add_argument("--num_docs", type=int, default=200, help="Number of synthetic documents to index")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Directory to save FAISS store")
    parser.add_argument("--embeddings_model", type=str, default=os.getenv("GRITLM_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"), help="Embedding model id (HF endpoint or sentence-transformers model)")
    parser.add_argument("--use_hf_endpoint", action="store_true", help="Use Hugging Face Inference API for embeddings instead of local model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load .env for optional HF token
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    # Lazy imports
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from huggingface_hub import InferenceClient

    random.seed(args.seed)

    print(f"Generating {args.num_docs} synthetic documents ...")
    docs = []
    for i in range(args.num_docs):
        content = generate_dummy_text(i)
        meta = {
            "source": f"dummy://doc/{i}",
            "id": "DUM" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8)),
        }
        docs.append(Document(page_content=content, metadata=meta))

    print(f"Loading embeddings model: {args.embeddings_model}")
    if args.use_hf_endpoint:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing. Add it to .env or omit --use_hf_endpoint.")
        # Force provider to hf-inference for feature-extraction support
        hf_client = InferenceClient(model=args.embeddings_model, provider="hf-inference")
        embeddings = HuggingFaceEndpointEmbeddings(client=hf_client)
    else:
        # Local sentence-transformers via HuggingFaceEmbeddings
        # Ensure model is a valid sentence-transformers or HF model id
        embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model)

    print("Building FAISS index (this may take a while for large num_docs) ...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving FAISS index to '{out_dir.as_posix()}' ...")
    vectorstore.save_local(out_dir.as_posix())
    print("Done.")


if __name__ == "__main__":
    main()


