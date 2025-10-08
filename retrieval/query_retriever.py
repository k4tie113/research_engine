#!/usr/bin/env python3
"""
Query Retriever with RRF Reranking

This script performs intelligent document retrieval using:
- Multi-query expansion (generates related queries from a seed query)
- FAISS vector search with embeddings
- Optional Reciprocal Rank Fusion (RRF) for combining results from multiple queries

Usage Examples:
    # Basic usage with RRF reranking (recommended)
    python query_retriever.py --query "machine learning in healthcare" --use_rrf

    # Custom parameters
    python query_retriever.py --query "your search query" --use_rrf --k 10 --num_queries 5

    # Without RRF (shows results for each query separately)
    python query_retriever.py --query "your search query" --k 5

    # Use different embedding model
    python query_retriever.py --query "your search query" --use_rrf --embeddings_model "all-MiniLM-L6-v2"

    To prevent issues with GritLM-7B taking a long time, use the command above ^^
 
Requirements:
    - HUGGINGFACEHUB_API_TOKEN environment variable (for LLM query expansion)
    - FAISS index built in data/faiss/ directory (will need to perform API scraping, chunking, and embedding first!)
    - Compatible embedding model (same as used to build the index, so I recommend using MiniLM for both for now rather than GritLM-7B)

Key Arguments:
    --query: Seed query to expand and search with
    --use_rrf: Enable RRF reranking (combines results from all generated queries)
    --k: Number of top results to return (default: 5)
    --num_queries: Number of related queries to generate (default: 10)
    --rrf_k: RRF constant for fusion (default: 60)
    --embeddings_model: HuggingFace model for embeddings (default: GritLM/GritLM-7B)
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from dotenv import load_dotenv

# Local FAISS adapter deps
import faiss  # type: ignore
import jsonlines  # type: ignore
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.retrievers import BaseRetriever  # type: ignore
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # type: ignore


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"Environment variable '{var_name}' is required.")
        sys.exit(1)
    return value


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Apply Reciprocal Rank Fusion to combine multiple ranked lists."""
    scores: Dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks, start=1):
            if doc_id < 0:
                continue
            scores[int(doc_id)] += 1.0 / (k + r)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Query retriever using MultiQueryRetriever + FAISS + GritLM-7B embeddings with optional RRF reranking")
    parser.add_argument("--query", type=str, default="NLP for scientific literature search", help="Seed query to expand and retrieve with")
    parser.add_argument("--k", type=int, default=5, help="Top k documents to return")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of related queries to generate")
    parser.add_argument("--embeddings_model", type=str, default=os.getenv("GRITLM_MODEL_ID", "GritLM/GritLM-7B"), help="Hugging Face model id for embeddings (GritLM-7B)")
    parser.add_argument("--use_local_embeddings", action="store_true", help="Use local sentence-transformers embeddings instead of HF endpoint")
    parser.add_argument("--llm_model", type=str, default=os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"), help="Hugging Face model id for query expansion LLM")
    parser.add_argument("--use_chat_llm", action="store_true", help="Use chat-capable HF endpoint (provider supports 'conversational' task)")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings", help="Directory containing FAISS store artifacts")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant k for reciprocal rank fusion")
    parser.add_argument("--use_rrf", action="store_true", help="Use Reciprocal Rank Fusion to combine results from multiple queries")
    args = parser.parse_args()

    # Load environment from project root .env then require HF token once
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    # Require HF token once; used for both embeddings and LLM
    _require_env("HUGGINGFACEHUB_API_TOKEN")

    # Lazy imports so base scripts remain lightweight
    from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
    try:
        # Optional import: only needed when --use_chat_llm is supplied
        from langchain_huggingface import ChatHuggingFace
    except Exception:  # noqa: BLE001
        ChatHuggingFace = None  # type: ignore[assignment]
    from langchain_community.vectorstores import FAISS
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.prompts import PromptTemplate
    from huggingface_hub import InferenceClient

    # Local FAISS adapter targeting this project's index format
    root = Path(__file__).resolve().parents[1]
    idx_path = root / "data" / "faiss" / "index_flatip.faiss"
    ids_path = root / "data" / "faiss" / "ids.jsonl"
    chunks_path = root / "data" / "chunks.jsonl"

    if not idx_path.exists() or not ids_path.exists():
        print("Failed to load local vector store.\n"
              f"Expected files: {idx_path} and {ids_path}.\n"
              "Run the embed + build steps first.")
        sys.exit(1)

    # Always use local ST embeddings for querying our FAISS (fast, reliable)
    try:
        st_model_name = args.embeddings_model
        print(f"Loading local Sentence-Transformers: {st_model_name} ...")
        st_model = SentenceTransformer(st_model_name, device="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load sentence-transformers model '{args.embeddings_model}': {exc}")
        sys.exit(1)

    index = faiss.read_index(str(idx_path))

    # Load ids metadata (row aligned)
    with jsonlines.open(ids_path, "r") as r:
        id_rows = list(r)
    if len(id_rows) != index.ntotal:
        print(f"ids.jsonl rows ({len(id_rows)}) != index.ntotal ({index.ntotal})")
        sys.exit(1)

    # Check dimension compatibility
    expected_dim = st_model.get_sentence_embedding_dimension()
    actual_dim = index.d
    if expected_dim != actual_dim:
        print(f"Dimension mismatch: Model embedding dimension ({expected_dim}) != FAISS index dimension ({actual_dim})")
        print(f"Model: {args.embeddings_model}")
        print(f"Index file: {idx_path}")
        print("\nTo fix this, you need to:")
        print("1. Use the same model that was used to create the FAISS index, OR")
        print("2. Rebuild the FAISS index with the current model")
        print(f"\nTry using: --embeddings_model <model_used_for_index>")
        sys.exit(1)

    # Optional chunk map to show snippets
    chunk_map = {}
    if chunks_path.exists():
        with jsonlines.open(chunks_path, "r") as reader:
            for rec in reader:
                try:
                    key = (rec.get("paper_id"), int(rec.get("chunk_index", -1)))
                except Exception:
                    continue
                chunk_map[key] = rec.get("chunk_text", "")

    class _LocalVS:
        def __init__(self):
            self.index = index
            self.ids = id_rows

        def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
            try:
                # Generate embedding with proper normalization
                vec = st_model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
                q = np.asarray(vec, dtype=np.float32)
                
                # Ensure dimension matches
                if q.shape[1] != self.index.d:
                    raise ValueError(f"Query embedding dimension {q.shape[1]} doesn't match index dimension {self.index.d}")
                
                D, I = self.index.search(q, k)
                out: List[Tuple[Document, float]] = []
                for score, ridx in zip(D[0], I[0]):
                    if ridx < 0:
                        continue
                    meta = self.ids[int(ridx)]
                    key = (meta.get("paper_id"), int(meta.get("chunk_index", -1)))
                    text = chunk_map.get(key, "")
                    doc = Document(page_content=text or "", metadata=meta)
                    # For normalized embeddings, score is cosine similarity (higher is better)
                    out.append((doc, float(score)))
                return out
            except Exception as exc:
                print(f"Error in similarity search: {exc}")
                return []

        def as_retriever(self, search_kwargs: dict):
            k = int(search_kwargs.get("k", 5))
            parent = self

            class LocalRetriever(BaseRetriever):
                def _get_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: CallbackManagerForRetrieverRun | None = None,
                ) -> List[Document]:
                    pairs = parent.similarity_search_with_score(query, k=k)
                    return [d for d, _ in pairs]

                async def _aget_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: CallbackManagerForRetrieverRun | None = None,
                ) -> List[Document]:
                    # simple sync fallback
                    return self._get_relevant_documents(query, run_manager=run_manager)

            return LocalRetriever()

    vectorstore = _LocalVS()

    print("Initializing LLM for multi-query expansion ...")
    # Always use conversational task; many providers/models only support it
    try:
        base_llm = HuggingFaceEndpoint(
            repo_id=args.llm_model,
            temperature=0.2,
            max_new_tokens=256,
            task="conversational",
        )
        if ChatHuggingFace is not None:
            llm = ChatHuggingFace(llm=base_llm)
        else:
            llm = base_llm
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to init conversational LLM: {exc}")
        sys.exit(1)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    print(f"Generating {args.num_queries} related queries using MultiQueryRetriever ...")
    multi_query_prompt = (
        "You are a helpful AI assistant generating alternative search queries.\n"
        "Generate {num_queries} diverse rephrasings that could retrieve complementary information.\n"
        "Return one query per line.\n\n"
        "Original question: {question}"
    )
    prompt = PromptTemplate.from_template(multi_query_prompt).partial(num_queries=str(args.num_queries))
    mqr = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=prompt,
        include_original=True,
    )

    seed_query = args.query
    print(f"\nSeed query:\n- {seed_query}\n")

    # Derive expanded queries explicitly for display and downstream scoring
    # Invoke the underlying LLM chain and robustly parse one-per-line
    try:
        response = mqr.llm_chain.invoke({"question": seed_query})  # type: ignore[attr-defined]
        raw_text = getattr(response, "content", None) or getattr(response, "text", "") or str(response)

        parsed_list: List[str] | None = None
        # Attempt to parse JSON/Python list output
        try:
            candidate = raw_text.strip()
            if candidate.startswith("[") and candidate.endswith("]"):
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, list):
                    parsed_list = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            parsed_list = None

        if parsed_list is not None:
            cleaned = []
            for item in parsed_list:
                s = item.strip().lstrip("-*").strip()
                # remove simple numbering prefixes like "1.", "1)"
                if len(s) > 2 and s[0].isdigit() and (s[1] in ".)" or s.startswith("(") and ")" in s):
                    s = s.lstrip("(0123456789)").lstrip(". )").strip()
                if s and s != seed_query and s not in cleaned:
                    cleaned.append(s)
            generated_queries = cleaned[: args.num_queries]
        else:
            # Fallback: split by lines and clean bullets/numbers
            lines = [ln.strip() for ln in str(raw_text).splitlines() if ln.strip()]
            cleaned = []
            for ln in lines:
                s = ln.strip().lstrip("-*").strip()
                if len(s) > 2 and s[0].isdigit() and (s[1] in ".)" or s.startswith("(") and ")" in s):
                    s = s.lstrip("(0123456789)").lstrip(". )").strip()
                if s and s != seed_query and s not in cleaned:
                    cleaned.append(s)
            generated_queries = cleaned[: args.num_queries]
    except Exception:
        generated_queries = []

    # MultiQueryRetriever returns aggregated retrieval results across expanded queries
    docs = mqr.get_relevant_documents(seed_query)

    print("Expanded queries (excluding the original):")
    if generated_queries:
        # Keep only up to args.num_queries, filter duplicates and the seed
        dedup = []
        for q in generated_queries:
            if q and q != seed_query and q not in dedup:
                dedup.append(q)
        for i, q in enumerate(dedup[: args.num_queries], start=1):
            print(f"{i:2d}. {q}")
    else:
        print("<none>")
    print()

    # Collect results from all queries (seed + expansions)
    queries_to_run = [seed_query]
    if generated_queries:
        queries_to_run.extend(generated_queries)

    if args.use_rrf:
        # Collect ranked lists for RRF fusion
        print(f"\nCollecting results from {len(queries_to_run)} queries for RRF fusion...")
        rank_lists: List[List[int]] = []
        all_results: List[List[Tuple[Document, float]]] = []
        
        for qi, q in enumerate(queries_to_run, start=1):
            print(f"Query {qi}/{len(queries_to_run)}: {q}")
            try:
                pairs = vectorstore.similarity_search_with_score(q, k=args.k)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to retrieve for query: {exc}")
                pairs = []
            
            all_results.append(pairs)
            
            # Extract document indices for RRF
            doc_indices = []
            for doc, _ in pairs:
                meta = getattr(doc, "metadata", {}) or {}
                # Find the index of this document in our ids list
                doc_id = meta.get("paper_id")
                chunk_idx = meta.get("chunk_index")
                for i, id_row in enumerate(id_rows):
                    if id_row.get("paper_id") == doc_id and id_row.get("chunk_index") == chunk_idx:
                        doc_indices.append(i)
                        break
                else:
                    # If not found, use a placeholder that won't affect RRF
                    doc_indices.append(-1)
            
            rank_lists.append(doc_indices)
        
        # Apply RRF fusion
        print(f"\nApplying Reciprocal Rank Fusion (k={args.rrf_k})...")
        fused_scores = reciprocal_rank_fusion(rank_lists, k=args.rrf_k)
        fused_sorted = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:args.k]
        
        print(f"\n=== RRF Fused Results (Top {args.k}) ===")
        for rank, (row_id, rrf_score) in enumerate(fused_sorted, 1):
            if row_id >= len(id_rows):
                continue
            meta = id_rows[row_id]
            key = (meta.get("paper_id"), int(meta.get("chunk_index", -1)))
            text = chunk_map.get(key, "")
            title = (meta.get("title") or "")[:150]
            pid = meta.get("paper_id") or "<unknown>"
            cidx = meta.get("chunk_index")
            
            print(f"\n[{rank}] RRF={rrf_score:.4f}  {pid}  #{cidx}  {title}")
            content = text.replace("\n", " ") if text else ""
            snippet = content[:1000]
            print(snippet + ("..." if len(content) > 1000 else ""))
    
    else:
        # Original behavior: show results for each query independently
        for qi, q in enumerate(queries_to_run, start=1):
            print(f"\n=== Query {qi} of {len(queries_to_run)} ===")
            print(q)
            try:
                pairs = vectorstore.similarity_search_with_score(q, k=args.k)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to retrieve for query: {exc}")
                pairs = []

            if not pairs:
                print("<no results>")
                continue

            for i, (doc, score) in enumerate(pairs, start=1):
                meta = getattr(doc, "metadata", {}) or {}
                pid = meta.get("paper_id") or meta.get("id") or "<unknown>"
                cidx = meta.get("chunk_index")
                title = (meta.get("title") or "")[:150]
                print(f"\n[{i}] score={score:.4f}  {pid}  #{cidx}  {title}")
                content = (getattr(doc, "page_content", "") or "").replace("\n", " ")
                snippet = content[:3000]
                print(snippet)


if __name__ == "__main__":
    main()

