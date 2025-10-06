import argparse
import ast
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"Environment variable '{var_name}' is required.")
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Query retriever using MultiQueryRetriever + FAISS + GritLM-7B embeddings")
    parser.add_argument("--query", type=str, default="NLP for scientific literature search", help="Seed query to expand and retrieve with")
    parser.add_argument("--k", type=int, default=5, help="Top k documents to return")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of related queries to generate")
    parser.add_argument("--embeddings_model", type=str, default=os.getenv("GRITLM_MODEL_ID", "GritLM/GritLM-7B"), help="Hugging Face model id for embeddings (GritLM-7B)")
    parser.add_argument("--use_local_embeddings", action="store_true", help="Use local sentence-transformers embeddings instead of HF endpoint")
    parser.add_argument("--llm_model", type=str, default=os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"), help="Hugging Face model id for query expansion LLM")
    parser.add_argument("--use_chat_llm", action="store_true", help="Use chat-capable HF endpoint (provider supports 'conversational' task)")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings", help="Directory containing FAISS store artifacts")
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

    # Embeddings: prefer HF endpoint unless --use_local_embeddings is set
    if args.use_local_embeddings:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to import local embeddings backend: {exc}")
            sys.exit(1)
        print("Loading local embeddings (sentence-transformers) ...")
        embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model)
    else:
        print("Loading GritLM-7B embeddings via Hugging Face Inference API ...")
        hf_embed_client = InferenceClient(model=args.embeddings_model, provider="hf-inference")
        embeddings = HuggingFaceEndpointEmbeddings(client=hf_embed_client)

    print(f"Loading FAISS vector store from '{args.embeddings_dir}' ...")
    try:
        # Expecting a standard LangChain FAISS export in the directory
        # This typically includes 'index.faiss' and 'index.pkl' (or similar) files
        vectorstore = FAISS.load_local(
            args.embeddings_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("Failed to load FAISS store.\n"
              "Ensure the directory contains a LangChain FAISS export (e.g., 'index.faiss' and 'index.pkl').\n"
              f"Path tried: {os.path.abspath(args.embeddings_dir)}\n"
              f"Error: {exc}")
        sys.exit(1)

    print("Initializing LLM for multi-query expansion ...")
    if args.use_chat_llm:
        if ChatHuggingFace is None:
            print("ChatHuggingFace not available. Please upgrade langchain-huggingface or omit --use_chat_llm.")
            sys.exit(1)
        base_llm = HuggingFaceEndpoint(
            repo_id=args.llm_model,
            temperature=0.2,
            max_new_tokens=256,
        )
        llm = ChatHuggingFace(llm=base_llm)
    else:
        llm = HuggingFaceEndpoint(
            repo_id=args.llm_model,
            temperature=0.2,
            max_new_tokens=256,
        )

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

    # For each query (seed + expansions), print top-k results independently
    queries_to_run = [seed_query]
    if generated_queries:
        queries_to_run.extend(generated_queries)

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
            source = meta.get("source") or meta.get("id") or meta.get("file_path") or "<unknown>"
            print(f"\n[{i}] distance={score:.4f} source={source}")
            content = getattr(doc, "page_content", "") or ""
            snippet = content[:500].replace("\n", " ")
            print(snippet + ("..." if len(content) > 500 else ""))


if __name__ == "__main__":
    main()


