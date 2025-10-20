#!/usr/bin/env python3
"""
Content Generator using Reranking Results

This script takes the reranking results from query_retriever.py and uses a lightweight LLM
to generate coherent content based on the retrieved document chunks.

Usage Examples:
    # Generate content from a query using RRF results
    python content_generator.py --query "machine learning in healthcare" --use_rrf
    
    # Generate with custom parameters
    python content_generator.py --query "your search query" --use_rrf --k 10 --generation_type "summary"
    
    # Generate different types of content
    python content_generator.py --query "your search query" --generation_type "research_synthesis" --max_tokens 512

Requirements:
    - FAISS index and related files (same as query_retriever.py)
    - Compatible embedding model (default: all-MiniLM-L6-v2 for fast, reliable embeddings)
    - Local transformers library for content generation (no API credits needed)
    - No API token required - runs completely locally

Key Arguments:
    --query: Seed query to expand and search with
    --use_rrf: Enable RRF reranking (combines results from all generated queries)
    --k: Number of top results to return for generation (default: 5)
    --generation_type: Type of content to generate (summary, research_synthesis, analysis, qa)
    --max_tokens: Maximum tokens for generation (default: 512)
    --embeddings_model: HuggingFace model for embeddings (default: all-MiniLM-L6-v2)
    --llm_model: Local HuggingFace model for generation (default: microsoft/DialoGPT-small)
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"Environment variable '{var_name}' is required.")
        sys.exit(1)
    return value

@dataclass
class GenerationConfig:
    """Configuration for content generation."""
    generation_type: str = "summary"
    max_tokens: int = 512
    temperature: float = 0.7
    llm_model: str = "microsoft/DialoGPT-medium"
    
    @property
    def prompt_template(self) -> str:
        """Get the prompt template for the generation type."""
        templates = {
            "summary": self._get_summary_prompt(),
            "research_synthesis": self._get_synthesis_prompt(),
            "analysis": self._get_analysis_prompt(),
            "qa": self._get_qa_prompt(),
            "insights": self._get_insights_prompt()
        }
        return templates.get(self.generation_type, templates["summary"])
    
    def _get_summary_prompt(self) -> str:
        return """Based on these research documents about {query}, write a brief summary:

{documents}

The main findings are:"""

    def _get_synthesis_prompt(self) -> str:
        return """Create a research synthesis from the following documents that:

1. Integrates findings across multiple sources
2. Identifies areas of agreement and disagreement
3. Suggests research gaps or opportunities
4. Provides a coherent narrative

Research Documents:
{documents}

Query: {query}

Research Synthesis:"""

    def _get_analysis_prompt(self) -> str:
        return """Analyze the following research documents to:

1. Evaluate the methodologies used
2. Assess the quality and reliability of findings
3. Identify strengths and limitations
4. Provide critical insights

Research Documents:
{documents}

Query: {query}

Analysis:"""

    def _get_qa_prompt(self) -> str:
        return """Answer the following question based on the research documents provided:

Question: {query}

Research Documents:
{documents}

Answer:"""

    def _get_insights_prompt(self) -> str:
        return """Extract key insights from the research documents that:

1. Reveal important trends or patterns
2. Suggest practical implications
3. Identify future research directions
4. Highlight innovative approaches

Research Documents:
{documents}

Query: {query}

Key Insights:"""

def run_query_retriever(query: str, k: int = 5, use_rrf: bool = True, 
                       embeddings_model: str = "all-MiniLM-L6-v2",
                       llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                       use_local_embeddings: bool = True,
                       use_chat_llm: bool = True) -> Dict[str, Any]:
    """Run the query retriever and capture its output."""
    # Use direct FAISS search instead of query_retriever.py to avoid API calls
    return run_direct_faiss_search(query, k, embeddings_model)

def run_direct_faiss_search(query: str, k: int = 5, embeddings_model: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Run direct FAISS search without API-dependent query expansion."""
    try:
        # Import required libraries
        import faiss
        import jsonlines
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from pathlib import Path
        
        # Load the FAISS index and related files
        root = Path(__file__).resolve().parents[1]
        idx_path = root / "data" / "faiss" / "index_flatip.faiss"
        ids_path = root / "data" / "faiss" / "ids.jsonl"
        chunks_path = root / "data" / "chunks.jsonl"
        
        if not idx_path.exists() or not ids_path.exists():
            return {"error": "FAISS index not found. Please build the index first."}
        
        # Load the embedding model
        print(f"Loading local Sentence-Transformers: {embeddings_model} ...")
        st_model = SentenceTransformer(embeddings_model, device="cpu")
        
        # Load FAISS index
        index = faiss.read_index(str(idx_path))
        
        # Load metadata
        with jsonlines.open(ids_path, "r") as r:
            id_rows = list(r)
        
        # Load chunks
        chunk_map = {}
        if chunks_path.exists():
            with jsonlines.open(chunks_path, "r") as reader:
                for rec in reader:
                    try:
                        key = (rec.get("paper_id"), int(rec.get("chunk_index", -1)))
                        chunk_map[key] = rec.get("chunk_text", "")
                    except Exception:
                        continue
        
        # Generate embedding for query
        vec = st_model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        q = np.asarray(vec, dtype=np.float32)
        
        # Search FAISS
        D, I = index.search(q, k)
        
        # Format results
        documents = []
        for rank, (score, ridx) in enumerate(zip(D[0], I[0]), 1):
            if ridx < 0:
                continue
            
            meta = id_rows[int(ridx)]
            key = (meta.get("paper_id"), int(meta.get("chunk_index", -1)))
            text = chunk_map.get(key, "")
            
            doc = {
                "rank": rank,
                "score": float(score),
                "rrf_score": float(score),  # Use same score for compatibility
                "paper_id": meta.get("paper_id", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "title": meta.get("title", "")[:150],
                "content": text  # This should be the FULL chunk text
            }
            documents.append(doc)
        
        return {
            "documents": documents,
            "total_docs": len(documents),
            "use_rrf": False  # Direct search, not RRF
        }
        
    except Exception as e:
        return {"error": f"Direct FAISS search failed: {str(e)}"}

def parse_retriever_output(output: str, use_rrf: bool) -> Dict[str, Any]:
    """Parse the query retriever output to extract document information."""
    documents = []
    lines = output.split('\n')
    
    if use_rrf:
        # Parse RRF results format
        in_results = False
        current_doc = {}
        
        for line in lines:
            line = line.strip()
            
            # Start of results section
            if "=== RRF Fused Results" in line:
                in_results = True
                continue
            
            if in_results and line.startswith('[') and ']' in line:
                # Parse document header: [rank] RRF=score paper_id #chunk_index title
                try:
                    # More flexible parsing - handle different spacing
                    import re
                    
                    # Extract rank
                    rank_match = re.search(r'\[(\d+)\]', line)
                    if not rank_match:
                        continue
                    rank = int(rank_match.group(1))
                    
                    # Extract RRF score
                    rrf_match = re.search(r'RRF=([\d.]+)', line)
                    if not rrf_match:
                        continue
                    rrf_score = float(rrf_match.group(1))
                    
                    # Extract paper_id and chunk_index
                    paper_match = re.search(r'RRF=[\d.]+  ([^#]+)#(\d+)', line)
                    if not paper_match:
                        continue
                    paper_id = paper_match.group(1).strip()
                    chunk_idx = int(paper_match.group(2))
                    
                    # Extract title (everything after the chunk index)
                    title_match = re.search(r'#\d+  (.+)', line)
                    title = title_match.group(1).strip() if title_match else ""
                    
                    current_doc = {
                        "rank": rank,
                        "rrf_score": rrf_score,
                        "paper_id": paper_id,
                        "chunk_index": chunk_idx,
                        "title": title,
                        "content": ""
                    }
                except (ValueError, IndexError, AttributeError):
                    continue
            
            elif in_results and current_doc and line and not line.startswith('['):
                # Content line
                current_doc["content"] += line + " "
            
            elif in_results and not line and current_doc:
                # End of document
                if current_doc.get("content"):
                    current_doc["content"] = current_doc["content"].strip()
                    documents.append(current_doc)
                    current_doc = {}
    
    else:
        # Parse individual query results format
        current_query = None
        current_doc = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("=== Query"):
                current_query = line
                continue
            
            if current_query and line.startswith('[') and ']' in line and 'score=' in line:
                # Parse document header: [rank] score=value paper_id #chunk_index title
                try:
                    parts = line.split('  ')
                    if len(parts) >= 4:
                        rank_part = parts[0]
                        score_part = parts[1]
                        paper_part = parts[2]
                        title_part = parts[3]
                        
                        rank = rank_part.strip('[]')
                        score = score_part.replace('score=', '')
                        paper_id = paper_part.split('#')[0].strip()
                        chunk_idx = paper_part.split('#')[1].strip() if '#' in paper_part else "0"
                        title = title_part
                        
                        current_doc = {
                            "rank": int(rank),
                            "score": float(score),
                            "paper_id": paper_id,
                            "chunk_index": int(chunk_idx),
                            "title": title,
                            "query": current_query,
                            "content": ""
                        }
                except (ValueError, IndexError):
                    continue
            
            elif current_query and current_doc and line and not line.startswith('['):
                # Content line
                current_doc["content"] += line + " "
            
            elif current_query and not line and current_doc:
                # End of document
                if current_doc.get("content"):
                    current_doc["content"] = current_doc["content"].strip()
                    documents.append(current_doc)
                    current_doc = {}
    
    return {
        "documents": documents,
        "total_docs": len(documents),
        "use_rrf": use_rrf
    }

def format_documents_for_prompt(documents: List[Dict[str, Any]], max_chars_per_doc: int = 2000) -> str:
    """Format documents for inclusion in the prompt."""
    formatted_docs = []
    
    for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents to keep prompt short
        # Truncate content if too long
        content = doc.get("content", "")
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        
        # Format document entry
        doc_entry = f"Document {i}:\n"
        doc_entry += f"Paper ID: {doc.get('paper_id', 'Unknown')}\n"
        doc_entry += f"Title: {doc.get('title', 'No title')}\n"
        
        if doc.get("rrf_score"):
            doc_entry += f"RRF Score: {doc['rrf_score']:.4f}\n"
        elif doc.get("score"):
            doc_entry += f"Similarity Score: {doc['score']:.4f}\n"
        
        doc_entry += f"Content: {content}\n"
        
        formatted_docs.append(doc_entry)
    
    return "\n".join(formatted_docs)

def create_simple_summary(documents: List[Dict[str, Any]], query: str) -> str:
    """Create a simple summary from the retrieved documents when LLM generation fails."""
    if not documents:
        return f"No documents found for query: {query}"
    
    summary_parts = []
    summary_parts.append(f"RESEARCH SUMMARY: {query}")
    summary_parts.append(f"Found {len(documents)} relevant documents\n")
    
    for i, doc in enumerate(documents[:3], 1):  # Limit to top 3
        paper_id = doc.get('paper_id', 'Unknown')
        title = doc.get('title', 'No title')
        content = doc.get('content', '')
        score = doc.get('score', 0)
        
        summary_parts.append(f"DOCUMENT {i} (Relevance: {score:.3f})")
        summary_parts.append(f"Paper ID: {paper_id}")
        summary_parts.append(f"Title: {title}")
        summary_parts.append("")
        
        if content:
            # Clean up the content and limit length for readability
            cleaned_content = content.replace('\n', ' ').strip()
            # Limit to first 1500 characters for readability
            if len(cleaned_content) > 1500:
                cleaned_content = cleaned_content[:1500] + "... [truncated]"
            summary_parts.append("Content:")
            summary_parts.append(cleaned_content)
        else:
            summary_parts.append("Content: [No content available]")
        
        summary_parts.append("\n" + "="*100 + "\n")
    
    return "\n".join(summary_parts)

def generate_content_with_llm(documents: List[Dict[str, Any]], query: str, 
                            config: GenerationConfig) -> str:
    """Generate content using a lightweight local LLM."""
    # Lazy imports for local models
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError:
        print("Error: transformers not available. Install with: pip install transformers torch")
        sys.exit(1)
    
    # Format documents for prompt
    formatted_docs = format_documents_for_prompt(documents)
    
    # Prepare the prompt
    prompt = config.prompt_template.format(
        documents=formatted_docs,
        query=query
    )
    
    try:
        # Use the configured local model
        model_name = config.llm_model
        print(f"Generating {config.generation_type} using local model: {model_name}...")
        
        # Initialize the pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Generate content
        response = generator(
            prompt,
            max_new_tokens=config.max_tokens,
            num_return_sequences=1,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            repetition_penalty=1.2
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Remove the original prompt from the response
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # If no meaningful content was generated, create a simple summary from the documents
        if not generated_text.strip() or len(generated_text.strip()) < 20:
            generated_text = create_simple_summary(documents, query)
        
        return generated_text
            
    except Exception as e:
        return f"Error generating content: {str(e)}"


def summarize_with_local_model(
    documents: List[Dict[str, Any]],
    query: str,
    model_id: str = "sshleifer/distilbart-cnn-12-6",
    max_summary_tokens: int = 256,
) -> str:
    """Map-reduce summarization over full chunks using a local summarization model.

    - Map: summarize each chunk independently (respecting model's max length)
    - Reduce: combine the per-chunk summaries and summarize again
    """
    if not documents:
        return f"No documents found for query: {query}"

    try:
        from transformers import pipeline
    except Exception as exc:
        return f"Error: transformers not available for summarization: {exc}"

    # Initialize summarization pipeline
    summarizer = pipeline("summarization", model=model_id)

    # Use the FULL chunks - don't split them artificially
    def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
        # Just return the full text - don't split it
        return [text]

    per_doc_summaries: List[str] = []
    for doc in documents:
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        # Split overly long chunks to avoid exceeding model limits
        text_segments = chunk_text(content)
        segment_summaries: List[str] = []
        for seg in text_segments:
            try:
                out = summarizer(
                    seg,
                    max_length=150,  # Fixed length for summary
                    min_length=50,   # Reasonable minimum
                    truncation=True,
                )
                if isinstance(out, list) and out and "summary_text" in out[0]:
                    segment_summaries.append(out[0]["summary_text"].strip())
            except Exception:
                # If summarizer fails on a segment, skip it
                continue

        # Combine segment summaries into one per-document summary
        if segment_summaries:
            combined = " \n".join(segment_summaries)
            # Optionally compress combined summary further
            try:
                reduced = summarizer(
                    combined,
                    max_length=150,
                    min_length=50,
                    truncation=True,
                )
                if isinstance(reduced, list) and reduced and "summary_text" in reduced[0]:
                    per_doc_summaries.append(reduced[0]["summary_text"].strip())
                else:
                    per_doc_summaries.append(combined)
            except Exception:
                per_doc_summaries.append(combined)

    if not per_doc_summaries:
        return create_simple_summary(documents, query)

    # Reduce across documents: global synthesis
    stitched = "\n".join(per_doc_summaries)
    try:
        global_sum = summarizer(
            f"Topic: {query}\n\nSummaries from top documents:\n{stitched}",
            max_length=200,
            min_length=100,
            truncation=True,
        )
        if isinstance(global_sum, list) and global_sum and "summary_text" in global_sum[0]:
            return global_sum[0]["summary_text"].strip()
    except Exception:
        pass

    return "\n".join(per_doc_summaries)

def main() -> None:
    parser = argparse.ArgumentParser(description="Content generator using reranking results from query_retriever.py")
    
    # Query and retrieval arguments
    parser.add_argument("--query", type=str, default="machine learning applications in healthcare", 
                       help="Seed query to expand and retrieve with")
    parser.add_argument("--k", type=int, default=5, 
                       help="Number of top documents to retrieve for generation")
    parser.add_argument("--use_rrf", action="store_true", 
                       help="Use Reciprocal Rank Fusion to combine results from multiple queries")
    parser.add_argument("--embeddings_model", type=str, default="all-MiniLM-L6-v2",
                       help="Hugging Face model for embeddings (default: all-MiniLM-L6-v2)")
    parser.add_argument("--retriever_llm_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Hugging Face model for query expansion in retriever")
    parser.add_argument("--use_local_embeddings", action="store_true", default=True,
                       help="Use local sentence-transformers embeddings (default: True)")
    parser.add_argument("--use_chat_llm", action="store_true", default=True,
                       help="Use chat-capable HF endpoint (default: True)")
    
    # Generation arguments
    parser.add_argument("--generation_type", type=str, default="summary",
                       choices=["summary", "research_synthesis", "analysis", "qa", "insights"],
                       help="Type of content to generate")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (0.0-1.0)")
    parser.add_argument("--llm_model", type=str, default="microsoft/DialoGPT-small",
                       help="Local Hugging Face model for content generation (default: microsoft/DialoGPT-small)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save generated content")
    parser.add_argument("--use_simple_summary", action="store_true",
                       help="Use simple document dump (debug) instead of LLM-based summarization")
    parser.add_argument("--use_summarizer", action="store_true",
                       help="Use local summarization model (map-reduce over full chunks)")
    parser.add_argument("--summarizer_model", type=str, default="sshleifer/distilbart-cnn-12-6",
                       help="Local HF summarization model id (default: sshleifer/distilbart-cnn-12-6)")
    
    args = parser.parse_args()
    
    # Load environment (no API token needed for local models)
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    # Note: No API token required since we use local models only
    
    # Create generation config
    config = GenerationConfig(
        generation_type=args.generation_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        llm_model=args.llm_model
    )
    
    print(f"Query: {args.query}")
    print(f"Generation type: {args.generation_type}")
    print(f"Using RRF: {args.use_rrf}")
    print(f"Retrieving top {args.k} documents...")
    print()
    
    # Run query retriever
    retriever_results = run_query_retriever(
        query=args.query,
        k=args.k,
        use_rrf=args.use_rrf,
        embeddings_model=args.embeddings_model,
        llm_model=args.retriever_llm_model,
        use_local_embeddings=args.use_local_embeddings,
        use_chat_llm=args.use_chat_llm
    )
    
    if "error" in retriever_results:
        print(f"Retrieval failed: {retriever_results['error']}")
        sys.exit(1)
    
    documents = retriever_results.get("documents", [])
    if not documents:
        print("No documents retrieved. Try adjusting your query or parameters.")
        sys.exit(1)
    
    print(f"Retrieved {len(documents)} documents for generation")
    
    # Debug: Show what we actually retrieved
    print("\nDEBUG - Retrieved documents:")
    for i, doc in enumerate(documents[:3], 1):
        print(f"  {i}. {doc.get('paper_id', 'Unknown')} - {doc.get('title', 'No title')[:100]}")
        print(f"     Score: {doc.get('score', 0):.3f}")
        print(f"     Content preview: {doc.get('content', '')[:200]}...")
        print()
    
    # Generate content
    if args.use_simple_summary:
        generated_content = create_simple_summary(documents, args.query)
    else:
        if args.use_summarizer:
            generated_content = summarize_with_local_model(
                documents=documents,
                query=args.query,
                model_id=args.summarizer_model,
                max_summary_tokens=args.max_tokens,
            )
        else:
            generated_content = generate_content_with_llm(documents, args.query, config)
    
    # Display results
    print("=" * 80)
    print(f"GENERATED {args.generation_type.upper()}")
    print("=" * 80)
    print(generated_content)
    print("=" * 80)
    
    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {args.query}\n")
            f.write(f"Generation Type: {args.generation_type}\n")
            f.write(f"Documents Used: {len(documents)}\n")
            f.write(f"Generated on: {Path(__file__).name}\n\n")
            f.write("=" * 80 + "\n")
            f.write(generated_content)
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\nContent saved to: {output_path}")

if __name__ == "__main__":
    main()
