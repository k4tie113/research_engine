#!/usr/bin/env python3
"""
content_generator_v2.py
-----------------------
Clean, focused content generator that:
1. Uses query_retriever.py for document retrieval with reranking
2. Processes FULL chunks from the results
3. Generates summaries using local lightweight LLM (DistilBART CNN)
4. No API calls, no subprocess complexity - direct integration
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import numpy as np
import faiss

# Import transformers for local LLM
try:
    from transformers import pipeline
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: Required libraries not installed. Run: pip install transformers torch sentence-transformers faiss-cpu")
    sys.exit(1)


def run_direct_faiss_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Directly search FAISS index to get documents with full chunks - no API calls.
    """
    print(f"Searching FAISS index directly for: '{query}' (k={k})")
    
    try:
        # Load FAISS index and metadata
        root_dir = Path(__file__).resolve().parents[1]
        faiss_dir = root_dir / "data" / "faiss"
        chunks_file = root_dir / "data" / "chunks.jsonl"
        
        # Load FAISS index
        index_path = faiss_dir / "index_flatip.faiss"
        ids_path = faiss_dir / "ids.jsonl"
        
        if not index_path.exists() or not ids_path.exists() or not chunks_file.exists():
            print(f"ERROR: Missing required files:")
            print(f"  - Index: {index_path.exists()}")
            print(f"  - IDs: {ids_path.exists()}")
            print(f"  - Chunks: {chunks_file.exists()}")
            return []
        
        # Load the index
        index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load metadata
        ids_data = []
        with open(ids_path, 'r', encoding='utf-8') as f:
            for line in f:
                ids_data.append(json.loads(line.strip()))
        
        # Load chunks mapping
        chunk_map = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_data = json.loads(line.strip())
                key = (chunk_data['paper_id'], chunk_data['chunk_index'])
                chunk_map[key] = chunk_data['chunk_text']
        
        print(f"Loaded {len(chunk_map)} chunks from chunks.jsonl")
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embed query
        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS with more results to allow for replacements
        search_k = min(k * 3, 50)  # Search more broadly for better chunks
        scores, indices = index.search(query_embedding, search_k)
        
        # Retrieve documents with content quality filtering
        documents = []
        used_indices = set()
        
        # First pass: collect initial documents (only the top k requested)
        initial_documents = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if i >= k or idx < 0:  # Only take the first k results
                break
                
            # Get metadata
            meta = ids_data[idx]
            paper_id = meta.get('paper_id')
            chunk_index = meta.get('chunk_index')
            
            # Get full chunk content
            key = (paper_id, chunk_index)
            content = chunk_map.get(key, "")
            
            if content:  # Only add if we have content
                initial_documents.append({
                    'paper_id': paper_id,
                    'title': meta.get('title', ''),
                    'score': float(score),
                    'content': content,
                    'idx': idx,
                    'chunk_index': chunk_index
                })
                used_indices.add(idx)
        
        # Second pass: check content quality and find replacements if needed
        for doc in initial_documents:
            raw_content = doc['content']
            cleaned_content = clean_content(raw_content)
            
            # Check if cleaned content is too short (less than 50% of raw)
            content_ratio = len(cleaned_content) / len(raw_content) if len(raw_content) > 0 else 0
            
            if content_ratio < 0.5 and len(cleaned_content) < 500:  # Mostly references/publication info
                print(f"Chunk {doc['paper_id']}-{doc['chunk_index']} is mostly references ({content_ratio:.2f} ratio), finding replacement...")
                
                # Find a better chunk from the same paper or similar papers
                replacement_found = False
                
                # Try to find other chunks from the same paper first
                for score, idx in zip(scores[0], indices[0]):
                    if idx in used_indices or idx < 0:
                        continue
                        
                    meta = ids_data[idx]
                    if meta.get('paper_id') == doc['paper_id']:
                        # Same paper, different chunk
                        chunk_index = meta.get('chunk_index')
                        key = (doc['paper_id'], chunk_index)
                        replacement_content = chunk_map.get(key, "")
                        
                        if replacement_content:
                            cleaned_replacement = clean_content(replacement_content)
                            replacement_ratio = len(cleaned_replacement) / len(replacement_content) if len(replacement_content) > 0 else 0
                            
                            if replacement_ratio >= 0.5 or len(cleaned_replacement) >= 500:
                                print(f"  -> Found better chunk from same paper (ratio: {replacement_ratio:.2f})")
                                documents.append({
                                    'paper_id': doc['paper_id'],
                                    'title': meta.get('title', doc['title']),
                                    'score': float(score),
                                    'content': replacement_content
                                })
                                used_indices.add(idx)
                                replacement_found = True
                                break
                
                # If no better chunk from same paper, try any other chunk
                if not replacement_found:
                    for score, idx in zip(scores[0], indices[0]):
                        if idx in used_indices or idx < 0:
                            continue
                            
                        meta = ids_data[idx]
                        paper_id = meta.get('paper_id')
                        chunk_index = meta.get('chunk_index')
                        key = (paper_id, chunk_index)
                        replacement_content = chunk_map.get(key, "")
                        
                        if replacement_content:
                            cleaned_replacement = clean_content(replacement_content)
                            replacement_ratio = len(cleaned_replacement) / len(replacement_content) if len(replacement_content) > 0 else 0
                            
                            if replacement_ratio >= 0.5 or len(cleaned_replacement) >= 500:
                                print(f"  -> Found better chunk from different paper (ratio: {replacement_ratio:.2f})")
                                documents.append({
                                    'paper_id': paper_id,
                                    'title': meta.get('title', ''),
                                    'score': float(score),
                                    'content': replacement_content
                                })
                                used_indices.add(idx)
                                replacement_found = True
                                break
                
                if not replacement_found:
                    print(f"  -> No better chunk found, keeping original")
                    documents.append({
                        'paper_id': doc['paper_id'],
                        'title': doc['title'],
                        'score': doc['score'],
                        'content': doc['content']
                    })
            else:
                # Content is good quality, keep it
                documents.append({
                    'paper_id': doc['paper_id'],
                    'title': doc['title'],
                    'score': doc['score'],
                    'content': doc['content']
                })
        
        print(f"Retrieved {len(documents)} documents from FAISS (after content quality filtering)")
        
        # Debug: Print full documents
        print("\n" + "="*80)
        print("RETRIEVED DOCUMENTS (FULL CONTENT)")
        print("="*80)
        for i, doc in enumerate(documents):
            print(f"\n--- DOCUMENT {i+1} ---")
            print(f"Paper ID: {doc['paper_id']}")
            print(f"Title: {doc['title']}")
            print(f"Score: {doc['score']:.3f}")
            print(f"Raw Content Length: {len(doc['content'])} characters")
            
            # Show cleaned content
            cleaned_content = clean_content(doc['content'])
            print(f"Cleaned Content Length: {len(cleaned_content)} characters")
            
            print("\nRAW CONTENT:")
            print("-" * 40)
            print(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
            print("-" * 40)
            
            print("\nCLEANED CONTENT:")
            print("-" * 40)
            print(cleaned_content[:500] + "..." if len(cleaned_content) > 500 else cleaned_content)
            print("-" * 40)
        print("="*80)
        
        return documents
        
    except Exception as e:
        print(f"ERROR in direct FAISS search: {e}")
        return []




def clean_content(content: str) -> str:
    """
    Remove publication information, references, and citations from content.
    """
    # Remove common reference patterns
    patterns_to_remove = [
        r'References?\s*$',  # "References" or "Reference" at end
        r'Bibliography\s*$',  # "Bibliography" at end
        r'^\s*\d+\.\s+.*?\.\s+.*?\d{4}\.',  # Numbered references like "1. Author, Title. 2023."
        r'^\s*\[[\d,\s-]+\]\s+.*?\d{4}',  # Bracket references like "[1,2,3] Author. 2023"
        r'^\s*\w+\s+et\s+al\.\s+.*?\d{4}',  # "Author et al. 2023" patterns
        r'doi:\s*[\d\.\/-]+',  # DOI patterns
        r'arXiv:\s*[\d\.]+v?\d*',  # arXiv patterns
        r'URL\s+https?://[^\s]+',  # URLs
        r'ISBN\s*[\d-]+',  # ISBN patterns
        r'Proceedings\s+of\s+the.*?\d{4}',  # Conference proceedings
        r'In\s+.*?\d{4}',  # "In Conference Name 2023"
        r'pp\.\s*\d+[-–]\d+',  # Page ranges
        r'volume\s+\d+',  # Volume numbers
        r'pages\s+\d+[-–]\d+',  # Page references
        r'doi:\s*10\.\d+\/[^\s]+',  # DOI with 10.x prefix
    ]
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append(line)
            continue
            
        # Check if line matches any removal pattern
        should_remove = False
        for pattern in patterns_to_remove:
            if re.search(pattern, line, re.IGNORECASE):
                should_remove = True
                break
        
        if not should_remove:
            cleaned_lines.append(line)
    
    # Join and clean up multiple newlines
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)  # Remove excessive newlines
    
    return cleaned_content.strip()


def generate_summary_with_llm(documents: List[Dict[str, Any]], query: str) -> str:
    """
    Generate a comprehensive summary using local DistilBART CNN summarizer.
    """
    if not documents:
        return "No documents found to summarize."
    
    print(f"Generating summary using local DistilBART CNN for {len(documents)} documents...")
    
    try:
        # Initialize the summarizer
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Combine all document content with cleaning
        all_content = []
        for i, doc in enumerate(documents):
            title = doc.get('title', 'Untitled')
            raw_content = doc.get('content', '')
            score = doc.get('score', 0.0)
            
            # Clean the content to remove references and publication info
            cleaned_content = clean_content(raw_content)
            
            # Format document
            doc_text = f"Document {i+1} (Relevance: {score:.3f})\n"
            doc_text += f"Title: {title}\n"
            doc_text += f"Paper ID: {doc.get('paper_id', 'Unknown')}\n\n"
            doc_text += f"Content: {cleaned_content}\n\n"
            
            all_content.append(doc_text)
        
        combined_text = "".join(all_content)
        
        # Use longer input length and generate longer summaries
        max_input_length = 2048  # Increased from 1024
        if len(combined_text) > max_input_length:
            combined_text = combined_text[:max_input_length] + "..."
        
        # Generate longer summary
        summary_result = summarizer(
            f"Query: {query}\n\nDocuments:\n{combined_text}",
            max_length=500,  # Increased from 300
            min_length=200,  # Increased from 100
            do_sample=False,
            truncation=True
        )
        
        if isinstance(summary_result, list) and len(summary_result) > 0:
            summary = summary_result[0]['summary_text']
            return summary.strip()
        else:
            return "Failed to generate summary."
            
    except Exception as e:
        print(f"Error generating summary with LLM: {e}")
        return f"Error generating summary: {str(e)}"


def create_simple_summary(documents: List[Dict[str, Any]], query: str) -> str:
    """
    Fallback: Create a simple summary by extracting key information.
    """
    if not documents:
        return "No relevant documents found."
    
    summary_parts = [f"RESEARCH SUMMARY: {query}"]
    summary_parts.append(f"Found {len(documents)} relevant documents:\n")
    
    for i, doc in enumerate(documents):
        title = doc.get('title', 'Untitled')
        paper_id = doc.get('paper_id', 'Unknown')
        score = doc.get('score', 0.0)
        raw_content = doc.get('content', '')
        
        # Clean the content to remove references
        cleaned_content = clean_content(raw_content)
        
        summary_parts.append(f"{i+1}. {title} (ID: {paper_id}, Relevance: {score:.3f})")
        
        # Extract first few sentences as key findings from cleaned content
        sentences = cleaned_content.split('.')[:5]  # First 5 sentences (increased from 3)
        if sentences:
            key_findings = '. '.join(sentences).strip()
            if key_findings:
                summary_parts.append(f"   Key findings: {key_findings}...")
        
        summary_parts.append("")
    
    return "\n".join(summary_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate content summaries from retrieved documents")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--use_simple", action="store_true", help="Use simple summary instead of LLM")
    parser.add_argument("--output", help="Output file to save summary")
    
    args = parser.parse_args()
    
    print(f"=== CONTENT GENERATOR V2 ===")
    print(f"Query: {args.query}")
    print(f"Documents to retrieve: {args.k}")
    print(f"Summary mode: {'Simple' if args.use_simple else 'LLM'}")
    print()
    
    # Step 1: Get documents from direct FAISS search
    documents = run_direct_faiss_search(args.query, args.k)
    
    if not documents:
        print("No documents retrieved. Exiting.")
        return
    
    print(f"Successfully retrieved {len(documents)} documents")
    
    # Step 2: Generate summary
    if args.use_simple:
        summary = create_simple_summary(documents, args.query)
    else:
        summary = generate_summary_with_llm(documents, args.query)
    
    # Step 3: Output results
    print("\n" + "="*80)
    print("GENERATED SUMMARY")
    print("="*80)
    print(summary)
    print("="*80)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"Query: {args.query}\n\n")
            f.write(summary)
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
