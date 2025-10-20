#!/usr/bin/env python3
"""
RAG Generator using Elasticsearch + Local LLM

This script provides a Retrieval-Augmented Generation (RAG) system that:
1. Searches the Elasticsearch chunk store for relevant documents
2. Uses a local LLM to generate responses based on retrieved chunks
3. Provides an interactive interface for NLP queries

Usage Examples:
    # Interactive mode
    python rag_generator.py --interactive
    
    # Single query
    python rag_generator.py --query "What are the latest advances in transformer architectures?"
    
    # Custom parameters
    python rag_generator.py --query "your question" --k 5 --model "microsoft/DialoGPT-medium"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Elasticsearch imports
from elasticsearch import Elasticsearch

# Local LLM imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    print("Error: transformers library not found. Install with: pip install transformers torch")
    sys.exit(1)

class RAGGenerator:
    def __init__(self, 
                 es_url: str = "http://localhost:9200",
                 index_name: str = "papers_chunks_final",
                 llm_model: str = "microsoft/DialoGPT-large",
                 device: str = "cpu",
                 max_length: int = 512):
        """
        Initialize the RAG Generator
        
        Args:
            es_url: Elasticsearch URL
            index_name: Name of the Elasticsearch index
            llm_model: HuggingFace model name for local LLM
            device: Device to run the model on ('cpu' or 'cuda')
            max_length: Maximum length for LLM generation
        """
        self.es_url = es_url
        self.index_name = index_name
        self.llm_model = llm_model
        self.device = device
        self.max_length = max_length
        
        # Initialize Elasticsearch client
        self.es = Elasticsearch(es_url)
        
        # Initialize local LLM
        self._load_llm()
        
    def _load_llm(self):
        """Load the local LLM model and tokenizer"""
        print(f"Loading local LLM: {self.llm_model}")
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if using CPU
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"LLM loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Falling back to a smaller model...")
            # Fallback to a smaller, more reliable model
            self.llm_model = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Fallback LLM loaded: {self.llm_model}")
        
        # Try to use a better model if available
        try:
            if self.llm_model in ["distilgpt2", "gpt2"]:
                print("Trying to load a better instruction-following model...")
                # Try microsoft/DialoGPT-small which is better at following instructions
                self.llm_model = "microsoft/DialoGPT-small"
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                self.model = AutoModelForCausalLM.from_pretrained(self.llm_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Better model loaded: {self.llm_model}")
        except Exception as e:
            print(f"Could not load better model, using fallback: {e}")
        
        # Set model to evaluation mode for consistent generation
        self.model.eval()
    
    def search_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in Elasticsearch
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Use a more sophisticated search with better field weights
            body = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["chunk_text^3", "title^2", "authors"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["chunk_text"],
                                    "type": "phrase",
                                    "boost": 2
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": ["paper_id", "chunk_index", "title", "authors", "chunk_text", "year"],
                "highlight": {
                    "fields": {
                        "chunk_text": {"fragment_size": 150, "number_of_fragments": 1}
                    }
                }
            }
            
            response = self.es.search(index=self.index_name, body=body)
            chunks = []
            
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                chunks.append({
                    "score": hit["_score"],
                    "paper_id": source.get("paper_id", ""),
                    "chunk_index": source.get("chunk_index", 0),
                    "title": source.get("title", ""),
                    "authors": source.get("authors", ""),
                    "chunk_text": source.get("chunk_text", ""),
                    "year": source.get("year", "")
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return []
    
    def generate_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the local LLM based on retrieved chunks
        
        Args:
            query: User's question
            chunks: Retrieved relevant chunks
            
        Returns:
            Generated response
        """
        if not chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Create context from retrieved chunks
        context = self._create_context(chunks)
        
        # Create prompt for the LLM
        prompt = self._create_prompt(query, context)
        
        try:
            
            # Tokenize input with more context
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Generate response with parameters optimized for summarization
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=250,  # Longer for more comprehensive output
                    num_return_sequences=1,
                    temperature=0.8,  # Higher temperature for more diverse output
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,  # Prevent repetition
                    top_p=0.95,  # More diverse sampling
                    top_k=60,  # More diverse sampling
                    repetition_penalty=1.3,  # Reduce repetition more
                    early_stopping=True  # Stop when EOS token is generated
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            # Extract only the generated part (remove the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            else:
                # If prompt not found, take everything after the last "Summary:" 
                if "Summary:" in response:
                    response = response.split("Summary:")[-1].strip()
                elif "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                elif "Unified Summary:" in response:
                    response = response.split("Unified Summary:")[-1].strip()
                elif "Comprehensive Summary:" in response:
                    response = response.split("Comprehensive Summary:")[-1].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Always return the LLM response, even if it's bad
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response."
    
    def _create_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks with better filtering"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            paper_id = chunk.get("paper_id", "unknown")
            chunk_index = chunk.get("chunk_index", 0)
            text = chunk.get("chunk_text", "")
            
            # Clean Unicode characters
            text = text.encode('ascii', 'ignore').decode('ascii')
            
            # Skip chunks that are too fragmented or have too many references
            if self._is_problematic_chunk(text):
                continue
            
            # Clean the text further
            text = self._clean_chunk_text(text)
            
            # Skip if text becomes too short after cleaning
            if len(text.strip()) < 100:
                continue
            
            # Keep more text for better summarization (up to 800 chars)
            if len(text) > 800:
                text = text[:800] + "..."
            
            context_part = f"Source {i} (Paper: {paper_id}):\n\n{text}\n"
            context_parts.append(context_part)
        
        return "\n" + "="*60 + "\n".join(context_parts)
    
    def _is_problematic_chunk(self, text: str) -> bool:
        """Check if a chunk has problematic characteristics"""
        # Count problematic elements
        figure_refs = len([m for m in text.split() if 'Figure' in m or 'Fig.' in m])
        table_refs = len([m for m in text.split() if 'Table' in m or 'Tab.' in m])
        citation_refs = len([m for m in text.split() if m.startswith('[') and m.endswith(']')])
        dashes = len([m for m in text.split() if m == '-' or m.startswith('-')])
        fragmented_sentences = len([s for s in text.split('.') if len(s.strip()) < 10])
        
        # Less aggressive filtering - allow more technical content for better generation
        if (figure_refs > 3 or table_refs > 3 or citation_refs > 12 or 
            dashes > 20 or fragmented_sentences > 5):
            return True
        
        # Skip if text has too many incomplete sentences or weird formatting
        if ('...' in text and text.count('...') > 3) or text.count('  ') > 10:
            return True
        
        # Skip chunks with excessive numbers (often from figures/tables)
        numbers = len([w for w in text.split() if w.replace('.', '').replace(',', '').isdigit()])
        if numbers > len(text.split()) * 0.4:  # More than 40% numbers
            return True
        
        # Skip chunks that are mostly citations or references
        if citation_refs > len(text.split()) * 0.15:  # More than 15% citations
            return True
            
        return False
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text to remove problematic elements"""
        # Remove figure and table references
        import re
        text = re.sub(r'Figure \d+[a-z]?', '', text)
        text = re.sub(r'Fig\. \d+[a-z]?', '', text)
        text = re.sub(r'Table \d+[a-z]?', '', text)
        text = re.sub(r'Tab\. \d+[a-z]?', '', text)
        
        # Remove citation references in brackets
        text = re.sub(r'\[[\d,\s-]+\]', '', text)
        
        # Clean up multiple spaces and dashes
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*-\s*', ' ', text)
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Only keep substantial sentences
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences).strip()
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM focused on proper summarization"""
        prompt = f"""Based on the following research sources, provide a single unified summary that answers: {query}

Sources:
{context}

Write ONE comprehensive summary that synthesizes information from all sources. Do not list sources individually. Create a flowing, unified response that answers the question using information from all the research papers combined.

Summary:"""
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Clean Unicode characters
        response = response.encode('ascii', 'ignore').decode('ascii')
        
        # Remove any remaining prompt fragments
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Context from research papers:') and not line.startswith('Answer:'):
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Clean up fragmented text and weird dashes
        import re
        response = re.sub(r'\s*-\s*', ' ', response)  # Remove standalone dashes
        response = re.sub(r'\s+', ' ', response)  # Remove multiple spaces
        response = re.sub(r'\[[\d,\s-]+\]', '', response)  # Remove citation references
        
        # If response is too short or empty, try to generate something from the context
        if len(response.strip()) < 10:
            return "Based on the research papers, there are several approaches and methods discussed in the literature that address this topic."
        
        # Ensure response ends with a period
        if response and not response.endswith('.'):
            response += '.'
        
        # Limit response length
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        return response
    
    def _is_valid_response(self, response: str, query: str) -> bool:
        """Check if the response is valid and relevant to the query"""
        if not response or len(response.strip()) < 10:
            return False
        
        # Check if response contains some key terms from the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # If there's some overlap, it's probably relevant
        overlap = len(query_words.intersection(response_words))
        return overlap > 0
    
    def _create_template_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a template-based summary that extracts meaningful information from chunks"""
        if not chunks:
            return f"I couldn't find specific information about '{query}' in the research papers."
        
        # Extract key information from chunks
        key_points = []
        sources_mentioned = set()
        
        for chunk in chunks[:3]:  # Use top 3 chunks
            paper_id = chunk.get('paper_id', 'unknown')
            text = chunk.get('chunk_text', '')
            
            # Clean text
            text = text.encode('ascii', 'ignore').decode('ascii')
            text = self._clean_chunk_text(text)
            
            # Extract meaningful sentences
            sentences = text.split('. ')
            for sentence in sentences[:2]:  # Take first 2 sentences per chunk
                sentence = sentence.strip()
                if (len(sentence) > 30 and  # Substantial sentences
                    not any(word in sentence.lower() for word in ['figure', 'table', 'fig.', 'tab.']) and
                    not sentence.startswith('[') and  # No citation refs
                    len(sentence.split()) > 5 and  # Substantial content
                    not any(word in sentence.lower() for word in ['arxiv:', 'url:', 'doi:'])):  # No URLs
                    key_points.append(sentence)
                    sources_mentioned.add(f"Paper {paper_id}")
                    break  # Only take one good sentence per chunk
        return response
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: search + generate
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        print(f"Searching for relevant chunks...")
        chunks = self.search_chunks(question, k)
        
        if not chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "num_sources": 0
            }
        
        print(f"Found {len(chunks)} relevant chunks")
        
        # Print full retrieved chunks for debugging
        print("\n" + "="*80)
        print("FULL RETRIEVED CHUNKS:")
        print("="*80)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- CHUNK {i} ---")
            print(f"Paper ID: {chunk.get('paper_id', 'N/A')}")
            print(f"Chunk Index: {chunk.get('chunk_index', 'N/A')}")
            
            # Clean Unicode characters for display
            title = chunk.get('title', 'N/A')
            authors = chunk.get('authors', 'N/A')
            year = chunk.get('year', 'N/A')
            content = chunk.get('chunk_text', 'N/A')
            
            if title != 'N/A':
                title = title.encode('ascii', 'ignore').decode('ascii')
            if authors != 'N/A':
                authors = authors.encode('ascii', 'ignore').decode('ascii')
            if content != 'N/A':
                content = content.encode('ascii', 'ignore').decode('ascii')
            
            print(f"Title: {title}")
            print(f"Authors: {authors}")
            print(f"Year: {year}")
            print(f"Relevance Score: {chunk.get('score', 'N/A'):.2f}")
            print(f"Content:")
            print(content)
            print("-" * 60)
        
        print("="*80)
        print(f"Generating response...")
        
        response = self.generate_response(question, chunks)
        
        # Prepare sources for display
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append({
                "source_id": i,
                "paper_id": chunk["paper_id"],
                "score": chunk["score"]
            })
        
        return {
            "response": response,
            "sources": sources,
            "num_sources": len(sources)
        }
    
    def interactive_mode(self):
        """Run in interactive mode for continuous queries"""
        print("RAG Generator Interactive Mode")
        print("=" * 50)
        print("Ask questions about NLP research papers!")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n" + "="*50)
                result = self.query(question)
                
                print(f"\nResponse:")
                print(result["response"])
                
                if result["sources"]:
                    print(f"\nSources ({result['num_sources']}):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. {source['title']} ({source['year']})")
                        print(f"     Authors: {source['authors'].encode('ascii', 'ignore').decode('ascii')}")
                        print(f"     Paper ID: {source['paper_id']}")
                        print(f"     Relevance Score: {source['score']:.2f}")
                        print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG Generator using Elasticsearch + Local LLM")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium", 
                       help="HuggingFace model name (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run the model on (default: cpu)")
    parser.add_argument("--es-url", type=str, default="http://localhost:9200",
                       help="Elasticsearch URL (default: http://localhost:9200)")
    parser.add_argument("--index", type=str, default="papers_chunks",
                       help="Elasticsearch index name (default: papers_chunks)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    
    # Initialize RAG generator
    try:
        rag = RAGGenerator(
            es_url=args.es_url,
            index_name=args.index,
            llm_model=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize RAG generator: {e}")
        sys.exit(1)
    
    # Check if Elasticsearch is running
    try:
        rag.es.ping()
        print("Elasticsearch connection successful")
    except Exception as e:
        print(f"Cannot connect to Elasticsearch: {e}")
        print("Make sure Elasticsearch is running on the specified URL")
        sys.exit(1)
    
    # Run in interactive mode or process single query
    if args.interactive:
        rag.interactive_mode()
    elif args.query:
        result = rag.query(args.query, args.k)
        
        print(f"\nResponse:")
        print(result["response"])
        
        if result["sources"]:
            print(f"\nSources ({result['num_sources']}):")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. Paper: {source['paper_id']}")
                print(f"     Relevance Score: {source['score']:.2f}")
                print()
    else:
        print("Please specify either --query or --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
