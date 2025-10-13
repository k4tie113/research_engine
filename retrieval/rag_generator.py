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
                 index_name: str = "papers_chunks",
                 llm_model: str = "microsoft/DialoGPT-medium",
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
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,  # Generate up to 150 new tokens
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    top_p=0.9,
                    top_k=50
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            else:
                # If prompt not found, take everything after the last "Answer:" 
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # If response is too short, contains the prompt, or doesn't make sense, use template-based response
            if (len(response) < 20 or 
                "Question:" in response or 
                "Context from research papers:" in response or
                not self._is_valid_response(response, query)):
                return self._create_template_response(query, chunks)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._create_template_response(query, chunks)
    
    def _create_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown Title")
            authors = chunk.get("authors", "Unknown Authors")
            text = chunk.get("chunk_text", "")
            year = chunk.get("year", "")
            
            # Clean Unicode characters
            title = title.encode('ascii', 'ignore').decode('ascii')
            authors = authors.encode('ascii', 'ignore').decode('ascii')
            text = text.encode('ascii', 'ignore').decode('ascii')
            
            # Truncate text to keep context manageable
            if len(text) > 500:
                text = text[:500] + "..."
            
            context_part = f"[Source {i}]"
            if title:
                context_part += f" Title: {title}"
            if authors:
                context_part += f" Authors: {authors}"
            if year:
                context_part += f" Year: {year}"
            context_part += f"\nContent: {text}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM"""
        prompt = f"""You are a helpful assistant that answers questions about NLP research papers. Use the provided context to answer the question.

Question: {query}

Context from research papers:
{context}

Based on the research papers above, please answer the question: {query}

Answer:"""
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
        """Create a template-based response when LLM generation fails"""
        if not chunks:
            return f"I couldn't find specific information about '{query}' in the research papers."
        
        # Extract key information from chunks
        titles = [chunk.get('title', '') for chunk in chunks if chunk.get('title')]
        authors = [chunk.get('authors', '') for chunk in chunks if chunk.get('authors')]
        
        # Create a simple summary
        response = f"Based on the research papers, here's what I found about '{query}':\n\n"
        
        for i, chunk in enumerate(chunks[:3], 1):  # Use top 3 chunks
            title = chunk.get('title', 'Unknown Title')
            text = chunk.get('chunk_text', '')[:200]  # First 200 chars
            
            if title and text:
                response += f"{i}. From '{title}': {text}...\n\n"
        
        if titles:
            response += f"Relevant papers include: {', '.join(titles[:3])}"
        
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
        print(f"Generating response...")
        
        response = self.generate_response(question, chunks)
        
        # Prepare sources for display
        sources = []
        for chunk in chunks:
            sources.append({
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "year": chunk["year"],
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
                print(f"  {i}. {source['title']} ({source['year']})")
                print(f"     Authors: {source['authors'].encode('ascii', 'ignore').decode('ascii')}")
                print(f"     Paper ID: {source['paper_id']}")
                print(f"     Relevance Score: {source['score']:.2f}")
                print()
    else:
        print("Please specify either --query or --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
