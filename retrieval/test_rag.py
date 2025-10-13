#!/usr/bin/env python3
"""
Test script for the RAG Generator

This script tests the RAG generator with sample NLP queries to ensure
everything is working correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.rag_generator import RAGGenerator

def test_rag_generator():
    """Test the RAG generator with sample queries"""
    
    print("Testing RAG Generator")
    print("=" * 50)
    
    # Initialize the generator
    try:
        rag = RAGGenerator(
            llm_model="distilgpt2",  # Use a smaller, faster model for testing
            device="cpu"
        )
        print("RAG Generator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG generator: {e}")
        return False
    
    # Test queries
    test_queries = [
        "What are transformer architectures?",
        "How does attention mechanism work?",
        "What is BERT and how does it work?",
        "Recent advances in natural language processing"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            result = rag.query(query, k=3)  # Use fewer chunks for faster testing
            
            print(f"Response: {result['response'][:200]}...")
            print(f"Sources found: {result['num_sources']}")
            
            if result['sources']:
                print("Top source:")
                source = result['sources'][0]
                print(f"  - {source['title']} ({source['year']})")
                print(f"  - Score: {source['score']:.2f}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return False
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_rag_generator()
    sys.exit(0 if success else 1)
