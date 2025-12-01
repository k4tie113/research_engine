#!/usr/bin/env python3
"""
Quick script to test query analysis.
Usage: python test_query.py "your query here"
"""

import sys
import json
from query_analyzer import analyze_query

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query.py 'your query here'")
        print("\nExample: python test_query.py 'Papers by Andrej Karpathy in NeurIPS 2024'")
        return
    
    query = " ".join(sys.argv[1:])
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    result = analyze_query(query)
    
    print(json.dumps(result, indent=2))
    print()

if __name__ == "__main__":
    main()
