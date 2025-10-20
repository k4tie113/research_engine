import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from elasticsearch import Elasticsearch
from typing import Dict, List, Any
import json

class SimpleRAGGenerator:
    def __init__(self, 
                 es_url: str = "http://localhost:9200",
                 index_name: str = "papers_chunks_final",
                 llm_model: str = "gpt2",
                 device: str = "cpu"):
        """
        Simple RAG Generator
        """
        self.es_url = es_url
        self.index_name = index_name
        self.device = device
        
        # Initialize Elasticsearch
        self.es = Elasticsearch(es_url)
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model)
        self.model.to(device)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Simple RAG Generator initialized with {llm_model}")

    def search_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks in Elasticsearch"""
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["chunk_text^3", "title^2", "authors"],
                    "fuzziness": "AUTO"
                }
            },
            "size": k
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        chunks = []
        
        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunks.append(chunk)
        
        return chunks

    def generate_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate response using LLM"""
        # Create simple context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('chunk_text', '')
            paper_id = chunk.get('paper_id', 'unknown')
            
            # Simple text cleaning
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Truncate to reasonable length
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            context_parts.append(f"Source {i} (Paper {paper_id}): {text}")
        
        context = "\n\n".join(context_parts)
        
        # Simple prompt
        prompt = f"""Question: {query}

Based on these research sources, provide a comprehensive answer:

{context}

Answer:"""
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            # Simple cleanup
            response = response.replace('\n', ' ').strip()
            if len(response) > 1000:
                response = response[:1000] + "..."
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error generating response."

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Main query method"""
        print(f"Searching for: {question}")
        
        # Search for chunks
        chunks = self.search_chunks(question, k)
        
        if not chunks:
            return {
                "response": "No relevant information found.",
                "sources": [],
                "num_sources": 0
            }
        
        print(f"Found {len(chunks)} chunks")
        
        # Print chunks for debugging
        print("\n" + "="*60)
        print("RETRIEVED CHUNKS:")
        print("="*60)
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Paper: {chunk.get('paper_id', 'N/A')}")
            print(f"Score: {chunk.get('score', 0):.2f}")
            text = chunk.get('chunk_text', '')[:200]
            print(f"Text: {text}...")
        print("="*60)
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(question, chunks)
        
        # Prepare sources
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

def main():
    """Test the simple RAG generator"""
    generator = SimpleRAGGenerator()
    
    # Test query
    result = generator.query("How is machine learning used in NLP?")
    
    print(f"\nResponse: {result['response']}")
    print(f"\nSources ({result['num_sources']}):")
    for source in result['sources']:
        print(f"  - Paper: {source['paper_id']} (Score: {source['score']:.2f})")

if __name__ == "__main__":
    main()
