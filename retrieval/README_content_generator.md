# Content Generator

The `content_generator.py` script uses the reranking results from `query_retriever.py` to generate coherent content using a lightweight LLM. It's designed to work seamlessly with the existing retrieval pipeline.

## Features

- **Multiple Generation Types**: Summary, research synthesis, analysis, Q&A, and insights
- **Lightweight LLM Integration**: Uses smaller, faster models like DialoGPT-medium for generation
- **Flexible Input**: Works with both RRF reranked results and individual query results
- **Configurable Output**: Adjustable token limits, temperature, and generation parameters
- **File Output**: Option to save generated content to files

## Usage

### Basic Usage

```bash
# Simple document summarization (recommended - fast and reliable)
python content_generator.py --query "machine learning in healthcare" --k 5 --use_simple_summary

# LLM-based generation (experimental)
python content_generator.py --query "machine learning in healthcare" --k 5 --max_tokens 256

# Generate with custom parameters
python content_generator.py --query "your search query" --k 10 --max_tokens 512 --use_simple_summary
```

### Generation Types

1. **`summary`** (default): Comprehensive summary of key findings
2. **`research_synthesis`**: Integration of findings across multiple sources
3. **`analysis`**: Critical analysis of methodologies and findings
4. **`qa`**: Question-answering based on retrieved documents
5. **`insights`**: Key insights and future research directions

### Advanced Options

```bash
# Use different local LLM models (no API credits needed)
python content_generator.py --query "your query" --llm_model "microsoft/DialoGPT-small"
python content_generator.py --query "your query" --llm_model "gpt2"  # Even smaller model
python content_generator.py --query "your query" --llm_model "distilgpt2"  # Lightweight alternative

# Save output to file
python content_generator.py --query "your query" --output_file "output/summary.txt"

# Use MiniLM for fast embeddings (default, but shown for clarity)
python content_generator.py --query "your query" --embeddings_model "all-MiniLM-L6-v2"
```

## Configuration

### Environment Variables

- **No API token required!** The script runs completely locally using your FAISS index and local models.

**Note**: The script bypasses API calls entirely by using direct FAISS search instead of the query expansion from `query_retriever.py`. This eliminates all API credit requirements.

### Default Models

- **Retrieval**: Direct FAISS search (no API calls, no query expansion)
- **Generation LLM**: `microsoft/DialoGPT-small` (local model, no API credits needed)
- **Embeddings**: `all-MiniLM-L6-v2` (fast, lightweight embeddings for vector search)

**Why Local Models?**
- **No API Credits**: Runs completely locally for content generation
- **Fast**: DialoGPT-small is optimized for speed and efficiency
- **Reliable**: Consistent performance without network dependencies
- **Privacy**: All generation happens locally on your machine

**Why MiniLM?**
- **Fast**: Much faster than larger models like GritLM-7B
- **Reliable**: Consistent performance across different queries
- **Lightweight**: Lower memory and computational requirements
- **Compatible**: Works well with most FAISS indices

## How It Works

1. **Direct FAISS Search**: Bypasses `query_retriever.py` and searches FAISS index directly
2. **Local Embeddings**: Uses MiniLM to embed the query locally
3. **Document Retrieval**: Gets top-k most similar documents from FAISS
4. **Content Generation**: Two modes available:
   - **Simple Summary**: Extracts key findings from documents (recommended)
   - **LLM Generation**: Uses local DialoGPT-small for content generation (experimental)
5. **Output**: Displays and optionally saves the generated content

## Example Output

```
Query: machine learning in healthcare
Generation type: summary
Using RRF: True
Retrieving top 5 documents...

Retrieved 5 documents for generation

================================================================================
GENERATED SUMMARY
================================================================================

Based on the retrieved research documents, machine learning applications in 
healthcare show significant potential across multiple domains. The studies 
demonstrate successful implementations in diagnostic imaging, predictive 
analytics, and treatment optimization. Key findings include improved accuracy 
in medical image analysis, enhanced patient outcome predictions, and more 
personalized treatment approaches. However, challenges remain in data privacy, 
model interpretability, and clinical validation processes...

================================================================================
```

## Dependencies

- `langchain`: Core framework for LLM integration
- `langchain-huggingface`: HuggingFace model integration
- `langchain-community`: Community extensions
- All dependencies from `query_retriever.py`

## Integration with Existing Pipeline

The content generator is designed to work seamlessly with the existing research engine:

1. **Data Flow**: `query_retriever.py` → `content_generator.py`
2. **Shared Configuration**: Uses same FAISS index and embedding models
3. **Consistent Interface**: Similar CLI patterns and error handling
4. **Extensible**: Easy to add new generation types or models

## Tips for Best Results

1. **Use RRF**: Enable `--use_rrf` for better document diversity
2. **Adjust k**: Increase `--k` for more comprehensive coverage
3. **Choose Generation Type**: Match generation type to your use case
4. **Experiment with Models**: Try different LLM models for different styles
5. **Fine-tune Parameters**: Adjust temperature and max_tokens for desired output

## Troubleshooting

- **No documents retrieved**: Check if FAISS index exists and query is appropriate
- **Generation errors**: Verify HF token and model availability
- **Poor quality output**: Try different generation types or adjust parameters
- **Slow performance**: Use smaller models or reduce k value
