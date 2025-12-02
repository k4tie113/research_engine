from __future__ import annotations
import logging
import os
import json as _json
from typing import TypedDict, Iterable, Any
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)

class CombineAnchorInput(TypedDict):
    query: str
    anchors_markdown: str

class CombineAnchorOutput(BaseModel):
    combined_query: str

def _chat_json(
    system_prompt: str, 
    user_prompt: str, 
    out_model: type[BaseModel], 
    max_tokens: int = 800
) -> BaseModel:
    """
    Call OpenAI API with JSON response format.
    Falls back gracefully if no API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - degrading gracefully")
        # Return original query as combined query (no-op)
        return out_model.model_validate({"combined_query": user_prompt})
    
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = _json.loads(raw)
        return out_model.model_validate(data)
    except Exception as e:
        logger.exception(f"Error calling OpenAI API: {e}")
        # Fall back to original query
        return out_model.model_validate({"combined_query": user_prompt})

_COMBINE_PROMPT = """
You will combine a user query with a list of anchor documents to produce a minimally
augmented query that preserves the original intent but resolves underspecified terms
when the anchors provide helpful context.

Rules:
- Keep the original query wording dominant.
- Only add small clarifications from anchors if they help disambiguate.
- Do not change the main intent.
- Output: {"combined_query": "<final query>"}

Original Query:
{{query}}

Anchor Documents (markdown):
{{anchors_markdown}}
"""

def _combine_call(inp: CombineAnchorInput) -> CombineAnchorOutput:
    """
    Internal helper to make the LLM call for combining query with anchors.
    """
    sys = "Return a single JSON object {'combined_query': string}."
    user = _COMBINE_PROMPT.replace("{{query}}", inp["query"]).replace(
        "{{anchors_markdown}}", inp["anchors_markdown"]
    )
    return _chat_json(sys, user, CombineAnchorOutput, max_tokens=800)

def combine_content_query_with_anchors(
    content_query: str, 
    anchor_docs: Iterable[Any] | None
) -> str:
    """
    Combine a content query with anchor documents to create an augmented query.
    
    Args:
        content_query: The original user query string
        anchor_docs: Iterable of documents (strings or objects with .markdown attribute)
    
    Returns:
        Combined query string (or original if combination fails/not needed)
    """
    # Early return if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("No OPENAI_API_KEY - returning original query")
        return content_query
    
    try:
        # If no anchor docs provided, return original
        if not anchor_docs:
            return content_query
        
        # Extract markdown content from anchor documents
        chunks: list[str] = []
        for a in anchor_docs:
            try:
                # Try to get .markdown attribute, fall back to string conversion
                markdown_content = getattr(a, "markdown", None)
                if markdown_content:
                    chunks.append(str(markdown_content))
                else:
                    str_content = str(a) if a else ""
                    if str_content:
                        chunks.append(str_content)
            except Exception as e:
                logger.warning(f"Error extracting content from anchor doc: {e}")
                # Try simple string conversion as last resort
                try:
                    str_content = str(a) if a else ""
                    if str_content:
                        chunks.append(str_content)
                except Exception:
                    continue
        
        # If no valid chunks extracted, return original
        if not chunks:
            logger.info("No valid anchor content extracted")
            return content_query
        
        # Combine chunks with newlines
        anchor_md = "\n".join(chunks)
        
        # Make the combination call
        out = _combine_call({
            "query": content_query, 
            "anchors_markdown": anchor_md
        })
        
        # Extract and validate result
        combined = (out.combined_query or "").strip()
        
        # Return combined query if valid, otherwise original
        return combined if combined else content_query
        
    except Exception as e:
        logger.exception(f"Failed to combine query with anchors: {e}")
        return content_query
