from __future__ import annotations
import logging
import os
import json as _json
from typing import TypedDict, Iterable
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)

class CombineAnchorInput(TypedDict):
    query: str
    anchors_markdown: str

class CombineAnchorOutput(BaseModel):
    combined_query: str

def _chat_json(system_prompt: str, user_prompt: str, out_model: type[BaseModel], max_tokens: int = 800) -> BaseModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Degrade gracefully: pretend we combined (i.e., no-op)
        return out_model.model_validate({"combined_query": user_prompt})
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    raw = resp.choices[0].message.content or "{}"
    data = _json.loads(raw)
    return out_model.model_validate(data)

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
    sys = "Return a single JSON object {'combined_query': string}."
    user = _COMBINE_PROMPT.replace("{{query}}", inp["query"]).replace("{{anchors_markdown}}", inp["anchors_markdown"])
    return _chat_json(sys, user, CombineAnchorOutput, max_tokens=800)

def combine_content_query_with_anchors(content_query: str, anchor_docs: Iterable[str] | None) -> str:
    """
    anchor_docs: iterable of markdown strings (or objects with .markdown attr).
    """
    try:
        if not anchor_docs:
            return content_query
        chunks: list[str] = []
        for a in anchor_docs:
            try:
                chunks.append(getattr(a, "markdown", None) or str(a) or "")
            except Exception:
                chunks.append(str(a) or "")
        anchor_md = "\n".join(chunks)
        out = _combine_call({"query": content_query, "anchors_markdown": anchor_md})
        return (out.combined_query or "").strip() or content_query
    except Exception as e:
        logger.exception(f"Failed to combine query with anchors: {e}")
        return content_query
