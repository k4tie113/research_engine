# semantic_alignment.py

import numpy as np
import re

def align_paragraphs_to_retrieved_chunks(answer_text, embed_model, retrieved_chunks, top_k=1, threshold=0.3):
    """
    Align each paragraph in the model's answer to the most semantically similar
    retrieved chunk (paragraph-level alignment instead of sentence-level).
    """

    # Split into paragraphs (double newline)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', answer_text.strip()) if p.strip()]
    if not paragraphs:
        return []

    # Encode paragraphs and retrieved chunks
    paragraph_embeddings = embed_model.encode(paragraphs, normalize_embeddings=True)
    chunk_texts = [r["chunk_text"] for r in retrieved_chunks]
    chunk_embeddings = embed_model.encode(chunk_texts, normalize_embeddings=True)

    # Compute cosine similarities
    similarities = np.dot(paragraph_embeddings, np.array(chunk_embeddings).T)

    alignments = []
    for i, sims in enumerate(similarities):
        best_indices = np.argsort(-sims)[:top_k]
        for idx in best_indices:
            score = float(sims[idx])
            if score < threshold:
                continue
            chunk_info = retrieved_chunks[idx]
            alignments.append({
                "paragraph": paragraphs[i],
                "paper_id": chunk_info["paper_id"],
                "title": chunk_info.get("title", "No title"),
                "similarity": score,
                "chunk_text": chunk_info["chunk_text"],
            })

    return alignments
