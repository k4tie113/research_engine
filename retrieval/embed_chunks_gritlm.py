#!/usr/bin/env python3
import os, json, jsonlines, numpy as np
from pathlib import Path
from tqdm import tqdm
from gritlm import GritLM
import torch

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("GRITLM_MODEL", "GritLM/GritLM-7B")
# Tuning knobs (via env): GRITLM_DEVICE=cuda|cpu, GRITLM_DTYPE=auto|bfloat16|float16|float32
GRITLM_DEVICE = os.environ.get("GRITLM_DEVICE")
GRITLM_DTYPE = os.environ.get("GRITLM_DTYPE", "auto")
# Reduce deadlock risk and noisy warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

def gritlm_instruction(instr: str) -> str:
    return "<|user|>\n" + instr + "\n<|embed|>\n" if instr else "<|embed|>\n"

def main(batch_size: int = 16, max_chunks: int | None = None):
    assert CHUNKS.exists(), f"Missing {CHUNKS}; run the PDF pipeline first."
    # Decide device and dtype
    cuda_ok = torch.cuda.is_available()
    device = (GRITLM_DEVICE or ("cuda" if cuda_ok else "cpu")).lower()
    if device not in ("cuda", "cpu"): device = "cpu"
    # Map dtype string
    dtype_map = {
        "auto": "auto",
        "bfloat16": "bfloat16",
        "float16": "float16",
        "float32": "float32",
    }
    dtype_arg = dtype_map.get(GRITLM_DTYPE.lower(), "auto")
    print(f"Loading model {MODEL_NAME} (mode=embedding, device={device}, dtype={dtype_arg})…")
    model = GritLM(MODEL_NAME, dtype=dtype_arg, mode="embedding")
    # Work around cache API mismatches by disabling KV caching if available
    try:
        if hasattr(model, "model") and getattr(model.model, "config", None) is not None:
            model.model.config.use_cache = False
            # Attempt to enable FlashAttention-2 if requested and available
            wanted_attn = os.environ.get("GRITLM_ATTN", "").lower()  # e.g. "flash_attention_2"
            if wanted_attn and hasattr(model.model, "config"):
                try:
                    model.model.config.attn_implementation = wanted_attn
                    print(f"Set attn_implementation={wanted_attn}")
                except Exception:
                    print("Could not set attn_implementation; proceeding without it")
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        # Move to requested device if supported
        if hasattr(model, "to"):
            model.to(device)
        print(f"Model device: {device}")
    except Exception:
        pass

    # Warmup small encode to ensure pipeline is functional
    try:
        print("Warming up encoder on 2 texts…")
        w = model.encode(["hello", "world"], instruction=gritlm_instruction(""))
        w = np.asarray(w, dtype=np.float32)
        print(f"Warmup OK: {w.shape}")
    except Exception as e:
        print(f"Warmup failed: {e}")
        raise

    texts, metas = [], []
    with jsonlines.open(CHUNKS, "r") as reader:
        for i, rec in enumerate(reader):
            if max_chunks and i >= max_chunks: break
            texts.append(rec["chunk_text"])
            metas.append({
                "paper_id": rec["paper_id"],
                "chunk_index": rec["chunk_index"],
                "title": rec.get("title", ""),
                "authors": rec.get("authors", ""),
                "token_count": rec.get("token_count", 0),
            })

    print(f"Encoding {len(texts)} chunks…")
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        print(f"  → batch {i}-{min(i+batch_size, len(texts))} (size={len(batch)})")
        # Use autocast on CUDA for half/bfloat16, else plain inference
        if device == "cuda":
            amp_dtype = torch.bfloat16 if dtype_arg == "bfloat16" else (torch.float16 if dtype_arg == "float16" else None)
            if amp_dtype is not None:
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=amp_dtype):
                    vecs = model.encode(batch, instruction=gritlm_instruction(""))
            else:
                with torch.inference_mode():
                    vecs = model.encode(batch, instruction=gritlm_instruction(""))
        else:
            with torch.inference_mode():
                vecs = model.encode(batch, instruction=gritlm_instruction(""))
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        embs.append(vecs)
    embs = np.vstack(embs)

    np.save(EMB_DIR / "embeddings.npy", embs)
    with jsonlines.open(EMB_DIR / "metadata.jsonl", "w") as w:
        for m in metas: w.write(m)

    print(f"Wrote {embs.shape} → {EMB_DIR/'embeddings.npy'}")
    print(f"Metadata → {EMB_DIR/'metadata.jsonl'}")

if __name__ == "__main__":
    main()

