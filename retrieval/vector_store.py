#!/usr/bin/env python3
"""
FAISS Vector Store (IndexFlatIP + metadata map)
-----------------------------------------------
- Stores L2-normalized embeddings in a FAISS index (cosine via inner product).
- Persists the index and a row_id→metadata map (JSONL).
- Provides search() by vector and optional search_text() if you pass an embedder.

File layout (defaults):
  data/faiss/index_flatip.faiss
  data/faiss/ids.jsonl           # one JSON object per row_id, same order as index

This is intentionally small & local: no sharding, no deletions (rebuild to delete).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import json
import jsonlines
import numpy as np

EPS = 1e-12


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return (X / (np.linalg.norm(X) + EPS)).astype(np.float32, copy=False)
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)).astype(np.float32, copy=False)


@dataclass
class VectorStorePaths:
    index_path: Path
    ids_path: Path


class FaissVectorStore:
    """
    Minimal vector store wrapper around FAISS IndexFlatIP with row-aligned metadata.

    Usage:
        vs = FaissVectorStore.default_paths(project_root)
        vs.build_from_files(emb_path, meta_jsonl)   # once
        vs = FaissVectorStore.default_paths(project_root).open()  # later
        results = vs.search(query_vec, k=5)
    """

    def __init__(self, index_path: Path, ids_path: Path):
        self.paths = VectorStorePaths(index_path=index_path, ids_path=ids_path)
        self.index: Optional[faiss.Index] = None
        self._metas: Optional[List[Dict[str, Any]]] = None
        self._dim: Optional[int] = None

    # ---------- path helpers ----------
    @staticmethod
    def default_paths(project_root: Path) -> "FaissVectorStore":
        out_dir = project_root / "data" / "faiss"
        return FaissVectorStore(
            index_path=out_dir / "index_flatip.faiss",
            ids_path=out_dir / "ids.jsonl",
        )

    # ---------- persistence ----------
    def open(self) -> "FaissVectorStore":
        """Load index + metadata from disk."""
        if not self.paths.index_path.exists() or not self.paths.ids_path.exists():
            raise FileNotFoundError("Vector store not found; build it first.")
        self.index = faiss.read_index(str(self.paths.index_path))
        self._dim = self.index.d
        with jsonlines.open(self.paths.ids_path, "r") as r:
            self._metas = list(r)
        if len(self._metas) != self.index.ntotal:
            raise ValueError(
                f"ids.jsonl rows ({len(self._metas)}) != index.ntotal ({self.index.ntotal})"
            )
        return self

    def save(self) -> None:
        """Persist current index and metadata."""
        assert self.index is not None and self._metas is not None
        self.paths.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.paths.index_path))
        with open(self.paths.ids_path, "w", encoding="utf-8") as f:
            for i, m in enumerate(self._metas):
                # include row_id for convenience on reload
                f.write(json.dumps({"row_id": i, **m}, ensure_ascii=False) + "\n")

    # ---------- building ----------
    def build_from_arrays(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> "FaissVectorStore":
        """
        Create a new IndexFlatIP and add all vectors. Overwrites existing files.
        """
        X = _l2_normalize(np.asarray(embs, dtype=np.float32))
        if len(metas) != X.shape[0]:
            raise ValueError("metas length must match number of embeddings")
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)

        self.index = index
        self._metas = list(metas)
        self._dim = dim
        self.save()
        return self

    def build_from_files(self, emb_path: Path, meta_jsonl: Path) -> "FaissVectorStore":
        """
        Read embeddings from .npy and metadata from .jsonl, then build & persist the index.
        """
        if not emb_path.exists() or not meta_jsonl.exists():
            raise FileNotFoundError("Missing embeddings or metadata file.")
        X = np.load(emb_path).astype(np.float32)
        with jsonlines.open(meta_jsonl, "r") as r:
            metas = list(r)
        return self.build_from_arrays(X, metas)

    # ---------- mutation ----------
    def add(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> List[int]:
        """
        Append new vectors + metadata. Returns assigned row_ids.
        Note: IndexFlatIP does NOT support deletion; to delete, rebuild.
        """
        assert self.index is not None and self._metas is not None and self._dim is not None
        X = _l2_normalize(np.asarray(embs, dtype=np.float32))
        if X.shape[1] != self._dim:
            raise ValueError(f"dim mismatch: index d={self._dim}, got {X.shape[1]}")
        if len(metas) != X.shape[0]:
            raise ValueError("metas length must match number of embeddings")

        start = self.index.ntotal
        self.index.add(X)
        self._metas.extend(metas)
        row_ids = list(range(start, start + X.shape[0]))
        # persist incremental updates
        with open(self.paths.ids_path, "a", encoding="utf-8") as f:
            for i, m in zip(row_ids, metas):
                f.write(json.dumps({"row_id": i, **m}, ensure_ascii=False) + "\n")
        faiss.write_index(self.index, str(self.paths.index_path))
        return row_ids

    # ---------- search ----------
    def search(self, q: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search by query vector. Returns a list of dicts:
        [{row_id, score, metadata}, ...]
        'score' is cosine (inner product on normalized vectors).
        """
        assert self.index is not None and self._metas is not None
        q = _l2_normalize(np.asarray(q, dtype=np.float32).reshape(1, -1))
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self._metas[idx]
            out.append({"row_id": int(idx), "score": float(score), "metadata": m})
        return out

    # optional, if you want a text-to-vec convenience:
    def search_text(
        self,
        text: str,
        embedder,            # any object with .encode([text]) -> np.ndarray
        instruction: str = "",
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        vec = embedder.encode([text], instruction=instruction)
        return self.search(np.asarray(vec, dtype=np.float32)[0], k=k)

    # ---------- accessors ----------
    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Vector store not opened/built yet.")
        return int(self._dim)

    def get_metadata(self, row_id: int) -> Dict[str, Any]:
        assert self._metas is not None
        return self._metas[row_id]
