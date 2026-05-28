"""
rag/embedder.py — Generate and cache embeddings using all-MiniLM-L6-v2.

Usage
-----
    from rag.embedder import Embedder

    emb = Embedder()
    chunks_with_vectors = emb.embed_chunks(chunks)   # adds "embedding" key
    query_vec = emb.embed_query("R&D spend Apple 2023")
"""

from __future__ import annotations
import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional

MODEL_NAME = "all-MiniLM-L6-v2"  # 22M params, 384-dim, fast on CPU
CACHE_DIR = Path(__file__).parent.parent / "data" / "embed_cache"
BATCH_SIZE = 64


class Embedder:
    """
    Wraps sentence-transformers with a file-level cache so re-runs
    don't re-embed chunks that haven't changed.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache
        self._model = None  # lazy-loaded

    # ── public ───────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Add an "embedding" key (list[float], 384-dim) to each chunk dict.
        Chunks that are already cached are not re-embedded.

        Returns the same list with embeddings added in-place.
        """
        model = self._load_model()
        os.makedirs(self.cache_dir, exist_ok=True)

        texts_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        # First pass: load from cache where possible
        for i, chunk in enumerate(chunks):
            key = _cache_key(chunk["text"])
            cached = self._load_cache(key)
            if cached is not None:
                chunk["embedding"] = cached
            else:
                texts_to_embed.append(chunk["text"])
                indices_to_embed.append(i)

        # Second pass: batch-embed cache misses
        if texts_to_embed:
            print(
                f"  [embedder] Embedding {len(texts_to_embed)} chunks "
                f"(model: {self.model_name})..."
            )
            vectors = self._batch_embed(model, texts_to_embed)

            for vec, idx in zip(vectors, indices_to_embed):
                vec_list = vec.tolist()
                chunks[idx]["embedding"] = vec_list
                if self.use_cache:
                    key = _cache_key(chunks[idx]["text"])
                    self._save_cache(key, vec_list)

            print(f"  [embedder] ✓ Done. Dim={vectors.shape[1]}")
        else:
            print(f"  [embedder] All {len(chunks)} chunks loaded from cache.")

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """Return a 384-dim numpy vector for a query string."""
        model = self._load_model()
        return model.encode(query, normalize_embeddings=True)

    def similarity(self, query_vec: np.ndarray, chunk_vec: list[float]) -> float:
        """Cosine similarity between query vector and a stored chunk vector."""
        v = np.array(chunk_vec, dtype=np.float32)
        return float(np.dot(query_vec, v))

    # ── internals ────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            print(f"  [embedder] Loading {self.model_name} ...")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            print(f"  [embedder] Model ready.")
        return self._model

    def _batch_embed(self, model, texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # unit-norm → dot product == cosine sim
            show_progress_bar=len(texts) > BATCH_SIZE,
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[list[float]]:
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save_cache(self, key: str, vec: list[float]) -> None:
        self._cache_path(key).write_text(json.dumps(vec))


# ── helpers ──────────────────────────────────────────────────────────────────


def _cache_key(text: str) -> str:
    """SHA-256 of text → 16-char hex used as filename."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
