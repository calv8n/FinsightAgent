"""
rag/store.py — In-memory vector + BM25 store with RRF fusion retrieval.

Keeps everything in RAM for now (Week 2). Qdrant swap-in is a one-file
change in Week 3 — the retrieve() interface stays identical.

Usage
-----
    from rag.store import RAGStore
    from rag.embedder import Embedder

    store = RAGStore()
    store.add(chunks_with_embeddings)

    results = store.retrieve("Apple R&D spend 2023", top_k=5)
"""

from __future__ import annotations
import math
import numpy as np
from collections import defaultdict
from typing import Optional


class RAGStore:
    """
    Hybrid retrieval: dense cosine (sentence-transformers) +
    sparse BM25 → fused with Reciprocal Rank Fusion.
    """

    def __init__(self):
        self._chunks: list[dict] = []  # full chunk dicts
        self._vectors: Optional[np.ndarray] = None  # (N, 384) float32
        self._bm25_index: Optional[_BM25] = None

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add(self, chunks: list[dict]) -> None:
        """
        Add chunks (must already have an "embedding" key) to the store.
        Can be called multiple times; new chunks are appended.
        """
        if not chunks:
            return

        for c in chunks:
            if "embedding" not in c:
                raise ValueError("Chunk missing 'embedding' key — run Embedder first.")

        self._chunks.extend(chunks)

        vecs = np.array([c["embedding"] for c in self._chunks], dtype=np.float32)
        self._vectors = vecs

        # Rebuild BM25 index over all chunks
        corpus = [c["text"] for c in self._chunks]
        self._bm25_index = _BM25(corpus)

        print(
            f"  [store] {len(self._chunks)} chunks indexed "
            f"(dense dim={vecs.shape[1]}, sparse vocab={self._bm25_index.vocab_size})"
        )

    def retrieve(
        self,
        query: str,
        embedder,  # Embedder instance
        top_k: int = 5,
        dense_k: int = 20,  # candidates from each retriever before fusion
        sparse_k: int = 20,
        rrf_k: int = 60,  # RRF constant
    ) -> list[dict]:
        """
        Hybrid retrieve: dense + sparse → RRF fusion → top_k results.

        Returns list of chunk dicts with added "score" key (RRF score).
        """
        if not self._chunks:
            return []

        # Dense retrieval
        q_vec = embedder.embed_query(query)
        dense_hits = self._dense_search(q_vec, dense_k)

        # Sparse BM25 retrieval
        sparse_hits = self._sparse_search(query, sparse_k)

        # RRF fusion
        fused = _rrf(dense_hits, sparse_hits, k=rrf_k)

        # Return top_k with chunk data
        results = []
        for idx, score in fused[:top_k]:
            chunk = dict(self._chunks[idx])
            chunk["score"] = round(score, 6)
            results.append(chunk)

        return results

    def __len__(self) -> int:
        return len(self._chunks)

    # ── internals ─────────────────────────────────────────────────────────────

    def _dense_search(self, q_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (chunk_idx, cosine_score) sorted descending."""
        scores = self._vectors @ q_vec  # (N,) — dot product = cosine (unit norm)
        top_k = min(k, len(scores))
        idxs = np.argpartition(scores, -top_k)[-top_k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        return [(int(i), float(scores[i])) for i in idxs]

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (chunk_idx, bm25_score) sorted descending."""
        if self._bm25_index is None:
            return []
        scores = self._bm25_index.score_all(query)
        top_k = min(k, len(scores))
        idxs = np.argpartition(scores, -top_k)[-top_k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        return [(int(i), float(scores[i])) for i in idxs]


# ── RRF ──────────────────────────────────────────────────────────────────────


def _rrf(
    dense_hits: list[tuple[int, float]],
    sparse_hits: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    score(d) = Σ 1 / (k + rank(d))  across retrievers
    """
    scores: dict[int, float] = defaultdict(float)

    for rank, (idx, _) in enumerate(dense_hits, start=1):
        scores[idx] += 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        scores[idx] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


# ── BM25 (from scratch, no extra dependency) ─────────────────────────────────


class _BM25:
    """
    BM25 implementation — no rank_bm25 dependency needed.
    k1=1.5, b=0.75 (standard defaults).
    """

    k1 = 1.5
    b = 0.75

    def __init__(self, corpus: list[str]):
        self.doc_count = len(corpus)
        self.tokenised = [self._tokenise(d) for d in corpus]
        self.doc_lens = np.array([len(t) for t in self.tokenised], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.doc_count else 1.0

        # Build inverted index: term → {doc_idx: term_freq}
        self._index: dict[str, dict[int, int]] = defaultdict(dict)
        for i, tokens in enumerate(self.tokenised):
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, freq in tf.items():
                self._index[t][i] = freq

        self.vocab_size = len(self._index)

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float32)
        for term in self._tokenise(query):
            if term not in self._index:
                continue
            df = len(self._index[term])
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            for doc_idx, tf in self._index[term].items():
                dl = self.doc_lens[doc_idx]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * num / den
        return scores

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        import re

        return re.findall(r"[a-z0-9]+", text.lower())
