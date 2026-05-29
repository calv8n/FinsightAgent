"""
rag/reranker.py — Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Sits on top of RRF as a final scoring pass.
The cross-encoder sees (query, passage) concatenated — much more accurate
than cosine similarity alone because it compares them jointly.

Usage
-----
    from rag.reranker import Reranker

    reranker = Reranker()
    results  = reranker.rerank(query, chunks, top_k=5)

Architecture position:
    dense ANN (Qdrant) ─┐
                         ├─► RRF fusion (20 candidates) ─► cross-encoder ─► top 5
    BM25 sparse      ───┘
"""

from __future__ import annotations

from typing import Optional

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free, local, 22 MB


class Reranker:
    """
    Cross-encoder reranker.

    Lazy-loads the model on first call so startup time is unaffected.
    Scores are logits (higher = more relevant) — we sort descending.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-score `chunks` against `query` using the cross-encoder.
        Returns top_k chunks sorted by cross-encoder score descending,
        with a "ce_score" key added.

        Falls back to the original RRF order if model load fails.
        """
        if not chunks:
            return chunks

        model = self._load()
        if model is None:
            return chunks[:top_k]

        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["ce_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["ce_score"], reverse=True)
        return reranked[:top_k]

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            print(f"  [reranker] Loading {self.model_name} ...")
            self._model = CrossEncoder(self.model_name, max_length=512)
            print(f"  [reranker] Ready.")
        except Exception as exc:
            print(f"  [reranker] ⚠ Could not load cross-encoder: {exc}")
            print(f"  [reranker] Falling back to RRF order (no reranking).")
            self._model = None
        return self._model
