"""
rag/__init__.py — RAG pipeline with Qdrant + cross-encoder reranker.
"""

from __future__ import annotations
from typing import Optional

from rag.ingest import fetch_10k
from rag.embedder import Embedder
from rag.store import RAGStore
from rag.reranker import Reranker


class RAGPipeline:
    def __init__(self, data_dir: Optional[str] = None, rerank: bool = True):
        self.embedder = Embedder()
        self.store = RAGStore()
        self.reranker = Reranker() if rerank else None
        self.data_dir = data_dir

    def ingest(self, ticker: str, years: list[int]) -> int:
        ticker = ticker.upper()
        to_fetch = [y for y in years if not self.store.has_filing(ticker, y)]
        if not to_fetch:
            print(f"[RAG] {ticker} {years} already in Qdrant — skipping.")
            return 0

        print(f"\n[RAG] Ingesting {ticker} {to_fetch} ...")
        chunks = fetch_10k(ticker, years=to_fetch, data_dir=self.data_dir)
        if not chunks:
            print(f"[RAG] No chunks for {ticker}.")
            return 0

        chunks = self.embedder.embed_chunks(chunks)
        self.store.add(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Retrieve 2× candidates, rerank down to top_k
        candidates = self.store.retrieve(
            query,
            self.embedder,
            top_k=top_k * 2 if self.reranker else top_k,
        )
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates

    def __len__(self) -> int:
        return len(self.store)
