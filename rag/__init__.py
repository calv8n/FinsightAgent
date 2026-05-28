"""
rag/__init__.py — RAG pipeline entry point.

Quick start
-----------
    from rag import RAGPipeline

    rag = RAGPipeline()
    rag.ingest("AAPL", years=[2022, 2023, 2024])
    rag.ingest("MSFT", years=[2022, 2023, 2024])

    results = rag.retrieve("Apple R&D spend as % of revenue")
    for r in results:
        print(r["ticker"], r["year"], r["section"], r["score"])
        print(r["text"][:300])
"""

from __future__ import annotations
from typing import Optional

from rag.ingest import fetch_10k
from rag.embedder import Embedder
from rag.store import RAGStore


class RAGPipeline:
    """
    Thin orchestrator that wires ingest → embed → store → retrieve.
    One instance per process; call ingest() for each ticker you need.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.embedder = Embedder()
        self.store = RAGStore()
        self.data_dir = data_dir

    def ingest(self, ticker: str, years: list[int]) -> int:
        """
        Download, chunk, embed, and index 10-K filings.
        Returns number of chunks added.
        """
        print(f"\n[RAG] Ingesting {ticker} {years} ...")
        chunks = fetch_10k(ticker, years=years, data_dir=self.data_dir)
        if not chunks:
            print(f"[RAG] No chunks produced for {ticker}.")
            return 0

        chunks = self.embedder.embed_chunks(chunks)
        self.store.add(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid dense + BM25 retrieval with RRF fusion.
        Returns top_k chunk dicts with a "score" key added.
        """
        return self.store.retrieve(query, self.embedder, top_k=top_k)

    def __len__(self) -> int:
        return len(self.store)
