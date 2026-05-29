"""
rag/store.py — Qdrant-backed hybrid store (dense + BM25 + RRF).

Drop-in replacement for the in-memory store.
The retrieve() signature is identical — RAGPipeline and app.py need no changes.

Setup
-----
    # Start Qdrant (Docker):
    docker run -d -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant

    pip install qdrant-client
"""

from __future__ import annotations

import math
import os
import re
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "finsight_chunks"
VECTOR_DIM = 384


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def _m():
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )

    return Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class RAGStore:
    """Qdrant dense ANN + in-memory BM25 + RRF fusion."""

    def __init__(self):
        self._qc = _client()
        self._col = COLLECTION_NAME
        self._cache: list[dict] = []  # payload mirror for BM25 + materialise
        self._bm25: Optional[_BM25] = None
        self._ensure_collection()
        self._rebuild_bm25()

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        _, _, PointStruct, *_ = _m()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=c["embedding"],
                payload={k: v for k, v in c.items() if k != "embedding"},
            )
            for c in chunks
        ]
        self._qc.upsert(collection_name=self._col, points=points)
        self._rebuild_bm25()
        total = self._qc.count(self._col).count
        print(f"  [qdrant] +{len(chunks)} chunks → {total} total")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 60,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[dict]:
        if self._count() == 0:
            return []
        q_vec = embedder.embed_query(query)
        qfilter = self._filter(ticker, year)
        dense_hits = self._dense(q_vec, dense_k, qfilter)
        sparse_hits = self._sparse(query, sparse_k, ticker, year)
        fused = _rrf(dense_hits, sparse_hits, k=rrf_k)
        results = []
        for doc_id, score in fused[:top_k]:
            c = self._by_id(doc_id)
            if c:
                c["score"] = round(score, 6)
                results.append(c)
        return results

    def has_filing(self, ticker: str, year: int) -> bool:
        _, _, _, Filter, FieldCondition, MatchValue = _m()
        f = Filter(
            must=[
                FieldCondition(key="ticker", match=MatchValue(value=ticker)),
                FieldCondition(key="year", match=MatchValue(value=year)),
            ]
        )
        return self._qc.count(self._col, count_filter=f).count > 0

    def __len__(self) -> int:
        return self._count()

    # ── internals ─────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        Distance, VectorParams, *_ = _m()
        names = [c.name for c in self._qc.get_collections().collections]
        if self._col not in names:
            self._qc.create_collection(
                collection_name=self._col,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def _rebuild_bm25(self):
        total = self._count()
        if total == 0:
            self._cache, self._bm25 = [], None
            return
        chunks, offset = [], None
        while True:
            result, next_off = self._qc.scroll(
                collection_name=self._col,
                with_payload=True,
                with_vectors=False,
                limit=500,
                offset=offset,
            )
            for p in result:
                c = dict(p.payload)
                c["_id"] = str(p.id)
                chunks.append(c)
            if next_off is None:
                break
            offset = next_off
        self._cache = chunks
        self._bm25 = _BM25([c.get("text", "") for c in chunks])

    def _dense(self, q_vec, k, f) -> list[tuple[str, float]]:
        hits = self._qc.search(
            collection_name=self._col,
            query_vector=q_vec.tolist(),
            limit=k,
            query_filter=f,
            with_payload=False,
        )
        return [(str(h.id), h.score) for h in hits]

    def _sparse(self, query, k, ticker, year) -> list[tuple[str, float]]:
        if not self._bm25 or not self._cache:
            return []
        scores = self._bm25.score_all(query)
        filtered = [
            (i, float(s))
            for i, (c, s) in enumerate(zip(self._cache, scores))
            if (not ticker or c.get("ticker") == ticker)
            and (not year or c.get("year") == year)
        ]
        filtered.sort(key=lambda t: t[1], reverse=True)
        return [(self._cache[i]["_id"], s) for i, s in filtered[:k]]

    def _by_id(self, doc_id: str) -> Optional[dict]:
        for c in self._cache:
            if c.get("_id") == doc_id:
                return {k: v for k, v in c.items() if k != "_id"}
        return None

    def _filter(self, ticker, year):
        _, _, _, Filter, FieldCondition, MatchValue = _m()
        conds = []
        if ticker:
            conds.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
        if year:
            conds.append(FieldCondition(key="year", match=MatchValue(value=year)))
        return Filter(must=conds) if conds else None

    def _count(self) -> int:
        try:
            return self._qc.count(self._col).count
        except:
            return 0


def _rrf(dense, sparse, k=60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rank, (did, _) in enumerate(dense, 1):
        scores[did] += 1 / (k + rank)
    for rank, (did, _) in enumerate(sparse, 1):
        scores[did] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


class _BM25:
    k1 = 1.5
    b = 0.75

    def __init__(self, corpus):
        self.n = len(corpus)
        self.tok = [self._t(d) for d in corpus]
        self.dl = np.array([len(t) for t in self.tok], dtype=np.float32)
        self.avdl = float(self.dl.mean()) if self.n else 1.0
        self.idx: dict[str, dict[int, int]] = defaultdict(dict)
        for i, tokens in enumerate(self.tok):
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, f in tf.items():
                self.idx[t][i] = f
        self.vocab_size = len(self.idx)

    def score_all(self, query):
        s = np.zeros(self.n, dtype=np.float32)
        for term in self._t(query):
            if term not in self.idx:
                continue
            df = len(self.idx[term])
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for i, tf in self.idx[term].items():
                dl = self.dl[i]
                s[i] += (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self.avdl))
                )
        return s

    @staticmethod
    def _t(text):
        return re.findall(r"[a-z0-9]+", text.lower())
