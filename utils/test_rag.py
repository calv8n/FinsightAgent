"""
tests/test_rag.py — RAG pipeline tests (no network, no SEC calls).

Tests chunker and store with synthetic text.
Embedder test is skipped if sentence-transformers isn't installed.

Run: python tests/test_rag.py
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chunker import chunk_filing
from rag.store import RAGStore, _BM25, _rrf

# ── synthetic filing ──────────────────────────────────────────────────────────

FAKE_FILING = """
ITEM 1. Business

Apple Inc. designs, manufactures and markets smartphones, personal computers,
tablets, wearables and accessories, and sells a variety of related services.

ITEM 1A. Risk Factors

The Company faces intense competition from well-resourced rivals including
Samsung, Google, and Microsoft. Macroeconomic conditions and foreign exchange
fluctuations may adversely affect revenues. Supply chain disruptions could
limit the availability of components. Failure to develop compelling new products
could cause customers to shift to competing platforms.

ITEM 7. Management Discussion and Analysis

Net sales for fiscal 2023 were $383.3 billion, a decrease of 2.8 percent
compared to the prior year. Products net sales decreased 6.6 percent.
Services net sales increased 9.1 percent to $85.2 billion.

Research and development expense was $29.9 billion, an increase of 14 percent.
R&D as a percentage of net sales was 7.8 percent, compared to 6.7 percent in 2022.

ITEM 8. Financial Statements

(Figures in millions)
Total net sales:  383,285
Cost of sales:    214,137
Gross margin:     169,148
R&D expense:       29,915
Operating income:  114,301
Net income:         96,995
"""


# ── helpers ───────────────────────────────────────────────────────────────────


def _pass(label):
    print(f"  ✅  {label}")


def _fail(label, reason):
    print(f"  ❌  {label}")
    print(f"       {reason}")


def run(label, fn):
    try:
        fn()
        _pass(label)
        return True
    except AssertionError as e:
        _fail(label, e)
        return False
    except Exception as e:
        _fail(label, f"exception: {e}")
        return False


# ── chunker tests ─────────────────────────────────────────────────────────────


def test_sections_detected():
    chunks = chunk_filing(FAKE_FILING, "AAPL", 2023)
    sections = {c["section"] for c in chunks}
    assert "Item 1" in sections, f"Item 1 missing, got {sections}"
    assert "Item 1A" in sections, f"Item 1A missing"
    assert "Item 7" in sections, f"Item 7 missing"


def test_metadata_present():
    chunks = chunk_filing(FAKE_FILING, "AAPL", 2023)
    for c in chunks:
        assert c["ticker"] == "AAPL"
        assert c["year"] == 2023
        assert c["form"] == "10-K"
        assert "text" in c
        assert "char_len" in c
        assert c["char_len"] == len(c["text"])


def test_min_length_filter():
    chunks = chunk_filing(FAKE_FILING, "AAPL", 2023)
    for c in chunks:
        assert c["char_len"] >= 120, f"chunk too short: {c['char_len']}"


def test_titles_populated():
    chunks = chunk_filing(FAKE_FILING, "AAPL", 2023)
    by_section = {c["section"]: c["title"] for c in chunks}
    assert by_section.get("Item 1A") == "Risk Factors"
    assert by_section.get("Item 7") == "Management Discussion and Analysis"


# ── BM25 tests ────────────────────────────────────────────────────────────────


def test_bm25_scores_nonzero():
    corpus = [
        "apple revenue growth",
        "microsoft cloud services",
        "research and development",
    ]
    bm25 = _BM25(corpus)
    scores = bm25.score_all("apple revenue")
    assert scores[0] > 0, "top doc should score > 0"
    assert scores[0] > scores[1], "apple doc should outscore microsoft doc"


def test_bm25_unknown_term():
    bm25 = _BM25(["hello world"])
    scores = bm25.score_all("zzzzunknown")
    assert scores[0] == 0.0


def test_rrf_merges_results():
    dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
    sparse = [(1, 12.0), (0, 10.0), (3, 8.0)]
    fused = _rrf(dense, sparse)
    idxs = [idx for idx, _ in fused]
    # doc 0 and 1 both appear in both lists → should rank highest
    assert idxs[0] in (0, 1)
    assert idxs[1] in (0, 1)


# ── store tests ───────────────────────────────────────────────────────────────


def _fake_chunks():
    texts = [
        "Apple research and development expense was 29.9 billion.",
        "Microsoft cloud revenue grew 22 percent year over year.",
        "Risk factors include supply chain disruptions and FX headwinds.",
        "Net income for fiscal 2023 was 96.9 billion dollars.",
    ]
    # Use tiny fake embeddings (4-dim) just to test store mechanics
    import numpy as np

    vecs = np.random.default_rng(42).standard_normal((len(texts), 4)).astype("float32")
    # Normalise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return [
        {
            "ticker": "AAPL",
            "year": 2023,
            "section": "Item 7",
            "title": "MD&A",
            "text": t,
            "char_len": len(t),
            "embedding": vecs[i].tolist(),
        }
        for i, t in enumerate(texts)
    ]


def test_store_add_and_len():
    store = RAGStore()
    store.add(_fake_chunks())
    assert len(store) == 4


def test_store_dense_search():
    import numpy as np

    store = RAGStore()
    chunks = _fake_chunks()
    store.add(chunks)

    # Query vector identical to first chunk embedding → should be top hit
    q_vec = np.array(chunks[0]["embedding"], dtype=np.float32)
    hits = store._dense_search(q_vec, k=4)
    assert hits[0][0] == 0, f"expected chunk 0 as top hit, got {hits[0][0]}"


def test_store_sparse_search():
    store = RAGStore()
    store.add(_fake_chunks())
    hits = store._sparse_search("research development Apple", k=4)
    # chunk 0 mentions research/development/Apple — should be top
    assert hits[0][0] == 0, f"expected chunk 0, got {hits[0][0]}"


# ── embedder smoke test (skipped if torch not installed) ─────────────────────


def test_embedder_smoke():
    try:
        import sentence_transformers  # noqa: F401
        from rag.embedder import Embedder
    except ImportError:
        print("  ⏭   Embedder skipped (sentence-transformers not installed)")
        return

    emb = Embedder(use_cache=False)
    vec = emb.embed_query("Apple R&D expense 2023")
    import numpy as np

    assert vec.shape == (384,), f"expected (384,), got {vec.shape}"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4, "vector should be unit-norm"


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "═" * 60)
    print("  RAG Pipeline Tests")
    print("═" * 60)

    tests = [
        ("Sections detected", test_sections_detected),
        ("Metadata present", test_metadata_present),
        ("Min-length filter", test_min_length_filter),
        ("Titles populated", test_titles_populated),
        ("BM25 scores non-zero", test_bm25_scores_nonzero),
        ("BM25 unknown term → 0", test_bm25_unknown_term),
        ("RRF merges results", test_rrf_merges_results),
        ("Store add + len", test_store_add_and_len),
        ("Store dense search", test_store_dense_search),
        ("Store sparse search", test_store_sparse_search),
        ("Embedder smoke test", test_embedder_smoke),
    ]

    passed = sum(run(label, fn) for label, fn in tests)
    total = len(tests)

    print(f"\n{'═'*60}")
    print(f"  {passed}/{total} passed" + (" 🎉" if passed == total else ""))
    print("═" * 60 + "\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
