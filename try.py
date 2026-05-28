from rag.store import RAGStore
from rag.embedder import Embedder

from rag.ingest import fetch_10k
from rag.embedder import Embedder

emb = Embedder()
chunks = fetch_10k("AAPL", years=[2022, 2023, 2024])
chunks_with_vectors = emb.embed_chunks(chunks)  # adds "embedding" key
query_vec = emb.embed_query("R&D spend Apple 2023")


store = RAGStore()
store.add(chunks_with_vectors)

results = store.retrieve("Apple R&D spend 2023", top_k=5, embedder=emb)
print(results[0]["text"])
