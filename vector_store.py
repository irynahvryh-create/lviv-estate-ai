import pathlib
from typing import Optional
import chromadb
from embedding_client import EmbeddingClient

class VectorStore:
    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        name: str = "real_estate_corpus",
        *,
        path: Optional[str] = None,
        reset: bool = False,
    ):
        self.embedder = embedder or EmbeddingClient()
        if path is None:
            path = str(pathlib.Path(__file__).resolve().parent / "chroma")
        self.path = path
        self._client = chromadb.PersistentClient(path=path)
        if reset:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._collection = self._client.get_or_create_collection(
            name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[str], metadatas: Optional[list[dict]] = None) -> None:
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(metadatas)} metadatas")
        vectors = self.embedder.embed_batch(chunks, task_type="RETRIEVAL_DOCUMENT")
        start = self._collection.count()
        ids = [str(start + i) for i in range(len(chunks))]
        self._collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)

    def search(self, query: str, *, n_results: int = 3) -> list[dict]:
        query_vector = self.embedder.embed(query, task_type="RETRIEVAL_QUERY")
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        metas = result.get("metadatas") or [[None] * len(result["documents"][0])]
        return [
            {"chunk": doc, "distance": dist, "metadata": meta or {}, "id": i}
            for doc, dist, meta, i in zip(
                result["documents"][0], result["distances"][0], metas[0], result["ids"][0]
            )
        ]