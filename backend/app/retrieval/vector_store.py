"""Persistent ChromaDB operations."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin ChromaDB adapter; retrieval can be extended with other backends."""

    collection_name = "research_paper_chunks"

    def __init__(self, persist_directory: Path) -> None:
        self.persist_directory = persist_directory
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB is not installed.") from exc
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            try:
                client = chromadb.PersistentClient(path=str(self.persist_directory))
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.exception("Unable to initialize ChromaDB")
                raise RuntimeError(f"Unable to initialize vector database: {exc}") from exc
        return self._collection

    def document_names(self) -> set[str]:
        """Return names represented in the current collection."""
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            logger.exception("Unable to list ChromaDB documents")
            raise RuntimeError(f"Unable to read vector database: {exc}") from exc
        return {
            str(metadata["filename"])
            for metadata in result.get("metadatas", [])
            if metadata and "filename" in metadata
        }

    def add_chunks(
        self,
        chunks: list[dict[str, int | str]],
        embeddings: list[list[float]],
    ) -> None:
        """Persist chunk text, metadata, and precomputed vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        try:
            self.collection.add(
                ids=[str(chunk["id"]) for chunk in chunks],
                documents=[str(chunk["text"]) for chunk in chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "filename": str(chunk["filename"]),
                        "document": str(chunk["document"]),
                        "page": int(chunk["page"]),
                        "chunk_index": int(chunk["chunk_index"]),
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            logger.exception("Unable to write chunks to ChromaDB")
            raise RuntimeError(f"Unable to write to vector database: {exc}") from exc
        logger.info("Persisted %d chunks to ChromaDB", len(chunks))

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return cosine-ranked chunks, including similarity and source metadata."""
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("ChromaDB retrieval failed")
            raise RuntimeError(f"Unable to search vector database: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        matches = []
        for content, metadata, distance, chunk_id in zip(
            documents, metadatas, distances, ids
        ):
            if content is None or metadata is None or distance is None:
                continue
            matches.append(
                {
                    "document": str(metadata.get("filename", "unknown")),
                    "page": int(metadata.get("page", 0)),
                    "content": str(content),
                    "score": round(
                        max(-1.0, min(1.0, 1.0 - float(distance))),
                        6,
                    ),
                    "chunk_id": str(chunk_id),
                }
            )
        logger.info("Retrieved %d chunks", len(matches))
        return matches

    def list_documents(self) -> list[dict[str, int | str]]:
        """Aggregate processed document metadata from stored chunks."""
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            logger.exception("Unable to list ChromaDB documents")
            raise RuntimeError(f"Unable to read vector database: {exc}") from exc
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"pages": set(), "chunks": 0}
        )
        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            name = str(metadata.get("filename", "unknown"))
            stats[name]["pages"].add(int(metadata.get("page", 0)))
            stats[name]["chunks"] += 1
        return [
            {
                "filename": name,
                "pages": len(values["pages"]),
                "chunks": values["chunks"],
            }
            for name, values in sorted(stats.items())
        ]

    def clear(self) -> None:
        """Delete and recreate the persistent application collection."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_directory))
            try:
                client.delete_collection(self.collection_name)
            except Exception as exc:
                if "does not exist" not in str(exc).lower():
                    raise
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Unable to clear ChromaDB")
            raise RuntimeError(f"Unable to clear vector database: {exc}") from exc
        logger.info("Cleared ChromaDB collection")


def create_vector_store(settings: Settings) -> VectorStore:
    """Build the vector store from application settings."""
    return VectorStore(settings.chroma_persist_directory)
