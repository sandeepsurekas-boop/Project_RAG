"""Semantic retriever with threshold filtering."""

import logging
from typing import Any

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Retrieve nearest chunks and discard low-similarity matches."""

    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(
        self, question: str, top_k: int, similarity_threshold: float
    ) -> list[dict[str, Any]]:
        """Search semantically and retain only matches above the threshold."""
        if not self.vector_store.document_names():
            logger.info("No processed documents are available for retrieval")
            return []
        query_embedding = self.embeddings.embed_query(question)
        matches = self.vector_store.search(query_embedding, top_k)
        relevant = [
            match for match in matches if match["score"] >= similarity_threshold
        ]
        logger.info(
            "Similarity filter retained %d of %d chunks (threshold %.2f)",
            len(relevant),
            len(matches),
            similarity_threshold,
        )
        return relevant
