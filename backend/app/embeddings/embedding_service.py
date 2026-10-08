"""Lazy Sentence Transformer embedding service."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate normalized embeddings with a locally hosted model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is not installed. Install requirements.txt."
                ) from exc
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document chunks as normalized vectors."""
        if not texts:
            return []
        try:
            vectors = self._get_model().encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Document embedding generation failed")
            raise RuntimeError(f"Unable to generate document embeddings: {exc}") from exc
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode one query using the same model as document chunks."""
        try:
            vector = self._get_model().encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Query embedding generation failed")
            raise RuntimeError(f"Unable to generate query embedding: {exc}") from exc
        return vector.tolist()
