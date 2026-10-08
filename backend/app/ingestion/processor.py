"""Orchestrate extraction, chunking, embedding, and persistence."""

import logging
from pathlib import Path

from backend.app.config import Settings
from backend.app.embeddings.embedding_service import EmbeddingService
from backend.app.ingestion.chunker import split_pages
from backend.app.ingestion.pdf_loader import load_pdf
from backend.app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process pending PDFs and store their page-aware chunks."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store

    def process_pending(self) -> dict[str, int | list[str]]:
        """Process all unindexed PDFs, embedding before one bulk DB write."""
        self.settings.papers_directory.mkdir(parents=True, exist_ok=True)
        indexed = self.vector_store.document_names()
        pending = [
            path
            for path in sorted(self.settings.papers_directory.glob("*.pdf"))
            if path.name not in indexed
        ]
        if not pending:
            return {"documents": [], "pages": 0, "chunks": 0}

        all_chunks: list[dict[str, int | str]] = []
        total_pages = 0
        for path in pending:
            pages = load_pdf(path)
            chunks = split_pages(
                pages,
                path.name,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            if not chunks:
                raise ValueError(f"No text chunks could be created from {path.name}")
            all_chunks.extend(chunks)
            total_pages += len(pages)

        logger.info(
            "Generating embeddings for %d chunks across %d documents",
            len(all_chunks),
            len(pending),
        )
        vectors = self.embeddings.embed_documents(
            [str(chunk["text"]) for chunk in all_chunks]
        )
        self.vector_store.add_chunks(all_chunks, vectors)
        names = [path.name for path in pending]
        logger.info("Processed %d documents and %d chunks", len(names), len(all_chunks))
        return {"documents": names, "pages": total_pages, "chunks": len(all_chunks)}
