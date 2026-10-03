"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.generation.rag_chain import RagAnswerGenerator
from app.ingestion.processor import DocumentProcessor
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.vector_store import VectorStore
from app.utils.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize lightweight services; external models load on demand."""
    settings = get_settings()
    settings.papers_directory.mkdir(parents=True, exist_ok=True)
    embeddings = EmbeddingService(settings.embedding_model)
    vector_store = VectorStore(settings.chroma_persist_directory)
    application.state.settings = settings
    application.state.vector_store = vector_store
    application.state.processor = DocumentProcessor(settings, embeddings, vector_store)
    application.state.retriever = SemanticRetriever(embeddings, vector_store)
    application.state.answer_generator = RagAnswerGenerator(settings)
    yield


app = FastAPI(
    title="Research Paper RAG API",
    description="Question answering over uploaded research papers with page citations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router)
