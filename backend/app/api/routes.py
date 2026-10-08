"""REST endpoints for document lifecycle and question answering."""

import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.app.api.schemas import (
    DocumentsResponse,
    HealthResponse,
    MessageResponse,
    ProcessResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from backend.app.config import Settings
from backend.app.generation.rag_chain import generate_answer
from backend.app.ingestion.chunker import safe_pdf_filename
from backend.app.ingestion.processor import DocumentProcessor
from backend.app.retrieval.retriever import SemanticRetriever
from backend.app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_DOCUMENTS = 4


def _services(request: Request) -> tuple[Settings, VectorStore]:
    return request.app.state.settings, request.app.state.vector_store


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process health without loading external models."""
    return HealthResponse(status="ok")


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_papers(
    request: Request,
    files: Annotated[list[UploadFile], File(description="PDF research papers")],
) -> UploadResponse:
    """Validate and save one or more PDFs for later processing."""
    settings, vector_store = _services(request)
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")
    settings.papers_directory.mkdir(parents=True, exist_ok=True)
    try:
        existing_files = {
            path.name for path in settings.papers_directory.glob("*.pdf")
        } | vector_store.document_names()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if len(existing_files) + len(files) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_DOCUMENTS} documents is supported.",
        )
    accepted: list[tuple[str, bytes]] = []
    accepted_names: set[str] = set()
    content_hashes: set[str] = set()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for upload in files:
        try:
            filename = safe_pdf_filename(upload.filename or "")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if filename in existing_files or filename in accepted_names:
            raise HTTPException(
                status_code=409, detail=f"Document already exists: {filename}"
            )
        content = await upload.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {settings.max_upload_mb} MB upload limit.",
            )
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400, detail=f"Uploaded file is not a valid PDF: {filename}"
            )
        digest = hashlib.sha256(content).hexdigest()
        if digest in content_hashes:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate PDF content detected in this upload: {filename}.",
            )
        accepted.append((filename, content))
        accepted_names.add(filename)
        content_hashes.add(digest)

    for filename, content in accepted:
        digest = hashlib.sha256(content).hexdigest()
        for existing_path in settings.papers_directory.glob("*.pdf"):
            try:
                if hashlib.sha256(existing_path.read_bytes()).hexdigest() == digest:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Duplicate PDF content already uploaded as {existing_path.name}.",
                    )
            except HTTPException:
                raise
            except OSError as exc:
                logger.exception("Unable to check existing PDF")
                raise HTTPException(
                    status_code=500, detail="Unable to validate existing documents."
                ) from exc
    written_paths = []
    try:
        for filename, content in accepted:
            destination = settings.papers_directory / filename
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
            written_paths.append(destination)
    except OSError as exc:
        logger.exception("Unable to save uploaded PDF")
        for path in written_paths:
            path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail="Unable to save uploaded documents."
        ) from exc
    logger.info("Uploaded %d PDF documents", len(accepted))
    return UploadResponse(
        uploaded=[filename for filename, _ in accepted],
        message=f"Uploaded {len(accepted)} PDF document(s). Process them to make them searchable.",
    )


@router.post("/process", response_model=ProcessResponse)
def process_documents(request: Request) -> ProcessResponse:
    """Index uploaded PDFs and persist their vectors."""
    processor: DocumentProcessor = request.app.state.processor
    try:
        result = processor.process_pending()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ProcessResponse(**result)


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(request: Request) -> DocumentsResponse:
    """List processed documents and their indexed page/chunk counts."""
    _, vector_store = _services(request)
    try:
        return DocumentsResponse(documents=vector_store.list_documents())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/documents", response_model=MessageResponse)
def clear_documents(request: Request) -> MessageResponse:
    """Clear indexed vectors but keep uploaded PDFs for optional reprocessing."""
    _, vector_store = _services(request)
    try:
        vector_store.clear()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MessageResponse(message="Vector database cleared. Uploaded PDFs were kept.")


@router.post("/query", response_model=QueryResponse)
def query_documents(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve source chunks and generate a grounded answer."""
    settings = request.app.state.settings
    try:
        retriever: SemanticRetriever = request.app.state.retriever
        sources = retriever.retrieve(
            body.question,
            body.top_k or settings.top_k,
            (
                body.similarity_threshold
                if body.similarity_threshold is not None
                else settings.similarity_threshold
            ),
        )
        answer = generate_answer(settings, body.question, sources)
    except RuntimeError as exc:
        message = str(exc)
        if "OPENAI_API_KEY" in message:
            raise HTTPException(status_code=503, detail=message) from exc
        if "Sentence Transformers" in message or "model" in message.lower():
            raise HTTPException(status_code=503, detail=message) from exc
        raise HTTPException(status_code=502, detail=message) from exc
    return QueryResponse(answer=answer, sources=sources)
