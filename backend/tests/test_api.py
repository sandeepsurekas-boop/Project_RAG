"""FastAPI endpoint tests with external services mocked."""

import io

import pymupdf
import pytest

from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "papers_directory", tmp_path / "papers")
    monkeypatch.setattr(settings, "chroma_persist_directory", tmp_path / "chroma")
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rejects_blank_question():
    with TestClient(app) as test_client:
        response = test_client.post("/query", json={"question": "  "})
    assert response.status_code == 422


def test_query_rejects_frontend_model_override():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/query",
            json={"question": "Question?", "model": "client-selected-model"},
        )
    assert response.status_code == 422


def test_query_returns_retrieved_sources(monkeypatch, client):
    from backend.app.retrieval.retriever import SemanticRetriever

    observed = {}

    def fake_retrieve(self, question, top_k, threshold):
        observed["retrieval_question"] = question
        return [
            {
                "document": "paper.pdf",
                "page": 2,
                "content": "Source passage",
                "score": 0.9,
                "chunk_id": "chunk-1",
            }
        ]

    monkeypatch.setattr(
        SemanticRetriever,
        "retrieve",
        fake_retrieve,
    )
    monkeypatch.setattr(
        "backend.app.api.routes.generate_answer",
        lambda settings, question, sources, history: "Answer [paper.pdf, page 2].",
    )
    response = client.post(
        "/query",
        json={
            "question": "What is described?",
            "history": [{"role": "user", "content": "Tell me about the paper."}],
        },
    )
    assert response.status_code == 200
    assert response.json()["sources"][0]["page"] == 2
    assert response.json()["sources"][0]["match_percent"] == 90
    assert response.json()["answer"].startswith("Answer")
    assert observed["retrieval_question"] == (
        "Tell me about the paper. What is described?"
    )


def test_upload_rejects_non_pdf(monkeypatch, client):
    from backend.app.retrieval.vector_store import VectorStore

    monkeypatch.setattr(VectorStore, "document_names", lambda self: set())
    response = client.post(
        "/upload",
        files=[("files", ("paper.pdf", b"not a pdf", "application/pdf"))],
    )
    assert response.status_code == 400


def test_upload_saves_pdf_on_backend(monkeypatch, client):
    from backend.app.retrieval.vector_store import VectorStore

    monkeypatch.setattr(VectorStore, "document_names", lambda self: set())
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Backend-owned paper storage")
    content = io.BytesIO()
    pdf.save(content)
    pdf.close()

    response = client.post(
        "/upload",
        files=[("files", ("backend-paper.pdf", content.getvalue(), "application/pdf"))],
    )

    assert response.status_code == 201
    saved_file = get_settings().papers_directory / "backend-paper.pdf"
    assert saved_file.read_bytes() == content.getvalue()
