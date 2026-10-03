"""FastAPI endpoint tests with external services mocked."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rejects_blank_question():
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "  "})
    assert response.status_code == 422


def test_query_returns_retrieved_sources(monkeypatch):
    from app.retrieval.retriever import SemanticRetriever
    from app.generation.rag_chain import RagAnswerGenerator

    monkeypatch.setattr(
        SemanticRetriever,
        "retrieve",
        lambda self, question, top_k, threshold: [
            {
                "document": "paper.pdf",
                "page": 2,
                "content": "Source passage",
                "score": 0.9,
                "chunk_id": "chunk-1",
            }
        ],
    )
    monkeypatch.setattr(
        RagAnswerGenerator,
        "generate",
        lambda self, question, sources, model=None: "Answer [paper.pdf, page 2].",
    )
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is described?"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["page"] == 2
    assert response.json()["answer"].startswith("Answer")


def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.retrieval.vector_store import VectorStore

    settings = get_settings()
    monkeypatch.setattr(settings, "papers_directory", tmp_path)
    monkeypatch.setattr(VectorStore, "document_names", lambda self: set())
    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files=[("files", ("paper.pdf", b"not a pdf", "application/pdf"))],
        )
    assert response.status_code == 400
