"""Tests for retrieval filtering and answer generation."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.embeddings.embedding_service import EmbeddingService
from app.generation.rag_chain import NOT_AVAILABLE_ANSWER, RagAnswerGenerator
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.vector_store import VectorStore


def test_retriever_filters_below_threshold():
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1, 0.2]
    store = Mock()
    store.search.return_value = [
        {"document": "a.pdf", "page": 1, "content": "Relevant", "score": 0.8, "chunk_id": "a"},
        {"document": "b.pdf", "page": 2, "content": "Weak", "score": 0.2, "chunk_id": "b"},
    ]
    matches = SemanticRetriever(embeddings, store).retrieve("question", 5, 0.3)
    assert len(matches) == 1
    assert matches[0]["document"] == "a.pdf"
    store.search.assert_called_once_with([0.1, 0.2], 5)


def test_retriever_skips_embedding_when_no_documents_are_indexed():
    embeddings = Mock()
    store = Mock()
    store.document_names.return_value = set()
    assert SemanticRetriever(embeddings, store).retrieve("question", 5, 0.3) == []
    embeddings.embed_query.assert_not_called()


def test_generator_returns_fallback_without_calling_llm():
    generator = RagAnswerGenerator(Mock())
    with patch("app.generation.rag_chain.create_chat_model") as create_model:
        assert generator.generate("question", []) == NOT_AVAILABLE_ANSWER
        create_model.assert_not_called()


def test_generator_passes_citations_in_context():
    settings = Mock()
    source = {
        "document": "paper.pdf",
        "page": 4,
        "content": "Attention text",
        "score": 0.82,
        "chunk_id": "chunk-1",
    }
    captured = {}

    def fake_llm(prompt_value):
        captured["prompt"] = prompt_value.to_string()
        return AIMessage(content="Grounded response.")

    fake_model = RunnableLambda(fake_llm)
    with patch(
        "app.generation.rag_chain.create_chat_model", return_value=fake_model
    ):
        answer = RagAnswerGenerator(settings).generate("What is attention?", [source])
    assert "Grounded response" in answer
    assert "[Source 1: paper.pdf, page 4]" in answer
    assert "document=paper.pdf" in captured["prompt"]
    assert "page=4" in captured["prompt"]
    assert "Attention text" in captured["prompt"]


def test_embedding_service_encodes_documents_and_query():
    service = EmbeddingService("test-model")
    model = Mock()
    model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    service._model = model
    assert service.embed_documents(["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]
    model.encode.return_value.tolist.return_value = [0.5, 0.6]
    assert service.embed_query("query") == [0.5, 0.6]
    assert model.encode.call_count == 2


def test_vector_store_converts_cosine_distance_to_similarity():
    store = VectorStore.__new__(VectorStore)
    collection = Mock()
    collection.query.return_value = {
        "documents": [["Text"]],
        "metadatas": [[{"filename": "paper.pdf", "page": 3}]],
        "distances": [[0.18]],
        "ids": [["chunk-1"]],
    }
    store._collection = collection
    matches = store.search([0.1, 0.2], 2)
    assert matches == [
        {
            "document": "paper.pdf",
            "page": 3,
            "content": "Text",
            "score": 0.82,
            "chunk_id": "chunk-1",
        }
    ]
