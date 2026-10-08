"""Tests for PDF extraction and page-aware chunking."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest

from backend.app.ingestion.chunker import safe_pdf_filename, split_pages
from backend.app.ingestion.pdf_loader import PDFLoadError, load_pdf
from backend.app.ingestion.processor import DocumentProcessor


def make_pdf(tmp_path: Path, text: str, name: str = "paper.pdf") -> Path:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    path = tmp_path / name
    pdf.save(path)
    pdf.close()
    return path


def test_pdf_loader_extracts_page_text(tmp_path):
    path = make_pdf(tmp_path, "Transformer attention paper")
    pages = load_pdf(path)
    assert pages == [{"page": 1, "text": "Transformer attention paper"}]


def test_pdf_loader_rejects_empty_pdf(tmp_path):
    path = tmp_path / "empty.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    with pytest.raises(PDFLoadError, match="no extractable text"):
        load_pdf(path)


def test_chunking_preserves_page_and_filename_metadata():
    chunks = split_pages(
        [{"page": 3, "text": "word " * 90}],
        "paper.pdf",
        chunk_size=100,
        chunk_overlap=20,
    )
    assert len(chunks) > 1
    assert all(chunk["page"] == 3 for chunk in chunks)
    assert all(chunk["filename"] == "paper.pdf" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)


def test_safe_filename_removes_path_and_rejects_non_pdf():
    assert safe_pdf_filename("../../paper.pdf") == "paper.pdf"
    with pytest.raises(ValueError, match=".pdf"):
        safe_pdf_filename("paper.txt")


def test_processor_embeds_and_persists_page_aware_chunks(tmp_path):
    make_pdf(tmp_path, "Transformer attention")
    settings = SimpleNamespace(
        papers_directory=tmp_path,
        chunk_size=100,
        chunk_overlap=20,
    )
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2]]
    vector_store = Mock()
    vector_store.document_names.return_value = set()

    result = DocumentProcessor(settings, embeddings, vector_store).process_pending()

    assert result == {"documents": ["paper.pdf"], "pages": 1, "chunks": 1}
    chunks = vector_store.add_chunks.call_args.args[0]
    assert chunks[0]["filename"] == "paper.pdf"
    assert chunks[0]["page"] == 1
    assert embeddings.embed_documents.call_args.args[0] == ["Transformer attention"]
