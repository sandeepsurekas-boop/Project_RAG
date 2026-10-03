"""PDF extraction using PyMuPDF."""

import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


class PDFLoadError(ValueError):
    """Raised when a PDF is unreadable or contains no extractable text."""


def load_pdf(path: Path) -> list[dict[str, int | str]]:
    """Extract non-empty text page by page, using one-based page numbers."""
    try:
        with fitz.open(path) as pdf:
            if pdf.is_encrypted:
                raise PDFLoadError(f"PDF is password-protected: {path.name}")
            pages = [
                {"page": page_number, "text": page.get_text("text").strip()}
                for page_number, page in enumerate(pdf, start=1)
            ]
    except PDFLoadError:
        raise
    except (fitz.FileDataError, OSError, RuntimeError) as exc:
        raise PDFLoadError(f"Unable to read PDF '{path.name}': {exc}") from exc

    text_pages = [page for page in pages if page["text"]]
    logger.info("Extracted %d text pages from %s", len(text_pages), path.name)
    if not text_pages:
        raise PDFLoadError(f"PDF contains no extractable text: {path.name}")
    return text_pages
