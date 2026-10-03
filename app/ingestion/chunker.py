"""Page-aware text chunking with stable source metadata."""

import hashlib
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(
    pages: list[dict[str, int | str]],
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, int | str]]:
    """Split pages independently so every chunk has an exact page citation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[dict[str, int | str]] = []
    safe_stem = Path(filename).stem
    for page in pages:
        page_number = int(page["page"])
        text = str(page["text"])
        for index, item in enumerate(splitter.create_documents([text])):
            content = item.page_content.strip()
            if not content:
                continue
            raw_id = f"{filename}:{page_number}:{index}:{content}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "text": content,
                    "filename": filename,
                    "document": safe_stem,
                    "page": page_number,
                    "chunk_index": index,
                }
            )
    return chunks


def safe_pdf_filename(filename: str) -> str:
    """Normalize an uploaded name to a safe, flat PDF filename."""
    basename = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", basename).strip(" .")
    if not cleaned or not cleaned.lower().endswith(".pdf"):
        raise ValueError("Each uploaded document must have a .pdf filename")
    return cleaned
