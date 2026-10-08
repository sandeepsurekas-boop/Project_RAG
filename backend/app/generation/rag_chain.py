"""Generate a cited answer from retrieved paper chunks."""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.app.config import Settings
from backend.app.generation.llm import create_chat_model

NOT_AVAILABLE = "The answer is not available in the provided documents."

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the research-paper excerpts below.
Do not use outside knowledge. If the answer is not in the excerpts, reply exactly:
"{not_available}"
Cite factual statements using [document, page N].

Excerpts:
{context}

Question:
{question}
"""
)


def generate_answer(
    settings: Settings,
    question: str,
    sources: list[dict[str, Any]],
) -> str:
    """Pass retrieved text to the backend LLM and include source citations."""
    if not sources:
        return NOT_AVAILABLE

    context = "\n\n".join(
        f"[{source['document']}, page {source['page']}]\n{source['content']}"
        for source in sources
    )
    try:
        answer = (
            PROMPT
            | create_chat_model(settings)
            | StrOutputParser()
        ).invoke(
            {
                "context": context,
                "question": question,
                "not_available": NOT_AVAILABLE,
            }
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"Answer generation failed: {exc}") from exc

    if not answer:
        raise RuntimeError("The language model returned an empty answer.")

    citations = ", ".join(
        f"[{source['document']}, page {source['page']}]" for source in sources
    )
    return f"{answer}\n\nSources: {citations}"
