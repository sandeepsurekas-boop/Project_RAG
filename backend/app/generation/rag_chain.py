"""Generate a cited answer from retrieved paper chunks."""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.config import Settings
from backend.app.generation.llm import create_chat_model

NOT_AVAILABLE = "The answer is not available in the provided documents."

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You answer questions about research papers. Use only the excerpts
below to support factual claims. Use the conversation only to understand
follow-up questions; do not treat previous answers as evidence. Cite claims
with [document, page N]. If the excerpts do not contain the answer, reply
exactly: "{not_available}" """,
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            """Retrieved paper excerpts:
{context}

Question:
{question}""",
        ),
    ]
)


def generate_answer(
    settings: Settings,
    question: str,
    sources: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Pass retrieved text to the backend LLM and include source citations."""
    if not sources:
        return NOT_AVAILABLE

    context = "\n\n".join(
        f"[{source['document']}, page {source['page']}]\n{source['content']}"
        for source in sources
    )
    chat_history = [
        HumanMessage(content=turn["content"])
        if turn["role"] == "user"
        else AIMessage(content=turn["content"])
        for turn in (history or [])
    ]
    try:
        answer = (
            CHAT_PROMPT
            | create_chat_model(settings)
            | StrOutputParser()
        ).invoke(
            {
                "context": context,
                "question": question,
                "not_available": NOT_AVAILABLE,
                "history": chat_history,
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
