"""Pydantic API request and response models."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    similarity_threshold: float | None = Field(default=None, ge=-1, le=1)
    model: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty.")
        return cleaned

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class Source(BaseModel):
    document: str
    page: int = Field(ge=1)
    content: str
    score: float = Field(ge=-1, le=1)
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    uploaded: list[str]
    message: str


class ProcessResponse(BaseModel):
    documents: list[str]
    pages: int
    chunks: int


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
