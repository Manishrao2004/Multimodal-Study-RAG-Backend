from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.schemas import AskRequest, AskResponse, ScreenshotAskResponse, EvidenceItem, _clean_query_text
from app.core.security import validate_query_text


class ConversationTitle(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=120)
    _validate_title = field_validator("title")(_clean_query_text)


class ConversationQuestion(AskRequest):
    request_id: UUID
    revision: int = Field(ge=0)


class StoredMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: float
    result: AskResponse | ScreenshotAskResponse | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    revision: int


class ConversationDetail(ConversationSummary):
    messages: list[StoredMessage]


class ImportedMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=30)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value):
        return validate_query_text(value, max_length=100000)


class ConversationImport(ConversationTitle):
    id: UUID
    messages: list[ImportedMessage] = Field(max_length=1000)
