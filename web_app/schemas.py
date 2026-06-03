"""Pydantic-схемы API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    card_template: str


class ProjectOut(BaseModel):
    id: str
    name: str
    card_template: str
    created_at: str
    individual_count: int = 0


class IndividualOut(BaseModel):
    individual_id: str
    metadata: dict[str, Any]
    photo_count: int = 0
    preview_url: str | None = None
    card_url: str | None = None


class IndividualDetailOut(IndividualOut):
    photos: list[dict[str, Any]] = []


class MetadataUpdate(BaseModel):
    metadata: dict[str, Any]
