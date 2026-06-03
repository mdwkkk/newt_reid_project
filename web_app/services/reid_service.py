"""Кэш NewtMatchEngine по проектам и индексация фото."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from inference import NewtMatchEngine

from . import store

PROJECT_ROOT = store.PROJECT_ROOT
MODEL_PATH = "models/best_model.pth"
THRESHOLD = 0.49


def _model_weights_path() -> Path:
    return PROJECT_ROOT / MODEL_PATH


def get_engine(app_state: Any, project_id: str) -> NewtMatchEngine:
    cache: dict[str, NewtMatchEngine] = app_state.engines
    if project_id in cache:
        return cache[project_id]

    db_path = store.vector_db_path(project_id)
    engine = NewtMatchEngine(
        model_weights_path=str(_model_weights_path()),
        db_pt_path=str(db_path) if db_path.is_file() else str(db_path),
        threshold=THRESHOLD,
    )
    cache[project_id] = engine
    return engine


def invalidate_engine(app_state: Any, project_id: str) -> None:
    app_state.engines.pop(project_id, None)


def index_photo_file(
    app_state: Any,
    project_id: str,
    individual_id: str,
    image_path: str,
    *,
    is_raw: bool = True,
) -> None:
    engine = get_engine(app_state, project_id)
    emb, _ = engine.extract_embedding_from_path(image_path, is_raw=is_raw)
    if emb is None:
        raise ValueError("Не удалось извлечь embedding")
    engine.add_embedding(individual_id, emb)
    db_path = store.vector_db_path(project_id)
    store.project_dir(project_id).mkdir(parents=True, exist_ok=True)
    engine.persist_db(str(db_path))


async def save_upload_and_index(
    app_state: Any,
    project_id: str,
    individual_id: str,
    upload: UploadFile,
    dest_path: Path,
) -> None:
    content = await upload.read()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=dest_path.suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        index_photo_file(app_state, project_id, individual_id, tmp_path, is_raw=True)
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def remove_individual_from_index(
    app_state: Any, project_id: str, individual_id: str
) -> None:
    engine = get_engine(app_state, project_id)
    engine.remove_label(individual_id)
    db_path = store.vector_db_path(project_id)
    engine.persist_db(str(db_path))


def reindex_individual(
    app_state: Any, project_id: str, individual_id: str
) -> None:
    """Переиндексировать все фото особи."""
    engine = get_engine(app_state, project_id)
    engine.remove_label(individual_id)
    gdir = store.individual_gallery_dir(project_id, individual_id)
    if not gdir.is_dir():
        db_path = store.vector_db_path(project_id)
        engine.persist_db(str(db_path))
        return

    exts = store._IMAGE_EXTS
    for img in sorted(gdir.iterdir()):
        if img.is_file() and img.suffix in exts:
            emb, err = engine.extract_embedding_from_path(str(img), is_raw=False)
            if emb is not None:
                engine.add_embedding(individual_id, emb)

    db_path = store.vector_db_path(project_id)
    engine.persist_db(str(db_path))
