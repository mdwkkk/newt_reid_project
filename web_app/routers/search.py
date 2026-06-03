"""Re-ID поиск в контексте проекта."""
from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from web_app.services import reid_service, store

router = APIRouter(prefix="/api", tags=["search"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = store.PROJECT_ROOT


@router.post("/projects/{project_id}/search")
async def api_search(
    request: Request,
    project_id: str,
    file: UploadFile = File(...),
) -> JSONResponse:
    if not store.get_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    if store.count_individuals(project_id) == 0:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error_code": "project_empty",
                "message": "В проекте нет особей. Добавьте особь, чтобы начать поиск.",
                "action_url": f"/projects/{project_id}/individuals/new",
            },
        )

    engine = reid_service.get_engine(request.app.state, project_id)

    if engine.is_empty:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error_code": "no_index",
                "message": "В проекте есть особи, но нет проиндексированных фото. Добавьте фотографии к особям.",
                "action_url": f"/projects/{project_id}/gallery",
            },
        )

    query_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    if not ext or ext.lower() not in {e.lower() for e in _IMAGE_EXTS}:
        ext = ".jpg"

    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        raw = engine.process_query(query_id, tmp_path, is_raw=True)
        payload: dict[str, Any] = json.loads(raw)

        if payload.get("error_code") == "project_empty":
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error_code": "project_empty",
                    "message": payload.get("error"),
                    "action_url": f"/projects/{project_id}/individuals/new",
                },
            )

        if "error" in payload:
            return JSONResponse(
                status_code=422,
                content={"success": False, "error": payload["error"]},
            )

        debug_path = PROJECT_ROOT / f"debug_input_{query_id}.jpg"
        unwrapped_b64: str | None = None
        if debug_path.is_file():
            unwrapped_b64 = base64.b64encode(debug_path.read_bytes()).decode("ascii")
            try:
                debug_path.unlink()
            except OSError:
                pass

        candidates_enriched: list[dict[str, Any]] = []
        for item in payload.get("top_20_candidates") or []:
            label = item.get("label")
            sid = str(label)
            preview_path = f"/api/projects/{project_id}/gallery/{sid}/preview"
            candidates_enriched.append(
                {
                    "id": label,
                    "label": label,
                    "score": item.get("score"),
                    "gallery_preview_path": preview_path,
                    "card_url": f"/projects/{project_id}/individuals/{sid}",
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "query_id": query_id,
                "unwrapped_mime": "image/jpeg",
                "unwrapped_base64": unwrapped_b64,
                "top_20_candidates": candidates_enriched,
                "best_match": payload.get("best_match"),
                "confidence": payload.get("confidence"),
                "is_new": payload.get("is_new"),
            },
        )
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
