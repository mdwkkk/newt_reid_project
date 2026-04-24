"""
FastAPI-оболочка для Re-ID. Запускать из корня репозитория (newt_reid_project-main),
чтобы CWD и относительные пути как в inference/crop_belly оставались корректны.

  uvicorn web_app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager

# Uvicorn на Windows часто пишет в cp1251; print() в test_model/predict (эмодзи) падает с
# UnicodeEncodeError и ломает сегментацию. Пишем в UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Корень репозитория (родитель папки web_app)
WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import NewtMatchEngine  # noqa: E402

_MODEL_PATH = "models/best_model.pth"
_DB_PATH = "data/vector_database.pt"
_GALLERY_ROOT = PROJECT_ROOT / "data" / "train_unwrapped"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}


def _first_image_in_id_folder(newt_id: str) -> Path | None:
    folder = _GALLERY_ROOT / newt_id
    if not folder.is_dir():
        return None
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix in _IMAGE_EXTS]
    if not files:
        return None
    return sorted(files, key=lambda p: p.name)[0]


def _sanitize_id_segment(raw: str) -> str:
    s = str(raw).strip()
    if not s or re.search(r"[^\w-]", s):
        raise HTTPException(status_code=400, detail="Invalid id")
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (PROJECT_ROOT / _MODEL_PATH).is_file():
        raise FileNotFoundError(f"Нет весов модели: {PROJECT_ROOT / _MODEL_PATH}")
    if not (PROJECT_ROOT / _DB_PATH).is_file():
        raise FileNotFoundError(f"Нет векторной БД: {PROJECT_ROOT / _DB_PATH}")
    # Рабочая директория = корень репозитория (как в CLI)
    os.chdir(PROJECT_ROOT)
    app.state.engine = NewtMatchEngine(
        model_weights_path=_MODEL_PATH,
        db_pt_path=_DB_PATH,
        threshold=0.49,
    )
    yield
    # shutdown: ничего обязательного


app = FastAPI(
    title="Newt Re-ID",
    description="Веб-API для идентификации тритонов (обёртка над inference.py)",
    lifespan=lifespan,
)


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    engine: NewtMatchEngine = request.app.state.engine
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
            preview_path = f"/api/gallery/{sid}/preview"
            candidates_enriched.append(
                {
                    "id": label,
                    "label": label,
                    "score": item.get("score"),
                    "gallery_preview_path": preview_path,
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


@app.get("/api/gallery/{newt_id}/preview")
def gallery_preview(newt_id: str) -> FileResponse:
    safe = _sanitize_id_segment(newt_id)
    one = _first_image_in_id_folder(safe)
    if one is None or not one.is_file():
        raise HTTPException(status_code=404, detail="Gallery image not found")
    mime, _ = mimetypes.guess_type(str(one))
    return FileResponse(one, media_type=mime or "image/jpeg")


# Статика и index.html в конце, чтобы /api/* не перехватывались
_static = WEB_APP_DIR / "static"
if _static.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_static), html=True),
        name="static",
    )
