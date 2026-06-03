"""
FastAPI: многопроектный Re-ID для тритонов.

Запуск из корня репозитория:
  uvicorn web_app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass

WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web_app.routers import individuals, pages, projects, search
from web_app.services import import_legacy, store

_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _MODEL_PATH.is_file():
        raise FileNotFoundError(f"Нет весов модели: {_MODEL_PATH}")
    os.chdir(PROJECT_ROOT)
    store.init_db()
    import_legacy.run_import_if_needed()
    app.state.engines = {}
    yield
    app.state.engines.clear()


app = FastAPI(
    title="Newt Re-ID",
    description="Многопроектная идентификация тритонов",
    lifespan=lifespan,
)

app.include_router(projects.router)
app.include_router(individuals.router)
app.include_router(search.router)
app.include_router(pages.router)

_static = WEB_APP_DIR / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")
