"""SQLite-хранилище проектов, особей и фото."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

WEB_APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "app.db"
PROJECTS_ROOT = DATA_DIR / "projects"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(raw: str) -> str:
    s = str(raw).strip()
    if not s or re.search(r"[^\w.\-]", s):
        raise ValueError("Invalid id")
    return s


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                card_template TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS individuals (
                project_id TEXT NOT NULL,
                individual_id TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_id, individual_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                individual_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id, individual_id)
                    REFERENCES individuals(project_id, individual_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_individuals_project
                ON individuals(project_id);
            CREATE INDEX IF NOT EXISTS idx_photos_individual
                ON photos(project_id, individual_id);
            """
        )


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def gallery_root(project_id: str) -> Path:
    return project_dir(project_id) / "gallery"


def vector_db_path(project_id: str) -> Path:
    return project_dir(project_id) / "vector_database.pt"


def individual_gallery_dir(project_id: str, individual_id: str) -> Path:
    safe = _sanitize_id(individual_id)
    return gallery_root(project_id) / safe


def list_projects() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.card_template, p.created_at,
                   COUNT(i.individual_id) AS individual_count
            FROM projects p
            LEFT JOIN individuals i ON i.project_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT p.id, p.name, p.card_template, p.created_at,
                   COUNT(i.individual_id) AS individual_count
            FROM projects p
            LEFT JOIN individuals i ON i.project_id = p.id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (project_id,),
        ).fetchone()
    return dict(row) if row else None


def create_project(name: str, card_template: str) -> dict[str, Any]:
    pid = str(uuid.uuid4())
    created = _now_iso()
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "gallery").mkdir(exist_ok=True)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, card_template, created_at) VALUES (?, ?, ?, ?)",
            (pid, name.strip(), card_template, created),
        )
    return {
        "id": pid,
        "name": name.strip(),
        "card_template": card_template,
        "created_at": created,
        "individual_count": 0,
    }


def delete_project(project_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        if cur.rowcount == 0:
            return False
    pdir = project_dir(project_id)
    if pdir.is_dir():
        shutil.rmtree(pdir, ignore_errors=True)
    return True


def list_individual_ids(project_id: str) -> set[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT individual_id FROM individuals WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    return {r["individual_id"] for r in rows}


def list_individuals(project_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT i.individual_id, i.metadata, i.created_at, i.updated_at,
                   COUNT(p.id) AS photo_count
            FROM individuals i
            LEFT JOIN photos p
                ON p.project_id = i.project_id AND p.individual_id = i.individual_id
            WHERE i.project_id = ?
            GROUP BY i.individual_id
            ORDER BY i.individual_id
            """,
            (project_id,),
        ).fetchall()

    result = []
    for r in rows:
        meta = json.loads(r["metadata"])
        iid = r["individual_id"]
        result.append(
            {
                "individual_id": iid,
                "metadata": meta,
                "photo_count": r["photo_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "preview_url": f"/api/projects/{project_id}/gallery/{iid}/preview",
                "card_url": f"/projects/{project_id}/individuals/{iid}",
            }
        )
    return result


def get_individual(project_id: str, individual_id: str) -> dict[str, Any] | None:
    safe = _sanitize_id(individual_id)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT individual_id, metadata, created_at, updated_at
            FROM individuals
            WHERE project_id = ? AND individual_id = ?
            """,
            (project_id, safe),
        ).fetchone()
        if not row:
            return None
        photos = conn.execute(
            """
            SELECT id, filename, sort_order, created_at
            FROM photos
            WHERE project_id = ? AND individual_id = ?
            ORDER BY sort_order, id
            """,
            (project_id, safe),
        ).fetchall()

    photo_list = []
    for p in photos:
        photo_list.append(
            {
                "id": p["id"],
                "filename": p["filename"],
                "sort_order": p["sort_order"],
                "url": f"/api/projects/{project_id}/gallery/{safe}/photos/{p['filename']}",
            }
        )

    meta = json.loads(row["metadata"])
    return {
        "individual_id": safe,
        "metadata": meta,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "photo_count": len(photo_list),
        "photos": photo_list,
        "preview_url": f"/api/projects/{project_id}/gallery/{safe}/preview",
        "card_url": f"/projects/{project_id}/individuals/{safe}",
    }


def create_individual(
    project_id: str,
    individual_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    safe = _sanitize_id(individual_id)
    now = _now_iso()
    meta_json = json.dumps(metadata, ensure_ascii=False)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO individuals (project_id, individual_id, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, safe, meta_json, now, now),
        )

    individual_gallery_dir(project_id, safe).mkdir(parents=True, exist_ok=True)
    return get_individual(project_id, safe)  # type: ignore


def update_individual_metadata(
    project_id: str,
    individual_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    safe = _sanitize_id(individual_id)
    now = _now_iso()
    meta_json = json.dumps(metadata, ensure_ascii=False)

    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE individuals SET metadata = ?, updated_at = ?
            WHERE project_id = ? AND individual_id = ?
            """,
            (meta_json, now, project_id, safe),
        )
        if cur.rowcount == 0:
            return None
    return get_individual(project_id, safe)


def delete_individual(project_id: str, individual_id: str) -> bool:
    safe = _sanitize_id(individual_id)
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM individuals WHERE project_id = ? AND individual_id = ?",
            (project_id, safe),
        )
        if cur.rowcount == 0:
            return False
    gdir = individual_gallery_dir(project_id, safe)
    if gdir.is_dir():
        shutil.rmtree(gdir, ignore_errors=True)
    return True


def add_photo_record(
    project_id: str,
    individual_id: str,
    filename: str,
    sort_order: int,
) -> dict[str, Any]:
    safe = _sanitize_id(individual_id)
    now = _now_iso()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO photos (project_id, individual_id, filename, sort_order, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, safe, filename, sort_order, now),
        )
        photo_id = cur.lastrowid
    return {
        "id": photo_id,
        "filename": filename,
        "sort_order": sort_order,
        "url": f"/api/projects/{project_id}/gallery/{safe}/photos/{filename}",
    }


def next_photo_filename(project_id: str, individual_id: str) -> str:
    gdir = individual_gallery_dir(project_id, individual_id)
    existing = list(gdir.glob("photo_*")) if gdir.is_dir() else []
    n = len(existing) + 1
    return f"photo_{n:03d}.jpg"


def first_image_path(project_id: str, individual_id: str) -> Path | None:
    gdir = individual_gallery_dir(project_id, individual_id)
    if not gdir.is_dir():
        return None
    files = [p for p in gdir.iterdir() if p.is_file() and p.suffix in _IMAGE_EXTS]
    if not files:
        return None
    return sorted(files, key=lambda p: p.name)[0]


def photo_path(project_id: str, individual_id: str, filename: str) -> Path | None:
    safe = _sanitize_id(individual_id)
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    p = individual_gallery_dir(project_id, safe) / filename
    return p if p.is_file() else None


def count_projects() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM projects").fetchone()
    return int(row["c"]) if row else 0


def count_individuals(project_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM individuals WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    return int(row["c"]) if row else 0
