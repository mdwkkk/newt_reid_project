"""Импорт демо-галереи train_unwrapped при первом запуске."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import store

PROJECT_ROOT = store.PROJECT_ROOT
LEGACY_GALLERY = PROJECT_ROOT / "data" / "train_unwrapped"
LEGACY_VECTOR_DB = PROJECT_ROOT / "data" / "vector_database.pt"
DEMO_NAME = "Демо (импорт)"
DEMO_TEMPLATE = "IK-1"


def run_import_if_needed() -> str | None:
    """Создать демо-проект, если БД пуста и есть legacy-данные. Возвращает project_id."""
    store.init_db()
    if store.count_projects() > 0:
        return None

    if not LEGACY_GALLERY.is_dir():
        return None

    proj = store.create_project(DEMO_NAME, DEMO_TEMPLATE)
    pid = proj["id"]
    dest_gallery = store.gallery_root(pid)
    dest_gallery.mkdir(parents=True, exist_ok=True)

    individual_ids: list[str] = []
    for folder in sorted(LEGACY_GALLERY.iterdir()):
        if not folder.is_dir():
            continue
        iid = folder.name
        individual_ids.append(iid)
        dest_folder = dest_gallery / iid
        if not dest_folder.exists():
            try:
                shutil.copytree(folder, dest_folder)
            except OSError:
                dest_folder.mkdir(parents=True, exist_ok=True)
                for f in folder.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dest_folder / f.name)

        meta = {"individual_id": iid}
        try:
            store.create_individual(pid, iid, meta)
        except (ValueError, Exception):
            continue

        gdir = store.individual_gallery_dir(pid, iid)
        sort = 0
        for img in sorted(gdir.iterdir()):
            if img.is_file() and img.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                store.add_photo_record(pid, iid, img.name, sort)
                sort += 1

    dest_vdb = store.vector_db_path(pid)
    if LEGACY_VECTOR_DB.is_file():
        shutil.copy2(LEGACY_VECTOR_DB, dest_vdb)
    else:
        dest_vdb.touch()

    return pid
