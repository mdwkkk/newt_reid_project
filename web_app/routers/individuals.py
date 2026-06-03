"""API особей и медиа."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from web_app.card_templates import validate_metadata
from web_app.schemas import MetadataUpdate
from web_app.services import reid_service, store

router = APIRouter(prefix="/api", tags=["individuals"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _require_project(project_id: str) -> dict[str, Any]:
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return proj


@router.get("/projects/{project_id}/individuals")
def api_list_individuals(project_id: str):
    _require_project(project_id)
    return {"individuals": store.list_individuals(project_id)}


@router.get("/projects/{project_id}/individuals/{individual_id}")
def api_get_individual(project_id: str, individual_id: str):
    _require_project(project_id)
    ind = store.get_individual(project_id, individual_id)
    if not ind:
        raise HTTPException(status_code=404, detail="Особь не найдена")
    return ind


@router.post("/projects/{project_id}/individuals")
async def api_create_individual(
    request: Request,
    project_id: str,
    metadata: str = File(...),
    photos: list[UploadFile] | None = File(None),
):
    photos = photos or []
    proj = _require_project(project_id)
    try:
        meta_in = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata должен быть JSON")

    existing = store.list_individual_ids(project_id)
    cleaned, errors = validate_metadata(proj["card_template"], meta_in, existing_ids=existing)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    iid = cleaned["individual_id"]
    try:
        store.create_individual(project_id, iid, cleaned)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved_photos = []
    sort = 0
    for upload in photos:
        if not upload.filename:
            continue
        ext = "." + upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ".jpg"
        if ext not in _IMAGE_EXTS:
            ext = ".jpg"
        fname = store.next_photo_filename(project_id, iid)
        if not fname.endswith(ext):
            fname = fname.rsplit(".", 1)[0] + ext
        dest = store.individual_gallery_dir(project_id, iid) / fname
        await reid_service.save_upload_and_index(
            request.app.state, project_id, iid, upload, dest
        )
        rec = store.add_photo_record(project_id, iid, fname, sort)
        saved_photos.append(rec)
        sort += 1

    ind = store.get_individual(project_id, iid)
    return JSONResponse(status_code=201, content=ind)


@router.put("/projects/{project_id}/individuals/{individual_id}")
def api_update_individual(
    project_id: str,
    individual_id: str,
    body: MetadataUpdate,
):
    proj = _require_project(project_id)
    if not store.get_individual(project_id, individual_id):
        raise HTTPException(status_code=404, detail="Особь не найдена")

    existing = store.list_individual_ids(project_id)
    cleaned, errors = validate_metadata(
        proj["card_template"],
        body.metadata,
        existing_ids=existing,
        exclude_individual_id=individual_id,
    )
    if cleaned.get("individual_id") != individual_id:
        raise HTTPException(
            status_code=400,
            detail="ID особи нельзя изменить через редактирование",
        )
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    ind = store.update_individual_metadata(project_id, individual_id, cleaned)
    return ind


@router.post("/projects/{project_id}/individuals/{individual_id}/photos")
async def api_add_photos(
    request: Request,
    project_id: str,
    individual_id: str,
    photos: list[UploadFile] = File(...),
):
    _require_project(project_id)
    if not store.get_individual(project_id, individual_id):
        raise HTTPException(status_code=404, detail="Особь не найдена")

    saved = []
    ind = store.get_individual(project_id, individual_id)
    sort = ind["photo_count"] if ind else 0

    for upload in photos:
        if not upload.filename:
            continue
        ext = "." + upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ".jpg"
        if ext not in _IMAGE_EXTS:
            ext = ".jpg"
        fname = store.next_photo_filename(project_id, individual_id)
        if not fname.endswith(ext):
            fname = fname.rsplit(".", 1)[0] + ext
        dest = store.individual_gallery_dir(project_id, individual_id) / fname
        await reid_service.save_upload_and_index(
            request.app.state, project_id, individual_id, upload, dest
        )
        rec = store.add_photo_record(project_id, individual_id, fname, sort)
        saved.append(rec)
        sort += 1

    return {"photos": saved}


@router.delete("/projects/{project_id}/individuals/{individual_id}", status_code=204)
def api_delete_individual(
    request: Request, project_id: str, individual_id: str
):
    _require_project(project_id)
    if not store.delete_individual(project_id, individual_id):
        raise HTTPException(status_code=404, detail="Особь не найдена")
    reid_service.remove_individual_from_index(
        request.app.state, project_id, individual_id
    )
    return None


@router.get("/projects/{project_id}/gallery/{individual_id}/preview")
def gallery_preview(project_id: str, individual_id: str):
    _require_project(project_id)
    import mimetypes

    one = store.first_image_path(project_id, individual_id)
    if one is None:
        raise HTTPException(status_code=404, detail="Фото не найдено")
    mime, _ = mimetypes.guess_type(str(one))
    return FileResponse(one, media_type=mime or "image/jpeg")


@router.get("/projects/{project_id}/gallery/{individual_id}/photos/{filename}")
def gallery_photo(project_id: str, individual_id: str, filename: str):
    _require_project(project_id)
    import mimetypes

    p = store.photo_path(project_id, individual_id, filename)
    if p is None:
        raise HTTPException(status_code=404, detail="Фото не найдено")
    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(p, media_type=mime or "image/jpeg")
