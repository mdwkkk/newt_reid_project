"""API проектов и шаблонов карточек."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from web_app.card_templates import get_template, is_valid_template, list_templates
from web_app.schemas import ProjectCreate, ProjectOut
from web_app.services import reid_service, store

router = APIRouter(prefix="/api", tags=["projects"])


@router.get("/card-templates")
def api_list_card_templates():
    return {"templates": list_templates()}


@router.get("/card-templates/{code}")
def api_get_card_template(code: str):
    tpl = get_template(code)
    if not tpl:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    return tpl


@router.get("/projects", response_model=list[ProjectOut])
def api_list_projects():
    return store.list_projects()


@router.post("/projects", response_model=ProjectOut, status_code=201)
def api_create_project(body: ProjectCreate):
    if not is_valid_template(body.card_template):
        raise HTTPException(status_code=400, detail="Недопустимый шаблон карточки")
    return store.create_project(body.name, body.card_template)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def api_get_project(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return proj


@router.delete("/projects/{project_id}", status_code=204)
def api_delete_project(project_id: str, request: Request):
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")
    reid_service.invalidate_engine(request.app.state, project_id)
    return None
