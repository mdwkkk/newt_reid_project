"""HTML-страницы (Jinja2)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from web_app.card_templates import get_template
from web_app.services import store

WEB_APP_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(WEB_APP_DIR / "templates"))

router = APIRouter(tags=["pages"])


def _project_or_404(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return proj


@router.get("/", response_class=HTMLResponse)
async def page_projects(request: Request):
    projects = store.list_projects()
    return templates.TemplateResponse(
        request,
        "projects.html",
        {"projects": projects, "active": "projects"},
    )


@router.get("/projects/{project_id}", response_class=HTMLResponse)
async def page_project(request: Request, project_id: str):
    proj = _project_or_404(project_id)
    return templates.TemplateResponse(
        request,
        "project.html",
        {"project": proj, "active": "project"},
    )


@router.get("/projects/{project_id}/search", response_class=HTMLResponse)
async def page_search(request: Request, project_id: str):
    proj = _project_or_404(project_id)
    return templates.TemplateResponse(
        request,
        "search.html",
        {"project": proj, "active": "search"},
    )


@router.get("/projects/{project_id}/gallery", response_class=HTMLResponse)
async def page_gallery(request: Request, project_id: str):
    proj = _project_or_404(project_id)
    individuals = store.list_individuals(project_id)
    return templates.TemplateResponse(
        request,
        "gallery.html",
        {
            "project": proj,
            "individuals": individuals,
            "active": "gallery",
        },
    )


@router.get("/projects/{project_id}/individuals/new", response_class=HTMLResponse)
async def page_individual_new(request: Request, project_id: str):
    proj = _project_or_404(project_id)
    tpl = get_template(proj["card_template"])
    return templates.TemplateResponse(
        request,
        "individual_form.html",
        {
            "project": proj,
            "template_schema": tpl,
            "individual": None,
            "mode": "create",
            "active": "new",
        },
    )


@router.get("/projects/{project_id}/individuals/{individual_id}", response_class=HTMLResponse)
async def page_individual(request: Request, project_id: str, individual_id: str):
    proj = _project_or_404(project_id)
    ind = store.get_individual(project_id, individual_id)
    if not ind:
        raise HTTPException(status_code=404, detail="Особь не найдена")
    tpl = get_template(proj["card_template"])
    all_ids = sorted(store.list_individual_ids(project_id))
    return templates.TemplateResponse(
        request,
        "individual.html",
        {
            "project": proj,
            "individual": ind,
            "template_schema": tpl,
            "all_individual_ids": all_ids,
            "active": "individual",
        },
    )


@router.get("/legacy", include_in_schema=False)
async def legacy_redirect():
    return RedirectResponse(url="/", status_code=302)
