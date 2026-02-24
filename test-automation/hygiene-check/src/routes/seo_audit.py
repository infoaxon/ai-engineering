"""SEO Audit routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..checkers.seo_auditor import SEOAuditor
from ..config import SiteManager
from ..models import CheckType, TriggeredBy
from ..storage import Storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/seo-audit")
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None
_checker: Optional[SEOAuditor] = None


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage, _checker
    _templates = templates
    _site_manager = site_manager
    _storage = storage
    _checker = SEOAuditor(storage, site_manager.tools, timeout=600)


@router.get("/")
async def seo_audit_index(request: Request, run_id: Optional[int] = None):
    """SEO audit page - form + results."""
    items = []
    run = None
    summary = {}

    if run_id:
        run = await _storage.get_run(run_id)
        items = await _storage.get_seo_items(run_id)
        if run and run.summary_json:
            try:
                summary = json.loads(run.summary_json)
            except (json.JSONDecodeError, TypeError):
                pass

    # Group items by section
    sections = {}
    for item in items:
        key = (item.section_number, item.section_name)
        if key not in sections:
            sections[key] = []
        sections[key].append(item)

    recent_runs = await _storage.get_recent_runs(limit=10, check_type=CheckType.SEO_AUDIT)

    return _templates.TemplateResponse("seo_audit/index.html", {
        "request": request,
        "run": run,
        "sections": sections,
        "summary": summary,
        "recent_runs": recent_runs,
        "sites": _site_manager.get_all_sites(),
    })


@router.post("/run")
async def run_seo_audit(request: Request, url: str = Form(...),
                         site_id: str = Form("default"),
                         environment: str = Form("production"),
                         remote: bool = Form(True)):
    """Trigger SEO audit for a URL (async — run in background, redirect immediately)."""
    run_id = await _checker.prepare_run(url, site_id, environment)

    async def _bg():
        try:
            await _checker.execute_run(run_id, url, remote=remote)
        except Exception as e:
            logger.error(f"SEO audit {run_id} failed: {e}")

    asyncio.create_task(_bg())
    return RedirectResponse(f"/seo-audit/?run_id={run_id}", status_code=303)


@router.get("/compare")
async def seo_compare(request: Request, site_id: str = ""):
    """Compare SEO audit results across environments."""
    site_config = _site_manager.get_site(site_id) if site_id else None
    env_results = {}

    if site_config:
        for env in site_config.get_environment_names():
            latest = await _storage.get_latest_run(CheckType.SEO_AUDIT, site_id, env)
            if latest:
                items = await _storage.get_seo_items(latest.id)
                env_results[env] = {
                    "run": latest,
                    "items": items,
                    "summary": json.loads(latest.summary_json) if latest.summary_json else {},
                }

    return _templates.TemplateResponse("seo_audit/compare.html", {
        "request": request,
        "site": site_config,
        "env_results": env_results,
        "sites": _site_manager.get_all_sites(),
    })
