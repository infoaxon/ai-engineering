"""Lighthouse audit routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..checkers.lighthouse_runner import LighthouseRunner
from ..config import SiteManager
from ..models import CheckType, TriggeredBy
from ..storage import Storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lighthouse")
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None
_checker: Optional[LighthouseRunner] = None


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage, _checker
    _templates = templates
    _site_manager = site_manager
    _storage = storage
    _checker = LighthouseRunner(
        storage, site_manager.tools,
        reports_dir=site_manager.get_reports_dir(),
        timeout=site_manager.get_timeout()
    )


@router.get("/")
async def lighthouse_index(request: Request, run_id: Optional[int] = None):
    """Lighthouse page - form + results with score gauges."""
    scores = []
    audits = []
    run = None

    if run_id:
        run = await _storage.get_run(run_id)
        scores = await _storage.get_lighthouse_scores(run_id)
        if scores:
            audits = await _storage.get_lighthouse_audits(scores[0].id)

    recent_runs = await _storage.get_recent_runs(limit=10, check_type=CheckType.LIGHTHOUSE)

    return _templates.TemplateResponse("lighthouse/index.html", {
        "request": request,
        "run": run,
        "scores": scores,
        "audits": audits,
        "recent_runs": recent_runs,
        "sites": _site_manager.get_all_sites(),
    })


@router.post("/run")
async def run_lighthouse(request: Request, url: str = Form(...),
                          site_id: str = Form("default"),
                          environment: str = Form("production"),
                          categories: list[str] = Form(["seo", "accessibility", "best-practices"])):
    """Trigger Lighthouse audit (async — run in background, redirect immediately)."""
    run_id = await _checker.prepare_run(url, site_id, environment)

    async def _bg():
        try:
            await _checker.execute_run(run_id, url, categories=categories)
        except Exception as e:
            logger.error(f"Lighthouse {run_id} failed: {e}")

    asyncio.create_task(_bg())
    return RedirectResponse(f"/lighthouse/?run_id={run_id}", status_code=303)


@router.get("/trends")
async def lighthouse_trends(request: Request, site_id: str = "", env: str = ""):
    """Score trends over time."""
    trend_data = []
    if site_id and env:
        trend_data = await _storage.get_score_trend(CheckType.LIGHTHOUSE, site_id, env)

    return _templates.TemplateResponse("lighthouse/trends.html", {
        "request": request,
        "trend_data": json.dumps(trend_data),
        "site_id": site_id,
        "env": env,
        "sites": _site_manager.get_all_sites(),
    })
