"""HTML Validation routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..checkers.html_validator import HTMLValidator
from ..config import SiteManager
from ..models import CheckType, TriggeredBy
from ..storage import Storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/html-validation")
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None
_checker: Optional[HTMLValidator] = None


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage, _checker
    _templates = templates
    _site_manager = site_manager
    _storage = storage
    _checker = HTMLValidator(storage, site_manager.tools, site_manager.get_timeout())


@router.get("/")
async def html_validation_index(request: Request, run_id: Optional[int] = None):
    """HTML validation page - form + results."""
    issues = []
    run = None
    error_count = warning_count = 0

    if run_id:
        run = await _storage.get_run(run_id)
        issues = await _storage.get_html_issues(run_id)
        error_count = sum(1 for i in issues if i.severity.value == "error")
        warning_count = sum(1 for i in issues if i.severity.value == "warn")

    # Get recent runs for history
    recent_runs = await _storage.get_recent_runs(limit=10, check_type=CheckType.HTML_VALIDATION)

    # Rule distribution
    rule_counts = {}
    for issue in issues:
        rule_counts[issue.rule] = rule_counts.get(issue.rule, 0) + 1

    return _templates.TemplateResponse("html_validation/index.html", {
        "request": request,
        "run": run,
        "issues": issues,
        "error_count": error_count,
        "warning_count": warning_count,
        "recent_runs": recent_runs,
        "rule_counts": rule_counts,
        "sites": _site_manager.get_all_sites(),
    })


@router.post("/run")
async def run_html_validation(request: Request, url: str = Form(...),
                               site_id: str = Form("default"),
                               environment: str = Form("production")):
    """Trigger HTML validation for a URL (async — run in background, redirect immediately)."""
    site_config = _site_manager.get_site(site_id)
    config_path = None
    if site_config:
        config_path = site_config.html_validation.get("config_path")

    run_id = await _checker.prepare_run(url, site_id, environment)

    async def _bg():
        try:
            await _checker.execute_run(run_id, url, config_path=config_path)
        except Exception as e:
            logger.error(f"HTML validation {run_id} failed: {e}")

    asyncio.create_task(_bg())
    return RedirectResponse(f"/html-validation/?run_id={run_id}", status_code=303)


@router.get("/history")
async def html_validation_history(request: Request):
    """Historical HTML validation runs."""
    runs = await _storage.get_recent_runs(limit=50, check_type=CheckType.HTML_VALIDATION)
    return _templates.TemplateResponse("html_validation/history.html", {
        "request": request,
        "runs": runs,
    })
