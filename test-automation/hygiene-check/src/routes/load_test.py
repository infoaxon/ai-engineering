"""Load Test routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..checkers.load_test_runner import LoadTestRunner
from ..config import SiteManager
from ..models import CheckType, TriggeredBy
from ..storage import Storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/load-test")
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None
_checker: Optional[LoadTestRunner] = None

SCENARIOS = ["smoke", "load", "stress", "soak", "spike"]


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage, _checker
    _templates = templates
    _site_manager = site_manager
    _storage = storage
    _checker = LoadTestRunner(storage, site_manager.tools, timeout=3600)


@router.get("/")
async def load_test_index(request: Request, run_id: Optional[int] = None):
    """Load test page - scenario selector + results."""
    run = None
    result = None
    summary = {}

    if run_id:
        run = await _storage.get_run(run_id)
        result = await _storage.get_load_test_result(run_id)
        if run and run.summary_json:
            try:
                summary = json.loads(run.summary_json)
            except (json.JSONDecodeError, TypeError):
                pass

    recent_runs = await _storage.get_recent_runs(limit=10, check_type=CheckType.LOAD_TEST)

    return _templates.TemplateResponse("load_test/index.html", {
        "request": request,
        "run": run,
        "result": result,
        "summary": summary,
        "recent_runs": recent_runs,
        "scenarios": SCENARIOS,
        "sites": _site_manager.get_all_sites(),
    })


@router.post("/run")
async def run_load_test(request: Request, url: str = Form(""),
                         site_id: str = Form("default"),
                         environment: str = Form("production"),
                         scenario: str = Form("smoke"),
                         host: Optional[str] = Form(None),
                         users: Optional[int] = Form(None),
                         duration: Optional[int] = Form(None)):
    """Trigger load test (async — run in background, redirect immediately)."""
    target_url = url
    if not target_url and site_id != "default":
        site_config = _site_manager.get_site(site_id)
        if site_config:
            target_url = site_config.get_url(environment) or ""

    run_id = await _checker.prepare_run(target_url, site_id, environment)

    async def _bg():
        try:
            await _checker.execute_run(
                run_id, target_url,
                scenario=scenario, host=host, users=users, duration=duration
            )
        except Exception as e:
            logger.error(f"Load test {run_id} failed: {e}")

    asyncio.create_task(_bg())
    return RedirectResponse(f"/load-test/?run_id={run_id}", status_code=303)


@router.get("/compare")
async def load_test_compare(request: Request, run1: Optional[int] = None, run2: Optional[int] = None):
    """Compare two load test runs side-by-side."""
    result1 = result2 = None
    run1_data = run2_data = None

    if run1:
        run1_data = await _storage.get_run(run1)
        result1 = await _storage.get_load_test_result(run1)
    if run2:
        run2_data = await _storage.get_run(run2)
        result2 = await _storage.get_load_test_result(run2)

    recent_runs = await _storage.get_recent_runs(limit=20, check_type=CheckType.LOAD_TEST)

    return _templates.TemplateResponse("load_test/compare.html", {
        "request": request,
        "run1": run1_data,
        "run2": run2_data,
        "result1": result1,
        "result2": result2,
        "recent_runs": recent_runs,
    })
