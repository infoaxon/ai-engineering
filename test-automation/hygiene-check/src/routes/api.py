"""JSON API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..checkers.html_validator import HTMLValidator
from ..checkers.seo_auditor import SEOAuditor
from ..checkers.lighthouse_runner import LighthouseRunner
from ..checkers.broken_link_checker import BrokenLinkChecker
from ..checkers.load_test_runner import LoadTestRunner
from ..config import SiteManager
from ..models import CheckType, TriggeredBy
from ..scoring import HealthScorer
from ..storage import Storage
from ..utils.run_logger import run_log_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None
_checkers: dict = {}


class RunRequest(BaseModel):
    url: str
    site_id: str = "default"
    environment: str = "production"


class LoadTestRequest(RunRequest):
    scenario: str = "smoke"
    host: Optional[str] = None
    users: Optional[int] = None
    duration: Optional[int] = None


def setup(site_manager: SiteManager, storage: Storage):
    global _site_manager, _storage, _checkers
    _site_manager = site_manager
    _storage = storage
    _checkers = {
        "html_validation": HTMLValidator(storage, site_manager.tools, site_manager.get_timeout()),
        "seo_audit": SEOAuditor(storage, site_manager.tools, site_manager.get_timeout()),
        "lighthouse": LighthouseRunner(storage, site_manager.tools, site_manager.get_reports_dir(), site_manager.get_timeout()),
        "broken_links": BrokenLinkChecker(storage, site_manager.tools, site_manager.get_timeout()),
        "load_test": LoadTestRunner(storage, site_manager.tools, timeout=3600),
    }


async def _execute_check_background(checker, run_id, url, **kwargs):
    """Execute a prepared check in the background."""
    try:
        await checker.execute_run(run_id, url, **kwargs)
    except Exception as e:
        logger.error(f"Background check {run_id} failed: {e}")


@router.get("/status")
async def api_status():
    """Overall status of all sites."""
    sites = _site_manager.get_all_sites()
    result = []
    for site in sites:
        for env in site.get_environment_names():
            site_status = {"site_id": site.site_id, "name": site.name, "environment": env, "checks": {}}
            for ct in CheckType:
                latest = await _storage.get_latest_run(ct, site.site_id, env)
                site_status["checks"][ct.value] = {
                    "status": latest.status.value if latest else "never_run",
                    "score": latest.score if latest else None,
                    "last_run": latest.started_at.isoformat() if latest else None,
                }
            result.append(site_status)
    return result


@router.get("/sites")
async def api_sites():
    """List configured sites."""
    return [
        {
            "site_id": s.site_id,
            "name": s.name,
            "environments": {e: s.get_url(e) for e in s.get_environment_names()},
        }
        for s in _site_manager.get_all_sites()
    ]


@router.get("/scores/{site_id}")
async def api_scores(site_id: str, env: str = ""):
    """Current health scores for a site."""
    site = _site_manager.get_site(site_id)
    if not site:
        raise HTTPException(404, "Site not found")
    environment = env or site.get_environment_names()[0]

    scorer = HealthScorer(_storage, _site_manager.get_weights())
    score = await scorer.calculate_score(site_id, environment)
    return score.model_dump()


@router.get("/scores/{site_id}/history")
async def api_score_history(site_id: str, env: str = "", days: int = 30):
    """Historical health scores."""
    history = await _storage.get_health_score_history(site_id, env, days)
    return [s.model_dump() for s in history]


@router.get("/runs/{run_id}/logs")
async def api_run_logs(run_id: int, since: int = 0):
    """Poll live log entries for a running check."""
    return run_log_store.get_entries_since(run_id, since)


@router.post("/html-validation/run")
async def api_run_html_validation(req: RunRequest):
    """Trigger HTML validation (async — returns immediately with run_id)."""
    checker = _checkers["html_validation"]
    run_id = await checker.prepare_run(req.url, req.site_id, req.environment)
    site_config = _site_manager.get_site(req.site_id)
    config_path = site_config.html_validation.get("config_path") if site_config else None
    asyncio.create_task(_execute_check_background(checker, run_id, req.url, config_path=config_path))
    return {"run_id": run_id, "status": "running"}


@router.post("/seo-audit/run")
async def api_run_seo_audit(req: RunRequest):
    """Trigger SEO audit (async — returns immediately with run_id)."""
    checker = _checkers["seo_audit"]
    run_id = await checker.prepare_run(req.url, req.site_id, req.environment)
    asyncio.create_task(_execute_check_background(checker, run_id, req.url))
    return {"run_id": run_id, "status": "running"}


@router.post("/lighthouse/run")
async def api_run_lighthouse(req: RunRequest):
    """Trigger Lighthouse audit (async — returns immediately with run_id)."""
    checker = _checkers["lighthouse"]
    run_id = await checker.prepare_run(req.url, req.site_id, req.environment)
    asyncio.create_task(_execute_check_background(checker, run_id, req.url))
    return {"run_id": run_id, "status": "running"}


@router.post("/broken-links/run")
async def api_run_broken_links(req: RunRequest):
    """Trigger broken link check (async — returns immediately with run_id)."""
    checker = _checkers["broken_links"]
    run_id = await checker.prepare_run(req.url, req.site_id, req.environment)
    asyncio.create_task(_execute_check_background(checker, run_id, req.url))
    return {"run_id": run_id, "status": "running"}


@router.post("/load-test/run")
async def api_run_load_test(req: LoadTestRequest):
    """Trigger load test (async — returns immediately with run_id)."""
    checker = _checkers["load_test"]
    run_id = await checker.prepare_run(req.url, req.site_id, req.environment)
    asyncio.create_task(_execute_check_background(
        checker, run_id, req.url,
        scenario=req.scenario, host=req.host, users=req.users, duration=req.duration
    ))
    return {"run_id": run_id, "status": "running"}


@router.get("/runs")
async def api_runs(check_type: Optional[str] = None, site_id: Optional[str] = None, limit: int = 20):
    """List check runs."""
    ct = CheckType(check_type) if check_type else None
    runs = await _storage.get_recent_runs(limit=limit, check_type=ct, site_id=site_id)
    return [r.model_dump() for r in runs]


@router.get("/runs/{run_id}")
async def api_run_detail(run_id: int):
    """Get run details + results."""
    run = await _storage.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    detail = run.model_dump()

    # Include type-specific results
    if run.check_type == CheckType.HTML_VALIDATION:
        issues = await _storage.get_html_issues(run_id)
        detail["issues"] = [i.model_dump() for i in issues]
    elif run.check_type == CheckType.SEO_AUDIT:
        items = await _storage.get_seo_items(run_id)
        detail["items"] = [i.model_dump() for i in items]
    elif run.check_type == CheckType.LIGHTHOUSE:
        scores = await _storage.get_lighthouse_scores(run_id)
        detail["scores"] = [s.model_dump() for s in scores]
    elif run.check_type == CheckType.BROKEN_LINKS:
        links = await _storage.get_broken_links(run_id)
        detail["links"] = [l.model_dump() for l in links]
    elif run.check_type == CheckType.LOAD_TEST:
        result = await _storage.get_load_test_result(run_id)
        detail["result"] = result.model_dump() if result else None

    return detail


@router.post("/check-all")
async def api_check_all(site_id: str, env: str = ""):
    """Run all checks for a site."""
    site = _site_manager.get_site(site_id)
    if not site:
        raise HTTPException(404, "Site not found")
    environment = env or site.get_environment_names()[0]
    url = site.get_url(environment)
    if not url:
        raise HTTPException(400, f"No URL for {site_id}/{environment}")

    results = {}
    for name, checker in _checkers.items():
        try:
            run_id, _ = await checker.run(url, site_id=site_id, environment=environment,
                                           triggered_by=TriggeredBy.MANUAL)
            results[name] = {"run_id": run_id, "status": "completed"}
        except Exception as e:
            results[name] = {"status": "failed", "error": str(e)}

    return results


@router.get("/trends/{site_id}/{check_type}")
async def api_trends(site_id: str, check_type: str, env: str = "", limit: int = 20):
    """Trend data for Chart.js."""
    ct = CheckType(check_type)
    site = _site_manager.get_site(site_id)
    environment = env or (site.get_environment_names()[0] if site else "")
    return await _storage.get_score_trend(ct, site_id, environment, limit)
