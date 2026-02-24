"""Dashboard routes - main overview page."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..config import SiteManager
from ..models import CheckType, DashboardData, CheckSummary, RunStatus
from ..scoring import HealthScorer
from ..storage import Storage

router = APIRouter()
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage
    _templates = templates
    _site_manager = site_manager
    _storage = storage


@router.get("/")
@router.get("/dashboard")
async def dashboard(request: Request, site: str = "", env: str = ""):
    """Main dashboard with composite health score and section cards."""
    sites = _site_manager.get_all_sites()
    site_id = site or (_site_manager.get_site_ids()[0] if _site_manager.get_site_ids() else "")
    site_config = _site_manager.get_site(site_id)

    environments = site_config.get_environment_names() if site_config else []
    environment = env or (environments[0] if environments else "")
    site_name = site_config.name if site_config else site_id

    # Build check summaries
    check_summaries = []
    check_meta = [
        (CheckType.HTML_VALIDATION, "HTML Validation", "bi-code-slash"),
        (CheckType.SEO_AUDIT, "SEO Audit", "bi-search"),
        (CheckType.LIGHTHOUSE, "Lighthouse", "bi-speedometer"),
        (CheckType.BROKEN_LINKS, "Broken Links", "bi-link-45deg"),
        (CheckType.LOAD_TEST, "Load Test", "bi-graph-up"),
    ]

    for check_type, name, icon in check_meta:
        latest = await _storage.get_latest_run(check_type, site_id, environment)
        summary_data = {}
        if latest and latest.summary_json:
            try:
                summary_data = json.loads(latest.summary_json)
            except (json.JSONDecodeError, TypeError):
                pass

        check_summaries.append({
            "check_type": check_type.value,
            "name": name,
            "icon": icon,
            "last_run": latest.started_at if latest else None,
            "last_score": latest.score if latest else None,
            "status": latest.status.value if latest else "unknown",
            "pass_count": summary_data.get("pass_count", 0),
            "fail_count": summary_data.get("fail_count", summary_data.get("broken_count", 0)),
            "warn_count": summary_data.get("warn_count", summary_data.get("warning_count", 0)),
        })

    # Get composite score
    scorer = HealthScorer(_storage, _site_manager.get_weights())
    score_history = await _storage.get_health_score_history(site_id, environment, days=30)
    composite_score = score_history[0].composite_score if score_history else 0

    # Recent runs (show all, not filtered by site — most checks are ad-hoc)
    recent_runs = await _storage.get_recent_runs(limit=10)

    return _templates.TemplateResponse("dashboard.html", {
        "request": request,
        "sites": sites,
        "current_site": site_id,
        "current_env": environment,
        "environments": environments,
        "site_name": site_name,
        "composite_score": composite_score,
        "score_color": HealthScorer.get_color(composite_score),
        "score_grade": HealthScorer.get_grade(composite_score),
        "checks": check_summaries,
        "recent_runs": recent_runs,
        "now": datetime.utcnow(),
    })
