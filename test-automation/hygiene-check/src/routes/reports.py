"""Reports browser routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..config import SiteManager
from ..models import CheckType
from ..storage import Storage

router = APIRouter(prefix="/reports")
_templates: Optional[Jinja2Templates] = None
_site_manager: Optional[SiteManager] = None
_storage: Optional[Storage] = None


def setup(templates: Jinja2Templates, site_manager: SiteManager, storage: Storage):
    global _templates, _site_manager, _storage
    _templates = templates
    _site_manager = site_manager
    _storage = storage


@router.get("/")
async def reports_index(request: Request, check_type: Optional[str] = None):
    """Browse all reports across all types."""
    ct = CheckType(check_type) if check_type else None
    runs = await _storage.get_recent_runs(limit=100, check_type=ct)

    # Group by type
    by_type = {}
    for run in runs:
        if run.check_type.value not in by_type:
            by_type[run.check_type.value] = []
        by_type[run.check_type.value].append(run)

    check_types = [
        ("html_validation", "HTML Validation"),
        ("seo_audit", "SEO Audit"),
        ("lighthouse", "Lighthouse"),
        ("broken_links", "Broken Links"),
        ("load_test", "Load Test"),
    ]

    return _templates.TemplateResponse("reports/index.html", {
        "request": request,
        "runs": runs,
        "by_type": by_type,
        "check_types": check_types,
        "active_filter": check_type,
    })
