"""Parser for Lighthouse JSON output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..models import LighthouseScore, LighthouseAudit

logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "performance": "performance",
    "accessibility": "accessibility",
    "best-practices": "best_practices",
    "seo": "seo",
}


def parse_lighthouse_json(json_path: str, url: str = "") -> dict:
    """Parse a Lighthouse JSON report file.

    Returns dict with keys: scores (LighthouseScore), audits (list[LighthouseAudit])
    """
    path = Path(json_path)
    if not path.exists():
        logger.error(f"Lighthouse JSON not found: {json_path}")
        return {"scores": None, "audits": []}

    with open(path) as f:
        data = json.load(f)

    target_url = data.get("requestedUrl", data.get("finalUrl", url))

    # Extract category scores
    categories = data.get("categories", {})
    score_kwargs = {"url": target_url, "report_json_path": str(json_path)}

    for lh_key, model_key in CATEGORY_MAP.items():
        cat = categories.get(lh_key, {})
        raw_score = cat.get("score")
        score_kwargs[model_key] = round(raw_score * 100, 1) if raw_score is not None else None

    scores = LighthouseScore(**score_kwargs)

    # Extract individual audits
    audits_data = data.get("audits", {})
    audits = []

    # Map audit IDs to categories
    audit_category_map = {}
    for cat_key, cat_data in categories.items():
        for audit_ref in cat_data.get("auditRefs", []):
            aid = audit_ref.get("id")
            if aid:
                audit_category_map[aid] = cat_key

    for audit_id, audit_data in audits_data.items():
        raw_score = audit_data.get("score")
        # Only include audits that have a score and aren't perfect (or are informative)
        if raw_score is not None and raw_score < 1.0:
            audits.append(LighthouseAudit(
                audit_id=audit_id,
                title=audit_data.get("title", ""),
                category=audit_category_map.get(audit_id, "other"),
                score=round(raw_score * 100, 1) if raw_score is not None else None,
                description=audit_data.get("description", "")[:500],
            ))

    return {"scores": scores, "audits": audits}


def calculate_lighthouse_score(scores: LighthouseScore) -> float:
    """Calculate average of non-null Lighthouse category scores."""
    values = [v for v in [scores.performance, scores.accessibility, scores.best_practices, scores.seo] if v is not None]
    return round(sum(values) / len(values), 1) if values else 0.0
