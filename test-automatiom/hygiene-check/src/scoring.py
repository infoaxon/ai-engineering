"""Composite Site Health Score calculation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .models import CheckType, SiteHealthScore
from .storage import Storage

logger = logging.getLogger(__name__)


class HealthScorer:
    """Calculates composite site health scores from individual check results."""

    def __init__(self, storage: Storage, weights: dict[str, float]):
        self.storage = storage
        self.weights = weights

    async def calculate_score(self, site_id: str, environment: str) -> SiteHealthScore:
        """Calculate and store composite health score from latest check results."""
        scores: dict[str, Optional[float]] = {}

        for check_type in CheckType:
            run = await self.storage.get_latest_run(check_type, site_id, environment)
            if run and run.score is not None:
                scores[check_type.value] = run.score
            else:
                scores[check_type.value] = None

        # Calculate weighted composite
        total_weight = 0.0
        weighted_sum = 0.0

        for check_name, weight in self.weights.items():
            score = scores.get(check_name)
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0

        health_score = SiteHealthScore(
            site_id=site_id,
            environment=environment,
            timestamp=datetime.utcnow(),
            html_score=scores.get("html_validation"),
            seo_score=scores.get("seo_audit"),
            lighthouse_score=scores.get("lighthouse"),
            broken_links_score=scores.get("broken_links"),
            load_test_score=scores.get("load_test"),
            composite_score=round(composite, 1),
        )

        await self.storage.store_health_score(health_score)
        return health_score

    @staticmethod
    def get_grade(score: float) -> str:
        if score >= 90:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "NEEDS WORK"
        else:
            return "CRITICAL"

    @staticmethod
    def get_color(score: float) -> str:
        if score >= 80:
            return "#198754"  # green
        elif score >= 60:
            return "#ffc107"  # yellow
        else:
            return "#dc3545"  # red
