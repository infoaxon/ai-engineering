"""APScheduler for periodic automated checks."""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .config import SiteManager
from .storage import Storage
from .scoring import HealthScorer

logger = logging.getLogger(__name__)


class CheckScheduler:
    """Manages periodic execution of hygiene checks."""

    def __init__(self, site_manager: SiteManager, storage: Storage):
        self.site_manager = site_manager
        self.storage = storage
        self.scorer = HealthScorer(storage, site_manager.get_weights())
        self._scheduler: Optional[AsyncIOScheduler] = None

    def start(self, interval_minutes: int = 60) -> None:
        """Start the scheduler with configured intervals."""
        self._scheduler = AsyncIOScheduler()

        # Periodic cleanup
        self._scheduler.add_job(
            self._cleanup,
            "interval",
            hours=24,
            id="cleanup",
            name="Database Cleanup",
        )

        # Health score recalculation
        self._scheduler.add_job(
            self._recalculate_scores,
            "interval",
            minutes=interval_minutes,
            id="health_scores",
            name="Health Score Recalculation",
        )

        self._scheduler.start()
        logger.info(f"Scheduler started (interval: {interval_minutes}m)")

    def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    async def _cleanup(self) -> None:
        retention = self.site_manager.settings.get("retention_days", 30)
        deleted = await self.storage.cleanup_old_records(retention)
        if deleted:
            logger.info(f"Cleaned up {deleted} old records")

    async def _recalculate_scores(self) -> None:
        """Recalculate health scores for all sites and environments."""
        for site in self.site_manager.get_all_sites():
            for env in site.get_environment_names():
                try:
                    score = await self.scorer.calculate_score(site.site_id, env)
                    logger.info(f"Health score for {site.site_id}/{env}: {score.composite_score}")
                except Exception as e:
                    logger.error(f"Error calculating score for {site.site_id}/{env}: {e}")
