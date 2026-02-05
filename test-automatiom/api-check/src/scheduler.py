"""APScheduler setup for periodic health checks."""

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .checker import check_environment
from .config import CustomerManager
from .storage import Storage

logger = logging.getLogger(__name__)


class HealthCheckScheduler:
    """Manages scheduled health checks for all customers and environments."""

    def __init__(self, customer_manager: CustomerManager, storage: Storage):
        self.customer_manager = customer_manager
        self.storage = storage
        self.scheduler = AsyncIOScheduler()
        self._running = False

    async def run_all_checks(self) -> None:
        """Run health checks for all customers and environments."""
        configs = self.customer_manager.get_all_configs()

        logger.info(f"Starting health check cycle at {datetime.utcnow().isoformat()}")

        for customer_id, config in configs.items():
            settings = config.settings

            for env_key, env_config in config.environments.items():
                if not env_config.apis:
                    continue

                try:
                    results = await check_environment(
                        env_key,
                        env_config.apis,
                        customer_id=customer_id,
                        default_timeout=settings.default_timeout_seconds,
                        default_latency_threshold=settings.latency_threshold_ms
                    )

                    await self.storage.store_results(results)

                    healthy = sum(1 for r in results if r.status.value == 'healthy')
                    degraded = sum(1 for r in results if r.status.value == 'degraded')
                    unhealthy = sum(1 for r in results if r.status.value == 'unhealthy')

                    logger.info(
                        f"Customer '{customer_id}' Environment '{env_key}': "
                        f"{healthy} healthy, {degraded} degraded, {unhealthy} unhealthy"
                    )

                except Exception as e:
                    logger.error(f"Error checking customer '{customer_id}' environment '{env_key}': {e}")

    def _sync_run_checks(self) -> None:
        """Synchronous wrapper for running checks (for APScheduler)."""
        try:
            loop = asyncio.get_running_loop()
            # If there's a running loop, schedule the task
            asyncio.ensure_future(self.run_all_checks(), loop=loop)
        except RuntimeError:
            # No running loop, create a new one
            asyncio.run(self.run_all_checks())

    def start(self, interval_minutes: int = 5) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self.scheduler.add_job(
            self._sync_run_checks,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id='health_check_job',
            name='API Health Checks',
            replace_existing=True
        )

        self.scheduler.add_job(
            self._sync_cleanup,
            trigger=IntervalTrigger(hours=24),
            id='cleanup_job',
            name='Cleanup Old Records',
            replace_existing=True
        )

        self.scheduler.start()
        self._running = True

        logger.info(f"Scheduler started. Checks every {interval_minutes} minutes.")

    def _sync_cleanup(self) -> None:
        """Synchronous wrapper for cleanup."""
        async def cleanup():
            deleted = await self.storage.cleanup_old_records(7)
            logger.info(f"Cleaned up {deleted} old records")

        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(cleanup(), loop=loop)
        except RuntimeError:
            asyncio.run(cleanup())

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown()
        self._running = False
        logger.info("Scheduler stopped.")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
