"""Abstract base class for all checkers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from ..models import CheckRun, CheckType, RunStatus, TriggeredBy
from ..storage import Storage
from ..utils.run_logger import RunLog, run_log_store

logger = logging.getLogger(__name__)


class BaseChecker(ABC):
    """Abstract base for hygiene checkers."""

    check_type: CheckType

    def __init__(self, storage: Storage, tools_config: dict, timeout: int = 300):
        self.storage = storage
        self.tools_config = tools_config
        self.timeout = timeout

    @abstractmethod
    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> Any:
        """Execute the check and return parsed results."""
        ...

    async def prepare_run(self, url: str, site_id: str = "default", environment: str = "production",
                          triggered_by: TriggeredBy = TriggeredBy.MANUAL) -> int:
        """Create a run record and live log. Returns run_id immediately."""
        run = CheckRun(
            check_type=self.check_type,
            site_id=site_id,
            environment=environment,
            target_url=url,
            started_at=datetime.utcnow(),
            triggered_by=triggered_by,
        )
        run_id = await self.storage.create_run(run)
        run_log = run_log_store.create(run_id, self.check_type.value, url)
        run_log.add(f"Target URL: {url}", "info")
        run_log.add(f"Environment: {environment}", "info")
        run_log.progress_pct = 10
        return run_id

    async def execute_run(self, run_id: int, url: str, **kwargs) -> Any:
        """Execute a previously prepared run. Called from background task."""
        run_log = run_log_store.get(run_id)
        try:
            result = await self.execute(url, run_log=run_log, **kwargs)

            if run_log:
                run_log.add("Calculating score...", "info")
                run_log.progress_pct = 80
            score = self._calculate_score(result)
            summary = self._build_summary(result)

            if run_log:
                run_log.add("Storing results...", "info")
                run_log.progress_pct = 90
            await self._store_results(run_id, result)
            await self.storage.complete_run(run_id, score=score, summary=summary)

            if run_log:
                run_log.add(f"Score: {score:.1f}", "success")
                run_log_store.complete(run_id, "completed")
            return result
        except Exception as e:
            logger.error(f"{self.check_type.value} check failed: {e}")
            if run_log:
                run_log.add(f"Error: {e}", "error")
                run_log_store.complete(run_id, "failed")
            await self.storage.complete_run(
                run_id, status=RunStatus.FAILED, error_message=str(e)
            )
            raise

    async def run(self, url: str, site_id: str = "default", environment: str = "production",
                  triggered_by: TriggeredBy = TriggeredBy.MANUAL, **kwargs) -> tuple[int, Any]:
        """Full run pipeline: create run -> execute -> store results -> complete run."""
        run_id = await self.prepare_run(url, site_id, environment, triggered_by)
        result = await self.execute_run(run_id, url, **kwargs)
        return run_id, result

    @abstractmethod
    def _calculate_score(self, result: Any) -> float:
        ...

    @abstractmethod
    def _build_summary(self, result: Any) -> dict:
        ...

    @abstractmethod
    async def _store_results(self, run_id: int, result: Any) -> None:
        ...
