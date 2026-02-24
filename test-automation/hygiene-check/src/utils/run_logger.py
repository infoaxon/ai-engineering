"""In-memory log store for live check progress tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogEntry:
    timestamp: float
    message: str
    level: str = "info"  # info, success, warning, error, progress


@dataclass
class RunLog:
    run_id: int
    check_type: str = ""
    url: str = ""
    status: str = "running"  # running, completed, failed
    started_at: float = 0.0
    completed_at: Optional[float] = None
    entries: list[LogEntry] = field(default_factory=list)
    progress_pct: int = 0

    def add(self, message: str, level: str = "info"):
        self.entries.append(LogEntry(timestamp=time.time(), message=message, level=level))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "check_type": self.check_type,
            "url": self.url,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "elapsed_seconds": round((self.completed_at or time.time()) - self.started_at, 1),
            "entries": [
                {"message": e.message, "level": e.level, "timestamp": e.timestamp}
                for e in self.entries
            ],
        }


class RunLogStore:
    """Thread-safe in-memory store for active run logs."""

    def __init__(self, max_runs: int = 100):
        self._logs: dict[int, RunLog] = {}
        self._max_runs = max_runs

    def create(self, run_id: int, check_type: str, url: str) -> RunLog:
        # Evict oldest if at capacity
        if len(self._logs) >= self._max_runs:
            oldest_key = min(self._logs, key=lambda k: self._logs[k].started_at)
            del self._logs[oldest_key]

        log = RunLog(run_id=run_id, check_type=check_type, url=url, started_at=time.time())
        log.add(f"Starting {check_type.replace('_', ' ')} check...", "info")
        self._logs[run_id] = log
        return log

    def get(self, run_id: int) -> Optional[RunLog]:
        return self._logs.get(run_id)

    def complete(self, run_id: int, status: str = "completed"):
        log = self._logs.get(run_id)
        if log:
            log.status = status
            log.completed_at = time.time()
            log.progress_pct = 100 if status == "completed" else log.progress_pct
            elapsed = round(log.completed_at - log.started_at, 1)
            if status == "completed":
                log.add(f"Check completed in {elapsed}s", "success")
            else:
                log.add(f"Check failed after {elapsed}s", "error")

    def get_entries_since(self, run_id: int, since_index: int = 0) -> dict:
        """Get new log entries since a given index (for polling)."""
        log = self._logs.get(run_id)
        if not log:
            return {"status": "not_found", "entries": []}

        new_entries = log.entries[since_index:]
        return {
            "run_id": run_id,
            "status": log.status,
            "progress_pct": log.progress_pct,
            "total_entries": len(log.entries),
            "entries": [
                {"message": e.message, "level": e.level}
                for e in new_entries
            ],
        }


# Global singleton
run_log_store = RunLogStore()
