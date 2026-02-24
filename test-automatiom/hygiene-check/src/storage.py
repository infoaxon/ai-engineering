"""SQLite storage for hygiene check results."""

from __future__ import annotations

import json
import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from .models import (
    CheckRun, CheckType, RunStatus, TriggeredBy, Severity, CheckStatus,
    LinkState, SLAVerdict,
    HTMLValidationIssue, SEOAuditItem, LighthouseScore, LighthouseAudit,
    BrokenLinkItem, LoadTestResult, SiteHealthScore,
)

# ---------- helpers ----------

def _row_to_check_run(row) -> CheckRun:
    return CheckRun(
        id=row["id"],
        check_type=CheckType(row["check_type"]),
        site_id=row["site_id"],
        environment=row["environment"],
        target_url=row["target_url"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        status=RunStatus(row["status"]),
        score=row["score"],
        summary_json=row["summary_json"],
        report_path=row["report_path"],
        error_message=row["error_message"],
        triggered_by=TriggeredBy(row["triggered_by"]),
    )


class Storage:
    """Async SQLite storage for all hygiene check data."""

    def __init__(self, db_path: Union[str, Path] = "hygiene_checks.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    # ========== Initialization ==========

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS check_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_type TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    score REAL,
                    summary_json TEXT,
                    report_path TEXT,
                    error_message TEXT,
                    triggered_by TEXT NOT NULL DEFAULT 'manual'
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS html_validation_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    url TEXT,
                    line_number INTEGER,
                    column_number INTEGER,
                    severity TEXT NOT NULL,
                    message TEXT,
                    rule TEXT,
                    FOREIGN KEY (run_id) REFERENCES check_runs(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS seo_audit_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    section_number INTEGER,
                    section_name TEXT,
                    status TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    FOREIGN KEY (run_id) REFERENCES check_runs(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS lighthouse_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    url TEXT,
                    performance REAL,
                    accessibility REAL,
                    best_practices REAL,
                    seo REAL,
                    report_json_path TEXT,
                    report_html_path TEXT,
                    FOREIGN KEY (run_id) REFERENCES check_runs(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS lighthouse_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lighthouse_id INTEGER NOT NULL,
                    audit_id TEXT,
                    title TEXT,
                    category TEXT,
                    score REAL,
                    description TEXT,
                    FOREIGN KEY (lighthouse_id) REFERENCES lighthouse_scores(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS broken_link_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    url TEXT,
                    status_code INTEGER,
                    link_state TEXT NOT NULL,
                    source_page TEXT,
                    response_ms REAL,
                    FOREIGN KEY (run_id) REFERENCES check_runs(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS load_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    scenario TEXT,
                    target_host TEXT,
                    users INTEGER,
                    duration_seconds INTEGER,
                    total_requests INTEGER,
                    error_rate REAL,
                    throughput REAL,
                    p95_response_ms REAL,
                    p99_response_ms REAL,
                    avg_response_ms REAL,
                    min_response_ms REAL,
                    max_response_ms REAL,
                    sla_overall TEXT,
                    FOREIGN KEY (run_id) REFERENCES check_runs(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS site_health_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    html_score REAL,
                    seo_score REAL,
                    lighthouse_score REAL,
                    broken_links_score REAL,
                    load_test_score REAL,
                    composite_score REAL NOT NULL DEFAULT 0
                )
            """)

            # Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_type_site ON check_runs(check_type, site_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON check_runs(started_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_html_issues_run ON html_validation_issues(run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_seo_items_run ON seo_audit_items(run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_lh_scores_run ON lighthouse_scores(run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_broken_links_run ON broken_link_results(run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_load_test_run ON load_test_results(run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_health_scores_site ON site_health_scores(site_id, environment)")

            await db.commit()
        self._initialized = True

    # ========== Check Runs ==========

    async def create_run(self, run: CheckRun) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO check_runs (check_type, site_id, environment, target_url, started_at, status, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run.check_type.value, run.site_id, run.environment, run.target_url,
                  run.started_at.isoformat(), run.status.value, run.triggered_by.value))
            await db.commit()
            return cursor.lastrowid

    async def complete_run(self, run_id: int, score: Optional[float] = None,
                           summary: Optional[dict] = None, report_path: Optional[str] = None,
                           error_message: Optional[str] = None, status: RunStatus = RunStatus.COMPLETED) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE check_runs SET completed_at = ?, status = ?, score = ?,
                    summary_json = ?, report_path = ?, error_message = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), status.value, score,
                  json.dumps(summary) if summary else None, report_path, error_message, run_id))
            await db.commit()

    async def get_run(self, run_id: int) -> Optional[CheckRun]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM check_runs WHERE id = ?", (run_id,)) as cur:
                row = await cur.fetchone()
                return _row_to_check_run(row) if row else None

    async def get_recent_runs(self, limit: int = 20, check_type: Optional[CheckType] = None,
                              site_id: Optional[str] = None) -> list[CheckRun]:
        await self.initialize()
        query = "SELECT * FROM check_runs WHERE 1=1"
        params: list = []
        if check_type:
            query += " AND check_type = ?"
            params.append(check_type.value)
        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [_row_to_check_run(r) for r in rows]

    async def get_latest_run(self, check_type: CheckType, site_id: str,
                             environment: str) -> Optional[CheckRun]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM check_runs
                WHERE check_type = ? AND site_id = ? AND environment = ?
                ORDER BY started_at DESC LIMIT 1
            """, (check_type.value, site_id, environment)) as cur:
                row = await cur.fetchone()
                return _row_to_check_run(row) if row else None

    # ========== HTML Validation ==========

    async def store_html_issues(self, run_id: int, issues: list[HTMLValidationIssue]) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO html_validation_issues (run_id, url, line_number, column_number, severity, message, rule)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(run_id, i.url, i.line_number, i.column_number, i.severity.value, i.message, i.rule) for i in issues])
            await db.commit()

    async def get_html_issues(self, run_id: int) -> list[HTMLValidationIssue]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM html_validation_issues WHERE run_id = ? ORDER BY line_number", (run_id,)) as cur:
                rows = await cur.fetchall()
                return [HTMLValidationIssue(
                    id=r["id"], run_id=r["run_id"], url=r["url"],
                    line_number=r["line_number"], column_number=r["column_number"],
                    severity=Severity(r["severity"]), message=r["message"], rule=r["rule"]
                ) for r in rows]

    # ========== SEO Audit ==========

    async def store_seo_items(self, run_id: int, items: list[SEOAuditItem]) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO seo_audit_items (run_id, section_number, section_name, status, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(run_id, i.section_number, i.section_name, i.status.value, i.message, i.details) for i in items])
            await db.commit()

    async def get_seo_items(self, run_id: int) -> list[SEOAuditItem]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM seo_audit_items WHERE run_id = ? ORDER BY section_number, id", (run_id,)) as cur:
                rows = await cur.fetchall()
                return [SEOAuditItem(
                    id=r["id"], run_id=r["run_id"], section_number=r["section_number"],
                    section_name=r["section_name"], status=CheckStatus(r["status"]),
                    message=r["message"], details=r["details"]
                ) for r in rows]

    # ========== Lighthouse ==========

    async def store_lighthouse_scores(self, run_id: int, scores: list[LighthouseScore]) -> list[int]:
        await self.initialize()
        ids = []
        async with aiosqlite.connect(self.db_path) as db:
            for s in scores:
                cursor = await db.execute("""
                    INSERT INTO lighthouse_scores (run_id, url, performance, accessibility, best_practices, seo, report_json_path, report_html_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (run_id, s.url, s.performance, s.accessibility, s.best_practices, s.seo, s.report_json_path, s.report_html_path))
                ids.append(cursor.lastrowid)
            await db.commit()
        return ids

    async def store_lighthouse_audits(self, audits: list[LighthouseAudit]) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO lighthouse_audits (lighthouse_id, audit_id, title, category, score, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(a.lighthouse_id, a.audit_id, a.title, a.category, a.score, a.description) for a in audits])
            await db.commit()

    async def get_lighthouse_scores(self, run_id: int) -> list[LighthouseScore]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM lighthouse_scores WHERE run_id = ?", (run_id,)) as cur:
                rows = await cur.fetchall()
                return [LighthouseScore(
                    id=r["id"], run_id=r["run_id"], url=r["url"],
                    performance=r["performance"], accessibility=r["accessibility"],
                    best_practices=r["best_practices"], seo=r["seo"],
                    report_json_path=r["report_json_path"], report_html_path=r["report_html_path"]
                ) for r in rows]

    async def get_lighthouse_audits(self, lighthouse_id: int) -> list[LighthouseAudit]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM lighthouse_audits WHERE lighthouse_id = ?", (lighthouse_id,)) as cur:
                rows = await cur.fetchall()
                return [LighthouseAudit(
                    id=r["id"], lighthouse_id=r["lighthouse_id"], audit_id=r["audit_id"],
                    title=r["title"], category=r["category"], score=r["score"], description=r["description"]
                ) for r in rows]

    # ========== Broken Links ==========

    async def store_broken_links(self, run_id: int, links: list[BrokenLinkItem]) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO broken_link_results (run_id, url, status_code, link_state, source_page, response_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(run_id, l.url, l.status_code, l.link_state.value, l.source_page, l.response_ms) for l in links])
            await db.commit()

    async def get_broken_links(self, run_id: int) -> list[BrokenLinkItem]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM broken_link_results WHERE run_id = ? ORDER BY link_state, url", (run_id,)) as cur:
                rows = await cur.fetchall()
                return [BrokenLinkItem(
                    id=r["id"], run_id=r["run_id"], url=r["url"],
                    status_code=r["status_code"], link_state=LinkState(r["link_state"]),
                    source_page=r["source_page"], response_ms=r["response_ms"]
                ) for r in rows]

    # ========== Load Test ==========

    async def store_load_test_result(self, run_id: int, result: LoadTestResult) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO load_test_results (run_id, scenario, target_host, users, duration_seconds,
                    total_requests, error_rate, throughput, p95_response_ms, p99_response_ms,
                    avg_response_ms, min_response_ms, max_response_ms, sla_overall)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, result.scenario, result.target_host, result.users, result.duration_seconds,
                  result.total_requests, result.error_rate, result.throughput, result.p95_response_ms,
                  result.p99_response_ms, result.avg_response_ms, result.min_response_ms,
                  result.max_response_ms, result.sla_overall.value))
            await db.commit()
            return cursor.lastrowid

    async def get_load_test_result(self, run_id: int) -> Optional[LoadTestResult]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM load_test_results WHERE run_id = ?", (run_id,)) as cur:
                r = await cur.fetchone()
                if not r:
                    return None
                return LoadTestResult(
                    id=r["id"], run_id=r["run_id"], scenario=r["scenario"],
                    target_host=r["target_host"], users=r["users"],
                    duration_seconds=r["duration_seconds"], total_requests=r["total_requests"],
                    error_rate=r["error_rate"], throughput=r["throughput"],
                    p95_response_ms=r["p95_response_ms"], p99_response_ms=r["p99_response_ms"],
                    avg_response_ms=r["avg_response_ms"], min_response_ms=r["min_response_ms"],
                    max_response_ms=r["max_response_ms"], sla_overall=SLAVerdict(r["sla_overall"])
                )

    # ========== Site Health Scores ==========

    async def store_health_score(self, score: SiteHealthScore) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO site_health_scores (site_id, environment, timestamp,
                    html_score, seo_score, lighthouse_score, broken_links_score, load_test_score, composite_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (score.site_id, score.environment, score.timestamp.isoformat(),
                  score.html_score, score.seo_score, score.lighthouse_score,
                  score.broken_links_score, score.load_test_score, score.composite_score))
            await db.commit()
            return cursor.lastrowid

    async def get_health_score_history(self, site_id: str, environment: str,
                                       days: int = 30) -> list[SiteHealthScore]:
        await self.initialize()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM site_health_scores
                WHERE site_id = ? AND environment = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (site_id, environment, cutoff)) as cur:
                rows = await cur.fetchall()
                return [SiteHealthScore(
                    id=r["id"], site_id=r["site_id"], environment=r["environment"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    html_score=r["html_score"], seo_score=r["seo_score"],
                    lighthouse_score=r["lighthouse_score"], broken_links_score=r["broken_links_score"],
                    load_test_score=r["load_test_score"], composite_score=r["composite_score"]
                ) for r in rows]

    # ========== Trends ==========

    async def get_score_trend(self, check_type: CheckType, site_id: str,
                              environment: str, limit: int = 20) -> list[dict]:
        """Get score trend data for Chart.js."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT started_at, score FROM check_runs
                WHERE check_type = ? AND site_id = ? AND environment = ? AND score IS NOT NULL
                ORDER BY started_at DESC LIMIT ?
            """, (check_type.value, site_id, environment, limit)) as cur:
                rows = await cur.fetchall()
                return [{"timestamp": r["started_at"], "score": r["score"]} for r in reversed(rows)]

    # ========== Cleanup ==========

    async def cleanup_old_records(self, retention_days: int = 30) -> int:
        await self.initialize()
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        total_deleted = 0
        async with aiosqlite.connect(self.db_path) as db:
            # Get old run IDs
            async with db.execute("SELECT id FROM check_runs WHERE started_at < ?", (cutoff,)) as cur:
                old_ids = [row[0] async for row in cur]

            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                for table in ["html_validation_issues", "seo_audit_items", "broken_link_results", "load_test_results"]:
                    cursor = await db.execute(f"DELETE FROM {table} WHERE run_id IN ({placeholders})", old_ids)
                    total_deleted += cursor.rowcount

                # lighthouse_audits via lighthouse_scores
                async with db.execute(f"SELECT id FROM lighthouse_scores WHERE run_id IN ({placeholders})", old_ids) as cur:
                    lh_ids = [row[0] async for row in cur]
                if lh_ids:
                    lh_ph = ",".join("?" * len(lh_ids))
                    cursor = await db.execute(f"DELETE FROM lighthouse_audits WHERE lighthouse_id IN ({lh_ph})", lh_ids)
                    total_deleted += cursor.rowcount
                cursor = await db.execute(f"DELETE FROM lighthouse_scores WHERE run_id IN ({placeholders})", old_ids)
                total_deleted += cursor.rowcount

                cursor = await db.execute(f"DELETE FROM check_runs WHERE id IN ({placeholders})", old_ids)
                total_deleted += cursor.rowcount

            # Health scores
            cursor = await db.execute("DELETE FROM site_health_scores WHERE timestamp < ?", (cutoff,))
            total_deleted += cursor.rowcount

            await db.commit()
        return total_deleted
