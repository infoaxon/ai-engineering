"""SQLite storage for health check results."""

import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path

from .models import HealthCheckResult, HealthStatus, APIStatus


class Storage:
    """SQLite storage for health check history."""

    def __init__(self, db_path: str | Path = "health_checks.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL DEFAULT 'default',
                    api_name TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL,
                    error_message TEXT,
                    response_valid INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_health_checks_customer_env_api
                ON health_checks(customer_id, environment, api_name)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_health_checks_timestamp
                ON health_checks(timestamp)
            """
            )

            await db.commit()

        self._initialized = True

    async def store_result(self, result: HealthCheckResult) -> None:
        """Store a single health check result."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO health_checks
                (customer_id, api_name, environment, timestamp, status, status_code, latency_ms, error_message, response_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.customer_id,
                    result.api_name,
                    result.environment,
                    result.timestamp.isoformat(),
                    result.status.value,
                    result.status_code,
                    result.latency_ms,
                    result.error_message,
                    1 if result.response_valid else 0,
                ),
            )
            await db.commit()

    async def store_results(self, results: list[HealthCheckResult]) -> None:
        """Store multiple health check results."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO health_checks
                (customer_id, api_name, environment, timestamp, status, status_code, latency_ms, error_message, response_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        r.customer_id,
                        r.api_name,
                        r.environment,
                        r.timestamp.isoformat(),
                        r.status.value,
                        r.status_code,
                        r.latency_ms,
                        r.error_message,
                        1 if r.response_valid else 0,
                    )
                    for r in results
                ],
            )
            await db.commit()

    async def get_latest_status(
        self, customer_id: str, environment: str, api_name: str
    ) -> HealthCheckResult | None:
        """Get the most recent health check result for an API."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM health_checks
                WHERE customer_id = ? AND environment = ? AND api_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (customer_id, environment, api_name),
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    return HealthCheckResult(
                        customer_id=row["customer_id"],
                        api_name=row["api_name"],
                        environment=row["environment"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        status=HealthStatus(row["status"]),
                        status_code=row["status_code"],
                        latency_ms=row["latency_ms"],
                        error_message=row["error_message"],
                        response_valid=bool(row["response_valid"]),
                    )
        return None

    async def get_api_status(
        self, customer_id: str, environment: str, api_name: str
    ) -> APIStatus:
        """Get current status with 24h statistics for an API."""
        await self.initialize()

        cutoff = datetime.utcnow() - timedelta(hours=24)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get latest result
            async with db.execute(
                """
                SELECT * FROM health_checks
                WHERE customer_id = ? AND environment = ? AND api_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (customer_id, environment, api_name),
            ) as cursor:
                latest = await cursor.fetchone()

            # Get 24h statistics
            async with db.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'healthy' THEN 1 ELSE 0 END) as healthy_count
                FROM health_checks
                WHERE customer_id = ? AND environment = ? AND api_name = ? AND timestamp >= ?
            """,
                (customer_id, environment, api_name, cutoff.isoformat()),
            ) as cursor:
                stats = await cursor.fetchone()

        if latest:
            uptime = None
            if stats["total"] > 0:
                uptime = (stats["healthy_count"] / stats["total"]) * 100

            return APIStatus(
                customer_id=customer_id,
                api_name=api_name,
                environment=environment,
                current_status=HealthStatus(latest["status"]),
                last_check=datetime.fromisoformat(latest["timestamp"]),
                last_latency_ms=latest["latency_ms"],
                last_error=latest["error_message"],
                uptime_24h=uptime,
                check_count_24h=stats["total"],
            )

        return APIStatus(
            customer_id=customer_id,
            api_name=api_name,
            environment=environment,
            current_status=HealthStatus.UNKNOWN,
        )

    async def get_environment_statuses(
        self, customer_id: str, environment: str, api_names: list[str]
    ) -> list[APIStatus]:
        """Get status for all APIs in an environment."""
        statuses = []
        for api_name in api_names:
            status = await self.get_api_status(customer_id, environment, api_name)
            statuses.append(status)
        return statuses

    async def get_history(
        self,
        customer_id: str,
        environment: str,
        api_name: str,
        hours: int = 24,
        limit: int = 100,
    ) -> list[HealthCheckResult]:
        """Get historical health check results."""
        await self.initialize()

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM health_checks
                WHERE customer_id = ? AND environment = ? AND api_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (customer_id, environment, api_name, cutoff.isoformat(), limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            HealthCheckResult(
                customer_id=row["customer_id"],
                api_name=row["api_name"],
                environment=row["environment"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                status=HealthStatus(row["status"]),
                status_code=row["status_code"],
                latency_ms=row["latency_ms"],
                error_message=row["error_message"],
                response_valid=bool(row["response_valid"]),
            )
            for row in rows
        ]

    async def cleanup_old_records(self, retention_days: int = 7) -> int:
        """Delete records older than retention period."""
        await self.initialize()

        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM health_checks
                WHERE timestamp < ?
            """,
                (cutoff.isoformat(),),
            )
            deleted = cursor.rowcount
            await db.commit()

        return deleted
