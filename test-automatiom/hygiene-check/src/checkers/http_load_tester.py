"""Built-in HTTP load tester using httpx + asyncio.

Provides immediate load testing without JMeter or any external tools.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

import httpx

from ..utils.run_logger import RunLog

logger = logging.getLogger(__name__)

# Scenario defaults matching run_loadtest.sh
SCENARIOS = {
    "smoke":  {"users": 5,   "duration": 60,  "rampup": 30,  "think_min": 2.0, "think_max": 3.0},
    "load":   {"users": 50,  "duration": 300, "rampup": 120, "think_min": 1.0, "think_max": 3.0},
    "stress": {"users": 200, "duration": 300, "rampup": 60,  "think_min": 0.5, "think_max": 1.5},
    "soak":   {"users": 30,  "duration": 600, "rampup": 60,  "think_min": 2.0, "think_max": 4.0},
    "spike":  {"users": 150, "duration": 120, "rampup": 10,  "think_min": 0.2, "think_max": 0.5},
}

SLA_DEFAULTS = {
    "error_rate": 2.0,    # Max 2% error rate
    "p95_ms": 3000,       # P95 <= 3000ms
    "p99_ms": 5000,       # P99 <= 5000ms
    "throughput": 10.0,   # Min 10 req/s
}


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calculate percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


class HttpLoadTester:
    """Built-in HTTP load tester using httpx + asyncio."""

    def __init__(self):
        self._results: list[dict] = []
        self._stop = False

    async def run(
        self,
        url: str,
        scenario: str = "smoke",
        users: int | None = None,
        duration: int | None = None,
        run_log: RunLog | None = None,
    ) -> dict:
        """Run load test and return parsed results dict.

        The returned dict matches the format of parse_loadtest_output()
        so it can be used directly by LoadTestRunner.
        """
        log = run_log.add if run_log else lambda msg, lvl="info": None

        cfg = SCENARIOS.get(scenario, SCENARIOS["smoke"])
        num_users = users if users is not None else cfg["users"]
        test_duration = duration if duration is not None else cfg["duration"]
        rampup = cfg["rampup"]
        think_min = cfg["think_min"]
        think_max = cfg["think_max"]

        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        log(f"Scenario: {scenario}", "info")
        log(f"Target: {url}", "info")
        log(f"Users: {num_users}, Duration: {test_duration}s, Ramp-up: {rampup}s", "info")
        log("Starting built-in HTTP load test...", "info")

        self._results = []
        self._stop = False
        start_time = time.monotonic()

        # Create shared client with connection pooling
        limits = httpx.Limits(
            max_connections=min(num_users + 10, 300),
            max_keepalive_connections=min(num_users, 200),
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            limits=limits,
        ) as client:
            tasks = []
            for worker_id in range(num_users):
                # Stagger worker start times over the ramp-up period
                delay = (rampup / max(num_users, 1)) * worker_id
                task = asyncio.create_task(
                    self._worker(
                        client, url, worker_id, delay,
                        test_duration, start_time,
                        think_min, think_max,
                    )
                )
                tasks.append(task)

            # Progress reporting while workers run
            total_elapsed = 0
            while total_elapsed < test_duration + rampup + 5:
                await asyncio.sleep(5)
                total_elapsed = time.monotonic() - start_time
                count = len(self._results)
                if run_log:
                    pct = min(70, int(20 + 50 * (total_elapsed / test_duration)))
                    run_log.progress_pct = pct
                log(f"Progress: {total_elapsed:.0f}s elapsed, {count} requests sent", "info")

                # Check if all workers are done
                if all(t.done() for t in tasks):
                    break

            self._stop = True
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_total = time.monotonic() - start_time
        log(f"Test completed in {elapsed_total:.1f}s", "info")

        # Calculate metrics
        return self._calculate_metrics(
            url=url,
            scenario=scenario,
            num_users=num_users,
            test_duration=test_duration,
            elapsed_total=elapsed_total,
            log=log,
        )

    async def _worker(
        self,
        client: httpx.AsyncClient,
        url: str,
        worker_id: int,
        delay: float,
        duration: int,
        start_time: float,
        think_min: float,
        think_max: float,
    ):
        """Single virtual user sending requests in a loop."""
        if delay > 0:
            await asyncio.sleep(delay)

        while not self._stop:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration + delay:
                break

            req_start = time.monotonic()
            try:
                response = await client.get(url)
                elapsed_ms = (time.monotonic() - req_start) * 1000
                self._results.append({
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "success": 200 <= response.status_code < 400,
                    "worker_id": worker_id,
                })
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
                elapsed_ms = (time.monotonic() - req_start) * 1000
                self._results.append({
                    "status_code": 0,
                    "elapsed_ms": elapsed_ms,
                    "success": False,
                    "worker_id": worker_id,
                    "error": type(e).__name__,
                })
            except Exception as e:
                elapsed_ms = (time.monotonic() - req_start) * 1000
                self._results.append({
                    "status_code": 0,
                    "elapsed_ms": elapsed_ms,
                    "success": False,
                    "worker_id": worker_id,
                    "error": str(e),
                })

            # Think time between requests
            think = random.uniform(think_min, think_max)
            await asyncio.sleep(think)

    def _calculate_metrics(
        self,
        url: str,
        scenario: str,
        num_users: int,
        test_duration: int,
        elapsed_total: float,
        log,
    ) -> dict:
        """Calculate all metrics from collected results."""
        results = self._results
        total = len(results)
        failures = sum(1 for r in results if not r["success"])
        error_rate = (failures / total * 100) if total > 0 else 0.0
        throughput = total / elapsed_total if elapsed_total > 0 else 0.0

        # Response time percentiles
        times = sorted(r["elapsed_ms"] for r in results) if results else []
        avg_ms = sum(times) / len(times) if times else 0.0
        min_ms = times[0] if times else 0.0
        max_ms = times[-1] if times else 0.0
        p50_ms = _percentile(times, 50)
        p90_ms = _percentile(times, 90)
        p95_ms = _percentile(times, 95)
        p99_ms = _percentile(times, 99)

        # SLA checks
        sla_pass = True
        if error_rate > SLA_DEFAULTS["error_rate"]:
            sla_pass = False
            log(f"SLA BREACH: Error rate {error_rate:.2f}% > {SLA_DEFAULTS['error_rate']}%", "error")
        if p95_ms > SLA_DEFAULTS["p95_ms"]:
            sla_pass = False
            log(f"SLA BREACH: P95 {p95_ms:.0f}ms > {SLA_DEFAULTS['p95_ms']}ms", "error")
        if p99_ms > SLA_DEFAULTS["p99_ms"]:
            sla_pass = False
            log(f"SLA BREACH: P99 {p99_ms:.0f}ms > {SLA_DEFAULTS['p99_ms']}ms", "error")
        if throughput < SLA_DEFAULTS["throughput"] and total > 0:
            sla_pass = False
            log(f"SLA BREACH: Throughput {throughput:.2f} req/s < {SLA_DEFAULTS['throughput']} req/s", "error")

        sla_overall = "pass" if sla_pass else "fail"

        # Extract host from URL for target_host
        from urllib.parse import urlparse
        parsed = urlparse(url)
        target_host = parsed.hostname or url

        # Log summary
        log(f"Total Requests: {total:,}", "info")
        log(f"Failures: {failures:,}", "info")
        err_level = "success" if error_rate < 1 else "error"
        err_sla = "PASS" if error_rate <= SLA_DEFAULTS["error_rate"] else "FAIL"
        log(f"Error Rate: {error_rate:.2f}% [{err_sla}]", err_level)
        tp_sla = "PASS" if throughput >= SLA_DEFAULTS["throughput"] else "FAIL"
        log(f"Throughput: {throughput:.2f} req/s [{tp_sla}]", "info")
        log(f"Min: {min_ms:.0f}ms", "info")
        log(f"Avg: {avg_ms:.0f}ms", "info")
        p95_sla = "PASS" if p95_ms <= SLA_DEFAULTS["p95_ms"] else "FAIL"
        log(f"P95: {p95_ms:.0f}ms [{p95_sla}]", "info")
        p99_sla = "PASS" if p99_ms <= SLA_DEFAULTS["p99_ms"] else "FAIL"
        log(f"P99: {p99_ms:.0f}ms [{p99_sla}]", "info")
        log(f"Max: {max_ms:.0f}ms", "info")

        if sla_pass:
            log("ALL SLAs PASSED", "success")
        else:
            log("SLA BREACH DETECTED", "error")

        return {
            "scenario": scenario,
            "target_host": target_host,
            "users": num_users,
            "duration_seconds": test_duration,
            "total_requests": total,
            "error_rate": round(error_rate, 2),
            "throughput": round(throughput, 2),
            "p95_response_ms": round(p95_ms, 1),
            "p99_response_ms": round(p99_ms, 1),
            "avg_response_ms": round(avg_ms, 1),
            "min_response_ms": round(min_ms, 1),
            "max_response_ms": round(max_ms, 1),
            "sla_overall": sla_overall,
        }
