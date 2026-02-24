"""Parser for JMeter / run_loadtest.sh output."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from ..models import LoadTestResult, SLAVerdict
from ..utils.shell import strip_ansi

logger = logging.getLogger(__name__)

# Result block patterns (after ANSI stripping)
TOTAL_REQ_RE = re.compile(r"Total Requests:\s+([\d,]+)")
FAILURES_RE = re.compile(r"Failures:\s+([\d,]+)")
ERROR_RATE_RE = re.compile(r"Error Rate:\s+([\d.]+)%\s+\[(PASS|FAIL)\]")
THROUGHPUT_RE = re.compile(r"Throughput:\s+([\d.]+)\s+req/s\s+\[(PASS|FAIL)\]")
P95_RE = re.compile(r"P95:\s+([\d,]+)ms\s+\[(PASS|FAIL)\]")
P99_RE = re.compile(r"P99:\s+([\d,]+)ms\s+\[(PASS|FAIL)\]")
MIN_RE = re.compile(r"Min:\s+([\d,]+)ms")
AVG_RE = re.compile(r"Avg:\s+([\d,]+)ms")
MAX_RE = re.compile(r"Max:\s+([\d,]+)ms")
OVERALL_PASS_RE = re.compile(r"ALL SLAs PASSED|Test PASSED")
OVERALL_FAIL_RE = re.compile(r"SLA BREACH DETECTED|Test FAILED")

# Config block
SCENARIO_RE = re.compile(r"Scenario:\s+(\w+)")
TARGET_RE = re.compile(r"Target:\s+(.+)")
USERS_RE = re.compile(r"Users:\s+(\d+)")
DURATION_RE = re.compile(r"Duration:\s+(\d+)s")


def _parse_int(s: str) -> int:
    return int(s.replace(",", ""))


def parse_loadtest_output(raw_output: str) -> dict:
    """Parse the console output of run_loadtest.sh.

    Returns dict suitable for building LoadTestResult.
    """
    output = strip_ansi(raw_output)

    result = {
        "scenario": "",
        "target_host": "",
        "users": 0,
        "duration_seconds": 0,
        "total_requests": 0,
        "error_rate": 0.0,
        "throughput": 0.0,
        "p95_response_ms": 0.0,
        "p99_response_ms": 0.0,
        "avg_response_ms": 0.0,
        "min_response_ms": 0.0,
        "max_response_ms": 0.0,
        "sla_overall": "pass",
    }

    for line in output.splitlines():
        stripped = line.strip()

        m = SCENARIO_RE.search(stripped)
        if m:
            result["scenario"] = m.group(1).lower()

        m = TARGET_RE.search(stripped)
        if m:
            result["target_host"] = m.group(1).strip()

        m = USERS_RE.search(stripped)
        if m:
            result["users"] = int(m.group(1))

        m = DURATION_RE.search(stripped)
        if m:
            result["duration_seconds"] = int(m.group(1))

        m = TOTAL_REQ_RE.search(stripped)
        if m:
            result["total_requests"] = _parse_int(m.group(1))

        m = ERROR_RATE_RE.search(stripped)
        if m:
            result["error_rate"] = float(m.group(1))

        m = THROUGHPUT_RE.search(stripped)
        if m:
            result["throughput"] = float(m.group(1))

        m = P95_RE.search(stripped)
        if m:
            result["p95_response_ms"] = float(_parse_int(m.group(1)))

        m = P99_RE.search(stripped)
        if m:
            result["p99_response_ms"] = float(_parse_int(m.group(1)))

        m = MIN_RE.search(stripped)
        if m:
            result["min_response_ms"] = float(_parse_int(m.group(1)))

        m = AVG_RE.search(stripped)
        if m:
            result["avg_response_ms"] = float(_parse_int(m.group(1)))

        m = MAX_RE.search(stripped)
        if m:
            result["max_response_ms"] = float(_parse_int(m.group(1)))

        if OVERALL_FAIL_RE.search(stripped):
            result["sla_overall"] = "fail"
        elif OVERALL_PASS_RE.search(stripped):
            result["sla_overall"] = "pass"

    return result


def parse_jtl_file(jtl_path: str) -> dict:
    """Parse a JMeter JTL (CSV) results file for more detailed metrics.

    Returns dict with summary stats computed from raw JTL data.
    """
    path = Path(jtl_path)
    if not path.exists():
        logger.warning(f"JTL file not found: {jtl_path}")
        return {}

    elapsed_times = []
    error_count = 0
    total_count = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            try:
                elapsed = int(row.get("elapsed", 0))
                elapsed_times.append(elapsed)
            except (ValueError, TypeError):
                pass
            success = row.get("success", "true").lower()
            if success == "false":
                error_count += 1

    if not elapsed_times:
        return {}

    elapsed_times.sort()
    n = len(elapsed_times)

    return {
        "total_requests": total_count,
        "error_count": error_count,
        "error_rate": round((error_count / total_count) * 100, 2) if total_count else 0,
        "min_ms": elapsed_times[0],
        "avg_ms": round(sum(elapsed_times) / n),
        "p50_ms": elapsed_times[int(n * 0.5)],
        "p90_ms": elapsed_times[int(n * 0.9)],
        "p95_ms": elapsed_times[int(n * 0.95)],
        "p99_ms": elapsed_times[min(int(n * 0.99), n - 1)],
        "max_ms": elapsed_times[-1],
    }


def calculate_score(error_rate: float, p95_ms: float, throughput: float) -> float:
    """Calculate load test score (0-100) based on key metrics."""
    score = 100.0
    # Error rate penalties
    if error_rate > 5:
        score -= 40
    elif error_rate > 1:
        score -= 20
    elif error_rate > 0:
        score -= 5

    # P95 response time penalties (ms)
    if p95_ms > 5000:
        score -= 30
    elif p95_ms > 2000:
        score -= 15
    elif p95_ms > 1000:
        score -= 5

    # Throughput bonus
    if throughput >= 100:
        score = min(100, score + 5)

    return max(0.0, score)
