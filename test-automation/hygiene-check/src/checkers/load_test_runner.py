"""Load Test checker - built-in HTTP tester with JMeter fallback."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseChecker
from ..models import CheckType, LoadTestRunResult, LoadTestResult, SLAVerdict, CheckRun
from ..parsers.jmeter_parser import parse_loadtest_output, calculate_score
from ..storage import Storage
from ..utils.run_logger import RunLog
from ..utils.shell import run_command

logger = logging.getLogger(__name__)


class LoadTestRunner(BaseChecker):
    check_type = CheckType.LOAD_TEST

    def __init__(self, storage: Storage, tools_config: dict, timeout: int = 3600):
        super().__init__(storage, tools_config, timeout)
        self.script = tools_config.get("loadtest_script", "run_loadtest.sh")

    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> LoadTestRunResult:
        """Run load test: built-in Python tester by default, JMeter if --jmeter."""
        use_jmeter = kwargs.get("jmeter", False)

        if use_jmeter:
            return await self._execute_jmeter(url, run_log=run_log, **kwargs)
        return await self._execute_builtin(url, run_log=run_log, **kwargs)

    async def _execute_builtin(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> LoadTestRunResult:
        """Run load test using the built-in Python HTTP tester."""
        from .http_load_tester import HttpLoadTester

        log = run_log.add if run_log else lambda msg, lvl="info": None

        scenario = kwargs.get("scenario", "smoke")
        host = kwargs.get("host")
        users = kwargs.get("users")
        duration = kwargs.get("duration")

        # Build target URL
        target_url = url
        if not target_url and host:
            target_url = host
        if not target_url:
            raise ValueError("No target URL or host specified")

        log("Using built-in HTTP load tester (no JMeter required)", "info")
        if run_log:
            run_log.progress_pct = 15

        tester = HttpLoadTester()
        parsed = await tester.run(
            url=target_url,
            scenario=scenario,
            users=users,
            duration=duration,
            run_log=run_log,
            sla=kwargs.get("sla"),
        )

        if run_log:
            run_log.progress_pct = 75

        lt_result = LoadTestResult(
            scenario=parsed.get("scenario", scenario),
            target_host=parsed.get("target_host", target_url),
            users=parsed.get("users", 0),
            duration_seconds=parsed.get("duration_seconds", 0),
            total_requests=parsed.get("total_requests", 0),
            error_rate=parsed.get("error_rate", 0.0),
            throughput=parsed.get("throughput", 0.0),
            p95_response_ms=parsed.get("p95_response_ms", 0.0),
            p99_response_ms=parsed.get("p99_response_ms", 0.0),
            avg_response_ms=parsed.get("avg_response_ms", 0.0),
            min_response_ms=parsed.get("min_response_ms", 0.0),
            max_response_ms=parsed.get("max_response_ms", 0.0),
            sla_overall=SLAVerdict(parsed.get("sla_overall", "pass")),
        )

        return LoadTestRunResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=target_url),
            result=lt_result,
        )

    async def _execute_jmeter(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> LoadTestRunResult:
        """Run JMeter load test via run_loadtest.sh and parse output."""
        log = run_log.add if run_log else lambda msg, lvl="info": None

        scenario = kwargs.get("scenario", "smoke")
        host = kwargs.get("host")
        users = kwargs.get("users")
        duration = kwargs.get("duration")

        log(f"Scenario: {scenario}", "info")

        # Fix command format: first positional arg is scenario, named args use = format
        cmd = ["bash", self.script, scenario]
        if host:
            cmd.append(f"--host={host}")
            log(f"Target host: {host}", "info")
        elif url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.hostname:
                cmd.append(f"--host={parsed.hostname}")
                if parsed.port:
                    cmd.append(f"--port={parsed.port}")
                cmd.append(f"--protocol={parsed.scheme or 'https'}")
                log(f"Target host: {parsed.hostname}", "info")

        if users:
            cmd.append(f"--users={users}")
            log(f"Virtual users: {users}", "info")
        if duration:
            cmd.append(f"--duration={duration}")
            log(f"Duration: {duration}s", "info")

        # Pass SLA overrides to shell script
        sla = kwargs.get("sla") or {}
        if "error_rate" in sla:
            cmd.append(f"--sla-error={sla['error_rate']}")
        if "p95_ms" in sla:
            cmd.append(f"--sla-p95={sla['p95_ms']}")
        if "p99_ms" in sla:
            cmd.append(f"--sla-p99={sla['p99_ms']}")

        log("Starting JMeter load test — this may take several minutes...", "info")
        if run_log:
            run_log.progress_pct = 20

        result = await run_command(cmd, timeout=self.timeout)

        if result.timed_out:
            log(f"Load test timed out after {self.timeout}s", "error")
            raise TimeoutError(f"Load test timed out after {self.timeout}s")

        log(f"JMeter finished (exit code {result.returncode})", "info")
        if run_log:
            run_log.progress_pct = 60

        log("Parsing load test results...", "info")
        parsed = parse_loadtest_output(result.stdout + result.stderr)

        # If exit code is non-zero and we got no parseable results, it's a real failure.
        # Exit code 1 with valid results means SLA breach — that's expected, not a crash.
        if result.returncode != 0 and not parsed.get("total_requests"):
            from ..utils.shell import strip_ansi
            stderr_clean = strip_ansi(result.stderr.strip())
            stdout_clean = strip_ansi(result.stdout.strip())
            # Filter out harmless Java/JMeter warnings from stderr
            stderr_lines = [
                ln for ln in stderr_clean.splitlines()
                if not ln.startswith("WARNING:") and not ln.startswith("WARN ")
            ]
            stderr_useful = "\n".join(stderr_lines).strip()
            last_line = stdout_clean.splitlines()[-1] if stdout_clean else ""
            error_msg = stderr_useful or last_line or f"JMeter script exited with code {result.returncode}"
            log(f"JMeter failed: {error_msg}", "error")
            raise RuntimeError(f"JMeter load test failed: {error_msg}")

        if parsed.get("total_requests"):
            log(f"Total requests: {parsed['total_requests']}", "info")
        if parsed.get("error_rate") is not None:
            err_rate = parsed["error_rate"]
            log(f"Error rate: {err_rate}%", "success" if err_rate < 1 else "error")
        if parsed.get("throughput"):
            log(f"Throughput: {parsed['throughput']} req/s", "info")
        if parsed.get("p95_response_ms"):
            log(f"P95 response: {parsed['p95_response_ms']}ms", "info")
        sla = parsed.get("sla_overall", "pass")
        log(f"SLA verdict: {sla.upper()}", "success" if sla == "pass" else "error")
        if run_log:
            run_log.progress_pct = 75

        lt_result = LoadTestResult(
            scenario=parsed.get("scenario", scenario),
            target_host=parsed.get("target_host", url),
            users=parsed.get("users", 0),
            duration_seconds=parsed.get("duration_seconds", 0),
            total_requests=parsed.get("total_requests", 0),
            error_rate=parsed.get("error_rate", 0.0),
            throughput=parsed.get("throughput", 0.0),
            p95_response_ms=parsed.get("p95_response_ms", 0.0),
            p99_response_ms=parsed.get("p99_response_ms", 0.0),
            avg_response_ms=parsed.get("avg_response_ms", 0.0),
            min_response_ms=parsed.get("min_response_ms", 0.0),
            max_response_ms=parsed.get("max_response_ms", 0.0),
            sla_overall=SLAVerdict(parsed.get("sla_overall", "pass")),
        )

        return LoadTestRunResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=url),
            result=lt_result,
        )

    def _calculate_score(self, result: LoadTestRunResult) -> float:
        if result.result:
            return calculate_score(
                result.result.error_rate,
                result.result.p95_response_ms,
                result.result.throughput,
            )
        return 0.0

    def _build_summary(self, result: LoadTestRunResult) -> dict:
        if result.result:
            r = result.result
            return {
                "scenario": r.scenario,
                "total_requests": r.total_requests,
                "error_rate": r.error_rate,
                "throughput": r.throughput,
                "p95_ms": r.p95_response_ms,
                "sla_overall": r.sla_overall.value,
            }
        return {}

    async def _store_results(self, run_id: int, result: LoadTestRunResult) -> None:
        if result.result:
            result.result.run_id = run_id
            await self.storage.store_load_test_result(run_id, result.result)
