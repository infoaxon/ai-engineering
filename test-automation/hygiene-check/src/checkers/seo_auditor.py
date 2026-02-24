"""SEO Audit checker - wraps seo-hygiene-check.sh."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseChecker
from ..models import CheckType, SEOAuditResult, CheckRun, CheckStatus
from ..parsers.seo_report_parser import parse_seo_output
from ..storage import Storage
from ..utils.run_logger import RunLog
from ..utils.shell import run_command

logger = logging.getLogger(__name__)


class SEOAuditor(BaseChecker):
    check_type = CheckType.SEO_AUDIT

    def __init__(self, storage: Storage, tools_config: dict, timeout: int = 600):
        super().__init__(storage, tools_config, timeout)
        self.seo_script = tools_config.get("seo_script", "seo-hygiene-check.sh")

    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> SEOAuditResult:
        """Run seo-hygiene-check.sh and parse output."""
        log = run_log.add if run_log else lambda msg, lvl="info": None

        cmd = ["bash", self.seo_script, url]

        remote = kwargs.get("remote", True)
        if remote:
            cmd.append("--remote")
            log("Mode: remote (HTTP-based checks only)", "info")
        else:
            log("Mode: local + remote checks", "info")

        log(f"Running SEO audit script: {self.seo_script}", "info")
        log("This may take a few minutes — scanning 12 SEO sections...", "info")
        if run_log:
            run_log.progress_pct = 30

        result = await run_command(cmd, timeout=self.timeout)

        if result.timed_out:
            log(f"SEO audit timed out after {self.timeout}s", "error")
            raise TimeoutError(f"SEO audit timed out after {self.timeout}s")

        log(f"SEO script finished (exit code {result.returncode})", "info")
        if run_log:
            run_log.progress_pct = 60

        log("Parsing SEO audit results...", "info")
        parsed = parse_seo_output(result.stdout + result.stderr)

        items = parsed["items"]
        pass_count = parsed["pass_count"]
        fail_count = parsed["fail_count"]
        warn_count = parsed["warn_count"]
        total_checks = parsed["total_checks"]
        overall_score = parsed["overall_score"]
        grade = parsed["grade"]

        # Fallback: if summary block wasn't found, count from parsed items
        if total_checks == 0 and items:
            pass_count = sum(1 for i in items if i.status == CheckStatus.PASS)
            fail_count = sum(1 for i in items if i.status == CheckStatus.FAIL)
            warn_count = sum(1 for i in items if i.status == CheckStatus.WARN)
            total_checks = pass_count + fail_count + warn_count
            if total_checks > 0 and not overall_score:
                overall_score = round((pass_count / total_checks) * 100, 1)
                if overall_score >= 90:
                    grade = "EXCELLENT"
                elif overall_score >= 70:
                    grade = "GOOD"
                elif overall_score >= 50:
                    grade = "NEEDS WORK"
                else:
                    grade = "CRITICAL"

        if run_log:
            run_log.progress_pct = 70

        if pass_count > 0:
            log(f"PASS: {pass_count} check(s)", "success")
        if fail_count > 0:
            log(f"FAIL: {fail_count} check(s)", "error")
        if warn_count > 0:
            log(f"WARN: {warn_count} check(s)", "warning")
        if grade:
            log(f"Grade: {grade} ({overall_score}%)", "info")

        return SEOAuditResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=url),
            items=items,
            pass_count=pass_count,
            fail_count=fail_count,
            warn_count=warn_count,
            total_checks=total_checks,
            overall_score=overall_score,
            grade=grade,
        )

    def _calculate_score(self, result: SEOAuditResult) -> float:
        if result.overall_score:
            return result.overall_score
        # Fallback: calculate from pass/fail/warn counts
        total = result.pass_count + result.fail_count + result.warn_count
        if total == 0:
            return 0.0
        return round((result.pass_count / total) * 100, 1)

    def _build_summary(self, result: SEOAuditResult) -> dict:
        return {
            "pass_count": result.pass_count,
            "fail_count": result.fail_count,
            "warn_count": result.warn_count,
            "total_checks": result.total_checks,
            "overall_score": result.overall_score,
            "grade": result.grade,
        }

    async def _store_results(self, run_id: int, result: SEOAuditResult) -> None:
        if result.items:
            for item in result.items:
                item.run_id = run_id
            await self.storage.store_seo_items(run_id, result.items)
