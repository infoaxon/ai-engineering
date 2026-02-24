"""Broken Link checker - wraps check-broken-links.sh / linkinator."""

from __future__ import annotations

import logging
import tempfile
from typing import Optional

from .base import BaseChecker
from ..models import CheckType, BrokenLinkResult, CheckRun
from ..parsers.broken_link_parser import parse_console_output, calculate_score
from ..storage import Storage
from ..utils.run_logger import RunLog
from ..utils.shell import run_command

logger = logging.getLogger(__name__)


class BrokenLinkChecker(BaseChecker):
    check_type = CheckType.BROKEN_LINKS

    def __init__(self, storage: Storage, tools_config: dict, timeout: int = 300):
        super().__init__(storage, tools_config, timeout)
        self.script = tools_config.get("broken_links_script", "check-broken-links.sh")

    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> BrokenLinkResult:
        """Run broken link checker and parse output."""
        log = run_log.add if run_log else lambda msg, lvl="info": None

        urls = kwargs.get("urls", [url])
        deep = kwargs.get("deep", False)
        strict = kwargs.get("strict", False)

        log(f"Checking {len(urls)} URL(s) for broken links", "info")
        if deep:
            log("Deep crawl enabled — following internal links", "info")
        if strict:
            log("Strict mode — treating redirects as failures", "info")

        # Scale timeout based on number of URLs (~3s per URL, minimum 300s)
        effective_timeout = max(self.timeout, len(urls) * 3, 300)
        if effective_timeout != self.timeout:
            log(f"Timeout scaled to {effective_timeout}s for {len(urls)} URLs", "info")

        # Write URLs to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for u in urls:
                f.write(u + "\n")
            urls_file = f.name

        cmd = ["bash", self.script, urls_file]
        if deep:
            cmd.append("--deep")
        if strict:
            cmd.append("--strict")

        log(f"Running broken link script: {self.script}", "info")
        if run_log:
            run_log.progress_pct = 30

        try:
            result = await run_command(cmd, timeout=effective_timeout)
        finally:
            import os
            os.unlink(urls_file)

        if result.timed_out:
            log(f"Broken link check timed out after {effective_timeout}s", "error")
            raise TimeoutError(f"Broken link check timed out after {effective_timeout}s")

        log(f"Script finished (exit code {result.returncode})", "info")
        if run_log:
            run_log.progress_pct = 60

        log("Parsing broken link results...", "info")
        parsed = parse_console_output(result.stdout + result.stderr)

        ok = parsed["ok_count"]
        broken = parsed["broken_count"]
        redirected = parsed["redirected_count"]
        timeout_cnt = parsed["timeout_count"]
        total = len(parsed["links"])

        log(f"Total links checked: {total}", "info")
        if ok > 0:
            log(f"OK: {ok} link(s)", "success")
        if broken > 0:
            log(f"BROKEN: {broken} link(s)", "error")
        if redirected > 0:
            log(f"Redirected: {redirected} link(s)", "warning")
        if timeout_cnt > 0:
            log(f"Timed out: {timeout_cnt} link(s)", "warning")
        if run_log:
            run_log.progress_pct = 75

        return BrokenLinkResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=url),
            links=parsed["links"],
            ok_count=parsed["ok_count"],
            broken_count=parsed["broken_count"],
            redirected_count=parsed["redirected_count"],
            timeout_count=parsed["timeout_count"],
        )

    def _calculate_score(self, result: BrokenLinkResult) -> float:
        return calculate_score(
            result.ok_count, result.broken_count,
            result.redirected_count, result.timeout_count
        )

    def _build_summary(self, result: BrokenLinkResult) -> dict:
        return {
            "ok_count": result.ok_count,
            "broken_count": result.broken_count,
            "redirected_count": result.redirected_count,
            "timeout_count": result.timeout_count,
            "total": len(result.links),
        }

    async def _store_results(self, run_id: int, result: BrokenLinkResult) -> None:
        if result.links:
            for link in result.links:
                link.run_id = run_id
            await self.storage.store_broken_links(run_id, result.links)
