"""Lighthouse checker - wraps lighthouse CLI."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseChecker
from ..models import CheckType, LighthouseResult, CheckRun
from ..parsers.lighthouse_parser import parse_lighthouse_json, calculate_lighthouse_score
from ..storage import Storage
from ..utils.run_logger import RunLog
from ..utils.shell import run_command

logger = logging.getLogger(__name__)


class LighthouseRunner(BaseChecker):
    check_type = CheckType.LIGHTHOUSE

    def __init__(self, storage: Storage, tools_config: dict, reports_dir: str = "./reports",
                 timeout: int = 120):
        super().__init__(storage, tools_config, timeout)
        self.lighthouse_cmd = tools_config.get("lighthouse", "lighthouse")
        self.reports_dir = Path(reports_dir) / "lighthouse"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> LighthouseResult:
        """Run lighthouse CLI and parse JSON output."""
        log = run_log.add if run_log else lambda msg, lvl="info": None

        categories = kwargs.get("categories", ["seo", "accessibility", "best-practices"])
        log(f"Categories: {', '.join(categories)}", "info")

        # Output paths
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = url.replace("https://", "").replace("http://", "").replace("/", "_")[:60]
        json_path = str(self.reports_dir / f"{slug}_{timestamp}.json")
        html_path = str(self.reports_dir / f"{slug}_{timestamp}.html")

        # Build command
        cmd = [
            self.lighthouse_cmd, url,
            "--output", "json,html",
            "--output-path", str(self.reports_dir / f"{slug}_{timestamp}"),
            "--chrome-flags=--headless --no-sandbox --disable-gpu",
            "--quiet",
        ]
        for cat in categories:
            cmd.extend(["--only-categories", cat])

        log("Launching Lighthouse (headless Chrome)...", "info")
        if run_log:
            run_log.progress_pct = 30
        result = await run_command(cmd, timeout=self.timeout)

        if result.timed_out:
            log(f"Lighthouse timed out after {self.timeout}s", "error")
            raise TimeoutError(f"Lighthouse timed out after {self.timeout}s")

        log(f"Lighthouse finished (exit code {result.returncode})", "info")
        if run_log:
            run_log.progress_pct = 60

        # Parse JSON report
        # Lighthouse names files as <path>.report.json and <path>.report.html
        actual_json = json_path
        if not Path(json_path).exists():
            alt = str(self.reports_dir / f"{slug}_{timestamp}.report.json")
            if Path(alt).exists():
                actual_json = alt

        log("Parsing Lighthouse JSON report...", "info")
        parsed = parse_lighthouse_json(actual_json, url)
        scores = parsed.get("scores")
        audits = parsed.get("audits", [])

        # Update report paths
        if scores:
            scores.report_json_path = actual_json
            scores.report_html_path = html_path if Path(html_path).exists() else None
            for attr in ["performance", "accessibility", "best_practices", "seo"]:
                val = getattr(scores, attr, None)
                if val is not None:
                    log(f"{attr.replace('_', ' ').title()}: {val}", "success" if val >= 80 else "warning" if val >= 60 else "error")
        else:
            log("No scores parsed from Lighthouse output", "warning")

        if audits:
            log(f"Found {len(audits)} audit detail(s)", "info")
        if run_log:
            run_log.progress_pct = 75

        return LighthouseResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=url),
            scores=[scores] if scores else [],
            audits=audits,
        )

    def _calculate_score(self, result: LighthouseResult) -> float:
        if result.scores:
            return calculate_lighthouse_score(result.scores[0])
        return 0.0

    def _build_summary(self, result: LighthouseResult) -> dict:
        if result.scores:
            s = result.scores[0]
            return {
                "performance": s.performance,
                "accessibility": s.accessibility,
                "best_practices": s.best_practices,
                "seo": s.seo,
                "failing_audits": len(result.audits),
            }
        return {}

    async def _store_results(self, run_id: int, result: LighthouseResult) -> None:
        if result.scores:
            for s in result.scores:
                s.run_id = run_id
            lh_ids = await self.storage.store_lighthouse_scores(run_id, result.scores)

            if result.audits and lh_ids:
                for audit in result.audits:
                    audit.lighthouse_id = lh_ids[0]
                await self.storage.store_lighthouse_audits(result.audits)
