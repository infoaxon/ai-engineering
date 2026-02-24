"""HTML Validation checker - wraps html-validate CLI."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from .base import BaseChecker
from ..models import CheckType, HTMLValidationResult, HTMLValidationIssue, Severity
from ..parsers.html_validate_parser import parse_json_output, calculate_score
from ..storage import Storage
from ..utils.run_logger import RunLog

logger = logging.getLogger(__name__)


class HTMLValidator(BaseChecker):
    check_type = CheckType.HTML_VALIDATION

    def __init__(self, storage: Storage, tools_config: dict, timeout: int = 120):
        super().__init__(storage, tools_config, timeout)
        self.html_validate_cmd = tools_config.get("html_validate", "html-validate")

    async def execute(self, url: str, run_log: Optional[RunLog] = None, **kwargs) -> HTMLValidationResult:
        """Fetch URL HTML, run html-validate on it, parse results."""
        config_path = kwargs.get("config_path")
        log = run_log.add if run_log else lambda msg, lvl="info": None

        # Fetch the HTML
        log(f"Fetching HTML from {url}...", "info")
        if run_log:
            run_log.progress_pct = 20
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, verify=False) as client:
            response = await client.get(url)
            html_content = response.text
        log(f"Fetched {len(html_content):,} bytes (HTTP {response.status_code})", "success")
        if run_log:
            run_log.progress_pct = 40

        # Write HTML to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_path = f.name

        # Create a temp file for JSON output (html-validate output can be very large
        # and overflow the 64KB subprocess pipe buffer, truncating the JSON)
        output_fd, output_path = tempfile.mkstemp(suffix=".json", prefix="hv_")

        try:
            # Build command — redirect stdout to a file to avoid pipe buffer truncation
            cmd_args = [self.html_validate_cmd, "--formatter", "json"]
            if config_path and Path(config_path).exists():
                cmd_args.extend(["--config", config_path])
                log(f"Using config: {Path(config_path).name}", "info")
            cmd_args.append(temp_path)

            log("Running html-validate...", "info")
            if run_log:
                run_log.progress_pct = 50

            # Run with stdout redirected to file to handle large outputs
            import os
            with os.fdopen(output_fd, "w") as out_f:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=out_f,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise TimeoutError(f"html-validate timed out after {self.timeout}s")

            returncode = process.returncode or 0
            log(f"html-validate finished (exit code {returncode})", "info")
            if run_log:
                run_log.progress_pct = 70

            # Read the full JSON output from the file
            json_output = Path(output_path).read_text(encoding="utf-8", errors="replace")
            output_size = len(json_output)
            log(f"Validation output: {output_size:,} bytes", "info")

            # Parse
            log("Parsing validation results...", "info")
            issues = parse_json_output(json_output, url)
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

        error_count = sum(1 for i in issues if i.severity == Severity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == Severity.WARN)

        if error_count > 0:
            log(f"Found {error_count} error(s)", "error")
        if warning_count > 0:
            log(f"Found {warning_count} warning(s)", "warning")
        if error_count == 0 and warning_count == 0:
            log("No issues found — HTML is clean", "success")

        from ..models import CheckRun
        return HTMLValidationResult(
            run=CheckRun(check_type=self.check_type, site_id="", environment="", target_url=url),
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
        )

    def _calculate_score(self, result: HTMLValidationResult) -> float:
        return calculate_score(result.error_count, result.warning_count)

    def _build_summary(self, result: HTMLValidationResult) -> dict:
        return {
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "total_issues": len(result.issues),
        }

    async def _store_results(self, run_id: int, result: HTMLValidationResult) -> None:
        if result.issues:
            for issue in result.issues:
                issue.run_id = run_id
            await self.storage.store_html_issues(run_id, result.issues)
