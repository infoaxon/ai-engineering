#!/usr/bin/env python3
"""Typer CLI entry point for Hygiene Check Dashboard."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.config import SiteManager
from src.storage import Storage
from src.models import CheckType, TriggeredBy
from src.scoring import HealthScorer

console = Console()
app = typer.Typer(name="hygiene", help="Hygiene Check Dashboard CLI", no_args_is_help=True)

# Sub-command groups
html_app = typer.Typer(name="html", help="HTML Validation commands")
seo_app = typer.Typer(name="seo", help="SEO Audit commands")
lighthouse_app = typer.Typer(name="lighthouse", help="Lighthouse audit commands")
links_app = typer.Typer(name="links", help="Broken link checking commands")
loadtest_app = typer.Typer(name="loadtest", help="Load test commands")
report_app = typer.Typer(name="report", help="Report management commands")

app.add_typer(html_app)
app.add_typer(seo_app)
app.add_typer(lighthouse_app)
app.add_typer(links_app)
app.add_typer(loadtest_app)
app.add_typer(report_app)


def _get_config() -> SiteManager:
    return SiteManager(BASE_DIR / "config" / "sites.yaml")


def _get_storage() -> Storage:
    return Storage(BASE_DIR / "hygiene_checks.db")


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _print_score(score: float, label: str = "Score"):
    color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
    console.print(f"  {label}: [{color}]{score:.1f}[/{color}]")


def _score_color(score: float) -> str:
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    return "red"


def _score_bar(score: float, width: int = 20) -> str:
    """Create a visual score bar like [████████░░░░] 75.0"""
    filled = int(score / 100 * width)
    color = _score_color(score)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] [{color}]{score:.1f}[/{color}]"


def _status_icon(status: str) -> str:
    """Return colored status icon."""
    icons = {
        "pass": "[green]✓ PASS[/green]",
        "fail": "[red]✗ FAIL[/red]",
        "warn": "[yellow]⚠ WARN[/yellow]",
        "info": "[dim]ℹ INFO[/dim]",
        "ok": "[green]✓ OK[/green]",
        "broken": "[red]✗ BROKEN[/red]",
        "redirected": "[yellow]→ REDIRECT[/yellow]",
        "timeout": "[red]⏱ TIMEOUT[/red]",
        "error": "[red]✗ ERROR[/red]",
        "completed": "[green]✓ COMPLETED[/green]",
        "running": "[yellow]⟳ RUNNING[/yellow]",
        "failed": "[red]✗ FAILED[/red]",
    }
    return icons.get(status, status)


def _metric_panel(label: str, value: str, style: str = "blue") -> Panel:
    """Create a compact metric panel for side-by-side display."""
    return Panel(f"[bold]{value}[/bold]", title=label, style=style, width=24, padding=(0, 1))


# ========== HTML Validation ==========

@html_app.command("validate")
def html_validate(
    url: str = typer.Argument(..., help="URL to validate"),
    config_path: Optional[str] = typer.Option(None, help="Path to .htmlvalidate.json"),
    ci: bool = typer.Option(False, "--ci", help="CI mode: JSON output + exit codes"),
    site_id: str = typer.Option("default", help="Site identifier"),
    environment: str = typer.Option("production", help="Environment name"),
):
    """Validate HTML for a URL."""
    from src.checkers.html_validator import HTMLValidator

    config = _get_config()
    storage = _get_storage()

    checker = HTMLValidator(storage, config.tools, config.get_timeout())

    async def _run():
        await storage.initialize()
        return await checker.run(url, site_id=site_id, environment=environment,
                                  triggered_by=TriggeredBy.CLI, config_path=config_path)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running HTML validation...", total=None)
        run_id, result = _run_async(_run())

    if ci:
        output = {
            "run_id": run_id,
            "score": checker._calculate_score(result),
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "issues": [{"line": i.line_number, "col": i.column_number,
                        "severity": i.severity.value, "message": i.message, "rule": i.rule}
                       for i in result.issues]
        }
        print(json.dumps(output, indent=2))
        sys.exit(1 if result.error_count > 0 else 0)
        return

    score = checker._calculate_score(result)
    total_issues = result.error_count + result.warning_count

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold]HTML Validation Results[/bold]\n"
        f"URL: [cyan]{url}[/cyan]\n"
        f"Run ID: [dim]#{run_id}[/dim]",
        style="blue",
    ))

    # Score + summary
    console.print(f"\n  Score: {_score_bar(score)}")
    console.print(f"  Total Issues: [bold]{total_issues}[/bold]  "
                  f"([red]{result.error_count} error(s)[/red], "
                  f"[yellow]{result.warning_count} warning(s)[/yellow])")

    if result.issues:
        # Rule distribution summary
        rule_counts: dict[str, dict] = {}
        for issue in result.issues:
            key = issue.rule or "unknown"
            if key not in rule_counts:
                rule_counts[key] = {"error": 0, "warn": 0}
            rule_counts[key][issue.severity.value] = rule_counts[key].get(issue.severity.value, 0) + 1

        console.print()
        console.print(Rule("Rule Distribution"))
        rule_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        rule_table.add_column("Rule", style="cyan")
        rule_table.add_column("Errors", justify="right", style="red")
        rule_table.add_column("Warnings", justify="right", style="yellow")
        rule_table.add_column("Total", justify="right", style="bold")

        for rule, counts in sorted(rule_counts.items(), key=lambda x: sum(x[1].values()), reverse=True):
            errs = counts.get("error", 0)
            warns = counts.get("warn", 0)
            rule_table.add_row(rule, str(errs) if errs else "-", str(warns) if warns else "-", str(errs + warns))
        console.print(rule_table)

        # Issues detail table
        console.print()
        console.print(Rule("Issue Details"))
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Line:Col", style="cyan", width=10)
        table.add_column("Sev", width=7)
        table.add_column("Message", ratio=3)
        table.add_column("Rule", style="dim", ratio=1)

        for idx, issue in enumerate(result.issues[:50], 1):
            sev_style = "red" if issue.severity.value == "error" else "yellow"
            loc = f"{issue.line_number or '?'}:{issue.column_number or '?'}"
            table.add_row(
                str(idx),
                loc,
                f"[{sev_style}]{issue.severity.value.upper()}[/{sev_style}]",
                issue.message[:100],
                issue.rule,
            )

        console.print(table)
        if len(result.issues) > 50:
            console.print(f"\n  [dim]... and {len(result.issues) - 50} more issues (use --ci for full JSON output)[/dim]")
    else:
        console.print("\n  [green]No issues found — HTML is clean![/green]")
    console.print()


@html_app.command("batch")
def html_batch(
    urls_file: str = typer.Argument(..., help="File containing URLs (one per line)"),
    ci: bool = typer.Option(False, "--ci", help="CI mode"),
):
    """Validate HTML for multiple URLs from a file."""
    urls = Path(urls_file).read_text().strip().splitlines()
    console.print(f"Validating {len(urls)} URLs...")

    for url in urls:
        url = url.strip()
        if url:
            html_validate(url=url, ci=ci, config_path=None, site_id="default", environment="production")


# ========== SEO Audit ==========

@seo_app.command("audit")
def seo_audit(
    url: str = typer.Argument(..., help="URL to audit"),
    remote: bool = typer.Option(True, "--remote/--local", help="Remote or local mode"),
    ci: bool = typer.Option(False, "--ci", help="CI mode"),
    site_id: str = typer.Option("default", help="Site identifier"),
    environment: str = typer.Option("production", help="Environment name"),
):
    """Run SEO audit for a URL."""
    from src.checkers.seo_auditor import SEOAuditor

    config = _get_config()
    storage = _get_storage()

    checker = SEOAuditor(storage, config.tools, config.get_timeout())

    async def _run():
        await storage.initialize()
        return await checker.run(url, site_id=site_id, environment=environment,
                                  triggered_by=TriggeredBy.CLI, remote=remote)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running SEO audit...", total=None)
        run_id, result = _run_async(_run())

    if ci:
        output = {
            "run_id": run_id,
            "score": result.overall_score,
            "grade": result.grade,
            "pass": result.pass_count,
            "fail": result.fail_count,
            "warn": result.warn_count,
        }
        print(json.dumps(output, indent=2))
        sys.exit(1 if result.fail_count > 0 else 0)
        return

    score = result.overall_score or 0
    grade_colors = {"EXCELLENT": "green", "GOOD": "green", "NEEDS WORK": "yellow", "CRITICAL": "red"}
    grade_color = grade_colors.get(result.grade, "white")

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold]SEO Audit Results[/bold]\n"
        f"URL: [cyan]{url}[/cyan]\n"
        f"Run ID: [dim]#{run_id}[/dim]",
        style="blue",
    ))

    # Score + grade + summary
    console.print(f"\n  Score: {_score_bar(score)}")
    console.print(f"  Grade: [{grade_color}][bold]{result.grade}[/bold][/{grade_color}]")
    console.print(f"  Checks: {result.total_checks} total — "
                  f"[green]{result.pass_count} passed[/green], "
                  f"[red]{result.fail_count} failed[/red], "
                  f"[yellow]{result.warn_count} warnings[/yellow]")

    if result.items:
        # Section summary table
        console.print()
        console.print(Rule("Section Summary"))
        section_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        section_table.add_column("#", style="dim", width=3, justify="right")
        section_table.add_column("Section", ratio=2)
        section_table.add_column("Status", width=10)
        section_table.add_column("Items", width=8, justify="right")

        sections: dict[int, dict] = {}
        for item in result.items:
            sn = item.section_number
            if sn not in sections:
                sections[sn] = {"name": item.section_name, "pass": 0, "fail": 0, "warn": 0, "info": 0}
            sections[sn][item.status.value] = sections[sn].get(item.status.value, 0) + 1

        for sn in sorted(sections):
            sec = sections[sn]
            total_items = sum(sec[k] for k in ["pass", "fail", "warn", "info"])
            if sec["fail"] > 0:
                status_str = "[red]✗ FAIL[/red]"
            elif sec["warn"] > 0:
                status_str = "[yellow]⚠ WARN[/yellow]"
            else:
                status_str = "[green]✓ PASS[/green]"
            section_table.add_row(str(sn), sec["name"], status_str, str(total_items))
        console.print(section_table)

        # Detailed items grouped by section
        console.print()
        console.print(Rule("Detailed Results"))
        current_section = 0
        for item in result.items:
            if item.section_number != current_section:
                current_section = item.section_number
                console.print(f"\n  [bold]{item.section_number}. {item.section_name}[/bold]")

            console.print(f"    {_status_icon(item.status.value)}  {item.message}")
            if item.details and item.status.value in ("fail", "warn"):
                # Show first 2 lines of details for failures/warnings
                detail_lines = item.details.strip().splitlines()
                for dl in detail_lines[:2]:
                    console.print(f"      [dim]{dl.strip()}[/dim]")
                if len(detail_lines) > 2:
                    console.print(f"      [dim]... +{len(detail_lines) - 2} more lines[/dim]")
    console.print()


@seo_app.command("compare")
def seo_compare(
    site_name: str = typer.Argument(..., help="Site ID to compare across environments"),
):
    """Compare SEO audit results across environments for a site."""
    config = _get_config()
    storage = _get_storage()
    site = config.get_site(site_name)

    if not site:
        console.print(f"[red]Site '{site_name}' not found[/red]")
        raise typer.Exit(1)

    async def _run():
        await storage.initialize()
        results = {}
        for env in site.get_environment_names():
            latest = await storage.get_latest_run(CheckType.SEO_AUDIT, site_name, env)
            if latest:
                results[env] = latest
        return results

    results = _run_async(_run())

    table = Table(title=f"SEO Comparison: {site.name}")
    table.add_column("Environment")
    table.add_column("Score")
    table.add_column("Status")
    table.add_column("Last Run")

    for env, run in results.items():
        score_str = f"{run.score:.1f}" if run.score else "N/A"
        table.add_row(env.upper(), score_str, run.status.value, str(run.started_at))

    console.print(table)


# ========== Lighthouse ==========

@lighthouse_app.command("audit")
def lighthouse_audit(
    url: str = typer.Argument(..., help="URL to audit"),
    categories: str = typer.Option("seo,accessibility,best-practices", help="Comma-separated categories"),
    ci: bool = typer.Option(False, "--ci", help="CI mode"),
    site_id: str = typer.Option("default", help="Site identifier"),
    environment: str = typer.Option("production", help="Environment name"),
):
    """Run Lighthouse audit for a URL."""
    from src.checkers.lighthouse_runner import LighthouseRunner

    config = _get_config()
    storage = _get_storage()

    checker = LighthouseRunner(storage, config.tools, config.get_reports_dir(), config.get_timeout())
    cats = [c.strip() for c in categories.split(",")]

    async def _run():
        await storage.initialize()
        return await checker.run(url, site_id=site_id, environment=environment,
                                  triggered_by=TriggeredBy.CLI, categories=cats)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running Lighthouse audit...", total=None)
        run_id, result = _run_async(_run())

    if ci:
        scores = result.scores[0] if result.scores else None
        output = {
            "run_id": run_id,
            "score": checker._calculate_score(result),
            "performance": scores.performance if scores else None,
            "accessibility": scores.accessibility if scores else None,
            "best_practices": scores.best_practices if scores else None,
            "seo": scores.seo if scores else None,
            "failing_audits": len(result.audits),
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)
        return

    overall_score = checker._calculate_score(result)

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold]Lighthouse Audit Results[/bold]\n"
        f"URL: [cyan]{url}[/cyan]\n"
        f"Run ID: [dim]#{run_id}[/dim]  Categories: [dim]{categories}[/dim]",
        style="blue",
    ))

    console.print(f"\n  Overall Score: {_score_bar(overall_score)}")

    if result.scores:
        s = result.scores[0]
        console.print()
        console.print(Rule("Category Scores"))
        for name, val in [("Performance", s.performance), ("Accessibility", s.accessibility),
                          ("Best Practices", s.best_practices), ("SEO", s.seo)]:
            if val is not None:
                console.print(f"  {name:20s} {_score_bar(val)}")

        # Report paths
        if s.report_html_path:
            console.print(f"\n  [dim]HTML Report: {s.report_html_path}[/dim]")
        if s.report_json_path:
            console.print(f"  [dim]JSON Report: {s.report_json_path}[/dim]")

    if result.audits:
        console.print()
        console.print(Rule("Failing Audits"))
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Audit", ratio=2)
        table.add_column("Category", style="cyan", ratio=1)
        table.add_column("Score", width=8, justify="right")
        table.add_column("Description", ratio=2, style="dim")

        for idx, audit in enumerate(result.audits[:30], 1):
            score_str = f"{audit.score:.0f}" if audit.score is not None else "N/A"
            score_style = _score_color(audit.score) if audit.score is not None else "dim"
            desc = (audit.description[:80] + "...") if len(audit.description) > 80 else audit.description
            table.add_row(
                str(idx),
                audit.title,
                audit.category,
                f"[{score_style}]{score_str}[/{score_style}]",
                desc,
            )
        console.print(table)
        if len(result.audits) > 30:
            console.print(f"\n  [dim]... and {len(result.audits) - 30} more audits (use --ci for full JSON output)[/dim]")
    else:
        console.print("\n  [green]All audits passed![/green]")
    console.print()


@lighthouse_app.command("bulk")
def lighthouse_bulk(
    urls_file: str = typer.Argument(..., help="File containing URLs"),
):
    """Run Lighthouse audit on multiple URLs."""
    urls = Path(urls_file).read_text().strip().splitlines()
    console.print(f"Auditing {len(urls)} URLs...")

    for url in urls:
        url = url.strip()
        if url:
            lighthouse_audit(url=url, ci=False, categories="seo,accessibility,best-practices",
                             site_id="default", environment="production")


# ========== Broken Links ==========

@links_app.command("check")
def links_check(
    urls_file: str = typer.Argument(..., help="File containing URLs to check"),
    deep: bool = typer.Option(False, "--deep", help="Deep crawl mode"),
    strict: bool = typer.Option(False, "--strict", help="Strict mode (fail on broken)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds (auto-scales if not set)"),
    ci: bool = typer.Option(False, "--ci", help="CI mode"),
):
    """Check for broken links."""
    from src.checkers.broken_link_checker import BrokenLinkChecker

    config = _get_config()
    storage = _get_storage()

    urls = Path(urls_file).read_text().strip().splitlines()
    effective_timeout = timeout if timeout else config.get_timeout()
    checker = BrokenLinkChecker(storage, config.tools, effective_timeout)

    async def _run():
        await storage.initialize()
        return await checker.run(urls[0] if urls else "", site_id="default", environment="production",
                                  triggered_by=TriggeredBy.CLI, urls=urls, deep=deep, strict=strict)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Checking links...", total=None)
        run_id, result = _run_async(_run())

    if ci:
        output = {
            "run_id": run_id,
            "score": checker._calculate_score(result),
            "ok": result.ok_count,
            "broken": result.broken_count,
            "redirected": result.redirected_count,
            "timeout": result.timeout_count,
        }
        print(json.dumps(output, indent=2))
        sys.exit(1 if result.broken_count > 0 else 0)
        return

    score = checker._calculate_score(result)
    total = result.ok_count + result.broken_count + result.redirected_count + result.timeout_count

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold]Broken Link Check Results[/bold]\n"
        f"URLs checked: [cyan]{len(urls)}[/cyan] from [dim]{urls_file}[/dim]\n"
        f"Run ID: [dim]#{run_id}[/dim]",
        style="blue",
    ))

    # Score + summary
    console.print(f"\n  Score: {_score_bar(score)}")
    console.print(f"  Total Links Found: [bold]{total}[/bold]")
    console.print(f"    [green]✓ OK:         {result.ok_count:>5}[/green]")
    console.print(f"    [red]✗ Broken:      {result.broken_count:>5}[/red]")
    console.print(f"    [yellow]→ Redirected:  {result.redirected_count:>5}[/yellow]")
    console.print(f"    [red]⏱ Timeout:     {result.timeout_count:>5}[/red]")

    # Show broken/redirected/timeout links in detail
    problem_links = [l for l in result.links if l.link_state.value != "ok"]
    if problem_links:
        console.print()
        console.print(Rule("Problem Links"))
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Status", width=12)
        table.add_column("Code", width=6, justify="right")
        table.add_column("URL", ratio=3)
        table.add_column("Source", ratio=1, style="dim")
        table.add_column("Time", width=8, justify="right")

        for idx, link in enumerate(problem_links[:50], 1):
            code_str = str(link.status_code) if link.status_code else "-"
            time_str = f"{link.response_ms:.0f}ms" if link.response_ms else "-"
            source = link.source_page[:40] if link.source_page else "-"
            table.add_row(
                str(idx),
                _status_icon(link.link_state.value),
                code_str,
                link.url[:100],
                source,
                time_str,
            )
        console.print(table)
        if len(problem_links) > 50:
            console.print(f"\n  [dim]... and {len(problem_links) - 50} more problem links (use --ci for full JSON output)[/dim]")
    elif total > 0:
        console.print(f"\n  [green]All {total} links are healthy![/green]")
    console.print()


# ========== Load Test ==========

@loadtest_app.command("run")
def loadtest_run(
    scenario: str = typer.Argument("smoke", help="Test scenario"),
    host: Optional[str] = typer.Option(None, help="Target host override"),
    users: Optional[int] = typer.Option(None, help="Number of users override"),
    duration: Optional[int] = typer.Option(None, help="Duration override (seconds)"),
    jmeter: bool = typer.Option(False, "--jmeter", help="Force JMeter mode (requires JMeter installed)"),
    sla_error: Optional[float] = typer.Option(None, "--sla-error", help="Max error rate % (default: 2.0)"),
    sla_p95: Optional[int] = typer.Option(None, "--sla-p95", help="Max P95 response time in ms (default: 3000)"),
    sla_p99: Optional[int] = typer.Option(None, "--sla-p99", help="Max P99 response time in ms (default: 5000)"),
    sla_throughput: Optional[float] = typer.Option(None, "--sla-throughput", help="Min throughput req/s (default: 10.0)"),
    ci: bool = typer.Option(False, "--ci", help="CI mode"),
    site_id: str = typer.Option("default", help="Site identifier"),
    environment: str = typer.Option("production", help="Environment name"),
):
    """Run a load test (built-in HTTP tester by default, --jmeter for JMeter)."""
    from src.checkers.load_test_runner import LoadTestRunner

    config = _get_config()
    storage = _get_storage()

    checker = LoadTestRunner(storage, config.tools, timeout=3600)

    # Build URL — host may already contain protocol
    if host:
        if host.startswith(("http://", "https://")):
            url = host
        else:
            url = f"https://{host}"
    else:
        url = ""

    # Build SLA overrides dict (only include values the user actually set)
    sla_overrides = {}
    if sla_error is not None:
        sla_overrides["error_rate"] = sla_error
    if sla_p95 is not None:
        sla_overrides["p95_ms"] = sla_p95
    if sla_p99 is not None:
        sla_overrides["p99_ms"] = sla_p99
    if sla_throughput is not None:
        sla_overrides["throughput"] = sla_throughput

    async def _run():
        await storage.initialize()
        return await checker.run(url, site_id=site_id, environment=environment,
                                  triggered_by=TriggeredBy.CLI,
                                  scenario=scenario, host=host, users=users,
                                  duration=duration, jmeter=jmeter,
                                  sla=sla_overrides if sla_overrides else None)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Running {scenario} load test...", total=None)
        run_id, result = _run_async(_run())

    if ci:
        r = result.result
        output = {
            "run_id": run_id,
            "score": checker._calculate_score(result),
            "scenario": r.scenario if r else scenario,
            "total_requests": r.total_requests if r else 0,
            "error_rate": r.error_rate if r else 0,
            "throughput": r.throughput if r else 0,
            "p95_ms": r.p95_response_ms if r else 0,
            "sla": r.sla_overall.value if r else "unknown",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1 if r and r.sla_overall.value == "fail" else 0)
        return

    score = checker._calculate_score(result)

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold]Load Test Results[/bold]\n"
        f"Scenario: [cyan]{scenario}[/cyan]  Host: [cyan]{host or 'N/A'}[/cyan]\n"
        f"Run ID: [dim]#{run_id}[/dim]",
        style="blue",
    ))

    console.print(f"\n  Score: {_score_bar(score)}")

    if result.result:
        r = result.result
        sla_color = "green" if r.sla_overall.value == "pass" else "red"

        # SLA verdict banner
        console.print()
        sla_text = f"  SLA VERDICT: [{sla_color}][bold]{r.sla_overall.value.upper()}[/bold][/{sla_color}]"
        console.print(sla_text)

        # Test configuration
        console.print()
        console.print(Rule("Test Configuration"))
        console.print(f"  Scenario:       [bold]{r.scenario}[/bold]")
        console.print(f"  Target Host:    [cyan]{r.target_host}[/cyan]")
        console.print(f"  Virtual Users:  [bold]{r.users}[/bold]")
        console.print(f"  Duration:       [bold]{r.duration_seconds}s[/bold]")

        # Key metrics table
        console.print()
        console.print(Rule("Performance Metrics"))

        metrics_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 3))
        metrics_table.add_column("Metric", style="bold", width=22)
        metrics_table.add_column("Value", justify="right", width=14)

        # Throughput & requests
        metrics_table.add_row("Total Requests", f"[bold]{r.total_requests:,}[/bold]")
        err_color = "green" if r.error_rate < 1 else "yellow" if r.error_rate < 5 else "red"
        metrics_table.add_row("Error Rate", f"[{err_color}]{r.error_rate:.2f}%[/{err_color}]")
        metrics_table.add_row("Throughput", f"[bold]{r.throughput:.2f} req/s[/bold]")
        console.print(metrics_table)

        # Response time table
        console.print()
        console.print(Rule("Response Times"))

        rt_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 3))
        rt_table.add_column("Metric", style="bold", width=22)
        rt_table.add_column("Value", justify="right", width=14)

        rt_table.add_row("Min", f"{r.min_response_ms:.0f} ms")
        rt_table.add_row("Average", f"[bold]{r.avg_response_ms:.0f} ms[/bold]")

        p95_color = "green" if r.p95_response_ms < 2000 else "yellow" if r.p95_response_ms < 5000 else "red"
        rt_table.add_row("P95 (95th pctile)", f"[{p95_color}][bold]{r.p95_response_ms:.0f} ms[/bold][/{p95_color}]")

        p99_color = "green" if r.p99_response_ms < 3000 else "yellow" if r.p99_response_ms < 8000 else "red"
        rt_table.add_row("P99 (99th pctile)", f"[{p99_color}][bold]{r.p99_response_ms:.0f} ms[/bold][/{p99_color}]")

        rt_table.add_row("Max", f"{r.max_response_ms:.0f} ms")
        console.print(rt_table)
    else:
        console.print("\n  [yellow]No test metrics available — the test may not have produced results.[/yellow]")
        console.print("  [dim]Check that JMeter is installed and the test plan is configured correctly.[/dim]")
    console.print()


# ========== Reports ==========

@report_app.command("list")
def report_list(
    type: Optional[str] = typer.Option(None, help="Filter by type: html|seo|lighthouse|links|loadtest"),
):
    """List recent check runs."""
    storage = _get_storage()

    type_map = {
        "html": CheckType.HTML_VALIDATION,
        "seo": CheckType.SEO_AUDIT,
        "lighthouse": CheckType.LIGHTHOUSE,
        "links": CheckType.BROKEN_LINKS,
        "loadtest": CheckType.LOAD_TEST,
    }
    ct = type_map.get(type) if type else None

    async def _run():
        await storage.initialize()
        return await storage.get_recent_runs(limit=30, check_type=ct)

    runs = _run_async(_run())

    filter_label = f" (type: {type})" if type else ""
    console.print()
    table = Table(title=f"Recent Runs{filter_label}", show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=5, justify="right")
    table.add_column("Type", width=18)
    table.add_column("URL", ratio=2)
    table.add_column("Score", width=8, justify="right")
    table.add_column("Status", width=14)
    table.add_column("Time", width=20)

    for run in runs:
        score_str = f"{run.score:.1f}" if run.score else "--"
        score_color = _score_color(run.score) if run.score else "dim"
        table.add_row(
            str(run.id),
            run.check_type.value.replace("_", " ").title(),
            run.target_url[:60] if run.target_url else "-",
            f"[{score_color}]{score_str}[/{score_color}]",
            _status_icon(run.status.value),
            str(run.started_at)[:19] if run.started_at else "-",
        )

    console.print(table)
    console.print(f"\n  [dim]{len(runs)} run(s) shown. Use 'report view <ID>' for details.[/dim]")
    console.print()


@report_app.command("view")
def report_view(
    report_id: int = typer.Argument(..., help="Run ID to view"),
):
    """View details of a specific check run."""
    storage = _get_storage()

    async def _run():
        await storage.initialize()
        run = await storage.get_run(report_id)
        if not run:
            return None, None

        detail = None
        if run.check_type == CheckType.HTML_VALIDATION:
            detail = await storage.get_html_issues(report_id)
        elif run.check_type == CheckType.SEO_AUDIT:
            detail = await storage.get_seo_items(report_id)
        elif run.check_type == CheckType.LIGHTHOUSE:
            detail = await storage.get_lighthouse_scores(report_id)
        elif run.check_type == CheckType.BROKEN_LINKS:
            detail = await storage.get_broken_links(report_id)
        elif run.check_type == CheckType.LOAD_TEST:
            detail = await storage.get_load_test_result(report_id)
        return run, detail

    run, detail = _run_async(_run())

    if not run:
        console.print(f"[red]Run #{report_id} not found[/red]")
        raise typer.Exit(1)

    score = run.score or 0
    console.print()
    console.print(Panel(
        f"[bold]Run #{run.id} — {run.check_type.value.replace('_', ' ').title()}[/bold]\n"
        f"URL: [cyan]{run.target_url}[/cyan]",
        style="blue",
    ))
    console.print(f"\n  Score:   {_score_bar(score)}")
    console.print(f"  Status:  {_status_icon(run.status.value)}")
    console.print(f"  Started: {run.started_at}")
    if run.completed_at:
        console.print(f"  Ended:   {run.completed_at}")
    if run.error_message:
        console.print(f"  Error:   [red]{run.error_message}[/red]")

    # Show summary JSON as formatted key-value pairs
    if run.summary_json:
        try:
            summary = json.loads(run.summary_json)
            console.print()
            console.print(Rule("Summary"))
            for k, v in summary.items():
                console.print(f"  {k.replace('_', ' ').title():20s} {v}")
        except (json.JSONDecodeError, TypeError):
            console.print(f"  Summary: {run.summary_json}")

    # Show detail items if available
    if detail:
        console.print()
        console.print(Rule("Details"))
        if run.check_type == CheckType.HTML_VALIDATION and isinstance(detail, list):
            console.print(f"  {len(detail)} issue(s) found")
            for item in detail[:15]:
                sev = getattr(item, "severity", None)
                sev_val = sev.value if sev else "info"
                sev_style = "red" if sev_val == "error" else "yellow"
                msg = getattr(item, "message", str(item))
                rule = getattr(item, "rule", "")
                line = getattr(item, "line_number", "")
                console.print(f"    [{sev_style}]{sev_val.upper():5s}[/{sev_style}]  L{line}  {msg}  [dim]({rule})[/dim]")
            if len(detail) > 15:
                console.print(f"    [dim]... and {len(detail) - 15} more[/dim]")

        elif run.check_type == CheckType.BROKEN_LINKS and isinstance(detail, list):
            console.print(f"  {len(detail)} link(s) found")
            for item in detail[:15]:
                state = getattr(item, "link_state", None)
                state_val = state.value if state else "ok"
                code = getattr(item, "status_code", "")
                console.print(f"    {_status_icon(state_val)}  [{'' if state_val == 'ok' else 'dim'}]{code}[/]  {getattr(item, 'url', item)}")
            if len(detail) > 15:
                console.print(f"    [dim]... and {len(detail) - 15} more[/dim]")

        elif run.check_type == CheckType.SEO_AUDIT and isinstance(detail, list):
            console.print(f"  {len(detail)} check(s)")
            current_section = 0
            for item in detail[:30]:
                sn = getattr(item, "section_number", 0)
                if sn != current_section:
                    current_section = sn
                    sname = getattr(item, "section_name", "")
                    console.print(f"\n  [bold]{sn}. {sname}[/bold]")
                status = getattr(item, "status", None)
                status_val = status.value if status else "info"
                msg = getattr(item, "message", str(item))
                console.print(f"    {_status_icon(status_val)}  {msg}")

        elif run.check_type == CheckType.LOAD_TEST:
            # detail is a LoadTestResult object, not a list
            r = detail
            console.print(f"  Scenario:       {getattr(r, 'scenario', 'N/A')}")
            console.print(f"  Target Host:    {getattr(r, 'target_host', 'N/A')}")
            console.print(f"  Virtual Users:  {getattr(r, 'users', 0)}")
            console.print(f"  Duration:       {getattr(r, 'duration_seconds', 0)}s")
            console.print(f"  Total Requests: {getattr(r, 'total_requests', 0):,}")
            err = getattr(r, 'error_rate', 0)
            err_color = "green" if err < 1 else "yellow" if err < 5 else "red"
            console.print(f"  Error Rate:     [{err_color}]{err:.2f}%[/{err_color}]")
            console.print(f"  Throughput:     {getattr(r, 'throughput', 0):.2f} req/s")
            console.print(f"  Avg Response:   {getattr(r, 'avg_response_ms', 0):.0f} ms")
            console.print(f"  P95 Response:   {getattr(r, 'p95_response_ms', 0):.0f} ms")
            console.print(f"  P99 Response:   {getattr(r, 'p99_response_ms', 0):.0f} ms")
            sla = getattr(r, 'sla_overall', None)
            sla_val = sla.value if sla else "unknown"
            sla_color = "green" if sla_val == "pass" else "red"
            console.print(f"  SLA Verdict:    [{sla_color}]{sla_val.upper()}[/{sla_color}]")

        elif run.check_type == CheckType.LIGHTHOUSE:
            # detail could be a LighthouseScore or list of scores
            scores_list = detail if isinstance(detail, list) else [detail]
            for s in scores_list:
                for name, attr in [("Performance", "performance"), ("Accessibility", "accessibility"),
                                   ("Best Practices", "best_practices"), ("SEO", "seo")]:
                    val = getattr(s, attr, None)
                    if val is not None:
                        console.print(f"  {name:20s} {_score_bar(val)}")
        else:
            console.print(f"  [dim]{detail}[/dim]")
    console.print()


# ========== Composite Score ==========

@app.command("score")
def show_score(
    site_name: str = typer.Argument(..., help="Site ID"),
    environment: str = typer.Option("", help="Environment (default: first)"),
):
    """Show composite health score for a site."""
    config = _get_config()
    storage = _get_storage()

    site = config.get_site(site_name)
    if not site:
        console.print(f"[red]Site '{site_name}' not found[/red]")
        raise typer.Exit(1)

    env = environment or site.get_environment_names()[0]

    async def _run():
        await storage.initialize()
        scorer = HealthScorer(storage, config.get_weights())
        return await scorer.calculate_score(site_name, env)

    score = _run_async(_run())

    console.print()
    console.print(Panel(
        f"[bold]{site.name}[/bold]\n"
        f"Environment: [cyan]{env.upper()}[/cyan]",
        style="blue",
    ))

    console.print(f"\n  Composite Score: {_score_bar(score.composite_score)}")
    console.print()
    console.print(Rule("Individual Scores"))

    score_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    score_table.add_column("Check Type", width=20)
    score_table.add_column("Score", width=30)
    score_table.add_column("Status", width=10)

    for name, val in [("HTML Validation", score.html_score), ("SEO Audit", score.seo_score),
                      ("Lighthouse", score.lighthouse_score), ("Broken Links", score.broken_links_score),
                      ("Load Test", score.load_test_score)]:
        if val is not None:
            status = "[green]✓[/green]" if val >= 80 else "[yellow]⚠[/yellow]" if val >= 60 else "[red]✗[/red]"
            score_table.add_row(name, _score_bar(val), status)
        else:
            score_table.add_row(name, "[dim]No data available[/dim]", "[dim]-[/dim]")

    console.print(score_table)
    console.print()


# ========== Dashboard Launcher ==========

@app.command("dashboard")
def launch_dashboard(
    port: int = typer.Option(8082, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the Hygiene Check Dashboard web server."""
    import uvicorn

    console.print(Panel("[bold]Starting Hygiene Check Dashboard[/bold]", style="blue"))
    console.print(f"  Dashboard: http://localhost:{port}/dashboard")
    console.print(f"  API Docs:  http://localhost:{port}/docs")

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()
