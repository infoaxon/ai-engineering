"""Pydantic models for all hygiene check types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class CheckType(str, Enum):
    HTML_VALIDATION = "html_validation"
    SEO_AUDIT = "seo_audit"
    LIGHTHOUSE = "lighthouse"
    BROKEN_LINKS = "broken_links"
    LOAD_TEST = "load_test"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


class LinkState(str, Enum):
    OK = "ok"
    BROKEN = "broken"
    REDIRECTED = "redirected"
    TIMEOUT = "timeout"


class SLAVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class TriggeredBy(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    CLI = "cli"


# --- Check Run (universal) ---

class CheckRun(BaseModel):
    id: Optional[int] = None
    check_type: CheckType
    site_id: str
    environment: str
    target_url: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    score: Optional[float] = None
    summary_json: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    triggered_by: TriggeredBy = TriggeredBy.MANUAL


# --- HTML Validation ---

class HTMLValidationIssue(BaseModel):
    id: Optional[int] = None
    run_id: int = 0
    url: str = ""
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    severity: Severity = Severity.ERROR
    message: str = ""
    rule: str = ""


class HTMLValidationResult(BaseModel):
    run: CheckRun
    issues: list[HTMLValidationIssue] = []
    error_count: int = 0
    warning_count: int = 0


# --- SEO Audit ---

class SEOAuditItem(BaseModel):
    id: Optional[int] = None
    run_id: int = 0
    section_number: int = 0
    section_name: str = ""
    status: CheckStatus = CheckStatus.INFO
    message: str = ""
    details: str = ""


class SEOAuditResult(BaseModel):
    run: CheckRun
    items: list[SEOAuditItem] = []
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    total_checks: int = 0
    overall_score: Optional[float] = None
    grade: str = ""


# --- Lighthouse ---

class LighthouseScore(BaseModel):
    id: Optional[int] = None
    run_id: int = 0
    url: str = ""
    performance: Optional[float] = None
    accessibility: Optional[float] = None
    best_practices: Optional[float] = None
    seo: Optional[float] = None
    report_json_path: Optional[str] = None
    report_html_path: Optional[str] = None


class LighthouseAudit(BaseModel):
    id: Optional[int] = None
    lighthouse_id: int = 0
    audit_id: str = ""
    title: str = ""
    category: str = ""
    score: Optional[float] = None
    description: str = ""


class LighthouseResult(BaseModel):
    run: CheckRun
    scores: list[LighthouseScore] = []
    audits: list[LighthouseAudit] = []


# --- Broken Links ---

class BrokenLinkItem(BaseModel):
    id: Optional[int] = None
    run_id: int = 0
    url: str = ""
    status_code: Optional[int] = None
    link_state: LinkState = LinkState.OK
    source_page: str = ""
    response_ms: Optional[float] = None


class BrokenLinkResult(BaseModel):
    run: CheckRun
    links: list[BrokenLinkItem] = []
    ok_count: int = 0
    broken_count: int = 0
    redirected_count: int = 0
    timeout_count: int = 0


# --- Load Test ---

class LoadTestResult(BaseModel):
    id: Optional[int] = None
    run_id: int = 0
    scenario: str = ""
    target_host: str = ""
    users: int = 0
    duration_seconds: int = 0
    total_requests: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0
    p95_response_ms: float = 0.0
    p99_response_ms: float = 0.0
    avg_response_ms: float = 0.0
    min_response_ms: float = 0.0
    max_response_ms: float = 0.0
    sla_overall: SLAVerdict = SLAVerdict.PASS


class LoadTestRunResult(BaseModel):
    run: CheckRun
    result: Optional[LoadTestResult] = None


# --- Site Health Score ---

class SiteHealthScore(BaseModel):
    id: Optional[int] = None
    site_id: str = ""
    environment: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    html_score: Optional[float] = None
    seo_score: Optional[float] = None
    lighthouse_score: Optional[float] = None
    broken_links_score: Optional[float] = None
    load_test_score: Optional[float] = None
    composite_score: float = 0.0


# --- Dashboard Summary ---

class CheckSummary(BaseModel):
    check_type: CheckType
    last_run: Optional[datetime] = None
    last_score: Optional[float] = None
    status: RunStatus = RunStatus.COMPLETED
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    trend: str = "stable"  # up, down, stable


class DashboardData(BaseModel):
    site_id: str = ""
    site_name: str = ""
    environment: str = ""
    composite_score: float = 0.0
    checks: list[CheckSummary] = []
    recent_runs: list[CheckRun] = []
