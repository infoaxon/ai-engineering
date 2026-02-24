"""Parser for seo-hygiene-check.sh output."""

from __future__ import annotations

import logging
import re

from ..models import SEOAuditItem, CheckStatus
from ..utils.shell import strip_ansi

logger = logging.getLogger(__name__)

# Section header: numbered title between separator lines
SECTION_RE = re.compile(r"^\s*(\d+)\.\s+(.+)$")

# Check result line: [PASS], [FAIL], [WARN], [INFO]
CHECK_RE = re.compile(r"^\s*\[(PASS|FAIL|WARN|INFO)\]\s+(.+)$")

# Summary counters
SUMMARY_RE = re.compile(
    r"PASS:\s*(\d+)\s*\|\s*FAIL:\s*(\d+)\s*\|\s*WARN:\s*(\d+)\s*\|\s*Total checks:\s*(\d+)"
)

# Overall score
SCORE_RE = re.compile(r"Overall Score:\s*(\d+)%\s*-\s*(.+)$")

SECTION_NAMES = {
    1: "HTTP Response & Redirect Chain",
    2: "Security Headers",
    3: "Cache-Control Headers - Static Assets",
    4: "Font Optimisation",
    5: "Compression (Gzip / Brotli)",
    6: "SEO Meta Tags",
    7: "Robots.txt & Sitemap",
    8: "Third-Party Script Loading",
    9: "Apache Configuration (Server-Side)",
    10: "Liferay Configuration (Server-Side)",
    11: "Performance Quick Checks",
    12: "Structured Data (Schema.org)",
}

STATUS_MAP = {
    "PASS": CheckStatus.PASS,
    "FAIL": CheckStatus.FAIL,
    "WARN": CheckStatus.WARN,
    "INFO": CheckStatus.INFO,
}


def parse_seo_output(raw_output: str) -> dict:
    """Parse the full output of seo-hygiene-check.sh.

    Returns dict with keys: items, pass_count, fail_count, warn_count,
    total_checks, overall_score, grade
    """
    output = strip_ansi(raw_output)
    lines = output.splitlines()

    items: list[SEOAuditItem] = []
    current_section_num = 0
    current_section_name = ""
    pass_count = 0
    fail_count = 0
    warn_count = 0
    total_checks = 0
    overall_score = 0.0
    grade = ""

    for line in lines:
        stripped = line.rstrip()

        # Check for section header
        section_match = SECTION_RE.match(stripped)
        if section_match:
            num = int(section_match.group(1))
            name = section_match.group(2).strip()
            if 1 <= num <= 12:
                current_section_num = num
                current_section_name = name
                continue

        # Check for result lines
        check_match = CHECK_RE.match(stripped)
        if check_match and current_section_num > 0:
            status_str = check_match.group(1)
            message = check_match.group(2).strip()
            items.append(SEOAuditItem(
                section_number=current_section_num,
                section_name=current_section_name,
                status=STATUS_MAP.get(status_str, CheckStatus.INFO),
                message=message,
            ))
            continue

        # Check for summary counters
        summary_match = SUMMARY_RE.search(stripped)
        if summary_match:
            pass_count = int(summary_match.group(1))
            fail_count = int(summary_match.group(2))
            warn_count = int(summary_match.group(3))
            total_checks = int(summary_match.group(4))
            continue

        # Check for overall score
        score_match = SCORE_RE.search(stripped)
        if score_match:
            overall_score = float(score_match.group(1))
            grade = score_match.group(2).strip()
            continue

    return {
        "items": items,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "total_checks": total_checks,
        "overall_score": overall_score,
        "grade": grade,
    }
