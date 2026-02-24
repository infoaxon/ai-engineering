"""Parser for html-validate CLI output."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..models import HTMLValidationIssue, Severity

logger = logging.getLogger(__name__)

# Pattern for stylish format: file:line:col: severity message [rule]
STYLISH_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s+(error|warning):\s+(.+?)\s+\[(.+?)\]$"
)


def parse_json_output(output: str, url: str = "") -> list[HTMLValidationIssue]:
    """Parse html-validate --formatter json output."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        logger.warning("Failed to parse html-validate JSON output, falling back to stylish parser")
        return parse_stylish_output(output, url)

    issues = []
    # JSON output is an array of file results
    if isinstance(data, list):
        for file_result in data:
            file_path = file_result.get("filePath", url)
            for msg in file_result.get("messages", []):
                severity_val = msg.get("severity", 2)
                severity = Severity.WARN if severity_val == 1 else Severity.ERROR
                issues.append(HTMLValidationIssue(
                    url=file_path,
                    line_number=msg.get("line"),
                    column_number=msg.get("column"),
                    severity=severity,
                    message=msg.get("message", ""),
                    rule=msg.get("ruleId", ""),
                ))
    return issues


def parse_stylish_output(output: str, url: str = "") -> list[HTMLValidationIssue]:
    """Parse html-validate --formatter stylish output."""
    issues = []
    for line in output.splitlines():
        line = line.strip()
        m = STYLISH_RE.match(line)
        if m:
            file_path, line_num, col_num, sev, message, rule = m.groups()
            severity = Severity.WARN if sev == "warning" else Severity.ERROR
            issues.append(HTMLValidationIssue(
                url=file_path or url,
                line_number=int(line_num),
                column_number=int(col_num),
                severity=severity,
                message=message,
                rule=rule,
            ))
    return issues


def calculate_score(error_count: int, warning_count: int) -> float:
    """Calculate HTML validation score (0-100)."""
    if error_count == 0 and warning_count == 0:
        return 100.0
    # Each error costs 5 points, each warning costs 1 point, min 0
    deduction = (error_count * 5) + (warning_count * 1)
    return max(0.0, 100.0 - deduction)
