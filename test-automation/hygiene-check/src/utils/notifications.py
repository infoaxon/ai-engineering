"""Notification utilities (Phase 2: Slack/email)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def send_slack_notification(webhook_url: str, message: str) -> bool:
    """Send a Slack notification (Phase 2 placeholder)."""
    logger.info(f"Slack notification (not configured): {message[:100]}")
    return False


async def send_email_notification(to: str, subject: str, body: str) -> bool:
    """Send an email notification (Phase 2 placeholder)."""
    logger.info(f"Email notification (not configured): {subject}")
    return False
