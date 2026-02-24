"""Async subprocess runner for shell commands."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ANSI escape code pattern
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color/escape codes from text."""
    return ANSI_RE.sub("", text)


@dataclass
class ShellResult:
    """Result of a shell command execution."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def clean_stdout(self) -> str:
        return strip_ansi(self.stdout)

    @property
    def clean_stderr(self) -> str:
        return strip_ansi(self.stderr)


async def run_command(
    cmd: list[str],
    timeout: int = 300,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> ShellResult:
    """Run a shell command asynchronously and capture output.

    Uses temp files for stdout/stderr to avoid 64KB pipe buffer truncation
    on commands that produce large output.
    """
    cmd_str = " ".join(cmd[:3]) + ("..." if len(cmd) > 3 else "")
    logger.info(f"Running command: {cmd_str}")

    stdout_fd, stdout_path = tempfile.mkstemp(prefix="sh_out_")
    stderr_fd, stderr_path = tempfile.mkstemp(prefix="sh_err_")

    try:
        with os.fdopen(stdout_fd, "w") as out_f, os.fdopen(stderr_fd, "w") as err_f:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=out_f,
                stderr=err_f,
                cwd=cwd,
                env=env,
            )

            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(f"Command timed out after {timeout}s: {cmd_str}")
                return ShellResult(
                    returncode=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    timed_out=True,
                )

        stdout_text = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
        stderr_text = Path(stderr_path).read_text(encoding="utf-8", errors="replace")

        return ShellResult(
            returncode=process.returncode or 0,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return ShellResult(
            returncode=-1,
            stdout="",
            stderr=f"Command not found: {cmd[0]}",
        )
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return ShellResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
        )
    finally:
        Path(stdout_path).unlink(missing_ok=True)
        Path(stderr_path).unlink(missing_ok=True)
