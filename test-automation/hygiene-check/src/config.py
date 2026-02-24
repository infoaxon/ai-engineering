"""YAML configuration loader and site manager."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def _resolve_vars(value: Any, variables: dict[str, str]) -> Any:
    """Recursively resolve ${var} placeholders in config values."""
    if isinstance(value, str):
        for var_name, var_value in variables.items():
            value = value.replace(f"${{{var_name}}}", var_value)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_vars(v, variables) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_vars(item, variables) for item in value]
    return value


class SiteConfig:
    """Configuration for a single site."""

    def __init__(self, site_id: str, data: dict):
        self.site_id = site_id
        self.name = data.get("name", site_id)
        self.environments = data.get("environments", {})
        self.html_validation = data.get("html_validation", {})
        self.lighthouse = data.get("lighthouse", {})
        self.load_test = data.get("load_test", {})

    def get_url(self, environment: str) -> Optional[str]:
        env = self.environments.get(environment, {})
        return env.get("url") if isinstance(env, dict) else None

    def get_environment_names(self) -> list[str]:
        return list(self.environments.keys())


class SiteManager:
    """Manages site configurations loaded from YAML."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._raw: dict = {}
        self._sites: dict[str, SiteConfig] = {}
        self.settings: dict = {}
        self.tools: dict = {}
        self.scoring: dict = {}
        self.load()

    def load(self) -> None:
        """Load and parse the YAML configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path) as f:
            self._raw = yaml.safe_load(f) or {}

        self.settings = self._raw.get("settings", {})
        tools_dir = self.settings.get("tools_dir", "")
        variables = {"tools_dir": tools_dir}

        self.tools = _resolve_vars(self._raw.get("tools", {}), variables)
        self.scoring = self._raw.get("scoring", {})

        sites_raw = self._raw.get("sites", {})
        resolved = _resolve_vars(sites_raw, variables)

        self._sites = {}
        for site_id, site_data in resolved.items():
            self._sites[site_id] = SiteConfig(site_id, site_data)

        logger.info(f"Loaded {len(self._sites)} site(s) from {self.config_path}")

    def get_site(self, site_id: str) -> Optional[SiteConfig]:
        return self._sites.get(site_id)

    def get_all_sites(self) -> list[SiteConfig]:
        return list(self._sites.values())

    def get_site_ids(self) -> list[str]:
        return list(self._sites.keys())

    def get_tool_path(self, tool_name: str) -> str:
        return self.tools.get(tool_name, tool_name)

    def get_weights(self) -> dict[str, float]:
        return self.scoring.get("weights", {
            "html_validation": 0.15,
            "seo_audit": 0.30,
            "lighthouse": 0.25,
            "broken_links": 0.15,
            "load_test": 0.15,
        })

    def get_reports_dir(self) -> str:
        return self.settings.get("reports_dir", "./reports")

    def get_timeout(self) -> int:
        return self.settings.get("default_timeout_seconds", 120)
