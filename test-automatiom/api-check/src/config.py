"""Configuration loader for YAML files with environment variable support."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Union

import yaml

from .models import APIConfig, AppConfig, CustomerConfig, EnvironmentConfig, Settings


def substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    pattern = r"\$\{([^}]+)\}"

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replace, value)


def process_config_values(obj: dict | list | str) -> dict | list | str:
    """Recursively process configuration values for env var substitution."""
    if isinstance(obj, dict):
        return {k: process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process_config_values(item) for item in obj]
    elif isinstance(obj, str):
        return substitute_env_vars(obj)
    return obj


def detect_content_type(body: str, options: dict) -> str:
    """Detect content type from body content and Postman options."""
    raw_options = options.get("raw", {})
    language = raw_options.get("language", "").lower()

    if language == "json":
        return "application/json"
    elif language == "xml":
        return "text/xml"

    body_stripped = body.strip()
    if body_stripped.startswith("{") or body_stripped.startswith("["):
        return "application/json"
    elif body_stripped.startswith("<"):
        return "text/xml"

    return "text/plain"


def parse_postman_collection(collection_path: Path) -> list[dict]:
    """Parse Postman collection and extract API definitions."""
    with open(collection_path, "r") as f:
        collection = json.load(f)

    apis = []

    for item in collection.get("item", []):
        name = item.get("name", "Unknown API")
        request = item.get("request", {})
        method = request.get("method", "GET")

        url = request.get("url", {})
        if isinstance(url, str):
            url_str = url
        else:
            url_str = url.get("raw", "")

        headers = {}
        for header in request.get("header", []):
            if not header.get("disabled", False):
                headers[header.get("key", "")] = header.get("value", "")

        body = request.get("body", {})
        raw_body = None
        content_type = None

        if body.get("mode") == "raw":
            raw_body = body.get("raw", "")
            body_options = body.get("options", {})
            content_type = detect_content_type(raw_body, body_options)

        api_config = {
            "name": name,
            "url": url_str,
            "method": method,
            "check_error_field": True,
            "error_field": "ErrorMessages",
            "latency_threshold_ms": 5000,
        }

        if headers:
            api_config["headers"] = headers
        if raw_body:
            api_config["raw_body"] = raw_body
        if content_type:
            api_config["content_type"] = content_type

        apis.append(api_config)

    return apis


def load_config(config_path: str | Path) -> AppConfig:
    """Load and parse YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    processed_config = process_config_values(raw_config)

    settings_data = processed_config.get("settings", {})
    settings = Settings(**settings_data)

    environments = {}
    env_data = processed_config.get("environments", {})

    for env_key, env_config in env_data.items():
        environments[env_key] = EnvironmentConfig(**env_config)

    return AppConfig(settings=settings, environments=environments)


class CustomerManager:
    """Manages customer configurations and their API configs."""

    def __init__(self, base_dir: Path, customers_config_path: Path):
        self.base_dir = base_dir
        self.customers_config_path = customers_config_path
        self._customers: dict[str, CustomerConfig] = {}
        self._configs: dict[str, AppConfig] = {}
        self._last_modified: Optional[float] = None

    def load_customers(self) -> dict[str, CustomerConfig]:
        """Load customer configurations from YAML file."""
        if not self.customers_config_path.exists():
            return {}

        with open(self.customers_config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if not raw_config or "customers" not in raw_config:
            return {}

        customers = {}
        for customer_id, customer_data in raw_config["customers"].items():
            if customer_data.get("active", True):
                customers[customer_id] = CustomerConfig(
                    customer_id=customer_id,
                    name=customer_data.get("name", customer_id),
                    description=customer_data.get("description", ""),
                    postman_collection=customer_data.get("postman_collection", ""),
                    postman_collections=customer_data.get("postman_collections", []),
                    environments=customer_data.get(
                        "environments", ["dev", "sit", "uat", "production"]
                    ),
                    active=customer_data.get("active", True),
                )

        self._customers = customers
        return customers

    def load_customer_config(self, customer_id: str) -> Optional[AppConfig]:
        """Load API configuration for a specific customer from their Postman collections."""
        if customer_id not in self._customers:
            self.load_customers()

        if customer_id not in self._customers:
            return None

        customer = self._customers[customer_id]

        # Parse and merge APIs from all collections
        all_apis = []
        for collection_rel_path in customer.all_collections:
            collection_path = self.base_dir / collection_rel_path
            if not collection_path.exists():
                print(
                    f"Warning: Postman collection not found for {customer_id}: {collection_path}"
                )
                continue
            apis = parse_postman_collection(collection_path)
            all_apis.extend(apis)

        if not all_apis:
            print(f"Warning: No APIs loaded for {customer_id}")
            return None

        api_configs = [APIConfig(**api) for api in all_apis]

        # Create environment configs
        env_names = {
            "dev": "Development",
            "sit": "System Integration Testing",
            "uat": "User Acceptance Testing",
            "production": "Production",
        }

        environments = {}
        for env_key in customer.environments:
            environments[env_key] = EnvironmentConfig(
                name=env_names.get(env_key, env_key.title()), apis=api_configs
            )

        config = AppConfig(
            customer_id=customer_id, settings=Settings(), environments=environments
        )

        self._configs[customer_id] = config
        return config

    def get_all_configs(self) -> dict[str, AppConfig]:
        """Get configurations for all active customers."""
        self.load_customers()

        for customer_id in self._customers:
            if customer_id not in self._configs:
                self.load_customer_config(customer_id)

        return self._configs

    @property
    def customers(self) -> dict[str, CustomerConfig]:
        """Get all customer configurations."""
        if not self._customers:
            self.load_customers()
        return self._customers

    def reload(self) -> None:
        """Force reload all configurations."""
        self._customers = {}
        self._configs = {}
        self.load_customers()
        self.get_all_configs()

    def add_customer(
        self,
        customer_id: str,
        name: str,
        description: str = "",
        postman_collection: str = "",
        postman_collections: Optional[list[str]] = None,
        environments: Optional[list[str]] = None,
    ) -> bool:
        """Add a new customer to the configuration.

        Args:
            customer_id: Unique identifier for the customer
            name: Display name for the customer
            description: Optional description
            postman_collection: Path to the Postman collection file (legacy single)
            postman_collections: List of paths to Postman collection files
            environments: List of environments (default: dev, sit, uat, production)

        Returns:
            True if customer was added successfully, False otherwise
        """
        if environments is None:
            environments = ["dev", "sit", "uat", "production"]
        if postman_collections is None:
            postman_collections = []

        customer_data = {
            "name": name,
            "description": description,
            "postman_collection": postman_collection,
            "postman_collections": postman_collections,
            "environments": environments,
            "active": True,
        }

        if not self.customers_config_path.exists():
            config_data = {"customers": {}}
        else:
            with open(self.customers_config_path, "r") as f:
                config_data = yaml.safe_load(f) or {"customers": {}}

        if "customers" not in config_data:
            config_data["customers"] = {}

        config_data["customers"][customer_id] = customer_data

        self._save_customers_config(config_data)
        self.reload()
        return True

    def update_customer(
        self,
        customer_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        postman_collection: Optional[str] = None,
        postman_collections: Optional[list[str]] = None,
        environments: Optional[list[str]] = None,
    ) -> bool:
        """Update an existing customer's configuration.

        Args:
            customer_id: Unique identifier for the customer
            name: New display name (optional)
            description: New description (optional)
            postman_collection: New path to Postman collection (optional, legacy)
            postman_collections: New list of paths to Postman collections (optional)
            environments: New list of environments (optional)

        Returns:
            True if customer was updated successfully, False otherwise
        """
        if not self.customers_config_path.exists():
            return False

        with open(self.customers_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {"customers": {}}

        if customer_id not in config_data.get("customers", {}):
            return False

        customer_data = config_data["customers"][customer_id]

        if name is not None:
            customer_data["name"] = name
        if description is not None:
            customer_data["description"] = description
        if postman_collection is not None:
            customer_data["postman_collection"] = postman_collection
        if postman_collections is not None:
            customer_data["postman_collections"] = postman_collections
        if environments is not None:
            customer_data["environments"] = environments

        config_data["customers"][customer_id] = customer_data

        self._save_customers_config(config_data)
        self.reload()
        return True

    def delete_customer(self, customer_id: str) -> bool:
        """Delete a customer from the configuration.

        Also removes the associated collection files if they exist.

        Args:
            customer_id: Unique identifier for the customer to delete

        Returns:
            True if customer was deleted successfully, False otherwise
        """
        if not self.customers_config_path.exists():
            return False

        with open(self.customers_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {"customers": {}}

        if customer_id not in config_data.get("customers", {}):
            return False

        customer_data = config_data["customers"][customer_id]

        # Delete legacy single collection
        collection_path = customer_data.get("postman_collection", "")
        if collection_path:
            full_path = self.base_dir / collection_path
            if full_path.exists():
                full_path.unlink()

        # Delete all collections in the list
        for coll_path in customer_data.get("postman_collections", []):
            if coll_path:
                full_path = self.base_dir / coll_path
                if full_path.exists():
                    full_path.unlink()

        # Delete customer directory if it exists and is empty
        customer_dir = self.base_dir / "collections" / customer_id
        if customer_dir.exists() and customer_dir.is_dir():
            try:
                customer_dir.rmdir()  # Only removes if empty
            except OSError:
                pass  # Directory not empty, leave it

        del config_data["customers"][customer_id]

        self._save_customers_config(config_data)
        self.reload()
        return True

    def _save_customers_config(self, config_data: dict) -> None:
        """Save the customers configuration to YAML file."""
        self.customers_config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.customers_config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    def get_customer_apis(self, customer_id: str) -> list[dict]:
        """Get the merged list of APIs for a customer from all collections.

        Args:
            customer_id: Unique identifier for the customer

        Returns:
            List of API configurations as dictionaries
        """
        if customer_id not in self._customers:
            self.load_customers()

        if customer_id not in self._customers:
            return []

        customer = self._customers[customer_id]
        all_apis = []

        for collection_rel_path in customer.all_collections:
            collection_path = self.base_dir / collection_rel_path
            if collection_path.exists():
                apis = parse_postman_collection(collection_path)
                all_apis.extend(apis)

        return all_apis

    def get_customer_collections(self, customer_id: str) -> list[dict]:
        """Get list of collections for a customer with metadata.

        Args:
            customer_id: Unique identifier for the customer

        Returns:
            List of collection info dicts with filename, api_count, modified_date, path
        """
        if customer_id not in self._customers:
            self.load_customers()

        if customer_id not in self._customers:
            return []

        customer = self._customers[customer_id]
        collections = []

        for collection_rel_path in customer.all_collections:
            collection_path = self.base_dir / collection_rel_path
            if collection_path.exists():
                apis = parse_postman_collection(collection_path)
                stat = collection_path.stat()
                collections.append(
                    {
                        "filename": collection_path.name,
                        "path": collection_rel_path,
                        "api_count": len(apis),
                        "modified_date": stat.st_mtime,
                        "size_bytes": stat.st_size,
                    }
                )

        return collections

    def delete_collection(self, customer_id: str, collection_path: str) -> bool:
        """Delete a specific collection from a customer.

        Args:
            customer_id: Unique identifier for the customer
            collection_path: Relative path to the collection file

        Returns:
            True if collection was deleted successfully, False otherwise
        """
        if not self.customers_config_path.exists():
            return False

        with open(self.customers_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {"customers": {}}

        if customer_id not in config_data.get("customers", {}):
            return False

        customer_data = config_data["customers"][customer_id]

        # Check if it's the legacy single collection
        if customer_data.get("postman_collection") == collection_path:
            customer_data["postman_collection"] = ""

        # Remove from postman_collections list
        collections_list = customer_data.get("postman_collections", [])
        if collection_path in collections_list:
            collections_list.remove(collection_path)
            customer_data["postman_collections"] = collections_list

        # Delete the actual file
        full_path = self.base_dir / collection_path
        if full_path.exists():
            full_path.unlink()

        config_data["customers"][customer_id] = customer_data
        self._save_customers_config(config_data)
        self.reload()
        return True

    def add_collections_to_customer(
        self, customer_id: str, collection_paths: list[str]
    ) -> bool:
        """Add collection paths to an existing customer.

        Args:
            customer_id: Unique identifier for the customer
            collection_paths: List of relative paths to collection files

        Returns:
            True if collections were added successfully, False otherwise
        """
        if not self.customers_config_path.exists():
            return False

        with open(self.customers_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {"customers": {}}

        if customer_id not in config_data.get("customers", {}):
            return False

        customer_data = config_data["customers"][customer_id]
        existing = set(customer_data.get("postman_collections", []))
        # Also check legacy field
        if customer_data.get("postman_collection"):
            existing.add(customer_data["postman_collection"])

        # Add new paths that don't already exist
        for path in collection_paths:
            if path not in existing:
                existing.add(path)

        customer_data["postman_collections"] = list(existing)
        config_data["customers"][customer_id] = customer_data

        self._save_customers_config(config_data)
        self.reload()
        return True


class ConfigManager:
    """Manages configuration with optional reload support."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
        self._last_modified: Optional[float] = None

    @property
    def config(self) -> AppConfig:
        """Get current configuration, reloading if file changed."""
        current_mtime = (
            self.config_path.stat().st_mtime if self.config_path.exists() else None
        )

        if self._config is None or current_mtime != self._last_modified:
            self._config = load_config(self.config_path)
            self._last_modified = current_mtime

        return self._config

    def reload(self) -> AppConfig:
        """Force reload configuration from file."""
        self._config = load_config(self.config_path)
        self._last_modified = self.config_path.stat().st_mtime
        return self._config
