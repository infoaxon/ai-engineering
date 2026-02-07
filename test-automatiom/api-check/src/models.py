"""Pydantic models for API configuration and health check results."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health status of an API endpoint."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HTTPMethod(str, Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"


class ContentType(str, Enum):
    """Supported content types."""

    JSON = "application/json"
    XML = "application/xml"
    TEXT_XML = "text/xml"
    FORM = "application/x-www-form-urlencoded"


class APIConfig(BaseModel):
    """Configuration for a single API endpoint."""

    name: str
    url: str
    method: HTTPMethod = HTTPMethod.GET
    expected_status: int = 200
    expected_response: Optional[dict[str, Any]] = None
    latency_threshold_ms: Optional[int] = None
    headers: Optional[dict[str, str]] = None
    body: Optional[dict[str, Any]] = None
    raw_body: Optional[str] = None  # For XML/raw content
    content_type: Optional[str] = None  # Content-Type header value
    timeout_seconds: Optional[int] = None
    # Error field validation - check response body instead of status code
    check_error_field: bool = False
    error_field: str = "ErrorMessages"  # Field to check for errors in response


class EnvironmentConfig(BaseModel):
    """Configuration for an environment."""

    name: str
    apis: list[APIConfig] = Field(default_factory=list)


class CustomerConfig(BaseModel):
    """Configuration for a customer."""

    customer_id: str
    name: str
    description: str = ""
    postman_collection: str = ""  # Legacy single collection (backward compatibility)
    postman_collections: list[str] = Field(default_factory=list)  # Multiple collections
    environments: list[str] = Field(
        default_factory=lambda: ["dev", "sit", "uat", "production"]
    )
    active: bool = True

    @property
    def all_collections(self) -> list[str]:
        """Get all collection paths (merges legacy and new list for backward compatibility)."""
        collections = []
        # Add legacy single collection if present
        if self.postman_collection:
            collections.append(self.postman_collection)
        # Add new multiple collections
        for coll in self.postman_collections:
            if coll and coll not in collections:
                collections.append(coll)
        return collections


class CustomerCreateRequest(BaseModel):
    """Request model for creating/updating a customer via admin interface."""

    name: str
    description: str = ""
    customer_id: Optional[str] = None
    environments: list[str] = Field(
        default_factory=lambda: ["dev", "sit", "uat", "production"]
    )


class Settings(BaseModel):
    """Global application settings."""

    check_interval_minutes: int = 5
    default_timeout_seconds: int = 30
    latency_threshold_ms: int = 2000
    retention_days: int = 7


class AppConfig(BaseModel):
    """Complete application configuration for a customer."""

    customer_id: str = "default"
    settings: Settings = Field(default_factory=Settings)
    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)


class HealthCheckResult(BaseModel):
    """Result of a single health check."""

    customer_id: str = "default"
    api_name: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: HealthStatus
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    response_valid: bool = True


class APIStatus(BaseModel):
    """Current status of an API with recent history."""

    customer_id: str = "default"
    api_name: str
    environment: str
    current_status: HealthStatus
    last_check: Optional[datetime] = None
    last_latency_ms: Optional[float] = None
    last_error: Optional[str] = None
    uptime_24h: Optional[float] = None
    check_count_24h: int = 0


class EnvironmentStatus(BaseModel):
    """Status summary for an environment."""

    customer_id: str = "default"
    env_key: str
    name: str
    total_apis: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    last_check: Optional[datetime] = None
    apis: list[APIStatus] = Field(default_factory=list)


class CustomerStatus(BaseModel):
    """Status summary for a customer."""

    customer_id: str
    name: str
    description: str = ""
    total_apis: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    environments: list[EnvironmentStatus] = Field(default_factory=list)
