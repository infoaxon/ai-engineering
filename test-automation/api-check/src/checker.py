"""Health check logic for API endpoints."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Tuple, Union

import httpx

from .models import APIConfig, HealthCheckResult, HealthStatus, HTTPMethod


def is_error_value_empty(value) -> bool:
    """Check if an error value should be considered empty/no error."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict):
        for v in value.values():
            if not is_error_value_empty(v):
                return False
        return True
    return False


def find_error_field(
    data: Union[dict, list], field_name: str
) -> Tuple[bool, Optional[str]]:
    """
    Recursively search for error field in response.
    Returns (has_errors, error_message).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == field_name:
                if is_error_value_empty(value):
                    return False, None

                if isinstance(value, list):
                    non_empty = [e for e in value if not is_error_value_empty(e)]
                    if not non_empty:
                        return False, None
                    error_msg = "; ".join(str(e) for e in non_empty[:3])
                    if len(non_empty) > 3:
                        error_msg += f" (+{len(non_empty)-3} more)"
                    return True, error_msg

                if isinstance(value, dict):
                    err_msg = (
                        value.get("ErrMessages")
                        or value.get("ErrorMessage")
                        or value.get("Message")
                    )
                    if err_msg and isinstance(err_msg, str) and err_msg.strip():
                        return True, str(value)[:200]
                    if is_error_value_empty(value):
                        return False, None

                return True, str(value)[:200]

            if isinstance(value, (dict, list)):
                found, error = find_error_field(value, field_name)
                if found:
                    return found, error

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                found, error = find_error_field(item, field_name)
                if found:
                    return found, error

    return False, None


async def check_api_health(
    api: APIConfig,
    environment: str,
    customer_id: str = "default",
    default_timeout: int = 30,
    default_latency_threshold: int = 2000,
) -> HealthCheckResult:
    """Perform health check on a single API endpoint."""
    timestamp = datetime.utcnow()
    timeout = api.timeout_seconds or default_timeout
    latency_threshold = api.latency_threshold_ms or default_latency_threshold

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            headers = dict(api.headers) if api.headers else {}

            content = None
            if api.raw_body:
                content = api.raw_body
                if api.content_type:
                    headers["Content-Type"] = api.content_type
                elif "<" in api.raw_body[:50]:
                    headers["Content-Type"] = "text/xml"
                elif api.raw_body.strip().startswith("{"):
                    headers["Content-Type"] = "application/json"

            json_body = api.body if not api.raw_body else None

            start_time = time.perf_counter()

            if api.method == HTTPMethod.GET:
                response = await client.get(api.url, headers=headers)
            elif api.method == HTTPMethod.POST:
                if content:
                    response = await client.post(
                        api.url, headers=headers, content=content
                    )
                else:
                    response = await client.post(
                        api.url, headers=headers, json=json_body
                    )
            elif api.method == HTTPMethod.PUT:
                if content:
                    response = await client.put(
                        api.url, headers=headers, content=content
                    )
                else:
                    response = await client.put(
                        api.url, headers=headers, json=json_body
                    )
            elif api.method == HTTPMethod.DELETE:
                response = await client.delete(api.url, headers=headers)
            elif api.method == HTTPMethod.PATCH:
                if content:
                    response = await client.patch(
                        api.url, headers=headers, content=content
                    )
                else:
                    response = await client.patch(
                        api.url, headers=headers, json=json_body
                    )
            elif api.method == HTTPMethod.HEAD:
                response = await client.head(api.url, headers=headers)
            else:
                response = await client.get(api.url, headers=headers)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if api.check_error_field:
                if response.status_code >= 400:
                    return HealthCheckResult(
                        customer_id=customer_id,
                        api_name=api.name,
                        environment=environment,
                        timestamp=timestamp,
                        status=HealthStatus.UNHEALTHY,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                        error_message=f"HTTP error: {response.status_code}",
                        response_valid=False,
                    )

                try:
                    response_json = response.json()
                    has_errors, error_msg = find_error_field(
                        response_json, api.error_field
                    )

                    if has_errors:
                        return HealthCheckResult(
                            customer_id=customer_id,
                            api_name=api.name,
                            environment=environment,
                            timestamp=timestamp,
                            status=HealthStatus.UNHEALTHY,
                            status_code=response.status_code,
                            latency_ms=latency_ms,
                            error_message=f"{api.error_field}: {error_msg}",
                            response_valid=False,
                        )
                except Exception as e:
                    return HealthCheckResult(
                        customer_id=customer_id,
                        api_name=api.name,
                        environment=environment,
                        timestamp=timestamp,
                        status=HealthStatus.UNHEALTHY,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                        error_message=f"Failed to parse response: {str(e)}",
                        response_valid=False,
                    )
            else:
                if response.status_code != api.expected_status:
                    return HealthCheckResult(
                        customer_id=customer_id,
                        api_name=api.name,
                        environment=environment,
                        timestamp=timestamp,
                        status=HealthStatus.UNHEALTHY,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                        error_message=f"Expected status {api.expected_status}, got {response.status_code}",
                        response_valid=False,
                    )

            if latency_ms > latency_threshold:
                return HealthCheckResult(
                    customer_id=customer_id,
                    api_name=api.name,
                    environment=environment,
                    timestamp=timestamp,
                    status=HealthStatus.DEGRADED,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error_message=f"Latency {latency_ms:.0f}ms exceeds threshold {latency_threshold}ms",
                    response_valid=True,
                )

            return HealthCheckResult(
                customer_id=customer_id,
                api_name=api.name,
                environment=environment,
                timestamp=timestamp,
                status=HealthStatus.HEALTHY,
                status_code=response.status_code,
                latency_ms=latency_ms,
                response_valid=True,
            )

    except httpx.TimeoutException:
        return HealthCheckResult(
            customer_id=customer_id,
            api_name=api.name,
            environment=environment,
            timestamp=timestamp,
            status=HealthStatus.UNHEALTHY,
            error_message=f"Request timed out after {timeout}s",
            response_valid=False,
        )
    except httpx.ConnectError as e:
        return HealthCheckResult(
            customer_id=customer_id,
            api_name=api.name,
            environment=environment,
            timestamp=timestamp,
            status=HealthStatus.UNHEALTHY,
            error_message=f"Connection failed: {str(e)}",
            response_valid=False,
        )
    except Exception as e:
        return HealthCheckResult(
            customer_id=customer_id,
            api_name=api.name,
            environment=environment,
            timestamp=timestamp,
            status=HealthStatus.UNHEALTHY,
            error_message=f"Unexpected error: {str(e)}",
            response_valid=False,
        )


async def check_environment(
    env_key: str,
    apis: list[APIConfig],
    customer_id: str = "default",
    default_timeout: int = 30,
    default_latency_threshold: int = 2000,
) -> list[HealthCheckResult]:
    """Check all APIs in an environment."""
    results = []

    for api in apis:
        result = await check_api_health(
            api, env_key, customer_id, default_timeout, default_latency_threshold
        )
        results.append(result)

    return results
