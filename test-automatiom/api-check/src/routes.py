"""FastAPI routes for the dashboard and API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .config import CustomerManager
from .models import CustomerStatus, EnvironmentStatus, HealthStatus
from .storage import Storage

router = APIRouter()

templates: Jinja2Templates = None
customer_manager: CustomerManager = None
storage: Storage = None


def setup_routes(
    _templates: Jinja2Templates,
    _customer_manager: CustomerManager,
    _storage: Storage
) -> None:
    """Initialize route dependencies."""
    global templates, customer_manager, storage
    templates = _templates
    customer_manager = _customer_manager
    storage = _storage


def no_cache_response(response: Response) -> Response:
    """Add no-cache headers to prevent browser caching."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.get("/", response_class=RedirectResponse)
async def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with all customers overview."""
    customers_status = []

    for customer_id, customer_config in customer_manager.customers.items():
        config = customer_manager.get_all_configs().get(customer_id)
        if not config:
            continue

        total_healthy = 0
        total_degraded = 0
        total_unhealthy = 0
        total_apis = 0
        environments = []

        for env_key, env_config in config.environments.items():
            api_names = [api.name for api in env_config.apis]
            statuses = await storage.get_environment_statuses(customer_id, env_key, api_names)

            healthy = sum(1 for s in statuses if s.current_status == HealthStatus.HEALTHY)
            degraded = sum(1 for s in statuses if s.current_status == HealthStatus.DEGRADED)
            unhealthy = sum(1 for s in statuses if s.current_status == HealthStatus.UNHEALTHY)

            total_healthy += healthy
            total_degraded += degraded
            total_unhealthy += unhealthy
            total_apis += len(env_config.apis)

            last_check = None
            for s in statuses:
                if s.last_check:
                    if last_check is None or s.last_check > last_check:
                        last_check = s.last_check

            env_status = EnvironmentStatus(
                customer_id=customer_id,
                env_key=env_key,
                name=env_config.name,
                total_apis=len(env_config.apis),
                healthy_count=healthy,
                degraded_count=degraded,
                unhealthy_count=unhealthy,
                last_check=last_check,
                apis=statuses
            )
            environments.append(env_status)

        customer_status = CustomerStatus(
            customer_id=customer_id,
            name=customer_config.name,
            description=customer_config.description,
            total_apis=total_apis,
            healthy_count=total_healthy,
            degraded_count=total_degraded,
            unhealthy_count=total_unhealthy,
            environments=environments
        )
        customers_status.append(customer_status)

    response = templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "customers": customers_status,
            "now": datetime.utcnow()
        }
    )
    return no_cache_response(response)


@router.get("/dashboard/{customer_id}", response_class=HTMLResponse)
async def customer_dashboard(request: Request, customer_id: str):
    """Customer-specific dashboard with all environments."""
    if customer_id not in customer_manager.customers:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Customer '{customer_id}' not found"},
            status_code=404
        )

    customer_config = customer_manager.customers[customer_id]
    config = customer_manager.get_all_configs().get(customer_id)

    if not config:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Configuration not found for '{customer_id}'"},
            status_code=404
        )

    environments = []
    total_healthy = 0
    total_degraded = 0
    total_unhealthy = 0

    for env_key, env_config in config.environments.items():
        api_names = [api.name for api in env_config.apis]
        statuses = await storage.get_environment_statuses(customer_id, env_key, api_names)

        healthy = sum(1 for s in statuses if s.current_status == HealthStatus.HEALTHY)
        degraded = sum(1 for s in statuses if s.current_status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for s in statuses if s.current_status == HealthStatus.UNHEALTHY)

        total_healthy += healthy
        total_degraded += degraded
        total_unhealthy += unhealthy

        last_check = None
        for s in statuses:
            if s.last_check:
                if last_check is None or s.last_check > last_check:
                    last_check = s.last_check

        env_status = EnvironmentStatus(
            customer_id=customer_id,
            env_key=env_key,
            name=env_config.name,
            total_apis=len(env_config.apis),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            last_check=last_check,
            apis=statuses
        )
        environments.append(env_status)

    customer_status = CustomerStatus(
        customer_id=customer_id,
        name=customer_config.name,
        description=customer_config.description,
        total_apis=sum(len(e.apis) for e in environments),
        healthy_count=total_healthy,
        degraded_count=total_degraded,
        unhealthy_count=total_unhealthy,
        environments=environments
    )

    response = templates.TemplateResponse(
        "customer.html",
        {
            "request": request,
            "customer": customer_status,
            "now": datetime.utcnow()
        }
    )
    return no_cache_response(response)


@router.get("/dashboard/{customer_id}/{env}", response_class=HTMLResponse)
async def environment_dashboard(request: Request, customer_id: str, env: str):
    """Environment-specific detailed view for a customer."""
    if customer_id not in customer_manager.customers:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Customer '{customer_id}' not found"},
            status_code=404
        )

    config = customer_manager.get_all_configs().get(customer_id)
    if not config or env not in config.environments:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Environment '{env}' not found"},
            status_code=404
        )

    customer_config = customer_manager.customers[customer_id]
    env_config = config.environments[env]
    api_names = [api.name for api in env_config.apis]
    statuses = await storage.get_environment_statuses(customer_id, env, api_names)

    api_configs = {api.name: api for api in env_config.apis}

    healthy = sum(1 for s in statuses if s.current_status == HealthStatus.HEALTHY)
    degraded = sum(1 for s in statuses if s.current_status == HealthStatus.DEGRADED)
    unhealthy = sum(1 for s in statuses if s.current_status == HealthStatus.UNHEALTHY)

    last_check = None
    for s in statuses:
        if s.last_check:
            if last_check is None or s.last_check > last_check:
                last_check = s.last_check

    env_status = EnvironmentStatus(
        customer_id=customer_id,
        env_key=env,
        name=env_config.name,
        total_apis=len(env_config.apis),
        healthy_count=healthy,
        degraded_count=degraded,
        unhealthy_count=unhealthy,
        last_check=last_check,
        apis=statuses
    )

    response = templates.TemplateResponse(
        "environment.html",
        {
            "request": request,
            "customer_id": customer_id,
            "customer_name": customer_config.name,
            "env": env_status,
            "api_configs": api_configs,
            "settings": config.settings,
            "now": datetime.utcnow()
        }
    )
    return no_cache_response(response)


@router.get("/api/status")
async def api_status_all():
    """JSON API for current status of all customers."""
    result = {}

    for customer_id, customer_config in customer_manager.customers.items():
        config = customer_manager.get_all_configs().get(customer_id)
        if not config:
            continue

        customer_data = {
            "name": customer_config.name,
            "environments": {}
        }

        for env_key, env_config in config.environments.items():
            api_names = [api.name for api in env_config.apis]
            statuses = await storage.get_environment_statuses(customer_id, env_key, api_names)

            customer_data["environments"][env_key] = {
                "name": env_config.name,
                "apis": [
                    {
                        "name": s.api_name,
                        "status": s.current_status.value,
                        "last_check": s.last_check.isoformat() if s.last_check else None,
                        "latency_ms": s.last_latency_ms,
                        "uptime_24h": s.uptime_24h,
                        "last_error": s.last_error
                    }
                    for s in statuses
                ]
            }

        result[customer_id] = customer_data

    return result


@router.get("/api/status/{customer_id}")
async def api_status_customer(customer_id: str):
    """JSON API for specific customer status."""
    if customer_id not in customer_manager.customers:
        return {"error": f"Customer '{customer_id}' not found"}

    customer_config = customer_manager.customers[customer_id]
    config = customer_manager.get_all_configs().get(customer_id)

    if not config:
        return {"error": f"Configuration not found for '{customer_id}'"}

    result = {
        "customer_id": customer_id,
        "name": customer_config.name,
        "environments": {}
    }

    for env_key, env_config in config.environments.items():
        api_names = [api.name for api in env_config.apis]
        statuses = await storage.get_environment_statuses(customer_id, env_key, api_names)

        result["environments"][env_key] = {
            "name": env_config.name,
            "apis": [
                {
                    "name": s.api_name,
                    "status": s.current_status.value,
                    "last_check": s.last_check.isoformat() if s.last_check else None,
                    "latency_ms": s.last_latency_ms,
                    "uptime_24h": s.uptime_24h,
                    "last_error": s.last_error
                }
                for s in statuses
            ]
        }

    return result


@router.get("/api/status/{customer_id}/{env}")
async def api_status_environment(customer_id: str, env: str):
    """JSON API for specific customer environment status."""
    if customer_id not in customer_manager.customers:
        return {"error": f"Customer '{customer_id}' not found"}

    config = customer_manager.get_all_configs().get(customer_id)
    if not config or env not in config.environments:
        return {"error": f"Environment '{env}' not found"}

    env_config = config.environments[env]
    api_names = [api.name for api in env_config.apis]
    statuses = await storage.get_environment_statuses(customer_id, env, api_names)

    return {
        "customer_id": customer_id,
        "environment": env,
        "name": env_config.name,
        "apis": [
            {
                "name": s.api_name,
                "status": s.current_status.value,
                "last_check": s.last_check.isoformat() if s.last_check else None,
                "latency_ms": s.last_latency_ms,
                "uptime_24h": s.uptime_24h,
                "last_error": s.last_error
            }
            for s in statuses
        ]
    }


@router.post("/api/check")
async def trigger_check():
    """Manually trigger a health check cycle."""
    from .main import scheduler as main_scheduler

    if main_scheduler:
        await main_scheduler.run_all_checks()
        return {"status": "ok", "message": "Health check triggered"}

    return {"status": "error", "message": "Scheduler not initialized"}
