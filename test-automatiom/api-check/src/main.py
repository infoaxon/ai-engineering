"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .admin import admin_router, setup_admin_routes
from .config import CustomerManager
from .routes import router, setup_routes
from .scheduler import HealthCheckScheduler
from .storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

scheduler: HealthCheckScheduler | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global scheduler

    base_dir = Path(__file__).parent.parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"
    db_path = base_dir / "health_checks.db"
    customers_config_path = base_dir / "config" / "customers.yaml"

    customer_manager = CustomerManager(base_dir, customers_config_path)
    storage = Storage(db_path)
    templates = Jinja2Templates(directory=templates_dir)

    scheduler = HealthCheckScheduler(customer_manager, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting API Health Check application...")
        await storage.initialize()

        logger.info("Running initial health check...")
        await scheduler.run_all_checks()

        scheduler.start(interval_minutes=5)

        yield

        logger.info("Shutting down...")
        scheduler.stop()

    app = FastAPI(
        title="API Health Check",
        description="Monitor API health across customers and environments",
        version="2.0.0",
        lifespan=lifespan
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    setup_routes(templates, customer_manager, storage)
    setup_admin_routes(templates, customer_manager, storage, base_dir)
    app.include_router(router)
    app.include_router(admin_router)

    return app


app = create_app()
