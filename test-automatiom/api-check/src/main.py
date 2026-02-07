"""FastAPI application entry point."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .admin import admin_router, setup_admin_routes
from .config import CustomerManager
from .routes import router, setup_routes
from .scheduler import HealthCheckScheduler
from .storage import Storage


def setup_logging() -> logging.Logger:
    """Configure logging based on environment."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_file = os.environ.get("LOG_FILE", "")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format)

    logger = logging.getLogger(__name__)

    # Add file handler if LOG_FILE is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


logger = setup_logging()
scheduler: Optional[HealthCheckScheduler] = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global scheduler

    # Read configuration from environment
    environment = os.environ.get("ENVIRONMENT", "development")
    check_interval = int(os.environ.get("CHECK_INTERVAL_MINUTES", "5"))

    base_dir = Path(__file__).parent.parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"
    db_path = base_dir / "health_checks.db"
    customers_config_path = base_dir / "config" / "customers.yaml"

    # Ensure required directories exist
    (base_dir / "logs").mkdir(exist_ok=True)
    (base_dir / "collections").mkdir(exist_ok=True)
    (base_dir / "config").mkdir(exist_ok=True)

    customer_manager = CustomerManager(base_dir, customers_config_path)
    storage = Storage(db_path)
    templates = Jinja2Templates(directory=templates_dir)

    scheduler = HealthCheckScheduler(customer_manager, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"Starting API Health Check application (env: {environment})...")
        await storage.initialize()

        # Run initial health check
        logger.info("Running initial health check...")
        await scheduler.run_all_checks()

        # Start scheduler with configured interval
        scheduler.start(interval_minutes=check_interval)
        logger.info(f"Scheduler started with {check_interval} minute interval")

        yield

        logger.info("Shutting down...")
        scheduler.stop()

    # Configure FastAPI based on environment
    docs_url = "/docs" if environment != "production" else None
    redoc_url = "/redoc" if environment != "production" else None

    app = FastAPI(
        title="API Health Check",
        description="Monitor API health across customers and environments",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    setup_routes(templates, customer_manager, storage)
    setup_admin_routes(templates, customer_manager, storage, base_dir)
    app.include_router(router)
    app.include_router(admin_router)

    return app


app = create_app()
