"""FastAPI application entry point for Hygiene Check Dashboard."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import SiteManager
from .routes import dashboard, html_validation, seo_audit, lighthouse, load_test, reports, api
from .scheduler import CheckScheduler
from .storage import Storage


def setup_logging() -> logging.Logger:
    """Configure logging based on environment."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger("hygiene-check")

    return logger


logger = setup_logging()
scheduler: Optional[CheckScheduler] = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global scheduler

    environment = os.environ.get("ENVIRONMENT", "development")
    check_interval = int(os.environ.get("CHECK_INTERVAL_MINUTES", "60"))

    base_dir = Path(__file__).parent.parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"
    db_path = base_dir / "hygiene_checks.db"
    config_path = base_dir / "config" / "sites.yaml"

    # Ensure required directories exist
    (base_dir / "logs").mkdir(exist_ok=True)
    (base_dir / "reports").mkdir(exist_ok=True)
    for sub in ["html-validation", "seo-audit", "lighthouse", "broken-links", "load-test"]:
        (base_dir / "reports" / sub).mkdir(exist_ok=True)

    # Set up log file
    log_file = base_dir / "logs" / "hygiene-check.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Initialize components
    site_manager = SiteManager(config_path)
    storage = Storage(db_path)
    templates = Jinja2Templates(directory=templates_dir)

    scheduler = CheckScheduler(site_manager, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"Starting Hygiene Check Dashboard (env: {environment})...")
        await storage.initialize()

        scheduler.start(interval_minutes=check_interval)
        logger.info(f"Scheduler started with {check_interval} minute interval")

        yield

        logger.info("Shutting down...")
        scheduler.stop()

    # Configure FastAPI
    docs_url = "/docs" if environment != "production" else None
    redoc_url = "/redoc" if environment != "production" else None

    app = FastAPI(
        title="Hygiene Check Dashboard",
        description="Unified Web Quality Monitoring — HTML Validation, SEO Audit, Lighthouse, Broken Links, Load Test",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Setup routes with dependencies
    dashboard.setup(templates, site_manager, storage)
    html_validation.setup(templates, site_manager, storage)
    seo_audit.setup(templates, site_manager, storage)
    lighthouse.setup(templates, site_manager, storage)
    load_test.setup(templates, site_manager, storage)
    reports.setup(templates, site_manager, storage)
    api.setup(site_manager, storage)

    # Include routers
    app.include_router(dashboard.router)
    app.include_router(html_validation.router)
    app.include_router(seo_audit.router)
    app.include_router(lighthouse.router)
    app.include_router(load_test.router)
    app.include_router(reports.router)
    app.include_router(api.router)

    return app


app = create_app()
