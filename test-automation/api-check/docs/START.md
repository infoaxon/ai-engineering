# Startup Guide - API Health Check

## Prerequisites

- Python 3.12+
- pip

## Quick Start

```bash
# 1. Navigate to the project directory
cd api-check

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the application
python run.py
```

The server starts on **http://localhost:8081** by default.

## Startup Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8081` | Port to bind to |
| `--reload` | off | Enable auto-reload (for development) |
| `--log-level` | `info` | Log level: `debug`, `info`, `warning`, `error` |

Example with all options:

```bash
python run.py --port 9000 --reload --log-level debug
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | App environment (`development` / `production`). Production disables `/docs` and `/redoc`. |
| `CHECK_INTERVAL_MINUTES` | `5` | Interval between scheduled health checks |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | _(none)_ | Path to log file. If set, logs are written to a rotating file (10 MB, 5 backups). |

## Accessing the Application

Once running, open these URLs in your browser:

- **Dashboard**: http://localhost:8081/dashboard
- **API Docs (Swagger)**: http://localhost:8081/docs (disabled in production)
- **ReDoc**: http://localhost:8081/redoc (disabled in production)

## Configuration

### Customer & API Configuration

- `config/customers.yaml` - Defines customers, their Postman collections, and active environments.
- `config/apis.yaml` - Auto-generated API definitions parsed from Postman collections.

### Postman Collections

Place Postman collection JSON files in the `collections/` directory. The app parses these to build the API list for health checking.

## What Happens on Startup

1. SQLite database (`health_checks.db`) is initialized in the project root.
2. An initial health check runs against all configured APIs.
3. A scheduler starts, repeating checks every `CHECK_INTERVAL_MINUTES`.

## Stopping the Application

### If running in the foreground

Press `Ctrl+C` in the terminal where the app is running.

### If the terminal is no longer available

Find and kill the process by port:

```bash
# Find the process using the port (default 8081)
lsof -i :8081

# Kill it using the PID from the output above
kill <PID>

# If the process doesn't stop, force kill
kill -9 <PID>
```

### Stop all running instances

```bash
# Find all python processes running the app
ps aux | grep 'run.py\|src.main:app' | grep -v grep

# Kill a specific instance by PID
kill <PID>
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Address already in use` | Another process is using the port. Run `lsof -i :<port>` to find it, or use `--port` to pick a different one. |
| `ModuleNotFoundError` | Ensure the virtual environment is activated and dependencies are installed. |
| No APIs shown on dashboard | Check that `config/customers.yaml` has active customers and valid Postman collection paths in `collections/`. |
