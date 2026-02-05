# API Health Check Utility

A Python-based API health monitoring utility that checks API availability, validates responses, measures latency, and displays results on a Bootstrap dashboard with environment-specific views.

## Features

- **Multi-environment support**: Monitor APIs across dev, sit, uat, and production environments
- **Three-tier health checks**:
  - Availability: HTTP request succeeds with expected status code
  - Response validation: Response matches expected values
  - Latency monitoring: Response time within configurable thresholds
- **Dashboard views**:
  - Overview of all environments
  - Detailed environment-specific pages
  - Auto-refresh every 30 seconds
- **REST API**: JSON endpoints for programmatic access
- **Persistent storage**: SQLite database for historical data
- **Configurable**: YAML-based configuration with environment variable support

## Installation

1. Create a virtual environment:
```bash
cd api-check
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/apis.yaml` to configure your APIs:

```yaml
settings:
  check_interval_minutes: 5
  default_timeout_seconds: 30
  latency_threshold_ms: 2000
  retention_days: 7

environments:
  dev:
    name: "Development"
    apis:
      - name: "User Service"
        url: "https://dev-api.example.com/health"
        method: GET
        expected_status: 200
        latency_threshold_ms: 1000
        expected_response:
          status: "healthy"
        headers:
          Authorization: "Bearer ${DEV_API_TOKEN}"
```

### Configuration Options

**Settings:**
- `check_interval_minutes`: How often to run health checks
- `default_timeout_seconds`: HTTP request timeout
- `latency_threshold_ms`: Default latency threshold for degraded status
- `retention_days`: How long to keep historical data

**API Configuration:**
- `name`: Display name for the API
- `url`: API endpoint URL
- `method`: HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD)
- `expected_status`: Expected HTTP status code (default: 200)
- `expected_response`: Expected JSON response values (partial matching)
- `latency_threshold_ms`: Per-API latency threshold (overrides default)
- `headers`: Custom HTTP headers
- `body`: Request body for POST/PUT/PATCH requests
- `timeout_seconds`: Per-API timeout (overrides default)

### Environment Variables

Use `${VAR_NAME}` syntax in YAML values to substitute environment variables:

```yaml
headers:
  Authorization: "Bearer ${API_TOKEN}"
```

## Usage

Start the server:

```bash
python run.py
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8080)
- `--reload`: Enable auto-reload for development
- `--log-level`: Log level (debug, info, warning, error)

Example:
```bash
python run.py --port 8000 --reload
```

## Accessing the Dashboard

- **Main Dashboard**: http://localhost:8080/dashboard
- **Environment View**: http://localhost:8080/dashboard/{env}
- **API Documentation**: http://localhost:8080/docs

## REST API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Current status of all environments |
| `GET /api/status/{env}` | Status for specific environment |
| `GET /api/history/{env}/{api_name}` | Historical data for an API |
| `POST /api/check` | Manually trigger health check |

## Project Structure

```
api-check/
├── config/
│   └── apis.yaml              # API configuration
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration loader
│   ├── models.py              # Pydantic models
│   ├── checker.py             # Health check logic
│   ├── scheduler.py           # APScheduler setup
│   ├── storage.py             # SQLite storage
│   └── routes.py              # API routes
├── templates/
│   ├── base.html              # Base template
│   ├── dashboard.html         # Main dashboard
│   ├── environment.html       # Environment view
│   └── error.html             # Error page
├── static/
│   └── style.css              # Custom styles
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── run.py                     # Entry point
```

## Health Status Definitions

- **Healthy**: API responded with expected status code, response validated, latency within threshold
- **Degraded**: API is functional but response time exceeds latency threshold
- **Unhealthy**: API failed to respond, returned unexpected status, or response validation failed
- **Unknown**: No health check data available yet
