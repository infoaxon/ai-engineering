# Hygiene Check Dashboard -- CLI Reference

Complete reference for all CLI commands in the Hygiene Check Dashboard.

```
python3 cli.py [COMMAND] [SUBCOMMAND] [ARGS] [OPTIONS]
```

---

## Table of Contents

- [HTML Validation](#html-validation)
  - [html validate](#html-validate)
  - [html batch](#html-batch)
- [SEO Audit](#seo-audit)
  - [seo audit](#seo-audit-1)
  - [seo compare](#seo-compare)
- [Lighthouse](#lighthouse)
  - [lighthouse audit](#lighthouse-audit)
  - [lighthouse bulk](#lighthouse-bulk)
- [Broken Links](#broken-links)
  - [links check](#links-check)
- [Load Testing](#load-testing)
  - [loadtest run](#loadtest-run)
- [Reports](#reports)
  - [report list](#report-list)
  - [report view](#report-view)
- [Composite Score](#composite-score)
  - [score](#score)
- [Dashboard](#dashboard)
  - [dashboard](#dashboard-1)
- [Common Options](#common-options)
- [CI Mode](#ci-mode)
- [Configuration](#configuration)
- [Scoring Weights](#scoring-weights)

---

## HTML Validation

### `html validate`

Validate HTML for a single URL.

```bash
python3 cli.py html validate <URL> [OPTIONS]
```

| Argument / Option | Type   | Default        | Description                     |
|-------------------|--------|----------------|---------------------------------|
| `URL`             | string | *(required)*   | URL to validate                 |
| `--config-path`   | string | `None`         | Path to `.htmlvalidate.json`    |
| `--ci`            | flag   | `False`        | CI mode: JSON output + exit codes |
| `--site-id`       | string | `default`      | Site identifier                 |
| `--environment`   | string | `production`   | Environment name                |

**Examples:**

```bash
# Basic validation
python3 cli.py html validate https://ppdolphin.brobotinsurance.com

# With custom config
python3 cli.py html validate https://example.com --config-path ./my-rules.htmlvalidate.json

# CI mode (JSON output, exit code 1 on errors)
python3 cli.py html validate https://example.com --ci
```

**CI output:** JSON with `run_id`, `score`, `error_count`, `warning_count`, `issues[]`

---

### `html batch`

Validate HTML for multiple URLs from a file.

```bash
python3 cli.py html batch <URLS_FILE> [OPTIONS]
```

| Argument / Option | Type   | Default   | Description                        |
|-------------------|--------|-----------|------------------------------------|
| `URLS_FILE`       | string | *(required)* | File containing URLs (one per line) |
| `--ci`            | flag   | `False`   | CI mode                            |

**Example:**

```bash
python3 cli.py html batch urls.txt
python3 cli.py html batch urls.txt --ci
```

---

## SEO Audit

### `seo audit`

Run SEO audit for a URL.

```bash
python3 cli.py seo audit <URL> [OPTIONS]
```

| Argument / Option     | Type   | Default      | Description              |
|-----------------------|--------|--------------|--------------------------|
| `URL`                 | string | *(required)* | URL to audit             |
| `--remote / --local`  | toggle | `--remote`   | Remote or local mode     |
| `--ci`                | flag   | `False`      | CI mode                  |
| `--site-id`           | string | `default`    | Site identifier          |
| `--environment`       | string | `production` | Environment name         |

**Examples:**

```bash
python3 cli.py seo audit https://ppdolphin.brobotinsurance.com
python3 cli.py seo audit https://example.com --local
python3 cli.py seo audit https://example.com --ci
```

**CI output:** JSON with `run_id`, `score`, `grade`, `pass`, `fail`, `warn`

---

### `seo compare`

Compare SEO audit results across environments for a site.

```bash
python3 cli.py seo compare <SITE_NAME>
```

| Argument    | Type   | Default      | Description                             |
|-------------|--------|--------------|-----------------------------------------|
| `SITE_NAME` | string | *(required)* | Site ID to compare across environments  |

**Example:**

```bash
python3 cli.py seo compare brobot
```

---

## Lighthouse

### `lighthouse audit`

Run Lighthouse audit for a URL.

```bash
python3 cli.py lighthouse audit <URL> [OPTIONS]
```

| Argument / Option | Type   | Default                              | Description              |
|-------------------|--------|--------------------------------------|--------------------------|
| `URL`             | string | *(required)*                         | URL to audit             |
| `--categories`    | string | `seo,accessibility,best-practices`   | Comma-separated categories |
| `--ci`            | flag   | `False`                              | CI mode                  |
| `--site-id`       | string | `default`                            | Site identifier          |
| `--environment`   | string | `production`                         | Environment name         |

**Examples:**

```bash
python3 cli.py lighthouse audit https://ppdolphin.brobotinsurance.com
python3 cli.py lighthouse audit https://example.com --categories seo,performance
python3 cli.py lighthouse audit https://example.com --ci
```

**CI output:** JSON with `run_id`, `score`, `performance`, `accessibility`, `best_practices`, `seo`, `failing_audits`

---

### `lighthouse bulk`

Run Lighthouse audit on multiple URLs from a file.

```bash
python3 cli.py lighthouse bulk <URLS_FILE>
```

| Argument    | Type   | Default      | Description            |
|-------------|--------|--------------|------------------------|
| `URLS_FILE` | string | *(required)* | File containing URLs   |

**Example:**

```bash
python3 cli.py lighthouse bulk urls.txt
```

---

## Broken Links

### `links check`

Check for broken links across one or more URLs.

```bash
python3 cli.py links check <URLS_FILE> [OPTIONS]
```

| Argument / Option | Type    | Default      | Description                               |
|-------------------|---------|--------------|-------------------------------------------|
| `URLS_FILE`       | string  | *(required)* | File containing URLs to check             |
| `--deep`          | flag    | `False`      | Deep crawl mode                           |
| `--strict`        | flag    | `False`      | Strict mode (fail on broken)              |
| `--timeout`       | integer | `None`       | Timeout in seconds (auto-scales if not set) |
| `--ci`            | flag    | `False`      | CI mode                                   |

**Examples:**

```bash
python3 cli.py links check urls.txt
python3 cli.py links check urls.txt --deep --strict
python3 cli.py links check urls.txt --timeout 120 --ci
```

**CI output:** JSON with `run_id`, `score`, `ok`, `broken`, `redirected`, `timeout`

---

## Load Testing

### `loadtest run`

Run a load test. Uses the built-in Python HTTP tester by default (no external dependencies). Pass `--jmeter` to use JMeter instead.

```bash
python3 cli.py loadtest run [SCENARIO] [OPTIONS]
```

| Argument / Option | Type    | Default      | Description                                    |
|-------------------|---------|--------------|------------------------------------------------|
| `SCENARIO`        | string  | `smoke`      | Test scenario: `smoke`, `load`, `stress`, `soak`, `spike` |
| `--host`          | string  | `None`       | Target host or URL                             |
| `--users`         | integer | `None`       | Override virtual user count                    |
| `--duration`      | integer | `None`       | Override duration in seconds                   |
| `--jmeter`        | flag    | `False`      | Force JMeter mode (requires JMeter installed)  |
| `--sla-error`     | float   | `2.0`        | Max error rate %                               |
| `--sla-p95`       | integer | `3000`       | Max P95 response time in ms                    |
| `--sla-p99`       | integer | `5000`       | Max P99 response time in ms                    |
| `--sla-throughput` | float  | `10.0`       | Min throughput in req/s                        |
| `--ci`            | flag    | `False`      | CI mode                                        |
| `--site-id`       | string  | `default`    | Site identifier                                |
| `--environment`   | string  | `production` | Environment name                               |

**Scenario defaults:**

| Scenario | Users | Duration | Ramp-up | Think Time   |
|----------|-------|----------|---------|--------------|
| smoke    | 5     | 60s      | 30s     | 2--3s        |
| load     | 50    | 300s     | 120s    | 1--3s        |
| stress   | 200   | 300s     | 60s     | 0.5--1.5s    |
| soak     | 30    | 600s     | 60s     | 2--4s        |
| spike    | 150   | 120s     | 10s     | 0.2--0.5s    |

**SLA thresholds (defaults, override with flags):**

| Metric       | Default          | Flag               |
|--------------|------------------|--------------------|
| Error rate   | <= 2%            | `--sla-error`      |
| P95 response | <= 3000ms        | `--sla-p95`        |
| P99 response | <= 5000ms        | `--sla-p99`        |
| Throughput   | >= 10 req/s      | `--sla-throughput`  |

**Examples:**

```bash
# Smoke test (5 users, 60s)
python3 cli.py loadtest run smoke --host ppdolphin.brobotinsurance.com

# Quick test with overrides
python3 cli.py loadtest run smoke --host ppdolphin.brobotinsurance.com --users 2 --duration 10

# Full load test
python3 cli.py loadtest run load --host ppdolphin.brobotinsurance.com

# Stress test
python3 cli.py loadtest run stress --host ppdolphin.brobotinsurance.com

# Soak test (long-running)
python3 cli.py loadtest run soak --host ppdolphin.brobotinsurance.com

# Spike test
python3 cli.py loadtest run spike --host ppdolphin.brobotinsurance.com

# Custom SLA thresholds
python3 cli.py loadtest run smoke --host ppdolphin.brobotinsurance.com --sla-error 5 --sla-p95 5000 --sla-throughput 5

# CI mode (JSON output, exit code 1 on SLA fail)
python3 cli.py loadtest run smoke --host ppdolphin.brobotinsurance.com --ci

# JMeter mode (requires JMeter installed)
python3 cli.py loadtest run smoke --host ppdolphin.brobotinsurance.com --jmeter
```

**CI output:** JSON with `run_id`, `score`, `scenario`, `total_requests`, `error_rate`, `throughput`, `p95_ms`, `sla`

---

## Reports

### `report list`

List recent check runs.

```bash
python3 cli.py report list [OPTIONS]
```

| Option   | Type   | Default | Description                                           |
|----------|--------|---------|-------------------------------------------------------|
| `--type` | string | `None`  | Filter by type: `html`, `seo`, `lighthouse`, `links`, `loadtest` |

**Examples:**

```bash
python3 cli.py report list
python3 cli.py report list --type loadtest
python3 cli.py report list --type seo
```

---

### `report view`

View details of a specific check run.

```bash
python3 cli.py report view <REPORT_ID>
```

| Argument    | Type    | Default      | Description        |
|-------------|---------|-------------|--------------------|
| `REPORT_ID` | integer | *(required)* | Run ID to view     |

**Example:**

```bash
python3 cli.py report view 42
```

---

## Composite Score

### `score`

Show the composite health score for a site, combining all check types.

```bash
python3 cli.py score <SITE_NAME> [OPTIONS]
```

| Argument / Option | Type   | Default      | Description                       |
|-------------------|--------|--------------|-----------------------------------|
| `SITE_NAME`       | string | *(required)* | Site ID                           |
| `--environment`   | string | `""`         | Environment (default: first configured) |

**Example:**

```bash
python3 cli.py score brobot
python3 cli.py score brobot --environment staging
```

---

## Dashboard

### `dashboard`

Start the Hygiene Check Dashboard web server.

```bash
python3 cli.py dashboard [OPTIONS]
```

| Option     | Type    | Default | Description         |
|------------|---------|---------|---------------------|
| `--port`   | integer | `8082`  | Port to bind to     |
| `--reload` | flag    | `False` | Enable auto-reload  |

**Example:**

```bash
python3 cli.py dashboard
python3 cli.py dashboard --port 9090 --reload
```

Starts the web UI at `http://localhost:{port}/dashboard` with API docs at `http://localhost:{port}/docs`.

---

## Common Options

These options are shared across most check commands:

| Option          | Type   | Default      | Description          |
|-----------------|--------|--------------|----------------------|
| `--ci`          | flag   | `False`      | CI mode: JSON output + non-zero exit codes on failure |
| `--site-id`     | string | `default`    | Site identifier for tracking |
| `--environment` | string | `production` | Environment name     |

---

## CI Mode

When `--ci` is passed, commands output structured JSON and use exit codes for CI/CD integration:

| Exit Code | Meaning                                        |
|-----------|------------------------------------------------|
| `0`       | Success / all checks passed                    |
| `1`       | Failures detected (HTML errors, SLA breach, broken links, etc.) |

**Example CI pipeline usage:**

```bash
# Fail the build if HTML has errors
python3 cli.py html validate https://staging.example.com --ci || exit 1

# Fail on broken links
python3 cli.py links check urls.txt --ci --strict || exit 1

# Fail on load test SLA breach
python3 cli.py loadtest run smoke --host staging.example.com --ci || exit 1
```

---

## Configuration

**Config file:** `config/sites.yaml`
**Database:** `hygiene_checks.db` (SQLite, auto-created)

---

## Scoring Weights

The composite health score is a weighted average of individual check scores:

| Check Type       | Weight |
|------------------|--------|
| HTML Validation  | 0.15   |
| SEO Audit        | 0.30   |
| Lighthouse       | 0.25   |
| Broken Links     | 0.15   |
| Load Test        | 0.15   |

Score color thresholds: green >= 80, yellow >= 60, red < 60.
