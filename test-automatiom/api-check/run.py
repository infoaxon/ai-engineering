#!/usr/bin/env python3
"""Application runner for API Health Check."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="API Health Check Dashboard")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Port to bind to (default: 8081)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("  API Health Check Dashboard")
    print(f"{'='*50}")
    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  Dashboard: http://localhost:{args.port}/dashboard")
    print(f"  API Docs: http://localhost:{args.port}/docs")
    print(f"{'='*50}\n")

    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
