"""Structured logging setup with GCP Cloud Logging integration."""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings

logger = logging.getLogger("src")


def setup_logging() -> None:
    """Configure logging: Cloud Logging on GCP, structured JSON locally."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.is_gcp:
        try:
            import google.cloud.logging

            client = google.cloud.logging.Client()
            client.setup_logging(log_level=log_level)
            logger.info("Cloud Logging initialized")
            return
        except Exception:
            logger.warning("Failed to init Cloud Logging, falling back to stdout")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with latency and status code."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
