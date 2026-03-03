from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.core.config import settings
from src.core.logger import logger


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces Bearer token authentication.
    Valdiates against PROXY_AUTH_TOKEN in configuration.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health checks if any
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authentication token."}
            )
        
        token = auth_header.split(" ")[1]
        if token != settings.PROXY_AUTH_TOKEN:
            logger.warning(f"Unauthorized access attempt from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized."}
            )

        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects a unique X-Request-ID into each request
    for tracing purposes and state mapping.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Store in request state for easy access in routes
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        # Ensure the header is present in the response
        response.headers["X-Request-ID"] = request_id
        return response
