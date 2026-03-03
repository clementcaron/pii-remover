from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routes import router
from src.api.middleware import AuthMiddleware, RequestIDMiddleware
from src.core.config import settings
from src.core.logger import logger
from src.proxy.client import ProxyClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting PII Remover Proxy...")
    # Initialize singleton client
    ProxyClient.get_client()
    yield
    # Shutdown
    logger.info("Shutting down PII Remover Proxy...")
    await ProxyClient.close_client()


app = FastAPI(
    title="PII Remover Proxy",
    description="Secure AI Anonymization Proxy (AI Gateway)",
    version="0.1.0",
    lifespan=lifespan
)

# Add Middleware (Order matters: RequestID first to be available for others)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(AuthMiddleware)

# Include Routes
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
