from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routes import router
from src.api.middleware import AuthMiddleware, RequestIDMiddleware
from src.core.config import settings
from src.core.logger import logger
from src.proxy.client import ProxyClient


from src.proxy.state import state_manager
import asyncio


# Global variable to hold the cleanup task so it can be cancelled on shutdown
cleanup_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting PII Remover Proxy...")
    # Initialize singleton client
    ProxyClient.get_client()
    
    # Start periodic cleanup task
    async def periodic_cleanup_task():
        while True:
            await asyncio.sleep(600)  # Clean up every 10 minutes
            try:
                state_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during state cleanup: {e}")

    global cleanup_task
    cleanup_task = asyncio.create_task(periodic_cleanup_task())
    
    yield
    # Shutdown
    logger.info("Shutting down PII Remover Proxy...")
    if cleanup_task:
        cleanup_task.cancel()
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

@app.get("/health/gpu")
async def health_check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "status": "healthy",
                "gpu": True,
                "device": torch.cuda.get_device_name(0)
            }
        else:
            return {"status": "degraded", "gpu": False, "reason": "CUDA not available"}
    except ImportError:
        return {"status": "degraded", "gpu": False, "reason": "PyTorch not installed"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
