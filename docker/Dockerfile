# ARG for base image selection
# CPU base: python:3.13-slim
# GPU base: nvidia/cuda:13.0.1-runtime-ubuntu24.04
# ---
# NOTE: This Dockerfile is now BuildKit-agnostic for maximum compatibility.
# If you have BuildKit, it will use it. If not, it will fall back to standard layers.
# Run with: docker compose up -d --build
# ---
ARG BASE_IMAGE=python:3.13-slim
# New ARG for GPU toggle
ARG USE_GPU=false

# Stage 1: Builder
FROM ${BASE_IMAGE} AS builder
ARG USE_GPU=false

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

WORKDIR /app

# Enable bytecode compilation and optimization
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies into a virtual environment
# We use --index-url for optionally downloading CUDA-enabled PyTorch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    /uv/bin/uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    if [ "$USE_GPU" = "true" ]; then \
        /uv/bin/uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt gliner uvicorn fastapi pydantic-settings python-dotenv; \
    else \
        /uv/bin/uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt gliner uvicorn fastapi pydantic-settings python-dotenv; \
    fi

# Stage 2: Final Runtime
FROM ${BASE_IMAGE} AS final

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy source code
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser .env.example /app/.env.example

# Environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Change ownership of the home directory for the appuser to allow cache writes (via volume)
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
