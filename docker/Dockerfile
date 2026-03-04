# ARG for base image selection
# CPU base: python:3.12-slim
# GPU base: nvidia/cuda:12.1.1-runtime-ubuntu22.04
ARG BASE_IMAGE=python:3.12-slim

# Stage 0: Common Python Setup
FROM ${BASE_IMAGE} AS python-base
USER root
RUN if command -v python3 &> /dev/null && python3 --version | grep -q "3.12"; then \
        echo "Python 3.12 already present"; \
    else \
        export DEBIAN_FRONTEND=noninteractive && \
        apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent wget ca-certificates \
        && add-apt-repository ppa:deadsnakes/ppa -y \
        && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        && rm -rf /var/lib/apt/lists/* \
        && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
        && ln -sf /usr/bin/python3.12 /usr/bin/python \
        && wget https://bootstrap.pypa.io/get-pip.py \
        && python3.12 get-pip.py \
        && rm get-pip.py; \
    fi

# Stage 1: Build & Model Download
FROM python-base AS builder

WORKDIR /app

# Install build dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install dependencies into venv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NuNER model weights (cached in /root/.cache/huggingface)
RUN python3 -c "from gliner import GLiNER; GLiNER.from_pretrained('numind/NuNER_Zero')"

# Stage 2: Final Runtime
FROM python-base AS final

# Create a non-root user early
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy virtual environment and model cache from builder using --chown to avoid slow recursive chown layer
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy source code
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser .env.example /app/.env.example

# Switch to non-root user
USER appuser

# Environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.12/site-packages:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
