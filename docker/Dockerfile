# ARG for base image selection
# CPU base: python:3.12-slim
# GPU base: nvidia/cuda:12.1.1-runtime-ubuntu22.04
ARG BASE_IMAGE=python:3.12-slim
# New ARG for GPU toggle
ARG USE_GPU=false

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

ARG USE_GPU=false

WORKDIR /app

# Install build dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install uv
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir uv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install dependencies into venv using uv for speed and better resolution
# We combine everything into a single install to ensure gliner/transformers resolve to the correct torch version
# We use --index-strategy unsafe-best-match because some packages exist on both indices with different versions
RUN if [ "$USE_GPU" = "true" ]; then \
        uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements.txt torch torchvision torchaudio huggingface-hub; \
    else \
        uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt torch torchvision torchaudio huggingface-hub && \
        # Explicit clean up of any accidental nvidia packages that might have been pulled by default resolution
        uv pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 triton || true; \
    fi

# NuNER model download (handled via setup script if run manually, or here for docker)
# Just creating the directory as the engine will download it on runtime if path doesn't exist
RUN mkdir -p /app/models

# Stage 2: Final Runtime
FROM python-base AS final

# Create a non-root user early
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy virtual environment and model cache from builder using --chown to avoid slow recursive chown layer
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/models /app/models

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

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
