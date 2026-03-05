# PII Remover Proxy

A lightweight, highly scalable, stateless reverse proxy (AI Gateway) designed to anonymize Personally Identifiable Information (PII) in requests to LLM providers and de-anonymize it in responses.

## Features

- Multi-Provider Support: Supports OpenAI, Anthropic, and Google Gemini API formats.
- Robust Anonymization: Uses Microsoft Presidio and spaCy for Named Entity Recognition (NER) and Regex for structured data.
- Ephemeral State: Request-scoped mapping storage with TTL-based cleanup to prevent memory leaks.
- Streaming Support: Handles Server-Sent Events (SSE) with a lookahead buffer to correctly de-anonymize fragmented chunks.
- Security-First: Non-root Docker execution, Bearer token authentication, and sanitized logging.

## Core Architecture

- api/: FastAPI routes and middleware for authentication and request tracking.
- core/: Configuration management and sanitized logging.
- engine/: Anonymization logic using NLP and Regex.
- proxy/: HTTP client management and SSE stream handling.

## Quick Start (Recommended)

For the easiest setup, use the interactive configuration script. It will auto-detect your hardware (CPU/GPU), help you configure your API keys, and set up your environment automatically.

```bash
./scripts/interactive_setup.sh
```

## Installation

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)
- Pytorch compatible GPU (optional, for 20x faster inference)

### Manual Setup

If you prefer not to use the interactive script:

1. **Environment**: 
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and USE_GPU preference
   ```

2. **Python Setup**:
   The automated setup script handles `torch` versions (CPU vs GPU) and pre-downloads the NuNER model weights.
   ```bash
   uv venv .venv
   .venv/bin/python3 scripts/setup.py
   ```

## Docker Deployment

We recommend using **Docker Compose** for the easiest deployment. It automatically reads your `.env` (configured via `interactive_setup.sh`) to decide whether to build a CPU or GPU-accelerated version.

```bash
# Build and start the container
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the proxy
docker compose down
```

> [!NOTE]
> By default, GPU support is commented out in `docker-compose.yml` to prevent errors on machines without the **NVIDIA Container Toolkit**. 
> To enable GPU acceleration in Docker, ensure the toolkit is installed on your host, then uncomment the `deploy:` section in `docker-compose.yml`.

## Usage Example

Once the proxy is running (default `http://localhost:8000`), you can route LLM requests through it.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "My name is John Doe and my email is john@example.com."}],
    "stream": false
  }'
```
